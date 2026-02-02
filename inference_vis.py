"""
FlowChroma Inference Script for Dataset Batch Processing

Colorize grayscale video sequences using the complete FlowChroma architecture:
- MemFlow: Temporal flow-based colorization (frozen)
- SwinTExCo: Reference-based single-frame colorization (fine-tuned)
- FusionNet: Multi-scale fusion network

Usage:
    python inference.py \
        --memflow_path MemFlow \
        --swintexco_path SwinSingle \
        --memflow_ckpt MemFlow/ckpt/memflow_colorization.pth \
        --swintexco_ckpt SwinSingle/ckpt/epoch_1 \
        --fusion_ckpt checkpoints/fusion_best.pth \
        --input_dirs /path/to/dataset1,/path/to/dataset2 \
        --output_dir /path/to/output \
        --target_size 224 224
"""

import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from tqdm import tqdm
import glob
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, '.')

from train.fusion_system import FusionSystem
from FusionNet.fusion_unet import FusionNetV1


def visualize_heatmap(tensor, colormap='turbo'):
    """
    Convert confidence/similarity map to colored heatmap

    Args:
        tensor: [1, 1, H, W] or [1, H, W] tensor, range [0, 1]
        colormap: 'turbo', 'viridis', 'jet', 'hot'

    Returns:
        PIL Image (RGB) - colored heatmap
    """
    # Extract data and move to CPU
    if tensor.ndim == 4:
        data = tensor.squeeze().cpu().numpy()  # [H, W]
    elif tensor.ndim == 3:
        data = tensor.squeeze(0).cpu().numpy()  # [H, W]
    else:
        data = tensor.cpu().numpy()

    # Ensure range [0, 1]
    data = np.clip(data, 0, 1)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(data)[:, :, :3]  # [H, W, 3] RGB (drop alpha)

    # Convert to uint8
    rgb_uint8 = (colored * 255).astype(np.uint8)

    return Image.fromarray(rgb_uint8)


def load_checkpoint(system, checkpoint_path, device):
    """Load trained FusionNet checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("   Using pre-trained MemFlow & SwinTExCo, untrained FusionNet")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load FusionNet weights
    if 'fusion_unet' in checkpoint:
        system.fusion_unet.load_state_dict(checkpoint['fusion_unet'])
        print("✅ FusionNet weights loaded")

    # Load fine-tuned SwinTExCo weights (if available)
    if 'swintexco_embed' in checkpoint:
        system.swintexco.embed_net.load_state_dict(checkpoint['swintexco_embed'])
        system.swintexco.nonlocal_net.load_state_dict(checkpoint['swintexco_nonlocal'])
        system.swintexco.colornet.load_state_dict(checkpoint['swintexco_colornet'])
        print("✅ Fine-tuned SwinTExCo weights loaded")

    epoch = checkpoint.get('epoch', 'unknown')
    best_loss = checkpoint.get('best_loss', 'unknown')
    print(f"   Epoch: {epoch}, Best Loss: {best_loss}")


def rgb_to_lab_tensor(pil_image, target_size=(224, 224)):
    """
    Convert PIL RGB image to LAB tensor [3, H, W]

    Args:
        pil_image: PIL Image (RGB)
        target_size: (H, W) tuple

    Returns:
        lab_tensor: [3, H, W] normalized to [-1, 1]
    """
    # Resize
    img_resized = pil_image.resize(target_size[::-1], Image.LANCZOS)  # PIL uses (W, H)

    # Convert to numpy array
    img_np = np.array(img_resized, dtype=np.uint8)

    # RGB to LAB
    lab_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Normalize
    lab_np[:, :, 0] = lab_np[:, :, 0] * 100.0 / 255.0  # L: [0, 100]
    lab_np[:, :, 1] = lab_np[:, :, 1] - 128.0          # a: [-128, 127]
    lab_np[:, :, 2] = lab_np[:, :, 2] - 128.0          # b: [-128, 127]

    # Further normalize to [-1, 1]
    lab_np[:, :, 0] = (lab_np[:, :, 0] / 50.0) - 1.0   # L: [-1, 1]
    lab_np[:, :, 1] = lab_np[:, :, 1] / 127.0          # a: [-1, 1]
    lab_np[:, :, 2] = lab_np[:, :, 2] / 127.0          # b: [-1, 1]

    # To tensor [3, H, W]
    lab_tensor = torch.from_numpy(lab_np).permute(2, 0, 1).float()

    return lab_tensor


def lab_tensor_to_rgb(lab_tensor):
    """
    Convert LAB tensor to RGB PIL Image

    Args:
        lab_tensor: [3, H, W] tensor

    Returns:
        PIL Image (RGB)
    """
    # To numpy [H, W, 3]
    lab_np = lab_tensor.detach().cpu().numpy().transpose(1, 2, 0)

    # Denormalize
    lab_np[:, :, 0] = (lab_np[:, :, 0] + 1.0) * 50.0  # L: [-1, 1] -> [0, 100]
    lab_np[:, :, 1] = lab_np[:, :, 1] * 127.0         # a: [-1, 1] -> [-127, 127]
    lab_np[:, :, 2] = lab_np[:, :, 2] * 127.0         # b: [-1, 1] -> [-127, 127]

    # Convert to OpenCV LAB format
    lab_cv = lab_np.copy()
    lab_cv[:, :, 0] = lab_np[:, :, 0] * 255.0 / 100.0  # L
    lab_cv[:, :, 1] = lab_np[:, :, 1] + 128.0          # a
    lab_cv[:, :, 2] = lab_np[:, :, 2] + 128.0          # b

    lab_cv = np.clip(lab_cv, 0, 255).astype(np.uint8)

    # LAB to RGB
    bgr_np = cv2.cvtColor(lab_cv, cv2.COLOR_LAB2BGR)
    rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)

    return Image.fromarray(rgb_np)


def process_scene(system, scene_path, output_scene_path, target_size=(224, 224),
                  debug_output_path=None, save_components=None, colormap='turbo', save_npy=False):
    """
    Process a single scene directory

    Args:
        system: FusionSystem
        scene_path: Path to scene directory containing frames
        output_scene_path: Path to output directory for this scene
        target_size: (H, W) tuple for resizing
        debug_output_path: Optional path to save intermediate results
        save_components: List of components to save ('memflow', 'swintexco', 'confidence', 'similarity')
        colormap: Colormap for heatmaps ('turbo', 'viridis', 'jet', 'hot')
        save_npy: Whether to save raw .npy files for confidence/similarity
    """
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    frame_files = []
    for ext in image_extensions:
        frame_files.extend(glob.glob(os.path.join(scene_path, ext)))
        frame_files.extend(glob.glob(os.path.join(scene_path, ext.upper())))

    frame_files = sorted(frame_files)

    if len(frame_files) < 1:
        print(f"  ⚠️  Skipping: no frames found")
        return

    print(f"  📹 Found {len(frame_files)} frames")

    # Load all frames as PIL Images
    print(f"  📥 Loading frames...")
    frames_pil = []
    for frame_path in tqdm(frame_files, desc="  Loading", leave=False):
        img = Image.open(frame_path).convert('RGB')
        img_resized = img.resize(target_size[::-1], Image.LANCZOS)  # PIL uses (W, H)
        frames_pil.append(img_resized)

    # First frame as reference
    reference_pil = frames_pil[0]
    print(f"  🎨 Using first frame as reference")

    # Convert frames to LAB tensors
    frames_lab = []
    for frame_pil in frames_pil:
        lab_tensor = rgb_to_lab_tensor(frame_pil, target_size)
        frames_lab.append(lab_tensor)

    # Reset memory for new video sequence
    system.reset_memory()

    # Create debug output directories if needed
    if debug_output_path and save_components:
        scene_name = os.path.basename(scene_path)
        debug_dirs = {}
        if 'memflow' in save_components:
            debug_dirs['memflow'] = os.path.join(debug_output_path, scene_name, 'memflow')
            os.makedirs(debug_dirs['memflow'], exist_ok=True)
        if 'swintexco' in save_components:
            debug_dirs['swintexco'] = os.path.join(debug_output_path, scene_name, 'swintexco')
            os.makedirs(debug_dirs['swintexco'], exist_ok=True)
        if 'confidence' in save_components:
            debug_dirs['confidence'] = os.path.join(debug_output_path, scene_name, 'confidence')
            os.makedirs(debug_dirs['confidence'], exist_ok=True)
            if save_npy:
                debug_dirs['confidence_npy'] = os.path.join(debug_output_path, scene_name, 'confidence_npy')
                os.makedirs(debug_dirs['confidence_npy'], exist_ok=True)
        if 'similarity' in save_components:
            debug_dirs['similarity'] = os.path.join(debug_output_path, scene_name, 'similarity')
            os.makedirs(debug_dirs['similarity'], exist_ok=True)
            if save_npy:
                debug_dirs['similarity_npy'] = os.path.join(debug_output_path, scene_name, 'similarity_npy')
                os.makedirs(debug_dirs['similarity_npy'], exist_ok=True)
    else:
        debug_dirs = None

    # Pre-compute reference features ONCE (SwinTExCo optimization)
    print(f"  🎨 Pre-computing reference features...")
    cached_ref_features = None
    with torch.no_grad():
        from torch.cuda.amp import autocast
        from src.utils import tensor_lab2rgb, uncenter_l, uncenter_ab

        with autocast(enabled=False):
            # Process reference image once
            ref_lab = system.swintexco.processor(reference_pil).unsqueeze(0).to(system.device)
            ref_l = ref_lab[:, 0:1, :, :]
            ref_ab = ref_lab[:, 1:3, :, :]
            ref_rgb = tensor_lab2rgb(torch.cat([uncenter_l(ref_l), uncenter_ab(ref_ab)], dim=1))

            # Extract features once (cache for reuse)
            features_B = system.swintexco.embed_net(ref_rgb)

            # Cache for reuse
            cached_ref_features = (ref_lab, features_B)

    # Process frames sequentially
    print(f"  🎬 Processing {len(frames_lab)} frames...")
    colorized_frames = []

    with torch.no_grad():
        for i in tqdm(range(len(frames_lab)), desc="  Colorizing", leave=False):
            # Add batch dimension [1, 3, H, W]
            frame_t1_batch = frames_lab[i].unsqueeze(0).to(system.device)

            if i == 0:
                # First frame: no previous frame
                results = system.forward_single_frame(
                    None,
                    frame_t1_batch,
                    reference_pil,
                    frames_pil[i],
                    is_first=True,
                    cached_ref_features=cached_ref_features,  # Use cached features
                    return_intermediates=(debug_dirs is not None)  # Return intermediates if debugging
                )
            else:
                # Subsequent frames: use PREVIOUS PREDICTION (not GT)
                # This enables error accumulation (like real inference should be)

                # Get previous frame's prediction (already in LAB format)
                prev_output_lab = colorized_frames[-1]  # This is PIL RGB

                # Convert RGB back to LAB tensor
                prev_output_lab_tensor = rgb_to_lab_tensor(prev_output_lab, target_size)
                frame_t_batch = prev_output_lab_tensor.unsqueeze(0).to(system.device)

                results = system.forward_single_frame(
                    frame_t_batch,
                    frame_t1_batch,
                    reference_pil,
                    frames_pil[i],
                    is_first=False,
                    cached_ref_features=cached_ref_features,  # Use cached features
                    return_intermediates=(debug_dirs is not None)  # Return intermediates if debugging
                )

            # Extract output based on return type
            if isinstance(results, dict):
                # Debugging mode: save intermediate results
                output_lab = results['fused']

                # Get frame name (preserve original filename)
                frame_name = os.path.basename(frame_files[i])

                # Save intermediate results (frame-by-frame to avoid memory accumulation)
                if 'memflow' in save_components:
                    memflow_rgb = lab_tensor_to_rgb(results['memflow'].squeeze(0))
                    memflow_rgb.save(os.path.join(debug_dirs['memflow'], frame_name))

                if 'swintexco' in save_components:
                    swintexco_rgb = lab_tensor_to_rgb(results['swintexco'].squeeze(0))
                    swintexco_rgb.save(os.path.join(debug_dirs['swintexco'], frame_name))

                if 'confidence' in save_components:
                    conf_heatmap = visualize_heatmap(results['memflow_conf'], colormap=colormap)
                    conf_heatmap.save(os.path.join(debug_dirs['confidence'], frame_name))
                    if save_npy:
                        conf_npy_path = os.path.join(debug_dirs['confidence_npy'],
                                                     os.path.splitext(frame_name)[0] + '.npy')
                        np.save(conf_npy_path, results['memflow_conf'].cpu().numpy())

                if 'similarity' in save_components:
                    sim_heatmap = visualize_heatmap(results['swintexco_sim'], colormap=colormap)
                    sim_heatmap.save(os.path.join(debug_dirs['similarity'], frame_name))
                    if save_npy:
                        sim_npy_path = os.path.join(debug_dirs['similarity_npy'],
                                                    os.path.splitext(frame_name)[0] + '.npy')
                        np.save(sim_npy_path, results['swintexco_sim'].cpu().numpy())
            else:
                # Normal mode: only final result
                output_lab = results

            # Convert to RGB and remove batch dimension
            output_rgb = lab_tensor_to_rgb(output_lab.squeeze(0))
            colorized_frames.append(output_rgb)

    # Save results with original filenames
    os.makedirs(output_scene_path, exist_ok=True)
    print(f"  💾 Saving colorized frames...")

    for original_path, colorized_frame in zip(frame_files, colorized_frames):
        # Preserve original filename
        frame_name = os.path.basename(original_path)
        output_path = os.path.join(output_scene_path, frame_name)
        colorized_frame.save(output_path)

    print(f"  ✅ Saved {len(colorized_frames)} frames to {output_scene_path}")


def process_datasets(system, input_dirs, output_dir, target_size=(224, 224),
                     debug_output=None, save_components=None, colormap='turbo', save_npy=False):
    """
    Process multiple datasets with scene directories

    Args:
        system: FusionSystem
        input_dirs: List of dataset root directories
        output_dir: Root output directory
        target_size: (H, W) tuple
        debug_output: Optional path to save intermediate results
        save_components: List of components to save
        colormap: Colormap for heatmaps
        save_npy: Whether to save raw .npy files
    """
    print("\n" + "="*80)
    print(f"📂 Scanning {len(input_dirs)} dataset(s)...")
    print("="*80)

    # Collect all scene directories from all input paths
    all_scenes = []
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"⚠️  Warning: directory not found: {input_dir}")
            continue

        print(f"\n📁 Scanning: {input_dir}")
        scene_count = 0
        for item in sorted(os.listdir(input_dir)):
            item_path = os.path.join(input_dir, item)
            if os.path.isdir(item_path):
                all_scenes.append((item, item_path))
                scene_count += 1
        print(f"   Found {scene_count} scenes")

    if len(all_scenes) == 0:
        print("❌ No scene directories found!")
        return

    print(f"\n📊 Total: {len(all_scenes)} scenes to process\n")

    # Process each scene
    for scene_idx, (scene_name, scene_path) in enumerate(all_scenes, 1):
        print(f"🎬 [{scene_idx}/{len(all_scenes)}] Processing: {scene_name}")

        output_scene_path = os.path.join(output_dir, scene_name)

        try:
            process_scene(
                system,
                scene_path,
                output_scene_path,
                target_size=target_size,
                debug_output_path=debug_output,
                save_components=save_components,
                colormap=colormap,
                save_npy=save_npy
            )
        except Exception as e:
            print(f"  ❌ Error processing {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        print()  # Empty line between scenes

    print("="*80)
    print(f"✅ All datasets processed!")
    print(f"📁 Results saved to: {output_dir}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='FlowChroma Dataset Batch Inference')

    # Model paths
    parser.add_argument('--memflow_path', type=str, required=True,
                        help='Path to MemFlow repository')
    parser.add_argument('--swintexco_path', type=str, required=True,
                        help='Path to SwinSingle repository')
    parser.add_argument('--memflow_ckpt', type=str, required=True,
                        help='Path to MemFlow checkpoint')
    parser.add_argument('--swintexco_ckpt', type=str, required=True,
                        help='Path to SwinTExCo checkpoint directory')
    parser.add_argument('--fusion_ckpt', type=str, required=True,
                        help='Path to trained fusion checkpoint')

    # Input/Output
    parser.add_argument('--input_dirs', type=str, required=True,
                        help='Comma-separated dataset root directories (e.g., /path1,/path2)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output root directory')

    # Processing
    parser.add_argument('--target_size', type=int, nargs=2, default=[224, 224],
                        help='Target frame size (H W), default: 224 224')

    # Debug output
    parser.add_argument('--debug_output', type=str, default=None,
                        help='Path to save intermediate results (optional)')
    parser.add_argument('--save_components', nargs='+',
                        default=None,
                        choices=['memflow', 'swintexco', 'confidence', 'similarity'],
                        help='Which components to save (default: None, save all if debug_output is set)')
    parser.add_argument('--heatmap_colormap', type=str, default='turbo',
                        choices=['turbo', 'viridis', 'jet', 'hot'],
                        help='Colormap for confidence/similarity heatmaps (default: turbo)')
    parser.add_argument('--save_npy', action='store_true',
                        help='Save raw .npy files for confidence/similarity maps')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Set default save_components if debug_output is provided
    if args.debug_output and args.save_components is None:
        args.save_components = ['memflow', 'swintexco', 'confidence', 'similarity']

    print("="*80)
    print(" FlowChroma Dataset Batch Inference".center(80))
    print("="*80)

    # Parse input directories
    input_dirs = [d.strip() for d in args.input_dirs.split(',')]
    print(f"\nInput directories: {len(input_dirs)}")
    for i, d in enumerate(input_dirs, 1):
        print(f"  [{i}] {d}")

    # Initialize system
    print("\nInitializing FlowChroma system...")
    system = FusionSystem(
        memflow_path=args.memflow_path,
        swintexco_path=args.swintexco_path,
        memflow_ckpt=args.memflow_ckpt,
        swintexco_ckpt=args.swintexco_ckpt,
        fusion_net=FusionNetV1(),
        device=args.device
    )
    system.eval()

    # Load trained checkpoint
    load_checkpoint(system, args.fusion_ckpt, args.device)

    # Process datasets
    target_size = tuple(args.target_size)
    process_datasets(
        system=system,
        input_dirs=input_dirs,
        output_dir=args.output_dir,
        target_size=target_size,
        debug_output=args.debug_output,
        save_components=args.save_components,
        colormap=args.heatmap_colormap,
        save_npy=args.save_npy
    )

    print("\n" + "="*80)
    print("✅ Inference completed!")
    print("="*80)


if __name__ == '__main__':
    main()

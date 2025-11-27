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

sys.path.insert(0, '.')

from train.fusion_system import FusionSystem
from FusionNet.fusion_unet import FusionNetV1


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


def process_scene(system, scene_path, output_scene_path, target_size=(224, 224)):
    """
    Process a single scene directory

    Args:
        system: FusionSystem
        scene_path: Path to scene directory containing frames
        output_scene_path: Path to output directory for this scene
        target_size: (H, W) tuple for resizing
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

    # Process frames sequentially
    print(f"  🎬 Processing {len(frames_lab)} frames...")
    colorized_frames = []

    with torch.no_grad():
        for i in tqdm(range(len(frames_lab)), desc="  Colorizing", leave=False):
            # Add batch dimension [1, 3, H, W]
            frame_t1_batch = frames_lab[i].unsqueeze(0).to(system.device)

            if i == 0:
                # First frame: no previous frame
                output_lab = system.forward_single_frame(
                    None,
                    frame_t1_batch,
                    reference_pil,
                    frames_pil[i],
                    is_first=True
                )
            else:
                # Subsequent frames: use PREVIOUS PREDICTION (not GT)
                # This enables error accumulation (like real inference should be)

                # Get previous frame's prediction (already in LAB format)
                prev_output_lab = colorized_frames[-1]  # This is PIL RGB

                # Convert RGB back to LAB tensor
                prev_output_lab_tensor = rgb_to_lab_tensor(prev_output_lab, target_size)
                frame_t_batch = prev_output_lab_tensor.unsqueeze(0).to(system.device)

                output_lab = system.forward_single_frame(
                    frame_t_batch,
                    frame_t1_batch,
                    reference_pil,
                    frames_pil[i],
                    is_first=False
                )

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


def process_datasets(system, input_dirs, output_dir, target_size=(224, 224)):
    """
    Process multiple datasets with scene directories

    Args:
        system: FusionSystem
        input_dirs: List of dataset root directories
        output_dir: Root output directory
        target_size: (H, W) tuple
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
                target_size=target_size
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

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

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
        target_size=target_size
    )

    print("\n" + "="*80)
    print("✅ Inference completed!")
    print("="*80)


if __name__ == '__main__':
    main()

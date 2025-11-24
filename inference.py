"""
FlowChroma Inference Script

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
        --fusion_ckpt fusion/checkpoints/fusion_best.pth \
        --input /path/to/grayscale/frames \
        --reference /path/to/reference/image.jpg \
        --output /path/to/output
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
from skimage import color


def load_checkpoint(system, checkpoint_path, device):
    """Load trained checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("   Using pre-trained SwinTExCo and untrained FusionNet")
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


def load_frames(input_path, max_frames=None):
    """
    Load grayscale frames from directory or video file

    Returns:
        List of PIL Images (RGB format, but grayscale content)
    """
    frames = []

    if os.path.isdir(input_path):
        # Load from image directory
        image_files = sorted(glob.glob(os.path.join(input_path, '*.jpg')) +
                           glob.glob(os.path.join(input_path, '*.png')))

        if max_frames:
            image_files = image_files[:max_frames]

        print(f"Loading {len(image_files)} frames from directory...")
        for img_path in tqdm(image_files):
            img = Image.open(img_path).convert('RGB')
            frames.append(img)

    elif os.path.isfile(input_path):
        # Load from video file
        print(f"Loading frames from video: {input_path}")
        cap = cv2.VideoCapture(input_path)
        frame_count = 0

        pbar = tqdm(desc="Loading video frames")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_count >= max_frames:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            frame_count += 1
            pbar.update(1)

        cap.release()
        pbar.close()
        print(f"Loaded {len(frames)} frames from video")

    else:
        raise ValueError(f"Input path not found: {input_path}")

    return frames


def rgb_to_lab_tensor(pil_image, target_size=(224, 224)):
    """Convert PIL RGB image to LAB tensor [3, H, W]"""
    # Resize
    img_resized = pil_image.resize(target_size, Image.LANCZOS)

    # Convert to numpy array
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    # RGB to LAB
    lab = color.rgb2lab(img_np)

    # Normalize
    lab[:, :, 0] = lab[:, :, 0] / 50.0 - 1.0  # L: [0, 100] -> [-1, 1]
    lab[:, :, 1] = lab[:, :, 1] / 127.0       # A: [-127, 127] -> [-1, 1]
    lab[:, :, 2] = lab[:, :, 2] / 127.0       # B: [-127, 127] -> [-1, 1]

    # To tensor [3, H, W]
    lab_tensor = torch.from_numpy(lab.transpose(2, 0, 1)).float()

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
    lab_np[:, :, 1] = lab_np[:, :, 1] * 127.0         # A: [-1, 1] -> [-127, 127]
    lab_np[:, :, 2] = lab_np[:, :, 2] * 127.0         # B: [-1, 1] -> [-127, 127]

    # LAB to RGB
    rgb_np = color.lab2rgb(lab_np)
    rgb_np = (rgb_np * 255).astype(np.uint8)

    return Image.fromarray(rgb_np)


def process_sequence(system, frames_pil, reference_pil, sequence_length=4, target_size=(224, 224)):
    """
    Process frames in sliding window sequences

    Args:
        system: FusionSystem
        frames_pil: List of PIL Images
        reference_pil: PIL Image (reference for all frames)
        sequence_length: Number of frames per sequence
        target_size: (H, W) tuple

    Returns:
        List of colorized PIL Images
    """
    num_frames = len(frames_pil)
    colorized_frames = []

    print(f"\nProcessing {num_frames} frames in sequences of {sequence_length}...")

    # Process frames in overlapping sequences
    for start_idx in tqdm(range(0, num_frames, sequence_length - 1)):
        end_idx = min(start_idx + sequence_length, num_frames)
        seq_frames = frames_pil[start_idx:end_idx]

        # Pad if last sequence is shorter
        if len(seq_frames) < sequence_length:
            # Repeat last frame
            last_frame = seq_frames[-1]
            seq_frames = seq_frames + [last_frame] * (sequence_length - len(seq_frames))

        # Convert to LAB tensors
        frames_lab = []
        frames_pil_seq = []
        references_pil = []

        for frame_pil in seq_frames:
            lab_tensor = rgb_to_lab_tensor(frame_pil, target_size)
            frames_lab.append(lab_tensor)
            frames_pil_seq.append(frame_pil.resize(target_size, Image.LANCZOS))
            references_pil.append(reference_pil.resize(target_size, Image.LANCZOS))

        # Inference
        with torch.no_grad():
            outputs = system.forward_sequence(frames_lab, frames_pil_seq, references_pil)

        # Convert outputs to RGB
        for i, output_lab in enumerate(outputs):
            # Only save non-overlapping frames (except for last sequence)
            frame_idx = start_idx + i
            if frame_idx < num_frames and (frame_idx < start_idx + 1 or start_idx == 0):
                rgb_pil = lab_tensor_to_rgb(output_lab)
                colorized_frames.append((frame_idx, rgb_pil))

        # For overlapping frames (except first sequence), only take the first frame
        if start_idx > 0:
            continue

    # Sort by frame index and return
    colorized_frames.sort(key=lambda x: x[0])
    return [frame for _, frame in colorized_frames]


def save_frames(frames, output_dir, prefix="frame"):
    """Save frames to directory"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving {len(frames)} frames to {output_dir}...")
    for i, frame in enumerate(tqdm(frames)):
        frame_path = os.path.join(output_dir, f"{prefix}_{i:05d}.png")
        frame.save(frame_path)

    print(f"✅ Frames saved to {output_dir}")


def save_video(frames, output_path, fps=30):
    """Save frames as video"""
    if len(frames) == 0:
        print("⚠️  No frames to save")
        return

    # Get frame size
    width, height = frames[0].size

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\nSaving video to {output_path}...")
    for frame in tqdm(frames):
        # Convert PIL to OpenCV format (RGB -> BGR)
        frame_np = np.array(frame)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"✅ Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='FlowChroma Video Colorization Inference')

    # Model paths
    parser.add_argument('--memflow_path', type=str, required=True,
                        help='Path to MemFlow repository')
    parser.add_argument('--swintexco_path', type=str, required=True,
                        help='Path to SwinSingle repository')
    parser.add_argument('--memflow_ckpt', type=str, required=True,
                        help='Path to MemFlow checkpoint')
    parser.add_argument('--swintexco_ckpt', type=str, required=True,
                        help='Path to SwinTExCo checkpoint directory')
    parser.add_argument('--fusion_ckpt', type=str, default=None,
                        help='Path to trained fusion checkpoint (optional)')

    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                        help='Input grayscale video or frames directory')
    parser.add_argument('--reference', type=str, required=True,
                        help='Reference color image (for style)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for colorized frames')

    # Processing
    parser.add_argument('--sequence_length', type=int, default=4,
                        help='Number of frames per sequence (default: 4)')
    parser.add_argument('--target_size', type=int, nargs=2, default=[224, 224],
                        help='Target frame size (H W), default: 224 224')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to process (for testing)')

    # Output format
    parser.add_argument('--save_video', action='store_true',
                        help='Save output as video (in addition to frames)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Output video FPS (if --save_video)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    print("="*80)
    print(" FlowChroma Video Colorization".center(80))
    print("="*80)

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
    if args.fusion_ckpt:
        load_checkpoint(system, args.fusion_ckpt, args.device)
    else:
        print("⚠️  No fusion checkpoint provided, using untrained FusionNet")

    # Load reference image
    print(f"\nLoading reference image: {args.reference}")
    reference_pil = Image.open(args.reference).convert('RGB')
    print(f"✅ Reference image loaded: {reference_pil.size}")

    # Load input frames
    frames_pil = load_frames(args.input, max_frames=args.max_frames)
    print(f"✅ Loaded {len(frames_pil)} frames")

    if len(frames_pil) == 0:
        print("❌ No frames found!")
        return

    # Process sequences
    target_size = tuple(args.target_size)
    colorized_frames = process_sequence(
        system, frames_pil, reference_pil,
        sequence_length=args.sequence_length,
        target_size=target_size
    )

    # Save results
    save_frames(colorized_frames, args.output)

    if args.save_video:
        video_path = os.path.join(args.output, 'colorized_video.mp4')
        save_video(colorized_frames, video_path, fps=args.fps)

    print("\n" + "="*80)
    print("✅ Inference completed!")
    print("="*80)


if __name__ == '__main__':
    main()

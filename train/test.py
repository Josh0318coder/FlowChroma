"""
Test Fusion System

Quick test to verify that the entire system works before training.
Tests 4-frame sequence processing with FusionSequenceDataset.

Usage:
    python train/test.py \
        --memflow_path MemFlow \
        --swintexco_path SwinSingle \
        --memflow_ckpt MemFlow/ckpt/memflow_colorization.pth \
        --swintexco_ckpt SwinSingle/ckpt/epoch_1 \
        --dataset /path/to/dataset \
        --imagenet /path/to/imagenet
"""

import argparse
import sys
import torch
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, '.')

from train.fusion_system import FusionSystem
from FusionNet.fusion_unet import PlaceholderFusion
from train.fusion_dataset import FusionSequenceDataset


def create_dummy_sequence(size=(224, 224)):
    """Create dummy 4-frame sequence for testing"""
    frames_lab = []
    frames_pil = []
    references_pil = []

    for _ in range(4):
        # Create dummy LAB tensor
        lab_tensor = torch.randn(3, *size)
        frames_lab.append(lab_tensor)

        # Create dummy PIL images
        dummy_rgb = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        frames_pil.append(Image.fromarray(dummy_rgb))
        references_pil.append(Image.fromarray(dummy_rgb))

    return frames_lab, frames_pil, references_pil


def test_fusion_system(args):
    """Test the complete fusion system with 4-frame sequences"""

    print("="*80)
    print(" Fusion System Test (4-Frame Sequence)".center(80))
    print("="*80)

    # Initialize system with PlaceholderFusion
    print("\n1. Initializing Fusion System...")
    try:
        system = FusionSystem(
            memflow_path=args.memflow_path,
            swintexco_path=args.swintexco_path,
            memflow_ckpt=args.memflow_ckpt,
            swintexco_ckpt=args.swintexco_ckpt,
            fusion_net=PlaceholderFusion(),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test sequence forward pass
    print("\n2. Testing 4-frame sequence processing...")
    try:
        frames_lab, frames_pil, references_pil = create_dummy_sequence()

        if torch.cuda.is_available():
            frames_lab = [f.cuda() for f in frames_lab]

        with torch.no_grad():
            outputs = system.forward_sequence(frames_lab, frames_pil, references_pil)

        print(f"✅ Sequence processing successful!")
        print(f"   Input: 4 frames")
        print(f"   Output: {len(outputs)} frames")
        for i, output in enumerate(outputs):
            print(f"   Frame {i}: shape={output.shape}, range=[{output.min():.3f}, {output.max():.3f}]")

        # Verify output
        assert len(outputs) == 4, f"Expected 4 outputs, got {len(outputs)}"
        for i, output in enumerate(outputs):
            assert output.shape == (3, 224, 224), f"Frame {i}: expected (3, 224, 224), got {output.shape}"
        print(f"✅ Output shapes verified!")

    except Exception as e:
        print(f"❌ Sequence processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with real dataset if provided
    if args.dataset and args.imagenet:
        print("\n3. Testing with real dataset...")
        try:
            dataset = FusionSequenceDataset(
                davis_root=args.dataset,
                imagenet_root=args.imagenet,
                sequence_length=4,
                target_size=(224, 224)
            )
            print(f"   Dataset loaded: {len(dataset)} sequences")

            # Test one sample
            sample = dataset[0]
            frames_lab = sample['frames_lab']
            frames_pil = sample['frames_pil']
            references_pil = sample['references_pil']

            if torch.cuda.is_available():
                frames_lab = [f.cuda() for f in frames_lab]

            with torch.no_grad():
                outputs = system.forward_sequence(frames_lab, frames_pil, references_pil)

            print(f"✅ Real data test successful!")
            print(f"   Video: {sample['video_name']}")
            print(f"   Processed {len(outputs)} frames")

        except Exception as e:
            print(f"❌ Real data test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\n3. Skipping real dataset test (--dataset and --imagenet not provided)")

    # Test parameter count
    print("\n4. Checking trainable parameters...")
    try:
        total_params = sum(p.numel() for p in system.parameters())
        trainable_params = sum(p.numel() for p in system.parameters() if p.requires_grad)

        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Frozen parameters: {total_params - trainable_params:,}")

        # For PlaceholderFusion, trainable should be 0
        assert trainable_params == 0, f"PlaceholderFusion should have 0 trainable params, got {trainable_params}"
        print(f"✅ Parameter freeze verified!")

    except Exception as e:
        print(f"❌ Parameter check failed: {e}")
        return False

    print("\n" + "="*80)
    print(" ✅ All tests passed! System is ready for training.".center(80))
    print("="*80)

    print("\nNext steps:")
    print("  1. Replace PlaceholderFusion with FusionNetV1")
    print("  2. Run training: python train/train.py")

    return True


def main():
    parser = argparse.ArgumentParser(description='Test Fusion System with 4-Frame Sequences')

    parser.add_argument('--memflow_path', type=str, required=True,
                        help='Path to MemFlow repository')
    parser.add_argument('--swintexco_path', type=str, required=True,
                        help='Path to SwinSingle repository')
    parser.add_argument('--memflow_ckpt', type=str, required=True,
                        help='Path to MemFlow checkpoint')
    parser.add_argument('--swintexco_ckpt', type=str, required=True,
                        help='Path to SwinTExCo checkpoint directory')
    parser.add_argument('--dataset', type=str, default=None,
                        help='(Optional) Dataset path for real data testing')
    parser.add_argument('--imagenet', type=str, default=None,
                        help='(Optional) ImageNet path for real data testing')

    args = parser.parse_args()

    success = test_fusion_system(args)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

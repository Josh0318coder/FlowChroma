"""
Test Fusion System

Quick test to verify that the entire system works before training.
Uses PlaceholderFusion to test data flow.

Usage:
    python fusion/test.py \
        --memflow_path ../MemFlow \
        --swintexco_path ../SwinSingle \
        --memflow_ckpt ../MemFlow/checkpoints/memflow_best.pth \
        --swintexco_ckpt ../SwinSingle/checkpoints/best/
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


def create_dummy_data(size=(224, 224), dtype=torch.float16):
    """Create dummy data for testing"""
    # Create dummy frames as PIL Images
    dummy_rgb = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    reference_pil = Image.fromarray(dummy_rgb)
    target_pil = Image.fromarray(dummy_rgb)

    # Create dummy LAB tensors (fp16 for FlashAttention compatibility)
    frame_t = torch.randn(1, 3, *size, dtype=dtype)
    frame_t1 = torch.randn(1, 3, *size, dtype=dtype)

    return frame_t, frame_t1, reference_pil, target_pil


def test_fusion_system(args):
    """Test the complete fusion system"""

    print("="*80)
    print(" Fusion System Test".center(80))
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
        return False

    # Test forward pass
    print("\n2. Testing forward pass...")
    try:
        frame_t, frame_t1, reference_pil, target_pil = create_dummy_data()

        if torch.cuda.is_available():
            frame_t = frame_t.cuda()
            frame_t1 = frame_t1.cuda()

        with torch.no_grad():
            output = system(frame_t, frame_t1, reference_pil, target_pil)

        print(f"✅ Forward pass successful!")
        print(f"   Input shape: {frame_t.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")

        # Verify output shape
        assert output.shape == (1, 2, 224, 224), f"Expected (1, 2, 224, 224), got {output.shape}"
        print(f"✅ Output shape verified!")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test multiple frames (memory management)
    print("\n3. Testing memory management...")
    try:
        system.reset_memory()

        for i in range(3):
            frame_t, frame_t1, reference_pil, target_pil = create_dummy_data()
            if torch.cuda.is_available():
                frame_t = frame_t.cuda()
                frame_t1 = frame_t1.cuda()

            with torch.no_grad():
                output = system(frame_t, frame_t1, reference_pil, target_pil)

            print(f"   Frame {i+1}: output shape = {output.shape}")

        print(f"✅ Memory management working!")

    except Exception as e:
        print(f"❌ Memory management failed: {e}")
        import traceback
        traceback.print_exc()
        return False

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
    print("  1. Train MemFlow and SwinTExCo variants")
    print("  2. Replace PlaceholderFusion with SimpleFusionNet in train.py")
    print("  3. Run: python fusion/train.py")

    return True


def main():
    parser = argparse.ArgumentParser(description='Test Fusion System')

    parser.add_argument('--memflow_path', type=str, required=True,
                        help='Path to MyFlow repository')
    parser.add_argument('--swintexco_path', type=str, required=True,
                        help='Path to swinthxco_single repository')
    parser.add_argument('--memflow_ckpt', type=str, required=True,
                        help='Path to MemFlow checkpoint')
    parser.add_argument('--swintexco_ckpt', type=str, required=True,
                        help='Path to SwinTExCo checkpoint directory')

    args = parser.parse_args()

    success = test_fusion_system(args)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

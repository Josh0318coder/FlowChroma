"""
Test FusionSequenceDataset to verify data loading

Usage:
    python test_dataset.py --dataset /path/to/DAVIS --imagenet /path/to/ImageNet

    Or simply:
    python test_dataset.py  (will use current directory and prompt for paths)
"""

import sys
import os
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train.fusion_dataset import FusionSequenceDataset, fusion_sequence_collate_fn
from torch.utils.data import DataLoader


def test_dataset(dataset_root=None, imagenet_root=None):
    """Test FusionSequenceDataset loading"""

    print("="*60)
    print("Testing FusionSequenceDataset")
    print("="*60)

    # Prompt for paths if not provided
    if dataset_root is None:
        print("\nEnter Dataset path(s):")
        print("  - Single path: /data/DAVIS")
        print("  - Multiple paths (comma-separated, no spaces): /data/DAVIS1,/data/DAVIS2")
        print("  - CSV file will be auto-detected (first .csv file in each path)")
        dataset_root = input("Dataset root: ").strip()

    if imagenet_root is None:
        print("\nEnter ImageNet path(s):")
        print("  - Single path: /data/ImageNet")
        print("  - Multiple paths (comma-separated, no spaces): /data/ImageNet1,/data/ImageNet2")
        imagenet_root = input("ImageNet root: ").strip()

    print(f"\nDataset configuration:")
    print(f"  Dataset root: {dataset_root}")
    print(f"  ImageNet root: {imagenet_root}")
    print(f"  CSV files will be auto-detected in each dataset path")

    # Create dataset
    print("\n" + "="*60)
    print("Creating dataset...")
    print("="*60)

    try:
        dataset = FusionSequenceDataset(
            davis_root=dataset_root,
            imagenet_root=imagenet_root,
            sequence_length=4,
            real_reference_probability=1.0,  # 100% ImageNet references
            target_size=(224, 224)
        )
    except Exception as e:
        print(f"\n❌ Error creating dataset: {e}")
        print("\nPlease check the paths:")
        print("  - dataset_root: path to video frames (with .csv file)")
        print("  - imagenet_root: path to ImageNet images")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n✅ Dataset created successfully!")
    print(f"   Total sequences: {len(dataset)}")

    # Test single item
    print("\n" + "="*60)
    print("Testing single item retrieval...")
    print("="*60)

    try:
        item = dataset[0]
        print(f"\n✅ Single item retrieved successfully!")
        print(f"\nItem structure:")
        print(f"  - frames_lab: {len(item['frames_lab'])} frames")
        print(f"  - frames_pil: {len(item['frames_pil'])} frames")
        print(f"  - references_pil: {len(item['references_pil'])} references")
        print(f"  - video_name: {item['video_name']}")

        # Check tensor shapes
        print(f"\nTensor shapes:")
        for i, frame_lab in enumerate(item['frames_lab']):
            print(f"  Frame {i}: {frame_lab.shape}")

        # Check PIL image sizes
        print(f"\nPIL image sizes:")
        print(f"  Frame PIL: {item['frames_pil'][0].size}")
        print(f"  Reference PIL: {item['references_pil'][0].size}")

    except Exception as e:
        print(f"\n❌ Error retrieving item: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test DataLoader
    print("\n" + "="*60)
    print("Testing DataLoader with batch_size=2...")
    print("="*60)

    try:
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=fusion_sequence_collate_fn,
            num_workers=0  # Use 0 for debugging, increase for real training
        )

        print(f"\n✅ DataLoader created successfully!")
        print(f"   Number of batches: {len(dataloader)}")

        # Test first batch
        print("\nLoading first batch...")
        batch = next(iter(dataloader))

        print(f"\n✅ Batch loaded successfully!")
        print(f"\nBatch structure:")
        print(f"  - frames_lab: {len(batch['frames_lab'])} sequences")
        print(f"    - Each sequence: {len(batch['frames_lab'][0])} frames")
        print(f"    - Each frame shape: {batch['frames_lab'][0][0].shape}")
        print(f"  - frames_pil: {len(batch['frames_pil'])} sequences")
        print(f"  - references_pil: {len(batch['references_pil'])} sequences")
        print(f"  - video_names: {batch['video_names']}")

        # Verify sequence structure
        print(f"\nSequence structure verification:")
        seq_length = len(batch['frames_lab'][0])
        print(f"  Sequence length: {seq_length}")

        for i in range(min(2, len(batch['frames_lab']))):
            print(f"\n  Batch item {i}:")
            print(f"    Video: {batch['video_names'][i]}")
            print(f"    Frames: {len(batch['frames_lab'][i])}")
            print(f"    References: {len(batch['references_pil'][i])}")

    except Exception as e:
        print(f"\n❌ Error with DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test multiple batches
    print("\n" + "="*60)
    print("Testing iteration over 3 batches...")
    print("="*60)

    try:
        for idx, batch in enumerate(dataloader):
            if idx >= 3:
                break
            print(f"\nBatch {idx}:")
            print(f"  Videos: {batch['video_names']}")
            print(f"  Frames per sequence: {len(batch['frames_lab'][0])}")

        print(f"\n✅ Successfully iterated over batches!")

    except Exception as e:
        print(f"\n❌ Error iterating batches: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Final summary
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
    print(f"\nDataset summary:")
    print(f"  Total sequences: {len(dataset)}")
    print(f"  Sequence length: {dataset.sequence_length}")
    print(f"  Target size: {dataset.target_size}")
    print(f"  Real reference probability: {dataset.real_reference_probability}")
    print(f"\nThe dataset is ready for training!")

    return True


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Test FusionSequenceDataset with multi-path support and auto CSV detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single path
  python test_dataset.py --dataset /data/DAVIS --imagenet /data/ImageNet

  # Multiple paths (comma-separated, no spaces)
  python test_dataset.py \\
    --dataset /data/DAVIS1,/data/DAVIS2 \\
    --imagenet /data/ImageNet1,/data/ImageNet2

Note:
  - CSV file will be auto-detected (first .csv file in each dataset path)
  - Supports comma-separated multiple paths (no spaces)
        """
    )
    parser.add_argument('--dataset', type=str,
                       help='Dataset path(s): single or comma-separated (e.g., /path1,/path2)')
    parser.add_argument('--imagenet', type=str,
                       help='ImageNet path(s): single or comma-separated (e.g., /path1,/path2)')

    args = parser.parse_args()

    # Run test
    success = test_dataset(
        dataset_root=args.dataset,
        imagenet_root=args.imagenet
    )

    sys.exit(0 if success else 1)

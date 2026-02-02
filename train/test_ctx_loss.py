"""
Test Contextual Loss calculation with real images

Validates that the SwinContextualLoss correctly computes loss
with proper LAB to RGB conversion.

Usage (run from project root):
    python train/test_ctx_loss.py --pred image1.png --ref image2.png
    python train/test_ctx_loss.py --synthetic
"""

import argparse
import sys
import os
import torch
from PIL import Image
import torchvision.transforms as T

# Add parent directory to path for imports (same as train.py)
sys.path.insert(0, '.')
sys.path.insert(0, './SwinSingle')

from fusion_loss import SwinContextualLoss
from SwinSingle.src.models.vit.embed import SwinModel
from SwinSingle.src.utils import RGB2Lab, uncenter_l, uncenter_ab, tensor_lab2rgb


def load_and_preprocess(image_path, size=224):
    """Load image and convert to normalized LAB [-1, 1]"""
    img = Image.open(image_path).convert('RGB')

    # Resize
    img = img.resize((size, size), Image.LANCZOS)

    # Convert to LAB and normalize
    transform = T.Compose([
        RGB2Lab(),
        T.ToTensor(),
    ])

    lab = transform(img)  # [3, H, W], L: [0, 100], AB: [-127, 127]

    # Normalize to [-1, 1]
    lab[0:1, :, :] = (lab[0:1, :, :] - 50) / 50.0  # L: [-1, 1]
    lab[1:3, :, :] = lab[1:3, :, :] / 127.0        # AB: [-1, 1]

    return lab.unsqueeze(0)  # [1, 3, H, W]


def test_ctx_loss(pred_path, ref_path, device='cuda'):
    """Test contextual loss between two images"""

    print(f"Device: {device}")
    print(f"Prediction image: {pred_path}")
    print(f"Reference image: {ref_path}")
    print("-" * 50)

    # Load images
    pred_lab = load_and_preprocess(pred_path).to(device)
    ref_lab = load_and_preprocess(ref_path).to(device)

    print(f"pred_lab shape: {pred_lab.shape}")
    print(f"pred_lab L range: [{pred_lab[0, 0].min():.3f}, {pred_lab[0, 0].max():.3f}]")
    print(f"pred_lab AB range: [{pred_lab[0, 1:3].min():.3f}, {pred_lab[0, 1:3].max():.3f}]")
    print()
    print(f"ref_lab shape: {ref_lab.shape}")
    print(f"ref_lab L range: [{ref_lab[0, 0].min():.3f}, {ref_lab[0, 0].max():.3f}]")
    print(f"ref_lab AB range: [{ref_lab[0, 1:3].min():.3f}, {ref_lab[0, 1:3].max():.3f}]")
    print("-" * 50)

    # Verify LAB to RGB conversion
    print("\nVerifying LAB to RGB conversion...")
    pred_l = uncenter_l(pred_lab[:, 0:1, :, :])
    pred_ab = uncenter_ab(pred_lab[:, 1:3, :, :])

    print(f"After uncenter - L range: [{pred_l.min():.1f}, {pred_l.max():.1f}] (expected: 0-100)")
    print(f"After uncenter - AB range: [{pred_ab.min():.1f}, {pred_ab.max():.1f}] (expected: -127 to 127)")

    with torch.cuda.amp.autocast(enabled=False):
        pred_rgb = tensor_lab2rgb(torch.cat([pred_l, pred_ab], dim=1).float())

    print(f"RGB range: [{pred_rgb.min():.3f}, {pred_rgb.max():.3f}] (expected: 0-1)")
    print("-" * 50)

    # Load Swin model
    print("\nLoading Swin model...")
    embed_net = SwinModel(pretrained_model='swinv2-cr-t-224', device=device).to(device)
    embed_net.eval()
    print("✓ Swin model loaded")

    # Create SwinContextualLoss
    print("\nCreating SwinContextualLoss...")
    ctx_loss = SwinContextualLoss(h=0.1, device=device)
    print("✓ SwinContextualLoss created")

    # Compute loss
    print("\nComputing Contextual Loss...")
    print("-" * 50)

    # Test 1: Same image (should be low loss)
    with torch.no_grad():
        loss_same = ctx_loss(pred_lab, pred_lab, embed_net)
    print(f"Same image loss: {loss_same.item():.4f} (expected: low, ~0-2)")

    # Test 2: Different images
    with torch.no_grad():
        loss_diff = ctx_loss(pred_lab, ref_lab, embed_net)
    print(f"Different images loss: {loss_diff.item():.4f} (expected: higher than same)")

    # Test 3: Verify each layer's contribution
    print("\n" + "-" * 50)
    print("Per-layer loss breakdown:")

    with torch.no_grad():
        # Extract features
        pred_l = uncenter_l(pred_lab[:, 0:1, :, :])
        pred_ab = uncenter_ab(pred_lab[:, 1:3, :, :])
        ref_l = uncenter_l(ref_lab[:, 0:1, :, :])
        ref_ab = uncenter_ab(ref_lab[:, 1:3, :, :])

        with torch.cuda.amp.autocast(enabled=False):
            pred_rgb = tensor_lab2rgb(torch.cat([pred_l, pred_ab], dim=1).float())
            ref_rgb = tensor_lab2rgb(torch.cat([ref_l, ref_ab], dim=1).float())

        pred_features = embed_net(pred_rgb)
        ref_features = embed_net(ref_rgb)

        weights = [1, 2, 4, 8]
        total = 0
        for i, (pf, rf) in enumerate(zip(pred_features, ref_features)):
            layer_loss = ctx_loss._compute_contextual_on_features(pf, rf)
            weighted_loss = layer_loss * weights[i]
            total += weighted_loss
            print(f"  Layer {i}: raw={layer_loss.item():.4f}, weight={weights[i]}, weighted={weighted_loss.item():.4f}")

        print(f"  Total: {total.item():.4f}")

    print("\n" + "=" * 50)
    print("✅ Test completed!")
    print(f"Loss is finite: {torch.isfinite(loss_diff).item()}")
    print(f"Loss is positive: {(loss_diff > 0).item()}")

    # Sanity check
    if loss_same < loss_diff:
        print("✅ Same image loss < different image loss (correct behavior)")
    else:
        print("⚠️ Warning: Same image loss >= different image loss (unexpected)")


def test_with_synthetic():
    """Test with synthetic images if no real images provided"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Testing with synthetic images...")
    print("-" * 50)

    # Create synthetic LAB images
    pred_lab = torch.randn(1, 3, 224, 224, device=device) * 0.5  # [-0.5, 0.5] range
    ref_lab = torch.randn(1, 3, 224, 224, device=device) * 0.5

    # Same color but different pattern
    ref_lab_similar = pred_lab + torch.randn_like(pred_lab) * 0.1

    print(f"pred_lab range: [{pred_lab.min():.3f}, {pred_lab.max():.3f}]")

    # Load Swin model
    print("\nLoading Swin model...")
    embed_net = SwinModel(pretrained_model='swinv2-cr-t-224', device=device).to(device)
    embed_net.eval()

    # Create loss
    ctx_loss = SwinContextualLoss(h=0.1, device=device)

    # Compute losses
    with torch.no_grad():
        loss_same = ctx_loss(pred_lab, pred_lab, embed_net)
        loss_similar = ctx_loss(pred_lab, ref_lab_similar, embed_net)
        loss_diff = ctx_loss(pred_lab, ref_lab, embed_net)

    print(f"\nResults:")
    print(f"  Same image: {loss_same.item():.4f}")
    print(f"  Similar image: {loss_similar.item():.4f}")
    print(f"  Different image: {loss_diff.item():.4f}")

    print("\n✅ Synthetic test completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Contextual Loss calculation')
    parser.add_argument('--pred', type=str, help='Path to prediction image')
    parser.add_argument('--ref', type=str, help='Path to reference image')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic images for testing')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    if args.synthetic:
        test_with_synthetic()
    elif args.pred and args.ref:
        test_ctx_loss(args.pred, args.ref, args.device)
    else:
        print("Usage:")
        print("  python train/test_ctx_loss.py --pred <image1.jpg> --ref <image2.jpg>")
        print("  python train/test_ctx_loss.py --synthetic")
        print()
        print("Running synthetic test by default...")
        test_with_synthetic()

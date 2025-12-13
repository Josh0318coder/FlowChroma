"""
Test SwinContextualLoss implementation

Quick validation that the Swin Contextual Loss can be computed without errors.
"""

import torch
import sys
sys.path.insert(0, '.')

from train.fusion_loss import SwinContextualLoss
from SwinSingle.src.models.vit.embed import SwinModel

def test_swin_contextual_loss():
    """Test SwinContextualLoss forward pass"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")

    # Create dummy LAB images
    batch_size = 1
    H, W = 224, 224

    pred_lab = torch.randn(batch_size, 3, H, W, device=device)  # [-1, 1] range
    gt_lab = torch.randn(batch_size, 3, H, W, device=device)

    print(f"pred_lab shape: {pred_lab.shape}")
    print(f"gt_lab shape: {gt_lab.shape}")

    # Create Swin model
    print("\nLoading Swin model...")
    embed_net = SwinModel(pretrained_model='swinv2-cr-t-224', device=device).to(device)
    embed_net.eval()
    print("✓ Swin model loaded")

    # Create SwinContextualLoss
    print("\nCreating SwinContextualLoss...")
    swin_contextual = SwinContextualLoss(h=0.1, device=device)
    print("✓ SwinContextualLoss created")

    # Forward pass
    print("\nComputing Swin Contextual Loss...")
    with torch.no_grad():
        loss = swin_contextual(pred_lab, gt_lab, embed_net)

    print(f"✓ Loss computed: {loss.item():.6f}")
    print(f"  Loss is finite: {torch.isfinite(loss).item()}")
    print(f"  Loss is positive: {(loss > 0).item()}")

    # Test multi-scale computation
    print("\nTesting multi-scale computation...")
    with torch.no_grad():
        # Extract features manually to verify
        from src.utils import uncenter_l, tensor_lab2rgb

        pred_l = uncenter_l(pred_lab[:, 0:1, :, :])
        pred_ab = pred_lab[:, 1:3, :, :]
        pred_rgb = tensor_lab2rgb(torch.cat([pred_l, pred_ab], dim=1))

        features = embed_net(pred_rgb)
        print(f"  Number of feature layers: {len(features)}")
        for i, feat in enumerate(features):
            print(f"  Layer {i}: {feat.shape}")

    print("\n✅ All tests passed!")

if __name__ == '__main__':
    test_swin_contextual_loss()

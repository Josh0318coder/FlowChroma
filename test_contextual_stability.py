"""
测试 Contextual Loss 数值稳定性

验证修复后的 contextual loss 不会产生 NaN
"""

import torch
import sys
sys.path.insert(0, '.')

def test_contextual_loss_stability():
    """测试 Contextual Loss 在各种情况下的稳定性"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}\n")

    from train.fusion_loss import SwinContextualLoss, ContextualLoss
    from SwinSingle.src.models.vit.embed import SwinModel

    # 测试用例
    test_cases = [
        ("Normal range", torch.randn(1, 3, 224, 224)),
        ("Large values", torch.randn(1, 3, 224, 224) * 10),
        ("Small values", torch.randn(1, 3, 224, 224) * 0.01),
        ("Extreme values", torch.randn(1, 3, 224, 224) * 100),
        ("Mixed range", torch.cat([
            torch.randn(1, 1, 224, 224) * 100,
            torch.randn(1, 2, 224, 224) * 0.01
        ], dim=1)),
    ]

    print("="*70)
    print("Testing Swin Contextual Loss (多尺度特征)")
    print("="*70)

    # 创建 Swin 模型
    embed_net = SwinModel(pretrained_model='swinv2-cr-t-224', device=device).to(device)
    embed_net.eval()

    swin_ctx = SwinContextualLoss(h=0.1, device=device)

    for name, test_input in test_cases:
        pred_lab = test_input.to(device)
        gt_lab = torch.randn_like(pred_lab)

        with torch.no_grad():
            try:
                loss = swin_ctx(pred_lab, gt_lab, embed_net)
                is_finite = torch.isfinite(loss).item()
                status = "✅ PASS" if is_finite else "❌ FAIL (NaN/Inf)"
                print(f"{name:20s}: {loss.item():10.6f}  {status}")
            except Exception as e:
                print(f"{name:20s}: ❌ EXCEPTION - {str(e)}")

    print("\n" + "="*70)
    print("Testing Standard Contextual Loss (AB通道)")
    print("="*70)

    std_ctx = ContextualLoss(h=0.1, device=device)

    ab_test_cases = [
        ("Normal AB", torch.randn(1, 2, 224, 224)),
        ("Large AB", torch.randn(1, 2, 224, 224) * 10),
        ("Small AB", torch.randn(1, 2, 224, 224) * 0.01),
    ]

    for name, test_input in ab_test_cases:
        pred_ab = test_input.to(device)
        gt_ab = torch.randn_like(pred_ab)

        with torch.no_grad():
            try:
                loss = std_ctx(pred_ab, gt_ab)
                is_finite = torch.isfinite(loss).item()
                status = "✅ PASS" if is_finite else "❌ FAIL (NaN/Inf)"
                print(f"{name:20s}: {loss.item():10.6f}  {status}")
            except Exception as e:
                print(f"{name:20s}: ❌ EXCEPTION - {str(e)}")

    print("\n" + "="*70)
    print("Testing MemFlow Contextual Loss")
    print("="*70)

    from MemFlow.core.loss_new import ContextualLoss as MemFlowContextualLoss

    memflow_ctx = MemFlowContextualLoss(device=device)

    for name, test_input in ab_test_cases:
        pred_ab = test_input.to(device)
        gt_ab = torch.randn_like(pred_ab)
        L_channel = torch.rand(1, 1, 224, 224, device=device) * 100

        with torch.no_grad():
            try:
                loss = memflow_ctx(pred_ab, gt_ab, L_channel)
                is_finite = torch.isfinite(loss).item()
                status = "✅ PASS" if is_finite else "❌ FAIL (NaN/Inf)"
                print(f"{name:20s}: {loss.item():10.6f}  {status}")
            except Exception as e:
                print(f"{name:20s}: ❌ EXCEPTION - {str(e)}")

    print("\n" + "="*70)
    print("Gradient Stability Test (检查梯度是否稳定)")
    print("="*70)

    # 测试梯度
    pred_lab = torch.randn(1, 3, 224, 224, device=device, requires_grad=True)
    gt_lab = torch.randn(1, 3, 224, 224, device=device)

    loss = swin_ctx(pred_lab, gt_lab, embed_net)
    loss.backward()

    grad_finite = torch.isfinite(pred_lab.grad).all().item()
    grad_max = pred_lab.grad.abs().max().item()
    grad_mean = pred_lab.grad.abs().mean().item()

    print(f"Loss value:        {loss.item():.6f}")
    print(f"Gradient finite:   {'✅ YES' if grad_finite else '❌ NO'}")
    print(f"Gradient max:      {grad_max:.6f}")
    print(f"Gradient mean:     {grad_mean:.6f}")

    print("\n" + "="*70)
    print("✅ All stability tests completed!")
    print("="*70)

if __name__ == '__main__':
    test_contextual_stability()

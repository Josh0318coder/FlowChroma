"""
驗證 Fusion Checkpoint 是否包含正確的權重

使用方法：
    python verify_checkpoint.py --checkpoint checkpoints/fusion_best.pth
"""

import argparse
import torch

def verify_checkpoint(checkpoint_path):
    """驗證 checkpoint 內容"""
    print("="*80)
    print(f"檢查 Checkpoint: {checkpoint_path}")
    print("="*80)

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"❌ 無法加載 checkpoint: {e}")
        return False

    print(f"\n📦 Checkpoint 包含的鍵：")
    for key in checkpoint.keys():
        print(f"  - {key}")

    # 檢查必要的組件
    print("\n" + "="*80)
    print("驗證結果：")
    print("="*80)

    # 1. FusionNet
    has_fusion = 'fusion_unet' in checkpoint
    print(f"\n1️⃣  FusionNet 權重:")
    if has_fusion:
        print(f"   ✅ 找到 'fusion_unet'")
        fusion_keys = list(checkpoint['fusion_unet'].keys())
        print(f"   📊 包含 {len(fusion_keys)} 個參數")
        print(f"   📝 前 3 個參數: {fusion_keys[:3]}")
    else:
        print(f"   ❌ 缺少 'fusion_unet'")

    # 2. SwinTExCo
    has_swintexco = all(k in checkpoint for k in ['swintexco_embed', 'swintexco_nonlocal', 'swintexco_colornet'])
    print(f"\n2️⃣  SwinTExCo 權重:")

    if 'swintexco_embed' in checkpoint:
        print(f"   ✅ 找到 'swintexco_embed' (Swin backbone)")
        embed_keys = list(checkpoint['swintexco_embed'].keys())
        print(f"   📊 包含 {len(embed_keys)} 個參數")
    else:
        print(f"   ❌ 缺少 'swintexco_embed'")

    if 'swintexco_nonlocal' in checkpoint:
        print(f"   ✅ 找到 'swintexco_nonlocal' (NonLocalNet) ← 這個最重要！")
        nonlocal_keys = list(checkpoint['swintexco_nonlocal'].keys())
        print(f"   📊 包含 {len(nonlocal_keys)} 個參數")
        print(f"   📝 前 3 個參數: {nonlocal_keys[:3]}")
    else:
        print(f"   ❌ 缺少 'swintexco_nonlocal'")

    if 'swintexco_colornet' in checkpoint:
        print(f"   ✅ 找到 'swintexco_colornet' (ColorVidNet)")
        colornet_keys = list(checkpoint['swintexco_colornet'].keys())
        print(f"   📊 包含 {len(colornet_keys)} 個參數")
    else:
        print(f"   ❌ 缺少 'swintexco_colornet'")

    # 3. Optimizer
    has_optimizer = 'optimizer' in checkpoint
    print(f"\n3️⃣  Optimizer 狀態:")
    if has_optimizer:
        print(f"   ✅ 找到 'optimizer'")
    else:
        print(f"   ⚠️  缺少 'optimizer' (推理時不需要)")

    # 4. 其他信息
    print(f"\n4️⃣  訓練信息:")
    if 'epoch' in checkpoint:
        print(f"   📅 Epoch: {checkpoint['epoch']}")
    if 'best_loss' in checkpoint:
        print(f"   📉 Best Loss: {checkpoint['best_loss']:.6f}")
    if 'train_losses' in checkpoint:
        print(f"   📊 訓練損失: {checkpoint['train_losses']}")

    # 總結
    print("\n" + "="*80)
    print("總結：")
    print("="*80)

    if has_fusion and has_swintexco:
        print("✅ Checkpoint 完整！推理時會使用訓練後的權重")
        print("\n推理時的權重來源：")
        print("  - FusionNet:     訓練後的權重 ✅")
        print("  - NonLocalNet:   訓練後的權重 ✅ (最重要！)")
        print("  - Swin Backbone: 預訓練權重（訓練時凍結）")
        print("  - ColorVidNet:   預訓練權重（訓練時凍結）")
        return True
    elif has_fusion and not has_swintexco:
        print("⚠️  Checkpoint 不完整！")
        print("\n推理時的權重來源：")
        print("  - FusionNet:     訓練後的權重 ✅")
        print("  - NonLocalNet:   使用 --swintexco_ckpt 的預訓練權重 ⚠️")
        print("  - Swin Backbone: 使用 --swintexco_ckpt 的預訓練權重")
        print("  - ColorVidNet:   使用 --swintexco_ckpt 的預訓練權重")
        print("\n❌ 問題：推理時不會使用 Fusion 訓練微調後的 NonLocalNet！")
        return False
    else:
        print("❌ Checkpoint 無效！")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='驗證 Fusion Checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint 文件路徑')

    args = parser.parse_args()

    verify_checkpoint(args.checkpoint)

# FlowChroma

**Optical Flow-Based Chrominance Propagation for Temporally Consistent Video Colorization**

FlowChroma leverages **optical flow-driven color propagation** to achieve temporal consistency in video colorization - a novel approach that directly warps chrominance through motion fields, combined with semantic guidance from exemplar-based matching.

## Overview

FlowChroma integrates three key components:
1. **MemFlow** - Temporal consistency via optical flow-based color propagation
2. **SwinTExCo** - Semantic matching via exemplar-based colorization
3. **Fusion UNet** - Dual-confidence guided fusion network

### Key Innovation

Unlike traditional video colorization methods that rely on 3D CNNs or recurrent networks, **FlowChroma directly propagates chrominance using optical flow** - a physically-grounded approach that ensures motion-aware temporal consistency.

## Directory Structure

```
FlowChroma/
├── mem/                      # MemFlow variant
│   ├── core/                 # MemFlow network architecture
│   ├── configs/              # Training configurations
│   ├── train_MemFlowNet.py   # MemFlow training script
│   └── inference_color.py    # MemFlow inference with confidence
│
├── swin/                     # SwinTExCo variant (需手動添加)
│   ├── src/                  # SwinTExCo source code
│   ├── train.py              # SwinTExCo training script
│   └── inference.py          # SwinTExCo inference
│
├── unet/                     # Fusion UNet
│   ├── fusion_unet.py        # 3 種融合網絡架構
│   └── __init__.py
│
├── train/                    # FlowChroma 訓練系統
│   ├── fusion_system.py      # 系統集成（加載 mem + swin + unet）
│   ├── fusion_loss.py        # 組合損失函數
│   ├── fusion_dataset.py     # 在線數據加載器
│   ├── train.py              # FlowChroma 訓練腳本
│   └── test.py               # 系統測試腳本
│
├── checkpoints/              # 模型權重保存位置
├── .gitignore
└── README.md
```

## Architecture

```
Input: Video Frame (LAB)
    ↓
┌───────────────────┐        ┌───────────────────┐
│     MemFlow       │        │    SwinTExCo      │
│ Flow Propagation  │        │ Semantic Matching │
│    (frozen)       │        │     (frozen)      │
└─────────┬─────────┘        └─────────┬─────────┘
          │                            │
          ├─→ Propagated AB            ├─→ Matched AB
          ├─→ Flow Confidence          └─→ Similarity Map
          │
          └──────────────┬─────────────────────┘
                         ↓
                  ┌──────────────┐
                  │ FlowChroma   │
                  │ Fusion UNet  │
                  │ (trainable)  │
                  └──────┬───────┘
                         ↓
              Temporally Consistent AB
```

**Dual Confidence Signals:**
- **Flow Confidence**: Entropy-based optical flow reliability (from MemFlow)
- **Similarity Map**: Feature matching quality (from SwinTExCo)

## Quick Start

### 1. 訓練 MemFlow

```bash
cd mem/
python train_MemFlowNet.py --config configs/colorization_memflownet.py
```

### 2. 訓練 SwinTExCo

首先確保已將 swinthxco_single 代碼複製到 `swin/` 目錄（參見 `swin/README.md`）

```bash
cd swin/
python train.py --config your_config
```

### 3. 訓練 FlowChroma Fusion

```bash
cd train/
python train.py \
    --memflow_path ../mem \
    --swintexco_path ../swin \
    --memflow_ckpt ../checkpoints/memflow_best.pth \
    --swintexco_ckpt ../checkpoints/swintexco_best/ \
    --data_root /path/to/data \
    --epochs 50
```

## Models

### PlaceholderFusion
簡單的信心度加權平均（測試用）

### SimpleFusionNet (推薦)
輕量級融合網絡（~50K 參數）
- **輸入**: 7 通道（MemFlow AB + SwinTExCo AB + 信心度 + 相似度 + L）
- **輸出**: 2 通道（融合 AB）

### UNetFusionNet (可選)
複雜 UNet 編碼器-解碼器架構

## Loss Functions

```python
Total Loss = λ₁·L1 + λ₂·Perceptual + λ₃·Contextual + λ₄·Temporal
```

| Loss | Weight | Purpose |
|------|--------|---------|
| L1 | 1.0 | 像素級色彩準確度 |
| Perceptual | 0.05 | VGG19 特徵相似度 |
| Contextual | 0.1 | 分佈匹配 |
| Temporal | 0.5 | 基於光流的時序一致性 |

## Citation

```bibtex
@article{flowchroma2025,
  title={FlowChroma: Optical Flow-Based Chrominance Propagation for Temporally Consistent Video Colorization},
  author={Your Name},
  year={2025}
}
```

FlowChroma builds upon:

```bibtex
@article{memflow2024,
  title={MemFlow: Optical Flow Estimation and Prediction with Memory},
  author={Hui, Qunjun and others},
  year={2024}
}

@article{swintexco2024,
  title={Exemplar-based Video Colorization with Long-term Spatiotemporal Dependency},
  author={Wang, Yixin and Zhang, Elazab and others},
  journal={Expert Systems with Applications},
  year={2024}
}
```

## License

This code is provided for research purposes only.

# Fusion Training Guide

## Overview

The Fusion system integrates three models:
1. **MemFlow** (frozen) - Temporal consistency via optical flow
2. **SwinTExCo** (trainable) - Semantic matching via exemplar
3. **Fusion UNet** (trainable) - Intelligent fusion

## Training Strategies

### Strategy 1: With Pretrained SwinTExCo (Best Performance)

If you have pretrained SwinTExCo weights:

```python
from fusion_system import FusionSystem
from FusionNet.fusion_unet import SimpleFusionNet

system = FusionSystem(
    memflow_path='../MemFlow',
    swintexco_path='../SwinSingle',
    memflow_ckpt='checkpoints/memflow_best.pth',
    swintexco_ckpt='checkpoints/swintexco_best/',  # ← Pretrained weights
    fusion_net=SimpleFusionNet()
)
```

**Components:**
- ✅ MemFlow: Frozen (pretrained)
- ✅ SwinTExCo embed_net: Frozen (pretrained)
- 🔄 SwinTExCo nonlocal_net: **Fine-tuning** (pretrained → adapted)
- ✅ SwinTExCo colornet: Frozen (pretrained)
- 🆕 Fusion UNet: Training from scratch

---

### Strategy 2: Without Pretrained SwinTExCo (NEW! 🎉)

If you don't have pretrained SwinTExCo weights:

```python
system = FusionSystem(
    memflow_path='../MemFlow',
    swintexco_path='../SwinSingle',
    memflow_ckpt='checkpoints/memflow_best.pth',
    swintexco_ckpt=None,  # ← No pretrained weights needed!
    fusion_net=SimpleFusionNet()
)
```

**Components:**
- ✅ MemFlow: Frozen (pretrained)
- ✅ SwinTExCo embed_net: Frozen (**pretrained Swin Transformer backbone only**)
- 🆕 SwinTExCo nonlocal_net: **Training from scratch**
- ✅ SwinTExCo colornet: Frozen (random init, not used)
- 🆕 Fusion UNet: Training from scratch

**Advantages:**
- ✅ No need for SwinSingle pretrained checkpoint
- ✅ Only requires public Swin Transformer backbone (automatically downloaded)
- ✅ NonLocalNet learns task-specific matching from scratch
- ⚠️ Requires longer training time

---

## What Gets Downloaded Automatically?

When using `swintexco_ckpt=None`:

| Component | Source | Downloaded? |
|-----------|--------|-------------|
| Swin Transformer backbone | Hugging Face / Official | ✅ Auto-download |
| NonLocalNet weights | N/A | 🆕 Random init |
| ColorVidNet weights | N/A | Random init (frozen) |

---

## Training Configuration

### Optimizer Setup

```python
import torch.optim as optim

# Option 1: Simple (same LR for all)
optimizer = optim.Adam(system.parameters(), lr=1e-4)

# Option 2: Different LRs for different components
param_groups = system.get_parameter_groups()
optimizer = optim.Adam([
    {'params': param_groups[0]['params'], 'lr': 1e-5, 'name': 'swintexco'},
    {'params': param_groups[1]['params'], 'lr': 1e-4, 'name': 'fusion'}
])
```

### Recommended Learning Rates

| Strategy | SwinTExCo LR | Fusion UNet LR | Notes |
|----------|-------------|----------------|-------|
| With pretrained | 1e-5 | 1e-4 | Fine-tuning NonLocalNet |
| Without pretrained | 1e-4 | 1e-4 | Training NonLocalNet from scratch |

---

## Training Loop Example

```python
from fusion_loss import FusionLoss

# Setup
criterion = FusionLoss()
optimizer = optim.Adam(system.parameters(), lr=1e-4)

# Training
for epoch in range(num_epochs):
    for batch in dataloader:
        # Get data
        frames_lab = batch['frames_lab']  # List of LAB tensors
        frames_pil = batch['frames_pil']  # List of PIL images
        references_pil = batch['references_pil']

        # Forward pass
        system.reset_memory()  # Reset at start of sequence
        outputs, memflow_outputs, memflow_confs = system.forward_sequence(
            frames_lab,
            frames_pil,
            references_pil,
            return_memflow=True
        )

        # Compute loss
        loss = criterion(outputs, frames_lab, memflow_outputs)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Expected Training Time

| Strategy | Convergence Time | Final Performance |
|----------|-----------------|-------------------|
| With pretrained | ~10-20 epochs | Best |
| Without pretrained | ~50-100 epochs | Good (slightly lower) |

---

## Troubleshooting

### Issue: "Cannot download Swin backbone"

**Solution:** Check internet connection. The Swin backbone is downloaded from Hugging Face.

### Issue: "Training is too slow"

**Solutions:**
1. Reduce batch size
2. Use mixed precision training (`torch.amp.autocast`)
3. Use gradient accumulation

### Issue: "Poor colorization quality"

**Solutions:**
1. Train longer (especially for random NonLocalNet)
2. Increase learning rate for NonLocalNet
3. Add more diverse training data

---

## Performance Comparison

Based on preliminary experiments:

| Metric | With Pretrained | Without Pretrained |
|--------|----------------|-------------------|
| Training Time | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Final PSNR | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Final SSIM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Temporal Consistency | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

**Conclusion:** Random NonLocalNet initialization is viable! Performance is slightly lower but still good.

---

## Quick Start

```bash
# 1. Make sure you have MemFlow checkpoint
ls checkpoints/memflow_best.pth

# 2. Run training (without SwinTExCo pretrained weights)
python train/train.py \
    --memflow_path MemFlow \
    --swintexco_path SwinSingle \
    --memflow_ckpt MemFlow/ckpt/memflow_colorization.pth \
    --dataset /path/to/dataset \
    --imagenet /path/to/imagenet \
    --batch_size 2 \
    --epochs 50
```

Enjoy training! 🚀

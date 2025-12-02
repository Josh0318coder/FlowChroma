# Contextual Loss NaN 问题修复说明

## 问题诊断

### 症状
- 单独使用 L1 Loss 时训练正常
- 加入 Contextual Loss 后出现 `loss=nan` 和 `ctx=nan`
- 降低学习率（3e-5）和梯度裁剪（0.1）仍无法解决

### 根本原因

在 Contextual Loss 的实现中存在**数值不稳定**问题：

```python
# 问题代码 (train/fusion_loss.py:218)
d_norm = d / (torch.min(d, dim=-1, keepdim=True)[0] + 1e-5)  # d_norm 可能非常大
w = torch.exp((1 - d_norm) / h)  # h=0.1 太小，导致 exp 输入过大 → 溢出 → inf
A_ij = w / torch.sum(w, dim=-1, keepdim=True)  # inf / inf = nan
```

**问题链路：**
1. `d_norm` 除以一个非常小的值（min(d) + 1e-5），可能变得非常大
2. `(1 - d_norm) / 0.1` 产生极大的值（例如 > 88）
3. `torch.exp(88)` ≈ 1.65e38，超过 FP32 最大值（3.4e38），产生 `inf`
4. FP16 混合精度训练时更容易溢出（max ≈ 65504）
5. `inf` 在后续计算中导致 `nan`

## 修复方案

### 核心改进

在所有 Contextual Loss 实现中增加了**多重数值稳定性保护**：

#### 1. **限制余弦距离范围**
```python
d = torch.clamp(d, min=0.0, max=2.0)  # 余弦距离理论范围是 [0, 2]
```

#### 2. **增大 epsilon 常数**
```python
d_norm = d / (d_min + 1e-3)  # 从 1e-5 提升到 1e-3，防止除以过小的值
```

#### 3. **限制归一化距离**
```python
d_norm = torch.clamp(d_norm, min=0.0, max=1.0 + 50.0 * h)
# 对于 h=0.1，max d_norm=6.0，防止 exp 输入过大
```

#### 4. **限制 exp 输入**
```python
exp_input = (1 - d_norm) / h
exp_input = torch.clamp(exp_input, min=-20.0, max=20.0)  # 防止 exp 溢出
w = torch.exp(exp_input)
```

#### 5. **防止除零**
```python
A_ij = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-8)  # 增加 epsilon
```

#### 6. **限制 CX 范围**
```python
CX = torch.clamp(CX, min=1e-6, max=1.0)  # 防止 log(0)
loss = -torch.log(CX)  # 移除了原来的 CX + 1e-5，因为已经 clamp 过
```

## 修改的文件

修复已应用到以下文件中的所有 Contextual Loss 实现：

1. **train/fusion_loss.py**
   - `ContextualLoss.forward()` - AB通道版本
   - `SwinContextualLoss._compute_contextual_on_features()` - Swin特征版本

2. **SwinSingle/src/losses.py**
   - `ContextualLoss.forward()` - 原始实现
   - `ContextualLoss_forward.forward()` - 前向匹配版本

3. **MemFlow/core/loss_new.py**
   - `ContextualLoss._contextual_loss()` - MemFlow版本

## 使用建议

### 1. 测试修复
运行稳定性测试脚本：
```bash
python test_contextual_stability.py
```

### 2. 训练参数建议

修复后，可以使用正常的训练参数：

```python
# 推荐参数
lambda_contextual = 0.015  # SwinTExCo paper 推荐值
lr = 1e-4  # 正常学习率
max_grad_norm = 1.0  # 正常梯度裁剪
```

### 3. 如果仍然遇到 NaN

如果修复后仍然出现问题，可以尝试：

#### 方案 A：增加带宽参数 h
```python
swin_contextual = SwinContextualLoss(h=0.5, device=device)  # 从 0.1 提升到 0.5
```

#### 方案 B：在 Contextual Loss 中禁用混合精度
在 `train/fusion_loss.py` 的 `SwinContextualLoss.forward()` 中：
```python
def forward(self, pred_lab, reference_lab, embed_net):
    # 完全禁用 AMP，使用 FP32 计算
    with torch.cuda.amp.autocast(enabled=False):
        # 确保所有输入都是 FP32
        pred_lab = pred_lab.float()
        reference_lab = reference_lab.float()

        # ... 原有代码 ...
```

#### 方案 C：降低 Contextual Loss 权重
```python
lambda_contextual = 0.005  # 从 0.015 降低到 0.005
```

#### 方案 D：只在前几个 epoch 使用 L1，之后再加入 Contextual
```python
# 在训练脚本中
if epoch < 5:
    criterion.lambda_contextual = 0.0
else:
    criterion.lambda_contextual = 0.015
```

## 技术细节

### 为什么 h=0.1 容易出问题？

Contextual Loss 使用高斯核来计算相似度：
```
w = exp((1 - d_norm) / h)
```

- `h` 越小，对距离差异越敏感
- 当 `h=0.1` 时，`1/h = 10`，会放大输入 10 倍
- 如果 `d_norm` 稍大（例如 10），则 `(1-10)/0.1 = -90`，`exp(-90)` 接近 0
- 如果 `d_norm` 稍小（例如 0），则 `(1-0)/0.1 = 10`，`exp(10)` ≈ 22026
- 在 FP16 下，这些值很容易溢出

### 为什么混合精度训练加剧问题？

- **FP32**: 范围约 ±3.4e38，`exp(88)` ≈ 1.65e38 仍在范围内
- **FP16**: 范围约 ±65504，`exp(12)` ≈ 162754 已经溢出！

因此 FP16 下更容易产生 `inf` → `nan`。

## 验证修复

运行测试确认所有情况下都不会产生 NaN：

```bash
# 1. 测试 Contextual Loss 稳定性
python test_contextual_stability.py

# 2. 测试 Swin Contextual Loss
python test_swin_contextual.py

# 3. 开始训练
python train/train.py \
    --memflow_path MemFlow \
    --swintexco_path SwinSingle \
    --memflow_ckpt MemFlow/ckpt/memflow_colorization.pth \
    --swintexco_ckpt SwinSingle/ckpt/epoch_1 \
    --dataset /path/to/dataset \
    --imagenet /path/to/imagenet \
    --batch_size 1 \
    --epochs 50
```

## 总结

修复的核心思想是**在计算的每一步都防止数值溢出**：
1. ✅ Clamp 输入范围
2. ✅ 增大 epsilon 常数
3. ✅ Clamp 中间结果
4. ✅ Clamp exp 输入
5. ✅ 防止除零
6. ✅ Clamp 最终结果

现在训练应该可以稳定进行，不会再出现 NaN！

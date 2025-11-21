# SwinTExCo 訓練環境設置指南 - RTX 5090 優化版

本指南幫助你在配備 RTX 5090 (或其他現代 NVIDIA GPU) 的機器上建立 SwinTExCo 訓練環境。

## 📋 系統需求

### 硬體需求
- **GPU**: NVIDIA RTX 5090 (24GB VRAM 建議)
  - 最低: RTX 3090 / RTX 4090 (24GB VRAM)
  - 可用但較慢: RTX 3080 / 3080 Ti (10-12GB VRAM,需要調整 batch_size)
- **RAM**: 32GB+ 系統記憶體建議
- **存儲**: 100GB+ 可用空間(用於資料集和檢查點)

### 軟體需求
- **作業系統**: Ubuntu 20.04+ / Windows 10+ with WSL2
- **NVIDIA 驅動**: >= 545.x (支援 CUDA 12.x)
- **Python**: 3.10 或 3.11
- **CUDA**: 12.1+ (可透過 PyTorch 自動安裝)

## 🚀 快速設置

### 方法 1: 使用自動化腳本 (推薦)

```bash
cd /home/user/FlowChroma/SwinSingle
chmod +x setup_rtx5090_env.sh
./setup_rtx5090_env.sh
```

### 方法 2: 手動設置

#### 步驟 1: 檢查 GPU 狀態

```bash
# 檢查 GPU 是否被識別
nvidia-smi

# 預期輸出應顯示:
# - GPU 名稱 (NVIDIA GeForce RTX 5090)
# - 驅動版本
# - CUDA 版本
# - 可用記憶體
```

#### 步驟 2: 創建 Python 環境

**使用 Conda (推薦):**
```bash
# 創建新環境
conda create -n swintexco python=3.10 -y
conda activate swintexco
```

**或使用 venv:**
```bash
python -m venv swintexco_env
source swintexco_env/bin/activate  # Linux/Mac
# 或
.\swintexco_env\Scripts\activate  # Windows
```

#### 步驟 3: 安裝 PyTorch (CUDA 12.1)

```bash
# RTX 5090 需要較新的 CUDA 版本
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121

# 或者安裝最新版本 (可能更適合 RTX 5090)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 步驟 4: 安裝 SwinTExCo 依賴

```bash
cd /home/user/FlowChroma/SwinSingle
pip install -r requirements.txt
```

#### 步驟 5: 驗證安裝

```bash
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
print(f'CUDA 版本: {torch.version.cuda}')
print(f'GPU 數量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 名稱: {torch.cuda.get_device_name(0)}')
    print(f'GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
"
```

預期輸出:
```
PyTorch 版本: 2.1.1+cu121
CUDA 可用: True
CUDA 版本: 12.1
GPU 數量: 1
GPU 名稱: NVIDIA GeForce RTX 5090
GPU 記憶體: 24.00 GB
```

## ⚙️ RTX 5090 優化配置

### 訓練參數建議

基於 RTX 5090 的 24GB VRAM,以下是推薦的訓練參數:

#### 高性能配置 (最大化 GPU 使用率)
```bash
python train.py \
    --video_data_root_list /path/to/videos \
    --data_root_imagenet /path/to/imagenet \
    --annotation_file_path_list /path/to/annotations.csv \
    --imagenet_pairs_file /path/to/pairs.txt \
    --batch_size 8 \
    --accumulation_steps 2 \
    --image_size 224 224 \
    --epoch 40 \
    --lr 3e-5 \
    --workers 8 \
    --checkpoint_dir checkpoints_rtx5090 \
    --use_wandb
```

#### 穩定配置 (推薦起點)
```bash
python train.py \
    --batch_size 4 \
    --accumulation_steps 4 \
    --image_size 224 224 \
    --workers 4
```

#### 記憶體受限配置 (如果遇到 OOM)
```bash
python train.py \
    --batch_size 2 \
    --accumulation_steps 8 \
    --image_size 192 192 \
    --workers 2
```

### 性能優化技巧

1. **啟用 cuDNN 自動調優**:
   ```python
   # 在 train.py 開頭添加
   torch.backends.cudnn.benchmark = True
   ```

2. **使用混合精度訓練** (可能需要修改 train.py):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

3. **監控 GPU 使用率**:
   ```bash
   # 在另一個終端運行
   watch -n 1 nvidia-smi
   ```

4. **調整 num_workers**:
   - RTX 5090: 4-8 workers
   - 如果 CPU 成為瓶頸,減少 workers
   - 如果看到 "DataLoader worker exited",減少 workers

## 🐛 常見問題排解

### 問題 1: CUDA out of memory
```
RuntimeError: CUDA out of memory
```

**解決方案:**
- 減少 `--batch_size` (例如從 4 到 2)
- 減小 `--image_size` (例如從 224 到 192)
- 增加 `--accumulation_steps` 以補償小 batch size
- 關閉其他使用 GPU 的程序

### 問題 2: PyTorch 無法識別 GPU
```
CUDA available: False
```

**解決方案:**
```bash
# 檢查 NVIDIA 驅動
nvidia-smi

# 重新安裝正確的 PyTorch 版本
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 檢查 CUDA 環境變數
echo $CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.1
```

### 問題 3: cuDNN 錯誤
```
RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED
```

**解決方案:**
```bash
# 清理 PyTorch 快取
python -c "import torch; torch.cuda.empty_cache()"

# 或重新安裝 PyTorch
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 問題 4: 訓練速度慢

**檢查清單:**
- [ ] 確認使用 GPU (不是 CPU)
- [ ] 啟用 `cudnn.benchmark = True`
- [ ] 適當的 `num_workers` 設置
- [ ] 資料存儲在 SSD (不是 HDD)
- [ ] 檢查 GPU 利用率: `nvidia-smi dmon -s u`

## 📊 性能基準

在 RTX 5090 上的預期性能:

| Batch Size | Image Size | 記憶體使用 | 訓練速度 (it/s) |
|------------|------------|-----------|----------------|
| 2          | 224x224    | ~8 GB     | ~2.5           |
| 4          | 224x224    | ~14 GB    | ~4.0           |
| 8          | 224x224    | ~22 GB    | ~6.5           |
| 4          | 256x256    | ~18 GB    | ~3.0           |

*注意: 實際性能取決於 CPU、RAM 速度和資料載入速度*

## 🔍 監控訓練

### 使用 Weights & Biases (推薦)

```bash
# 設置 W&B
pip install wandb
wandb login

# 訓練時啟用 wandb
python train.py --use_wandb --wandb_name "swintexco_rtx5090_experiment1"
```

### 手動監控

```bash
# 終端 1: 訓練
python train.py ...

# 終端 2: 監控 GPU
watch -n 1 nvidia-smi

# 終端 3: 監控檢查點大小
watch -n 10 du -sh checkpoints/
```

## 📚 資料集準備

確保資料集路徑正確設置:

```bash
# 資料集結構範例
/path/to/data/
├── DAVIS/
│   └── davis_videos/
├── ImageNet/
│   └── imagenet-vc/
└── annotations/
    ├── davis_annot.csv
    └── pairs.txt
```

## 🎯 下一步

1. ✅ 完成環境設置
2. 📁 準備資料集
3. 🧪 運行小規模測試訓練
4. 🚀 啟動完整訓練
5. 📊 監控和調優

## 💡 額外資源

- [PyTorch CUDA 語義](https://pytorch.org/docs/stable/notes/cuda.html)
- [NVIDIA CUDA 安裝指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [SwinTExCo 原始倉庫](https://github.com/Chronopt-Research/SwinTExCo)

---

**需要幫助?** 請在 GitHub Issues 中報告問題。

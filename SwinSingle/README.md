# SwinTExCo Module

此目錄包含 SwinTExCo (Swin Transformer Exemplar-based Colorization) 變體的代碼。

## 🚀 快速開始 (RTX 5090 / 現代 GPU)

### 1. 環境設置

```bash
# 自動設置 (推薦)
./setup_rtx5090_env.sh

# 或手動安裝
pip install -r requirements.txt
```

### 2. 驗證環境

```bash
./quick_test.sh
```

### 3. 開始訓練

```bash
# 使用預配置腳本 (需要先修改資料集路徑)
./train_rtx5090.sh

# 或使用命令行
python train.py --help
```

**完整設置指南**: 請查看 [SETUP_RTX5090.md](SETUP_RTX5090.md)

---

## 如何獲取原始代碼

從 SwinTExCo 官方倉庫複製以下文件到此目錄：

```bash
# 方法 1: 從 GitHub 克隆
git clone https://github.com/Chronopt-Research/SwinTExCo.git temp_swintexco
cp -r temp_swintexco/src ./SwinSingle/
cp temp_swintexco/train.py ./SwinSingle/
cp temp_swintexco/inference.py ./SwinSingle/
rm -rf temp_swintexco

# 方法 2: 從本地 swinthxco_single 倉庫複製
cp -r swinthxco_single/src ./SwinSingle/
cp swinthxco_single/inference.py ./SwinSingle/
cp swinthxco_single/train.py ./SwinSingle/
```

## 必要修改

請確保已修改 SwinTExCo 以返回 `similarity_map`：

### 修改 1: src/models/CNN/FrameColor.py
```python
# 修改返回值從:
return IA_ab_predict, nonlocal_BA_lab

# 改為:
return IA_ab_predict, nonlocal_BA_lab, similarity_map
```

### 修改 2: train.py 和 inference.py
相應更新接收返回值。

## 目錄結構

```
SwinSingle/
├── src/                        # SwinTExCo 源代碼
│   ├── models/
│   │   ├── CNN/
│   │   │   ├── FrameColor.py
│   │   │   ├── WarpNet.py
│   │   │   └── ...
│   │   ├── Transformer/
│   │   └── vit/
│   ├── data/                   # 資料載入器
│   ├── losses/                 # 損失函數
│   └── utils/                  # 工具函數
├── inference.py                # 推理腳本
├── train.py                    # 訓練腳本
├── requirements.txt            # Python 依賴
├── setup_rtx5090_env.sh       # 環境設置腳本
├── train_rtx5090.sh           # 訓練啟動腳本 (RTX 5090 優化)
├── quick_test.sh              # 環境測試腳本
├── configs_rtx5090.yaml       # 訓練配置文件
├── SETUP_RTX5090.md           # 詳細設置指南
└── README.md                   # 本文件
```

## 📚 相關文件

- **[SETUP_RTX5090.md](SETUP_RTX5090.md)** - RTX 5090 環境設置完整指南
- **[configs_rtx5090.yaml](configs_rtx5090.yaml)** - 訓練配置範例
- **[requirements.txt](requirements.txt)** - Python 依賴列表

## 🎯 訓練配置建議

| GPU 型號 | VRAM | Batch Size | Image Size | Workers |
|----------|------|------------|------------|---------|
| RTX 5090 | 24GB | 6-8        | 224x224    | 4-8     |
| RTX 4090 | 24GB | 6-8        | 224x224    | 4-8     |
| RTX 4080 | 16GB | 4          | 224x224    | 4       |
| RTX 3090 | 24GB | 4-6        | 224x224    | 4       |
| RTX 3080 | 10GB | 2          | 192x192    | 2       |

#!/bin/bash
# 快速測試腳本 - 驗證環境設置是否正確

echo "======================================"
echo "SwinTExCo 環境測試"
echo "======================================"
echo ""

# 測試 1: Python 版本
echo "[1/6] 檢查 Python 版本..."
PYTHON_VERSION=$(python --version 2>&1)
echo "✓ $PYTHON_VERSION"
echo ""

# 測試 2: NVIDIA 驅動和 GPU
echo "[2/6] 檢查 GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "✓ GPU 檢測成功"
else
    echo "❌ 未找到 nvidia-smi"
    exit 1
fi
echo ""

# 測試 3: PyTorch
echo "[3/6] 檢查 PyTorch..."
python -c "
import sys
try:
    import torch
    print(f'✓ PyTorch {torch.__version__}')
    if torch.cuda.is_available():
        print(f'✓ CUDA {torch.version.cuda}')
        print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
        print(f'✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    else:
        print('❌ CUDA 不可用')
        sys.exit(1)
except ImportError as e:
    print(f'❌ PyTorch 未安裝: {e}')
    sys.exit(1)
" || exit 1
echo ""

# 測試 4: 核心依賴
echo "[4/6] 檢查核心依賴..."
python -c "
import sys
missing = []
for pkg in ['einops', 'timm', 'wandb', 'pandas', 'tqdm', 'cv2', 'PIL']:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'❌ {pkg} 未安裝')
        missing.append(pkg)

if missing:
    print(f'\n請安裝缺失的套件: pip install {\" \".join(missing)}')
    sys.exit(1)
" || exit 1
echo ""

# 測試 5: 檢查目錄結構
echo "[5/6] 檢查目錄結構..."
required_dirs=("src" "src/models" "src/data" "src/losses" "src/utils")
all_exist=true
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✓ $dir"
    else
        echo "❌ $dir 不存在"
        all_exist=false
    fi
done
echo ""

# 測試 6: 檢查關鍵文件
echo "[6/6] 檢查關鍵文件..."
required_files=("train.py" "inference.py" "requirements.txt")
all_exist=true
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "❌ $file 不存在"
        all_exist=false
    fi
done
echo ""

# 最終結果
echo "======================================"
if [ "$all_exist" = true ]; then
    echo "✅ 所有檢查通過!"
    echo "======================================"
    echo ""
    echo "下一步:"
    echo "1. 準備資料集"
    echo "2. 修改 train_rtx5090.sh 中的資料集路徑"
    echo "3. 運行: ./train_rtx5090.sh"
    echo ""
    echo "或者運行快速訓練測試 (使用小資料集):"
    echo "python train.py --help"
else
    echo "❌ 部分檢查失敗"
    echo "======================================"
    echo "請修復上述問題後重試"
    exit 1
fi

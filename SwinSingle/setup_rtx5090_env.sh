#!/bin/bash
# SwinTExCo 訓練環境設置腳本 - 針對 RTX 5090 優化

echo "======================================"
echo "SwinTExCo RTX 5090 環境設置"
echo "======================================"

# 檢查 NVIDIA 驅動
echo -e "\n[1/6] 檢查 GPU 和 CUDA..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 錯誤: 未找到 nvidia-smi"
    echo "請先安裝 NVIDIA 驅動程式 (建議 >= 545.x 支援 CUDA 12.x)"
    exit 1
fi

nvidia-smi
echo "✓ GPU 檢測成功"

# 檢查 CUDA 版本
if command -v nvcc &> /dev/null; then
    echo "CUDA 版本:"
    nvcc --version
else
    echo "⚠️  警告: 未找到 nvcc,但可以使用 conda 或 pip 安裝的 CUDA runtime"
fi

# 創建 conda 環境 (可選)
echo -e "\n[2/6] 創建 Python 環境..."
read -p "是否使用 conda 創建新環境? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ENV_NAME="swintexco_rtx5090"
    conda create -n $ENV_NAME python=3.10 -y
    echo "✓ 請執行: conda activate $ENV_NAME"
    echo "  然後重新運行此腳本"
    exit 0
fi

# 檢查 Python 版本
echo -e "\n[3/6] 檢查 Python 版本..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "當前 Python 版本: $PYTHON_VERSION"

# 安裝 PyTorch (CUDA 12.1)
echo -e "\n[4/6] 安裝 PyTorch..."
echo "為 RTX 5090 安裝 PyTorch 2.1.1 with CUDA 12.1"
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121

# 驗證 PyTorch 安裝
echo -e "\n[5/6] 驗證 PyTorch GPU 支援..."
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'CUDA 版本: {torch.version.cuda}'); print(f'GPU 數量: {torch.cuda.device_count()}'); print(f'GPU 名稱: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 安裝其他依賴
echo -e "\n[6/6] 安裝其他依賴..."
pip install -r requirements.txt

echo -e "\n======================================"
echo "✅ 環境設置完成!"
echo "======================================"
echo -e "\n下一步:"
echo "1. 檢查 GPU 記憶體: nvidia-smi"
echo "2. 測試訓練: python train.py --help"
echo "3. 開始訓練: 請參考 training_guide_rtx5090.md"

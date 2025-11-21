#!/bin/bash
# SwinTExCo 訓練啟動腳本 - RTX 5090 優化配置

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================"
echo "SwinTExCo 訓練 - RTX 5090 配置"
echo -e "======================================${NC}\n"

# 檢查 GPU
echo -e "${YELLOW}[檢查] 驗證 GPU 可用性...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ 錯誤: 未找到 nvidia-smi${NC}"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
echo -e "${GREEN}✓ 檢測到 GPU: $GPU_NAME${NC}"
echo -e "${GREEN}✓ GPU 記憶體: ${GPU_MEMORY} MB${NC}\n"

# 檢查 PyTorch
echo -e "${YELLOW}[檢查] 驗證 PyTorch CUDA 支援...${NC}"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✓ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')" || exit 1
echo ""

# 配置參數
# ============================================
# 請根據你的資料集路徑修改以下變數
# ============================================

# 資料集路徑 (請修改為實際路徑)
VIDEO_DATA_ROOT="${VIDEO_DATA_ROOT:-../dataset/DAVIS/davis_videos,../dataset/Hollywood2_Pixabay/images}"
DATA_ROOT_IMAGENET="${DATA_ROOT_IMAGENET:-../dataset/ImageNet/imagenet-vc}"
ANNOTATION_FILES="${ANNOTATION_FILES:-../dataset/DAVIS/davis_annot.csv}"
IMAGENET_PAIRS="${IMAGENET_PAIRS:-../dataset/pairs.txt}"

# 訓練參數
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224 224}"
EPOCHS="${EPOCHS:-40}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# 檢查點目錄
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints_rtx5090}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-100}"

# W&B 設置 (可選)
USE_WANDB="${USE_WANDB:-false}"
WANDB_NAME="${WANDB_NAME:-swintexco_rtx5090_$(date +%Y%m%d_%H%M%S)}"

# 其他訓練參數
WEIGHT_L1="${WEIGHT_L1:-1.0}"
WEIGHT_CONTEXTUAL="${WEIGHT_CONTEXTUAL:-0.015}"
WEIGHT_PERCEPTUAL="${WEIGHT_PERCEPTUAL:-0.15}"
WEIGHT_SMOOTHNESS="${WEIGHT_SMOOTHNESS:-150.0}"
WEIGHT_GAN="${WEIGHT_GAN:-0.015}"

# ============================================
# 自動配置建議 (基於 GPU 記憶體)
# ============================================

if [ "$GPU_MEMORY" -gt 20000 ]; then
    echo -e "${GREEN}檢測到高階 GPU (>20GB VRAM)${NC}"
    echo -e "${GREEN}建議使用: batch_size=8, image_size=224${NC}"
    SUGGESTED_BS=8
    SUGGESTED_ACCUM=2
elif [ "$GPU_MEMORY" -gt 10000 ]; then
    echo -e "${YELLOW}檢測到中階 GPU (10-20GB VRAM)${NC}"
    echo -e "${YELLOW}建議使用: batch_size=4, image_size=224${NC}"
    SUGGESTED_BS=4
    SUGGESTED_ACCUM=4
else
    echo -e "${RED}檢測到較小 GPU (<10GB VRAM)${NC}"
    echo -e "${RED}建議使用: batch_size=2, image_size=192${NC}"
    SUGGESTED_BS=2
    SUGGESTED_ACCUM=8
fi

echo ""

# 顯示配置
echo -e "${YELLOW}======================================"
echo "訓練配置"
echo -e "======================================${NC}"
echo "資料集路徑:"
echo "  - Video: $VIDEO_DATA_ROOT"
echo "  - ImageNet: $DATA_ROOT_IMAGENET"
echo "  - Annotations: $ANNOTATION_FILES"
echo "  - Pairs: $IMAGENET_PAIRS"
echo ""
echo "訓練參數:"
echo "  - Batch Size: $BATCH_SIZE (建議: $SUGGESTED_BS)"
echo "  - Accumulation Steps: $ACCUMULATION_STEPS (建議: $SUGGESTED_ACCUM)"
echo "  - Image Size: $IMAGE_SIZE"
echo "  - Epochs: $EPOCHS"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - Workers: $NUM_WORKERS"
echo ""
echo "輸出:"
echo "  - Checkpoint Dir: $CHECKPOINT_DIR"
echo "  - Checkpoint Step: $CHECKPOINT_STEP"
echo "  - W&B: $USE_WANDB"
if [ "$USE_WANDB" = "true" ]; then
    echo "  - W&B Name: $WANDB_NAME"
fi
echo ""

# 確認開始訓練
read -p "按 Enter 開始訓練,或按 Ctrl+C 取消..."

# 創建檢查點目錄
mkdir -p "$CHECKPOINT_DIR"

# 構建訓練命令
TRAIN_CMD="python train.py \
    --video_data_root_list $VIDEO_DATA_ROOT \
    --data_root_imagenet $DATA_ROOT_IMAGENET \
    --annotation_file_path_list $ANNOTATION_FILES \
    --imagenet_pairs_file $IMAGENET_PAIRS \
    --batch_size $BATCH_SIZE \
    --accumulation_steps $ACCUMULATION_STEPS \
    --image_size $IMAGE_SIZE \
    --epoch $EPOCHS \
    --lr $LEARNING_RATE \
    --workers $NUM_WORKERS \
    --checkpoint_dir $CHECKPOINT_DIR \
    --checkpoint_step $CHECKPOINT_STEP \
    --weight_l1 $WEIGHT_L1 \
    --weight_contextual $WEIGHT_CONTEXTUAL \
    --weight_perceptual $WEIGHT_PERCEPTUAL \
    --weight_smoothness $WEIGHT_SMOOTHNESS \
    --weight_gan $WEIGHT_GAN"

# 添加 W&B 支援
if [ "$USE_WANDB" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --use_wandb --wandb_name $WANDB_NAME"
fi

# 顯示完整命令
echo -e "\n${YELLOW}執行命令:${NC}"
echo "$TRAIN_CMD"
echo ""

# 啟用 cuDNN benchmark
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# 開始訓練
echo -e "${GREEN}======================================"
echo "開始訓練..."
echo -e "======================================${NC}\n"

eval $TRAIN_CMD

# 訓練結束
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}======================================"
    echo "✅ 訓練完成!"
    echo -e "======================================${NC}"
    echo "檢查點保存在: $CHECKPOINT_DIR"
else
    echo -e "\n${RED}======================================"
    echo "❌ 訓練失敗"
    echo -e "======================================${NC}"
    exit 1
fi

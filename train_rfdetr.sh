#!/bin/bash
# RF-DETR fine-tuning
# 실행 전: conda activate rfdetr && cd ~/REDRED

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rfdetr

DATASET_DIR="data/coco_rfdetr"
NUM_CLASSES=60
EPOCHS=${1:-50}
BATCH=${2:-8}
OUT_DIR="runs/rfdetr"

# Step 1: 기존 학습 데이터 COCO 변환 (이미 있으면 스킵)
if [ ! -f "${DATASET_DIR}/train/_annotations.coco.json" ]; then
    echo "=== YOLO → COCO 변환 ==="
    python tools/yolo_to_coco.py \
        --train_txt ~/yolov7/data/train.txt \
        --names     data/names.txt \
        --out_dir   ${DATASET_DIR} \
        --symlink
fi

# Step 2: (선택) SAM2 도메인 데이터 추출
if [ "$3" = "--with_sam2" ]; then
    echo "=== SAM2 도메인 데이터 추출 ==="
    python tools/sam2_video_label.py \
        --videos ~/Dataset/4.TestVideo_Sample/cam0/Sample_1.mp4 \
                 ~/Dataset/4.TestVideo_Sample/cam1/Sample_1.mp4 \
                 ~/Dataset/4.TestVideo_Sample/cam2/Sample_1.mp4 \
                 ~/Dataset/4.TestVideo_Sample/cam3/Sample_1.mp4 \
                 ~/Dataset/4.TestVideo_Sample/cam4/Sample_1.mp4 \
        --weights ~/Dataset/yolov7_custom.pt \
        --names   data/names.txt \
        --sam2_ckpt ~/checkpoints/sam2/sam2.1_hiera_large.pt \
        --out_dir  ${DATASET_DIR} \
        --interval 30
fi

# Step 3: RF-DETR 학습
echo "=== RF-DETR 학습 (epochs=${EPOCHS}, batch=${BATCH}) ==="
python - <<EOF
from rfdetr import RFDETRBase
import os

model = RFDETRBase(num_classes=${NUM_CLASSES}, resolution=640)
model.train(
    dataset_dir="${DATASET_DIR}",
    epochs=${EPOCHS},
    batch_size=${BATCH},
    lr=1e-4,
    output_dir="${OUT_DIR}",
    checkpoint_interval=10,
)
print("학습 완료 →", os.path.abspath("${OUT_DIR}"))
EOF

echo "=== Done. 모델: ${OUT_DIR}/checkpoint_best.pth ==="

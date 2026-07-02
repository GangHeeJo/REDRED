#!/bin/bash
# RF-DETR fine-tuning
# 실행 전: conda activate rfdetr && cd ~/REDRED
# Usage: bash train_rfdetr.sh [epochs=30] [batch=8] [--with_sam2] [--rebuild]
# 2026-07-02: epoch1/50 만에 이미 mAP 0.887(지난 50에폭 최종치 0.898에 근접) --
# 수렴이 빨라서 기본 에폭을 50->30으로 낮춤. 필요하면 checkpoint_best_regular.pth
# 를 pretrain_weights로 넣고 추가로 더 돌릴 수 있음(진짜 resume은 아니고 이어서
# fine-tune하는 방식).
#
# 2026-07-02: --rebuild 추가 -- 미탐지 클래스(WEAK_CLASS_IDS)+타이밍 오차 큰
# 클래스(TIMING_ISSUE_CLASS_IDS, 둘 다 tools/yolo_to_coco.py 참고) 보강을 위해
# COCO 변환을 --oversample_weak 5로 다시 만듦(3→5, 효과 애매하면 강도를
# 높이자는 결정). 기존 방식은 train/_annotations.coco.json이 있으면 그냥
# 스킵해서 이 보강이 반영 안 됨.

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rfdetr

DATASET_DIR="data/coco_rfdetr"
NUM_CLASSES=60
EPOCHS=${1:-30}
BATCH=${2:-8}
OUT_DIR="runs/rfdetr"

# Step 1: 기존 학습 데이터 COCO 변환 (이미 있으면 스킵, --rebuild면 강제 재생성)
if [ "$4" = "--rebuild" ] || [ ! -f "${DATASET_DIR}/train/_annotations.coco.json" ]; then
    echo "=== YOLO → COCO 변환 (약한/타이밍오차 클래스 5배 오버샘플링) ==="
    python tools/yolo_to_coco.py \
        --train_txt ~/yolov7/data/train.txt \
        --names     data/names.txt \
        --out_dir   ${DATASET_DIR} \
        --oversample_weak 5 \
        --symlink
fi

# Step 2: (선택) SAM2 도메인 데이터 추출 + train에 병합
if [ "$3" = "--with_sam2" ]; then
    echo "=== SAM2 도메인 데이터 추출 (RF-DETR 체크포인트 사용) ==="
    python tools/sam2_video_label.py \
        --videos ~/Dataset/4.TestVideo_Sample/cam0/Sample_1.mp4 \
                 ~/Dataset/4.TestVideo_Sample/cam1/Sample_1.mp4 \
                 ~/Dataset/4.TestVideo_Sample/cam2/Sample_1.mp4 \
                 ~/Dataset/4.TestVideo_Sample/cam3/Sample_1.mp4 \
                 ~/Dataset/4.TestVideo_Sample/cam4/Sample_1.mp4 \
        --weights ${OUT_DIR}/checkpoint_best_total.pth \
        --names   data/names.txt \
        --sam2_ckpt ~/checkpoints/sam2/sam2.1_hiera_large.pt \
        --out_dir  ${DATASET_DIR} \
        --interval 30

    echo "=== SAM2 데이터 → train 병합 ==="
    python - <<'PYEOF'
import json
from pathlib import Path

dataset_dir = Path("data/coco_rfdetr")
train_json  = dataset_dir / "train" / "_annotations.coco.json"
sam2_json   = dataset_dir / "annotations" / "instances_video_domain.json"

if not sam2_json.exists():
    print("SAM2 JSON 없음, 스킵")
    exit()

with open(train_json) as f:
    train = json.load(f)
with open(sam2_json) as f:
    extra = json.load(f)

# ID 충돌 방지: offset 적용
img_offset = max(img["id"] for img in train["images"]) + 1
ann_offset = max(ann["id"] for ann in train["annotations"]) + 1

for img in extra["images"]:
    old_id = img["id"]
    img["id"] += img_offset
    for ann in extra["annotations"]:
        if ann["image_id"] == old_id:
            ann["image_id"] = img["id"]

for ann in extra["annotations"]:
    ann["id"] += ann_offset

# video_domain 이미지는 train/images/ 대신 images/video_domain/에 있음
# file_name에 경로 prefix 추가 (rfdetr는 dataset_dir 기준으로 읽음)
for img in extra["images"]:
    if not img["file_name"].startswith("../"):
        img["file_name"] = "../images/" + img["file_name"]

train["images"]      += extra["images"]
train["annotations"] += extra["annotations"]

with open(train_json, "w") as f:
    json.dump(train, f)

print(f"병합 완료: train {len(train['images'])}장, {len(train['annotations'])}개 어노테이션")
PYEOF
fi

# Step 3: RF-DETR 학습 (resolution=672, 56의 배수)
echo "=== RF-DETR 학습 (epochs=${EPOCHS}, batch=${BATCH}) ==="
python - <<EOF
from rfdetr import RFDETRBase
import os

model = RFDETRBase(num_classes=${NUM_CLASSES}, resolution=672)
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

echo "=== Done. 모델: ${OUT_DIR}/checkpoint_best_total.pth ==="

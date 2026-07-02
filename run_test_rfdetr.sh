#!/bin/bash
# RF-DETR 파이프라인 테스트 (YOLOv7 드롭인 교체)
# Usage: bash run_test_rfdetr.sh [skip=2]

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rfdetr

SKIP=${1:-2}
WEIGHTS="runs/rfdetr/checkpoint_best_total.pth"
CAM_DIR=~/Dataset/4.TestVideo_Sample
OUT="output/submission_rfdetr_skip${SKIP}.csv"
DEBUG_LOG="output/debug_frame_counts_rfdetr.csv"
PER_CAM_LOG="output/per_cam_rfdetr.csv"

python src/run_pipeline_rfdetr.py \
    --videos  ${CAM_DIR}/cam0/Sample_1.mp4 \
              ${CAM_DIR}/cam1/Sample_1.mp4 \
              ${CAM_DIR}/cam2/Sample_1.mp4 \
              ${CAM_DIR}/cam3/Sample_1.mp4 \
              ${CAM_DIR}/cam4/Sample_1.mp4 \
    --weights ${WEIGHTS} \
    --names   data/names.txt \
    --prices  data/prices.csv \
    --out     ${OUT} \
    --skip    ${SKIP} \
    --conf    0.4 \
    --device  0 \
    --debug_log ${DEBUG_LOG} \
    --per_cam_log ${PER_CAM_LOG}

echo "=== 채점 ==="
python tools/score.py \
    --sub ${OUT} \
    --gt  data/ground_truth_v2.csv

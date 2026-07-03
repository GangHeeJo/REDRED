#!/bin/bash
# RF-DETR 전용 새 파이프라인(rfdetr_native_pipeline.py) 테스트
# Usage: bash run_test_rfdetr_native.sh [skip=3] [conf=0.35] [weights=...] [fusion_mode=vote|noisy_or] [use_margin=0|1]
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rfdetr

SKIP=${1:-3}
CONF=${2:-0.35}
WEIGHTS="${3:-runs/rfdetr/checkpoint_best_total.pth}"
FUSION_MODE="${4:-vote}"
USE_MARGIN="${5:-0}"
MARGIN_FLAG=""
MARGIN_TAG="nomargin"
if [ "$USE_MARGIN" = "1" ]; then
    MARGIN_FLAG="--use_margin"
    MARGIN_TAG="margin"
fi
CAM_DIR=~/Dataset/4.TestVideo_Sample
TAG=$(basename "$WEIGHTS" | sed 's/\.[^.]*$//')
OUT="output/submission_native_skip${SKIP}_conf${CONF}_${FUSION_MODE}_${MARGIN_TAG}_${TAG}.csv"
DEBUG_LOG="output/debug_frame_counts_native.csv"
PER_CAM_LOG="output/per_cam_native.csv"
TIMED_LOG="output/timed_native.csv"

python src/rfdetr_native_pipeline.py \
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
    --conf    ${CONF} \
    --device  0 \
    --fusion_mode ${FUSION_MODE} \
    ${MARGIN_FLAG} \
    --class_config config/rfdetr_native_class_config.json \
    --debug_log ${DEBUG_LOG} \
    --per_cam_log ${PER_CAM_LOG} \
    --timed_log ${TIMED_LOG}

echo "=== 채점 ==="
python tools/score.py \
    --sub ${OUT} \
    --gt  data/ground_truth_v2.csv

echo "=== 3종 채점 (count/order/time) ==="
python tools/score_methods.py \
    --sub ${OUT} \
    --gt  data/ground_truth_v2.csv \
    --timed ${TIMED_LOG}

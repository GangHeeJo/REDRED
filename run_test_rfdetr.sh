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

# 2026-07-02: 6개 전부 적용했을 때 milano(42)/lindt(50)처럼 이 딕셔너리에 없는
# 클래스까지 깨지는 원인불명 부작용 발견(event_detector.py는 클래스별 완전
# 독립이라 코드상 이유를 못 찾음). 확실히 개선 확인된 3개만 남김
# (31=macadamia, 36=nature_valley, 46=chewy_dips_peanut_butter).
# a1_steak_sauce(17)/aunt_jemima(0)/dove_white(54)는 효과 없었거나 형태만
# 바뀌어서 제외.
PER_CLASS_CONFIRM='{"31":60,"36":60,"46":60}'

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
    --per_cam_log ${PER_CAM_LOG} \
    --per_class_confirm "${PER_CLASS_CONFIRM}"

echo "=== 채점 ==="
python tools/score.py \
    --sub ${OUT} \
    --gt  data/ground_truth_v2.csv

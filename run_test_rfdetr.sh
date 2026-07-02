#!/bin/bash
# RF-DETR 파이프라인 테스트 (YOLOv7 드롭인 교체)
# Usage: bash run_test_rfdetr.sh [skip=3] [conf=0.5]
#
# 2026-07-02: skip=2/conf=0.4는 RTF=1.26로 기준(<=1) 초과 -- score.py 기준
# RTF>1은 "상대평가, 공식 미공개"라 몇 점 깎이는지 알 수 없어 리스크가 큼.
# skip=3/conf=0.5(RTF=0.86, order F1 83.6%, 33.4/40)를 새 기본값으로 확정.

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rfdetr

SKIP=${1:-3}
CONF=${2:-0.5}
WEIGHTS="runs/rfdetr/checkpoint_best_total.pth"
CAM_DIR=~/Dataset/4.TestVideo_Sample
OUT="output/submission_rfdetr_skip${SKIP}_conf${CONF}.csv"
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
    --conf    ${CONF} \
    --device  0 \
    --debug_log ${DEBUG_LOG} \
    --per_cam_log ${PER_CAM_LOG} \
    --per_class_confirm "${PER_CLASS_CONFIRM}"

echo "=== 채점 ==="
python tools/score.py \
    --sub ${OUT} \
    --gt  data/ground_truth_v2.csv

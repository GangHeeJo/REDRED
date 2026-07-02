#!/bin/bash
# RF-DETR conf 임계값 스윕 (재학습 없이 기존 체크포인트로 conf만 바꿔 비교)
# 실행: bash tools/sweep_rfdetr_conf.sh [skip=2]
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rfdetr

SKIP=${1:-2}
WEIGHTS="runs/rfdetr/checkpoint_best_total.pth"
CAM_DIR=~/Dataset/4.TestVideo_Sample
PER_CLASS_CONFIRM='{"31":60,"36":60,"46":60}'
SUMMARY="output/conf_sweep_summary.txt"
> "$SUMMARY"

for CONF in 0.4 0.5 0.6; do
    echo "=== conf=${CONF} ==="
    OUT="output/submission_rfdetr_conf${CONF}.csv"
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
        --per_class_confirm "${PER_CLASS_CONFIRM}"

    echo "--- conf=${CONF} score ---" | tee -a "$SUMMARY"
    python tools/score.py --sub ${OUT} --gt data/ground_truth_v2.csv | tee -a "$SUMMARY"
    echo "" >> "$SUMMARY"
done

echo ""
echo "=== conf 스윕 요약 (order F1만) ==="
grep -E "conf=|F1 \(order\)" "$SUMMARY"
echo ""
echo "전체 로그: $SUMMARY"

#!/bin/bash
# 동일 설정으로 RF-DETR 파이프라인을 N번 반복 실행해 실행간 비결정성 폭 측정
# 실행: bash tools/measure_rfdetr_variance.sh [N=3] [skip=2]
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rfdetr

N=${1:-3}
SKIP=${2:-2}
WEIGHTS="runs/rfdetr/checkpoint_best_total.pth"
CAM_DIR=~/Dataset/4.TestVideo_Sample
PER_CLASS_CONFIRM='{"0":60,"17":60,"31":60,"36":60,"46":60,"54":60}'

SUMMARY="output/variance_summary.txt"
> "$SUMMARY"

for i in $(seq 1 $N); do
    echo "=== RUN $i/$N ==="
    OUT="output/submission_rfdetr_variance_run${i}.csv"
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
        --per_class_confirm "${PER_CLASS_CONFIRM}"

    echo "--- RUN $i score ---" | tee -a "$SUMMARY"
    python tools/score.py --sub ${OUT} --gt data/ground_truth_v2.csv | tee -a "$SUMMARY"
    echo "" >> "$SUMMARY"
done

echo ""
echo "=== 3회 요약 (order F1 라인만) ==="
grep "F1 (order)" "$SUMMARY"
echo ""
echo "전체 로그: $SUMMARY"
echo "각 회차 제출 CSV: output/submission_rfdetr_variance_run{1..${N}}.csv"

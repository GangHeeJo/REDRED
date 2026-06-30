#!/bin/bash
# Event-Triggered 파이프라인 v3 테스트
# v3: trigger마다 before(N장 median) vs after(N장 median) 자체 비교
#     stable_counts 전역 상태 없음 → 오류 누적 차단
# Usage: bash run_test_et.sh [weights_path]

WEIGHTS=${1:-~/runs/kd/yolo11m_kd_0630_0036/weights/best.pt}

echo "=============================="
echo " Event-Triggered Pipeline v3"
echo " weights: $WEIGHTS"
echo " n_before=5  n_after=5"
echo "=============================="

python src/run_event_triggered.py \
    --videos ~/Dataset/4.TestVideo_Sample/cam0/Sample_1.mp4 \
             ~/Dataset/4.TestVideo_Sample/cam1/Sample_1.mp4 \
             ~/Dataset/4.TestVideo_Sample/cam2/Sample_1.mp4 \
             ~/Dataset/4.TestVideo_Sample/cam3/Sample_1.mp4 \
             ~/Dataset/4.TestVideo_Sample/cam4/Sample_1.mp4 \
    --weights "$WEIGHTS" \
    --names data/names.txt \
    --prices data/prices.csv \
    --out output/submission_et.csv \
    --conf 0.4 \
    --device 0 \
    --quorum 2 \
    --n_before 5 \
    --n_after 5 \
    --timed_log output/sub_et_timed.csv

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Event-Triggered 채점 ==="
    python tools/score_methods.py \
        --gt data/ground_truth_v2.csv \
        --sub output/submission_et.csv \
        --timed output/sub_et_timed.csv

    echo ""
    echo "=== 비교: ET vs ByteSort(quorum=2) vs Phase24 ==="
    echo "[ByteSort quorum=2]"
    python tools/score_methods.py \
        --gt data/ground_truth_v2.csv \
        --sub output/submission_bytetrack.csv \
        --timed output/sub_bytetrack_timed.csv 2>/dev/null || echo "  (결과 없음)"

    echo "[Phase24 YOLOv7]"
    python tools/score_methods.py \
        --gt data/ground_truth_v2.csv \
        --sub output/submission_skip2.csv \
        --timed output/sub_events_timed.csv 2>/dev/null || echo "  (결과 없음)"
fi

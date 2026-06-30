#!/bin/bash
# KD 학습된 YOLO11 모델로 파이프라인 테스트
# Usage: bash run_test_kd.sh [skip] [weights_path]
# Example: bash run_test_kd.sh 2 ~/runs/kd/yolo11m_kd_0630_0036/weights/best.pt

SKIP=${1:-2}
WEIGHTS=${2:-~/runs/kd/yolo11m_kd_0630_0036/weights/best.pt}
DESC="KD yolo11m skip=${SKIP} $(date '+%m-%d %H:%M')"

echo "=============================="
echo " KD Pipeline Test"
echo " weights: $WEIGHTS"
echo " skip: $SKIP"
echo "=============================="

python src/run_pipeline.py \
    --videos ~/Dataset/4.TestVideo_Sample/cam0/Sample_1.mp4 \
             ~/Dataset/4.TestVideo_Sample/cam1/Sample_1.mp4 \
             ~/Dataset/4.TestVideo_Sample/cam2/Sample_1.mp4 \
             ~/Dataset/4.TestVideo_Sample/cam3/Sample_1.mp4 \
             ~/Dataset/4.TestVideo_Sample/cam4/Sample_1.mp4 \
    --weights "$WEIGHTS" \
    --names data/names.txt \
    --prices data/prices.csv \
    --out output/submission_kd_skip${SKIP}.csv \
    --skip $SKIP \
    --conf 0.4 \
    --device 0 \
    --use_tracker \
    --tracker_max_age 15 \
    --debug_log output/debug_kd_frame_counts.csv \
    --timed_log output/sub_kd_events_timed.csv \
    --per_cam_log output/per_cam_kd_tuned.csv

if [ $? -eq 0 ] && [ -f output/run_stats.json ]; then
    RTF=$(python -c "import json; print(json.load(open('output/run_stats.json'))['rtf'])")
    echo ""
    echo "=== KD 모델 채점 (RTF=$RTF) ==="
    python tools/score_methods.py \
        --gt data/ground_truth_v2.csv \
        --sub output/submission_kd_skip${SKIP}.csv \
        --timed output/sub_kd_events_timed.csv

    echo ""
    echo "=== Phase 24(YOLOv7) vs KD(YOLO11) 비교 ==="
    echo "Phase 24 기준선:"
    python tools/score_methods.py \
        --gt data/ground_truth_v2.csv \
        --sub output/submission_skip2.csv \
        --timed output/sub_events_timed.csv 2>/dev/null || echo "  (Phase 24 결과 없음)"

    echo ""
    echo "=== GitHub 업로드 ==="
    git add output/submission_kd_skip${SKIP}.csv \
            output/sub_kd_events_timed.csv \
            output/debug_kd_frame_counts.csv \
            output/per_cam_kd_tuned.csv 2>/dev/null || true
    git commit -m "result: KD skip=${SKIP} RTF=${RTF} $(date '+%m-%d %H:%M')"
    git push
fi

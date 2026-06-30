#!/bin/bash
# YOLO11 KD 모델 + --no_tuning (YOLOv7 특화 튜닝 전면 제거) 베이스라인 테스트
# Usage: bash run_test_kd_clean.sh [skip] [weights_path]

SKIP=${1:-2}
WEIGHTS=${2:-~/runs/kd/yolo11m_kd_0630_0036/weights/best.pt}
DESC="KD_clean no_tuning skip=${SKIP} $(date '+%m-%d %H:%M')"

echo "=============================="
echo " KD Clean Baseline Test"
echo " weights: $WEIGHTS"
echo " skip: $SKIP"
echo " no_tuning: ON"
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
    --out output/submission_kd_clean_skip${SKIP}.csv \
    --skip $SKIP \
    --conf 0.4 \
    --device 0 \
    --use_tracker \
    --tracker_max_age 15 \
    --quorum 2 \
    --init_min_detections 1 \
    --per_class_confirm '{"8":120,"28":60}' \
    --debug_log output/debug_kd_clean_frame_counts.csv \
    --timed_log output/sub_kd_clean_events_timed.csv \
    --per_cam_log output/per_cam_kd_clean.csv

if [ $? -eq 0 ]; then
    echo ""
    echo "=== KD Clean 채점 ==="
    python tools/score_methods.py \
        --gt data/ground_truth_v2.csv \
        --sub output/submission_kd_clean_skip${SKIP}.csv \
        --timed output/sub_kd_clean_events_timed.csv

    echo ""
    echo "=== 비교: KD_clean vs KD_tuned vs Phase24 ==="
    echo "[KD_clean (no_tuning)]"
    python tools/score_methods.py \
        --gt data/ground_truth_v2.csv \
        --sub output/submission_kd_clean_skip${SKIP}.csv \
        --timed output/sub_kd_clean_events_timed.csv 2>/dev/null

    echo "[KD_tuned (YOLOv7 파라미터 그대로)]"
    python tools/score_methods.py \
        --gt data/ground_truth_v2.csv \
        --sub output/submission_kd_skip${SKIP}.csv \
        --timed output/sub_kd_events_timed.csv 2>/dev/null || echo "  (KD_tuned 결과 없음)"

    echo "[Phase24 YOLOv7 baseline]"
    python tools/score_methods.py \
        --gt data/ground_truth_v2.csv \
        --sub output/submission_skip2.csv \
        --timed output/sub_events_timed.csv 2>/dev/null || echo "  (Phase24 결과 없음)"

    echo ""
    echo "=== Per-camera 분석 ==="
    python tools/analyze_per_cam.py \
        --per_cam output/per_cam_kd_clean.csv \
        --names data/names.txt

    echo ""
    echo "=== GitHub 업로드 ==="
    git add output/submission_kd_clean_skip${SKIP}.csv \
            output/sub_kd_clean_events_timed.csv \
            output/per_cam_kd_clean.csv \
            output/debug_kd_clean_frame_counts.csv 2>/dev/null || true
    git commit -m "result: KD_clean skip=${SKIP} $(date '+%m-%d %H:%M')"
    git push
fi

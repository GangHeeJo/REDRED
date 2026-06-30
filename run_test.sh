#!/bin/bash
# Usage: bash run_test.sh [skip] ["설명"]
# Example: bash run_test.sh 2 "UNKNOWN 제거 + history reset"

SKIP=${1:-2}
DESC=${2:-"skip=${SKIP} $(date '+%m-%d %H:%M')"}
YOLOV7=~/yolov7
WEIGHTS=~/Dataset/yolov7_custom.pt

PYTHONPATH=$YOLOV7 python src/run_pipeline.py \
    --videos ~/Dataset/4.TestVideo_Sample/cam0/Sample_1.mp4 \
             ~/Dataset/4.TestVideo_Sample/cam1/Sample_1.mp4 \
             ~/Dataset/4.TestVideo_Sample/cam2/Sample_1.mp4 \
             ~/Dataset/4.TestVideo_Sample/cam3/Sample_1.mp4 \
             ~/Dataset/4.TestVideo_Sample/cam4/Sample_1.mp4 \
    --weights $WEIGHTS \
    --names data/names.txt \
    --prices data/prices.csv \
    --out output/submission_skip${SKIP}.csv \
    --skip $SKIP \
    --conf 0.4 \
    --device 0 \
    --use_tracker \
    --tracker_max_age 15 \
    --debug_log output/debug_frame_counts.csv \
    --timed_log output/sub_events_timed.csv \
    --per_class_confirm '{"11":99,"43":31}'

# 파이프라인 성공 시 자동 채점 + 리더보드 갱신 + GitHub push
if [ $? -eq 0 ] && [ -f output/run_stats.json ]; then
    RTF=$(python -c "import json; print(json.load(open('output/run_stats.json'))['rtf'])")
    echo ""
    echo "=== 자동 채점 중 (RTF=$RTF) ==="
    python tools/score.py \
        --sub output/submission_skip${SKIP}.csv \
        --desc "$DESC" \
        --rtf "$RTF"

    echo ""
    echo "=== 3종 채점 방식 (count/order/time) ==="
    python tools/score_methods.py \
        --gt data/ground_truth_v2.csv \
        --sub output/submission_skip${SKIP}.csv \
        --timed output/sub_events_timed.csv

    echo ""
    echo "=== 리더보드 GitHub 업로드 ==="
    git add output/leaderboard.csv output/leaderboard.html
    git commit -m "leaderboard: $DESC (RTF=$RTF)"
    git push
fi

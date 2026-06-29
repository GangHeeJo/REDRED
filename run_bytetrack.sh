#!/bin/bash
SKIP=${1:-2}
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
    --out output/submission_bytetrack.csv \
    --skip $SKIP \
    --conf 0.4 \
    --device 0 \
    --use_tracker \
    --tracker_type bytetrack \
    --tracker_max_age 15 \
    --debug_log output/debug_bytetrack.csv \
    --timed_log output/sub_bytetrack_timed.csv

if [ $? -eq 0 ]; then
    echo ""
    echo "=== 3종 채점 (ByteTrack) ==="
    python tools/score_methods.py \
        --gt data/ground_truth_v2.csv \
        --sub output/submission_bytetrack.csv \
        --timed output/sub_bytetrack_timed.csv

    echo ""
    echo "=== 비교 (main Phase 24) ==="
    python tools/score_methods.py \
        --gt data/ground_truth_v2.csv \
        --sub output/submission_skip2.csv \
        --timed output/sub_events_timed.csv
fi

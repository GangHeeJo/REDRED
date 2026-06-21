#!/bin/bash
# Usage: bash run_test.sh [skip]
# Example: bash run_test.sh 2

SKIP=${1:-2}
YOLOV7=~/yolov7
WEIGHTS=~/Dataset/yolov7_custom.pt
VIDEOS="~/Dataset/4.TestVideo_Sample/cam0/Sample_1.mp4 \
        ~/Dataset/4.TestVideo_Sample/cam1/Sample_1.mp4 \
        ~/Dataset/4.TestVideo_Sample/cam2/Sample_1.mp4 \
        ~/Dataset/4.TestVideo_Sample/cam3/Sample_1.mp4 \
        ~/Dataset/4.TestVideo_Sample/cam4/Sample_1.mp4"

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
    --conf 0.5 \
    --device 0

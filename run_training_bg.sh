#!/bin/bash
# 창(xterm) 닫혀도 학습이 안 죽게 nohup+disown으로 백그라운드 실행
# Usage: bash run_training_bg.sh [epochs=30] [batch=8]
LOGFILE="train_rfdetr_$(date +%Y%m%d_%H%M%S).log"
nohup bash train_rfdetr.sh "${1:-30}" "${2:-8}" --with_sam2 --rebuild > "$LOGFILE" 2>&1 &
PID=$!
disown
echo "백그라운드 시작됨. PID: $PID"
echo "로그 파일: $LOGFILE"
echo "진행 확인: tail -f $LOGFILE"

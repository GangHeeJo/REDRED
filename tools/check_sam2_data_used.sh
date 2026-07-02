#!/bin/bash
# 특정 RF-DETR 학습이 SAM2 도메인 데이터를 실제로 포함했는지 확인
# 실행: bash tools/check_sam2_data_used.sh

DATASET_DIR="data/coco_rfdetr"
TRAIN_JSON="${DATASET_DIR}/train/_annotations.coco.json"
SAM2_JSON="${DATASET_DIR}/annotations/instances_video_domain.json"
CKPT="runs/rfdetr/checkpoint_best_total.pth"

echo "=== 1. SAM2 추출 결과 파일 존재 여부 ==="
if [ -f "$SAM2_JSON" ]; then
    echo "있음: $SAM2_JSON"
    python3 -c "
import json
d = json.load(open('$SAM2_JSON'))
print(f'  이미지 {len(d[\"images\"])}장, 어노테이션 {len(d[\"annotations\"])}개')
"
else
    echo "없음: $SAM2_JSON  → SAM2 라벨링 단계 자체가 실행 안 됐을 가능성"
fi

echo ""
echo "=== 2. train 어노테이션에 video_domain 이미지가 실제로 병합됐는지 ==="
if [ -f "$TRAIN_JSON" ]; then
    python3 -c "
import json
d = json.load(open('$TRAIN_JSON'))
vd = [img for img in d['images'] if 'video_domain' in img['file_name']]
print(f'  train 전체 이미지: {len(d[\"images\"])}장')
print(f'  그중 video_domain(SAM2) 유래 이미지: {len(vd)}장')
if vd:
    print(f'  예시 파일명: {vd[0][\"file_name\"]}')
"
else
    echo "없음: $TRAIN_JSON"
fi

echo ""
echo "=== 3. 파일 수정 시각 순서 (병합이 이번 체크포인트 학습 전에 일어났는지) ==="
ls -la --time-style=full-iso "$SAM2_JSON" "$TRAIN_JSON" "$CKPT" 2>/dev/null

echo ""
echo "=== 4. 최근 학습 로그에 SAM2 관련 출력이 있는지 (있으면) ==="
find . -maxdepth 2 -iname "*.log" -newer "$TRAIN_JSON" 2>/dev/null
echo "(로그 파일 경로를 알고 있으면 grep -i 'sam2\|병합' <logfile> 로 직접 확인)"

#!/bin/bash
# Final_submission/ 폴더를 조립하는 스크립트.
# main 브랜치(YOLOv7)와 feat/rfdetr-sam2 브랜치(RF-DETR+SAM2, 메인 제출모델)에서
# git archive로 실행에 필요한 파일만 뽑아온다. 브랜치 전환(checkout) 없이 동작하므로
# 현재 어느 브랜치에 있든 실행 가능.
#
# Usage: bash tools/build_final_submission.sh
set -e

OUT=~/Final_submission
echo "=== $OUT 생성 ==="
rm -rf "$OUT"
mkdir -p "$OUT/rfdetr_sam2/src" "$OUT/rfdetr_sam2/config" "$OUT/rfdetr_sam2/tools" "$OUT/rfdetr_sam2/data"
mkdir -p "$OUT/yolov7/src" "$OUT/yolov7/tools" "$OUT/yolov7/data"
mkdir -p "$OUT/weights_file"

echo "=== RF-DETR+SAM2 (메인) 파일 추출 — feat/rfdetr-sam2 브랜치 ==="
for f in src/rfdetr_native_pipeline.py src/infer_rfdetr.py src/rfdetr_margin_infer.py; do
    git show feat/rfdetr-sam2:"$f" > "$OUT/rfdetr_sam2/$f"
done
git show feat/rfdetr-sam2:config/rfdetr_native_class_config_v2_reinforced.json > "$OUT/rfdetr_sam2/config/rfdetr_native_class_config_v2_reinforced.json"
for f in tools/score.py tools/score_methods.py tools/yolo_to_coco.py tools/sam2_video_label.py; do
    git show feat/rfdetr-sam2:"$f" > "$OUT/rfdetr_sam2/$f"
done
for f in data/names.txt data/prices.csv data/ground_truth_v2.csv; do
    git show feat/rfdetr-sam2:"$f" > "$OUT/rfdetr_sam2/$f"
done
git show feat/rfdetr-sam2:run_test_rfdetr_native.sh > "$OUT/rfdetr_sam2/run_test_rfdetr_native.sh"
git show feat/rfdetr-sam2:requirements.txt > "$OUT/rfdetr_sam2/requirements.txt"
git show feat/rfdetr-sam2:setup_rfdetr_env.sh > "$OUT/rfdetr_sam2/setup_rfdetr_env.sh"
chmod +x "$OUT/rfdetr_sam2/run_test_rfdetr_native.sh" "$OUT/rfdetr_sam2/setup_rfdetr_env.sh"

echo "=== YOLOv7 (비교모델) 파일 추출 — main 브랜치 ==="
for f in src/run_pipeline.py src/event_detector.py src/multi_view_fusion.py src/tracker.py src/csv_generator.py; do
    git show main:"$f" > "$OUT/yolov7/$f"
done
for f in tools/score.py tools/score_methods.py; do
    git show main:"$f" > "$OUT/yolov7/$f"
done
for f in data/names.txt data/prices.csv data/ground_truth_v2.csv; do
    git show main:"$f" > "$OUT/yolov7/$f"
done
git show main:run_test.sh > "$OUT/yolov7/run_test.sh"
git show main:requirements.txt > "$OUT/yolov7/requirements.txt"
chmod +x "$OUT/yolov7/run_test.sh"

echo "=== 가중치 파일 복사 ==="
cp runs/rfdetr/checkpoint_best_total.pth "$OUT/weights_file/" 2>&1 || echo "!! checkpoint_best_total.pth 못 찾음, 경로 확인 필요"
cp ~/Dataset/yolov7_custom.pt "$OUT/weights_file/" 2>&1 || echo "!! yolov7_custom.pt 못 찾음, 경로 확인 필요"

echo "=== 완료 ==="
du -sh "$OUT"
find "$OUT" -type f | sort

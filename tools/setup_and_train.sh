#!/bin/bash
# REDRED — 서버 초기 셋업 + 학습 명령어 모음
# 실행 전: conda activate yolov7
# 위치: 이 스크립트는 ~/REDRED/tools/ 에 두고 ~/yolov7/ 에서 실행

YOLO_DIR="$HOME/yolov7"
DATA_DIR="$HOME/Dataset"
REDRED_DIR="$HOME/REDRED"

# ─────────────────────────────────────────────
# STEP 1: 데이터셋 압축 해제 (최초 1회)
# ─────────────────────────────────────────────
step1_extract() {
    cd "$DATA_DIR"
    tar -xf 1.competition_trainset.tar
    tar -xf 2.backsub_images_100.tar
    tar -xf "2.backsub_images_100_trans.tar" 2>/dev/null || true
    echo "[step1] extraction done"
}

# ─────────────────────────────────────────────
# STEP 2: train.txt / val.txt 생성
# ─────────────────────────────────────────────
step2_split() {
    cd "$DATA_DIR/1.competition_trainset"
    python make_list_dir.py          # → target.txt

    python "$REDRED_DIR/tools/make_split.py" \
        --list_file "$DATA_DIR/1.competition_trainset/target.txt" \
        --out_dir   "$YOLO_DIR/data" \
        --val_ratio 0.1 \
        --seed 42
    echo "[step2] train/val split done"
}

# ─────────────────────────────────────────────
# STEP 3: custom.yaml 생성
# ─────────────────────────────────────────────
step3_yaml() {
    python "$REDRED_DIR/tools/make_yaml.py" \
        --names "$DATA_DIR/names.txt" \
        --train "$YOLO_DIR/data/train.txt" \
        --val   "$YOLO_DIR/data/val.txt" \
        --out   "$YOLO_DIR/data/custom.yaml"
    echo "[step3] custom.yaml done"
}

# ─────────────────────────────────────────────
# STEP 4: 베이스라인 검증 (제공 가중치)
# ─────────────────────────────────────────────
step4_baseline_test() {
    cd "$YOLO_DIR"
    python test.py \
        --data    data/custom.yaml \
        --weights "$DATA_DIR/yolov7_custom.pt" \
        --batch-size 16 \
        --img-size 640 \
        --conf-thres 0.4 \
        --iou-thres  0.45 \
        --verbose \
        --name baseline
    echo "[step4] baseline test done → runs/test/baseline/"
}

# ─────────────────────────────────────────────
# STEP 5: Cut&Paste 증강 데이터 생성
# ─────────────────────────────────────────────
step5_augment() {
    python "$REDRED_DIR/tools/cut_paste_aug.py" \
        --seg_dir    "$DATA_DIR/2.backsub_images_100" \
        --bg_dir     "$DATA_DIR/3.Background_Images" \
        --out_dir    "$DATA_DIR/augmented" \
        --num_images 10000 \
        --max_objects 4

    # 증강 이미지를 train.txt에 추가
    find "$DATA_DIR/augmented/images" -name "*.jpg" >> "$YOLO_DIR/data/train.txt"
    echo "[step5] augmentation done, train.txt updated"
}

# ─────────────────────────────────────────────
# STEP 6: 파인튜닝 (제공 가중치에서 시작)
# ─────────────────────────────────────────────
step6_finetune() {
    cd "$YOLO_DIR"
    python train.py \
        --weights "$DATA_DIR/yolov7_custom.pt" \
        --data    data/custom.yaml \
        --cfg     cfg/training/yolov7.yaml \
        --img     640 640 \
        --batch-size 8 \
        --epochs  50 \
        --hyp     data/hyp.scratch.p5.yaml \
        --name    finetune_v1 \
        --device  0 \
        --workers 4
    echo "[step6] finetune done → runs/train/finetune_v1/"
}

# ─────────────────────────────────────────────
# STEP 7: 처음부터 풀 학습 (더 나은 가중치 필요 시)
# ─────────────────────────────────────────────
step7_full_train() {
    cd "$YOLO_DIR"
    python train.py \
        --weights yolov7.pt \
        --data    data/custom.yaml \
        --cfg     cfg/training/yolov7.yaml \
        --img     640 640 \
        --batch-size 8 \
        --epochs  100 \
        --hyp     data/hyp.scratch.p5.yaml \
        --name    full_train_v1 \
        --device  0 \
        --workers 4
    echo "[step7] full train done → runs/train/full_train_v1/"
}

# ─────────────────────────────────────────────
# STEP 8: RTF 측정 (샘플 영상)
# ─────────────────────────────────────────────
step8_rtf() {
    local WEIGHTS="${1:-$DATA_DIR/yolov7_custom.pt}"
    local VIDEO="${2:-$DATA_DIR/4.TestVideo_Sample/cam0/Sample_1.mp4}"

    cd "$YOLO_DIR"
    python detect.py \
        --weights "$WEIGHTS" \
        --source  "$VIDEO" \
        --img-size 640 \
        --conf-thres 0.4 \
        --device 0 \
        --name rtf_test

    echo "[step8] check runs/detect/rtf_test/ for results"
    echo "RTF = (processing time) / (video duration)"
    echo "Run: python $REDRED_DIR/tools/measure_rtf.py --video $VIDEO --weights $WEIGHTS"
}

# ─────────────────────────────────────────────
# 한 번에 전체 실행 (최초 셋업)
# ─────────────────────────────────────────────
run_all() {
    step1_extract
    step2_split
    step3_yaml
    step4_baseline_test
}

# 인자 없으면 사용법 출력
if [ $# -eq 0 ]; then
    echo "Usage: bash tools/setup_and_train.sh <step>"
    echo "Steps: step1_extract | step2_split | step3_yaml | step4_baseline_test"
    echo "       step5_augment | step6_finetune | step7_full_train | step8_rtf"
    echo "       run_all  (step1~4 순서대로)"
else
    "$@"
fi

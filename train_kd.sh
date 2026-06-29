#!/bin/bash
# Knowledge Distillation: YOLOv7(teacher) → YOLO11(student)
#
# Step 1: teacher soft label 생성  (이미 있으면 자동 스킵)
# Step 2: YOLO11 KD 학습
#
# Usage (Singularity 컨테이너 내부, ~/REDRED):
#   bash train_kd.sh [--skip_softlabel] [--epochs 100] [--batch 16]
#
# 전제:
#   - ~/yolov7/data/train.txt   : 학습 이미지 경로 목록
#   - ~/Dataset/yolov7_custom.pt: teacher weights
#   - ~/Dataset/1.competition_trainset/: 이미지+라벨 (symlink OK)
#   - ~/REDRED/data/custom.yaml : YOLO11용 data yaml

set -e
cd "$(dirname "$0")"

# ── 기본 설정 ───────────────────────────────────────────────
TRAIN_TXT="${TRAIN_TXT:-$HOME/yolov7/data/train.txt}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$HOME/Dataset/yolov7_custom.pt}"
SOFT_LABEL_DIR="${SOFT_LABEL_DIR:-$HOME/Dataset/soft_labels}"
DATA_YAML="${DATA_YAML:-$HOME/REDRED/data/custom.yaml}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
ALPHA="${ALPHA:-0.5}"
TAU="${TAU:-4.0}"
NAME="${NAME:-yolo11m_kd_$(date +%m%d_%H%M)}"
PROJECT="${PROJECT:-$HOME/runs/kd}"
SKIP_SOFTLABEL=0

# ── 인자 파싱 ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip_softlabel) SKIP_SOFTLABEL=1 ;;
        --epochs)   EPOCHS="$2";  shift ;;
        --batch)    BATCH="$2";   shift ;;
        --alpha)    ALPHA="$2";   shift ;;
        --tau)      TAU="$2";     shift ;;
        --name)     NAME="$2";    shift ;;
        --device)   DEVICE="$2";  shift ;;
    esac
    shift
done

echo "=============================="
echo " KD Training Pipeline"
echo " epochs=$EPOCHS  batch=$BATCH  alpha=$ALPHA  tau=$TAU"
echo " name=$NAME"
echo "=============================="

# ── Step 1: soft label 생성 ────────────────────────────────
if [[ $SKIP_SOFTLABEL -eq 0 ]]; then
    echo "[Step 1] Generating teacher soft labels..."
    mkdir -p "$SOFT_LABEL_DIR"
    # 이미 완료된 파일 수 확인
    existing=$(ls "$SOFT_LABEL_DIR"/*.npy 2>/dev/null | wc -l)
    total=$(wc -l < "$TRAIN_TXT")
    echo "  existing=$existing / total=$total"
    if [[ $existing -lt $total ]]; then
        python tools/gen_soft_labels.py \
            --train_txt "$TRAIN_TXT" \
            --weights   "$TEACHER_WEIGHTS" \
            --out_dir   "$SOFT_LABEL_DIR" \
            --device    "$DEVICE" \
            --img_size  "$IMGSZ"
    else
        echo "  All soft labels already generated, skipping."
    fi
else
    echo "[Step 1] Skipped (--skip_softlabel)"
fi

# ── Step 2: YOLO11 KD 학습 ────────────────────────────────
echo "[Step 2] Training YOLO11 with KD..."
python src/kd_trainer.py \
    --model      yolo11m.pt \
    --data       "$DATA_YAML" \
    --soft_label_dir "$SOFT_LABEL_DIR" \
    --epochs     "$EPOCHS" \
    --batch      "$BATCH" \
    --imgsz      "$IMGSZ" \
    --device     "$DEVICE" \
    --project    "$PROJECT" \
    --name       "$NAME" \
    --alpha      "$ALPHA" \
    --tau        "$TAU"

echo ""
echo "=============================="
echo " Training complete!"
echo " Weights: $PROJECT/$NAME/weights/best.pt"
echo " Next: update MODEL_PATH in src/run_pipeline.py and run run_test.sh"
echo "=============================="

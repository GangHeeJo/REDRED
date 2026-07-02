#!/bin/bash
# RF-DETR 파이프라인 테스트 (YOLOv7 드롭인 교체)
# Usage: bash run_test_rfdetr.sh [skip=3] [capture_conf=0.35]
#
# 2026-07-02: skip=2/conf=0.4는 RTF=1.26로 기준(<=1) 초과 -- score.py 기준
# RTF>1은 "상대평가, 공식 미공개"라 몇 점 깎이는지 알 수 없어 리스크가 큼.
# skip=3/conf=0.5(RTF=0.86, order F1 83.6%, 33.4/40)를 새 기본값으로 확정.
#
# 2026-07-02 추가: --conf는 이제 "모델 캡처 하한선"(느슨하게, 0.35)이고,
# 실제 효과적인 문턱은 multi_view_fusion.py의 CLASS_CONF_OVERRIDE/
# DEFAULT_EFFECTIVE_CONF(기본 0.5, cheerios/campbells/hunts_sauce만 0.35)가
# fusion 직전에 클래스별로 재적용함 -- 약한 신호는 살리고 노이즈 심한 클래스는
# 그대로 0.5 유지.

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rfdetr

SKIP=${1:-3}
CONF=${2:-0.35}
WEIGHTS="runs/rfdetr/checkpoint_best_total.pth"
CAM_DIR=~/Dataset/4.TestVideo_Sample
OUT="output/submission_rfdetr_skip${SKIP}_capconf${CONF}.csv"
DEBUG_LOG="output/debug_frame_counts_rfdetr.csv"
PER_CAM_LOG="output/per_cam_rfdetr.csv"
TIMED_LOG="output/timed_rfdetr.csv"

# 2026-07-02: 31(macadamia)은 이제 whitelist=[0] 단일카메라로 처리해서 제거
# (confirm=60까지 겹치면 이미 약해진 신호를 더 필터링할 위험). 36/46은 skip=3/
# conf=0.5에서도 계속 유효(과다발화 없음) 확인돼서 유지. 54(dove_white)는
# whitelist를 다 맞게 좁혀도 여전히 과다발화 -- 순수 confidence flicker로 판단해
# confirm 적용. 8(hunts_sauce)는 whitelist 시도 자체를 되돌려서 confirm도 같이
# 제거(효과 없었음). 17(a1_steak_sauce)는 confirm이 아니라 init inventory
# override 문제로 재진단돼서 confirm 목록에서 제거(run_pipeline_rfdetr.py의
# CLASS_INIT_INVENTORY_OVERRIDE 참고).
PER_CLASS_CONFIRM='{"36":60,"46":60,"54":60}'

python src/run_pipeline_rfdetr.py \
    --videos  ${CAM_DIR}/cam0/Sample_1.mp4 \
              ${CAM_DIR}/cam1/Sample_1.mp4 \
              ${CAM_DIR}/cam2/Sample_1.mp4 \
              ${CAM_DIR}/cam3/Sample_1.mp4 \
              ${CAM_DIR}/cam4/Sample_1.mp4 \
    --weights ${WEIGHTS} \
    --names   data/names.txt \
    --prices  data/prices.csv \
    --out     ${OUT} \
    --skip    ${SKIP} \
    --conf    ${CONF} \
    --device  0 \
    --debug_log ${DEBUG_LOG} \
    --per_cam_log ${PER_CAM_LOG} \
    --timed_log ${TIMED_LOG} \
    --per_class_confirm "${PER_CLASS_CONFIRM}"

echo "=== 채점 ==="
python tools/score.py \
    --sub ${OUT} \
    --gt  data/ground_truth_v2.csv

echo "=== 3종 채점 (count/order/time) ==="
python tools/score_methods.py \
    --sub ${OUT} \
    --gt  data/ground_truth_v2.csv \
    --timed ${TIMED_LOG}

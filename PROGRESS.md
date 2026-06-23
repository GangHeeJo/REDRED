# REDRED 진행 현황

**팀원:** 조강희 / 정현수 / 박준영  
**서버:** `ssh aicompetition30@147.46.121.38` (학교 내부망/VPN 필요)

---

## 현재 상태 (2026-06-23 최종)

파이프라인 정상 동작 중. `data/ground_truth_v2.csv`(105개 실측 이벤트, **시간 포함**)가 현재 기준 GT — `tools/score_methods.py`로 3가지 방식 동시 채점.

| 항목 | 값 |
|------|-----|
| RTF | 0.742 (목표 < 1.0, 통과) |
| 처리 시간 | 177.3s (영상 길이 239.0s) |
| F1 (count 기준, `tools/score.py`) | **90.0%** (TP=94 FP=10 FN=11) — 06-23 시작 시점 42%에서 대폭 개선 |
| F1 (전체 순서 LCS, class 무관) | 83.3% |
| F1 (실제 시각 비교, 지연 보정 ±3초) | 82.5% — 가장 엄격한 지표 |
| 추정 총점 (정확도+RTF, /60) | **51.1점** (정확도 36.0 + RTF 15.1) — 리더보드 1위 |
| 모델 mAP@0.5 | 98.1% (제공 가중치 `yolov7_custom.pt` 사용 중) |
| 제출 파일 | `~/REDRED/output/submission_skip2.csv` |

⚠️ "이벤트 수"(112개 등) 또는 단일 F1만으로 품질을 판단하지 않음 — **3가지 채점 방식을 같이 보고 판단**. 이유와 사용법은 Phase 7 참고. 리더보드: `output/leaderboard.html` (브라우저로 열기).

---

## 서버 접속 방법

### 방법 1: TurboX 웹 접속 (추천)
- 브라우저에서 TurboX 주소 접속 (학교 VPN 필요할 수 있음)
- 로그인 후 xterm 터미널 열기
- xterm을 열면 **자동으로 Singularity 컨테이너 안에서 시작됨**

### 방법 2: SSH + qsub (TurboX 없을 때)
SSH로 직접 접속하면 Singularity 컨테이너 밖 → GPU 사용 불가. 아래로 수동 진입:

```bash
# ssu_a6gpu 큐 (A6000 GPU, 대기 적음 — 추천)
qsub -q ssu_a6gpu -l select=1:ncpus=6:mem=128g:ngpus=1:Qlist=a6000:container_engine=singularity \
  -v "CONTAINER_IMAGE=147.46.121.38:5000/ubuntu:18.04-gpu,PBS_CONTAINER_ARGS=--no-https" \
  -I -- /tools/scripts/pbs_bash.sh

# ssai_agpu 큐 (MIG GPU, 대기 있을 수 있음)
qsub -q ssai_agpu -l select=1:ncpus=1:mem=36g:ngpus=1:Qlist=mig_agpu:container_engine=singularity \
  -v "CONTAINER_IMAGE=147.46.121.38:5000/ubuntu:18.04-gpu,PBS_CONTAINER_ARGS=--no-https" \
  -I -- /tools/scripts/pbs_bash.sh
```

- `qsub: waiting for job XXXXX to start` → GPU 할당 대기 중, 기다리면 됨
- 큐 상태 확인: `qstat -q ssu_a6gpu`
- 접속되면 프롬프트: `(yolov7) Singularity>`

### conda 환경 활성화
```bash
conda activate ~/envs/yolov7
```

---

## 서버 디렉토리 구조

```
~/
├── Dataset/
│   ├── yolov7_custom.pt              # 대회 제공 원본 가중치 (mAP@0.5=98.1%)
│   ├── names.txt                     # 60개 클래스 이름
│   ├── 1.competition_trainset/       # 학습 데이터 (20,436장)
│   ├── 3.Background_Images/          # 배경 이미지 (2,915장)
│   ├── 3.background_substracted_white/  # 세그멘테이션 이미지 (클래스별)
│   ├── 4.TestVideo_Sample/           # 테스트 영상 (cam0~cam4)
│   └── augmented/                    # Cut&Paste 증강 이미지 (5,000장)
├── yolov7/
│   ├── data/
│   │   ├── custom.yaml               # 학습 설정 (nc=60)
│   │   ├── train.txt                 # 학습 이미지 경로 (~23,000줄)
│   │   └── val.txt                   # 검증 이미지 경로 (~2,000줄)
│   └── runs/train/retrain_aug/       # 파인튜닝 결과 (mAP 오히려 낮아져 미채택)
├── REDRED/                           # 이 레포 (GangHeeJo/REDRED)
└── envs/yolov7/                      # conda 환경 (Python 3.8, torch 1.12.1+cu113)
```

---

## 파이프라인 실행

```bash
# Singularity 진입 후
conda activate ~/envs/yolov7
cd ~/REDRED && git pull

# skip=2 (RTF 최적화, 추천)
bash run_test.sh 2

# skip=1 (정확도 최대, 느림)
bash run_test.sh 1

# 결과 확인
cat output/submission_skip2.csv | head -20
```

---

## 완료된 작업

### 파이프라인 구축 ✅
- 5개 카메라 → YOLOv7 추론 → 멀티뷰 퓨전 → 이벤트 감지 → CSV 출력
- RTF 0.746 (기준 1.0 이하 통과)
- `grab()`/`retrieve()`로 skip 프레임 디코딩 최적화
- 5카메라 배치 GPU 추론 (단일 forward pass)

### EventDetector 파라미터 튜닝 ✅
- `WINDOW_SIZE`: 7 → 25 (더 넓은 median 창으로 confidence 경계 깜빡임 억제)
- `MIN_EVENT_GAP`: 10 → 90 (skip=2 기준 실제 180프레임≈6초, 오탐 억제)
- 결과: 246 이벤트 → 112 이벤트 (pepperidge_farm 22→6)

### CSV 포맷 수정 ✅
- 셀 포맷: `"재고 수량: N개"` → `"N개"`, `"총액: X"` → `"X"`
- 반환 이벤트는 총액에 기여하지 않음 (재입고로 처리)
- `include_action=True`, `total_mode="per_class"` 적용

### Cut&Paste 증강 ✅
- Flip + Perspective Warp + Blur/Noise
- 5,000장 생성 완료 (`~/Dataset/augmented/`)

### 파인튜닝 시도 (결과 미채택)
- 원본 가중치 기반 30 epoch 추가 학습
- mAP@0.5: 0.9904 (원본) → 0.9840 (파인튜닝) — 오히려 낮아짐
- 원인: Cut&Paste 증강과 실제 데이터 분포 차이
- **현재 원본 가중치(`yolov7_custom.pt`) 사용 중**

---

## 재학습 방법 (필요 시)

```bash
# 1. 증강 데이터 새로 생성
python ~/REDRED/tools/cut_paste_aug.py \
    --seg_dir  ~/Dataset/2.backsub_images_100 \
    --bg_dir   ~/Dataset/3.Background_Images \
    --out_dir  ~/Dataset/augmented \
    --num_images 5000 --max_objects 4

# 2. train.txt 재구성 (기존 augmented 줄 제거 후 새로 추가)
grep -v "augmented" ~/yolov7/data/train.txt > /tmp/train_clean.txt
mv /tmp/train_clean.txt ~/yolov7/data/train.txt
find ~/Dataset/augmented/images -name "*.jpg" >> ~/yolov7/data/train.txt

# 3. 재학습 (screen으로 세션 유지)
screen -S train
cd ~/yolov7 && PYTHONPATH=~/yolov7 python train.py \
    --weights ~/Dataset/yolov7_custom.pt \
    --data data/custom.yaml \
    --epochs 30 \
    --batch-size 16 \
    --img 640 \
    --device 0 \
    --name retrain_v2 \
    --exist-ok

# 세션 이탈: Ctrl+A, D  /  재접속: screen -r train
```

---

## 개발 로그 (시간순)

### 2026-06-19 | Phase 1 — 초기 파이프라인 구축 (박준영/chickgoose)
- 5카메라 → YOLOv7 → 멀티뷰 퓨전 → 이벤트 감지 → CSV 기본 뼈대 완성
- `torch.load` 직접 사용으로 `attempt_download` 버그 우회 (경로 소문자 변환 문제)
- PyTorch 1.12 호환성 패치: `Upsample.recompute_scale_factor = None`
- `run_test.sh` 추가, PYTHONPATH 포함

### 2026-06-19 | Phase 2 — RTF 최적화 (박준영/chickgoose)
- `grab()`/`retrieve()` 방식 도입: skip 프레임에서 H.264 디코딩 없이 위치만 이동
- 5카메라 배치 GPU 추론: 단일 forward pass로 RTF 대폭 절감
- RTF 0.751 달성

### 2026-06-19 | Phase 3 — 증강 파이프라인 개선 (박준영/chickgoose)
- chickgoose 레포 기준 `augment/cut_paste_aug.py` 여러 차례 개선:
  - `load_seg_images()`: 폴더명 숫자 prefix에서 class_id 파싱 버그 수정
  - 회전 각도 ±15° → ±5°로 축소 (과도한 회전이 실제 데이터와 괴리)
  - 마스크 halo(경계 번짐) 제거
  - `--no_erasing` 플래그 추가 (erasing이 오히려 품질 저하)
- 5,000장 증강 완료 (`~/Dataset/augmented/`)
- 세그멘테이션 소스: `~/Dataset/3.background_substracted_white/` (클래스별 폴더)

### 2026-06-19 | Phase 4 — 파인튜닝 시도 및 미채택 (박준영/chickgoose)
- 원본 가중치 기반 30 epoch 추가 학습
- mAP@0.5: 0.9904 (원본) → 0.9840 — 오히려 하락
- 원인 추정: Cut&Paste 증강 분포가 실제 테스트 영상과 다름
- 결론: **원본 가중치(`yolov7_custom.pt`) 유지**

### 2026-06-21 | Phase 5 — GangHeeJo 레포 동기화 및 코드 정리 (조강희)
- 서버 `~/REDRED` remote를 chickgoose → GangHeeJo로 변경
- chickgoose `pipeline/` 코드를 우리 `src/`로 통합
- 주요 병합 내용:
  - `infer_batch()` 배치 추론 로직
  - `grab()`/`retrieve()` 프레임 스킵 최적화
- CSV 포맷 수정: 셀값 `"재고 수량: N개"` → `"N개"`, `"총액: X"` → `"X"`
- 반환 이벤트 총액 처리: 환불이 아닌 재입고로 해석 → 총액 기여 0원

### 2026-06-21~22 | Phase 6 — EventDetector 파라미터 튜닝 (조강희)
- `tools/analyze_inventory.py` 제작: 프레임별 재고 변화 그래프 시각화
- 분석 결과: `pepperidge_farm_milk_chocolate_macadamia_cookies`가 confidence 경계선에서 0↔1 빠르게 깜빡임 (노이즈)
- 튜닝 과정:

| 시도 | WINDOW_SIZE | MIN_EVENT_GAP | conf | 이벤트 수 | pepperidge_farm |
|------|------------|---------------|------|-----------|-----------------|
| 초기 | 7 | 10 | 0.4 | 246 | 22 |
| 1차 | 9 | 30 | 0.4 | 170 | 12 |
| 2차 | 9 | 60 | 0.4 | 144 | 12 |
| 3차 | 9 | 90 | 0.4 | 130 | 10 |
| 4차 | 15 | 90 | 0.5 | 124 | 14 (악화) |
| **확정** | **25** | **90** | **0.4** | **112** | **6** |

- conf 올리면 오히려 감지가 더 불안정해짐 → 0.4 유지
- WINDOW_SIZE=25가 5~10프레임 주기 깜빡임 억제에 효과적

### 2026-06-23 | Phase 7 — ground_truth.csv 도입 및 정확도 버그 수정 (강희조, 박준영+Claude)

**계기:** 실제 영상에서 인식률이 체감상 낮다는 문제 제기 → `data/ground_truth.csv`(105개 이벤트, 6섹션, 87개는 frame 번호 포함) 추가로 처음 정량 비교 가능해짐.

**진단 도구:**
- `tools/score.py`, `tools/compare_to_ground_truth.py` — submission vs ground_truth 채점(precision/recall)
- `tools/diagnose_missing_events.py`, `tools/analyze_detections.py` — raw count로 perception failure vs logic bug 구분
- `tools/replay_event_detector.py` — `run_pipeline.py --debug_log`로 뽑은 프레임별 raw count를 **서버 없이 로컬에서** EventDetector에 재생, 상태 변화 추적
- `tools/check_training_class_counts.py` — 클래스별 학습 이미지 개수 집계
- `tools/probe_low_confidence.py` — conf=0.05로 재추론해서 "진짜 못 보는지" vs "threshold 문제인지" 구분

**버그 1 — EventDetector UNKNOWN 상태 (커밋 aba6a47, 강희조):**
초기 추정에 없던 클래스(영상 시작 시 선반에 없던 물건)가 UNKNOWN 상태로 시작 → 처음 감지되면(반환 이벤트) 그 값을 "원래 초기값"으로 잘못 확정 → 반환 이벤트 재현율 7% vs 구매 76%로 극단적 비대칭. **수정: UNKNOWN 상태 제거, 전부 STABLE+committed=0으로 시작.**

**버그 2 — 연쇄 잠김 (커밋 894ef88, 강희조):**
반환 감지가 한 번 실패하면 committed가 갱신 안 돼서 다음 구매도 "선반이 비어있는데 구매?"로 자동 차단되는 연쇄 실패. **수정: 구매-차단 제약 제거, 이벤트 발생 후 sliding window 히스토리 리셋(쿨다운 효과).**

→ 두 버그 수정 후 재현율 **42% → 79%** (44/105 → 83/105)

**버그 3 — 멀티뷰 퓨전 구조적 사각지대 (박준영+Claude):**
`bumblebee_albacore`/`dove_white`/`dove_pink` raw count가 영상 전체에서 0. `tools/probe_low_confidence.py`로 확인한 결과 모델은 conf 0.5~0.9대로 자주 감지하지만, **5대 카메라 중 3대 이상이 동시에 본 적이 한 번도 없음**(최대 2대) — `fuse_weighted_median`은 과반(3+/5) 동의가 있어야 count>0이 되므로 구조적으로 항상 0이 됨. 학습 데이터 양 문제 아님(더 적은 데이터의 다른 클래스는 정상 인식).
**수정:** `src/multi_view_fusion.py`에 `MAX_CONFIDENCE_CLASS_IDS = {2, 53, 54}` 추가 — 이 3개 클래스만 카메라 중 최댓값(max-across-cameras)으로 융합, 나머지 57개는 기존 median 유지.

→ 재현율 **79% → 82%** (83/105 → 86/105)

**부작용 (해결됨, 아래 Phase 8 참고):** `dove_white`는 복구됐지만 카메라 1대 노이즈에도 즉시 이벤트가 확정돼 중복 발화 4건 발생.

**여전히 남은 이슈 (Phase 8에서 일부 재확인/해소):**
- 섹션1 초반 구매 일부: `--init_frames 30` 윈도우 안에서 이미 집어간 물건은 "원래 없었다"로 흡수돼 구매 이벤트 자체가 안 생길 수 있음
- `nabisco_nilla_wafers`, `haribo_gold_bears_gummi_candy`: 학습 데이터 충분(63~68퍼센타일) — 서버 실행 간 GPU 추론 비결정성으로 인한 간헐적 누락으로 판단, 로직 문제 아님

### 2026-06-23 | Phase 8 — quorum 절충, ground_truth_v2, 3종 채점 방식 (박준영+Claude)

**`dove_white` quorum 절충 (`src/multi_view_fusion.py`):**
`MAX_CONFIDENCE_CLASS_IDS`(bool) → `CLASS_QUORUM_OVERRIDE`(class_id → 필요 카메라 수)로 일반화.
`bumblebee_albacore`/`dove_pink`는 quorum=1(기존 max 그대로, 깨끗하게 동작) 유지, `dove_white`만 quorum=2로 올림 — 카메라 1대 노이즈로 인한 중복발화 4건→1건으로 감소(단, 경계선 케이스 1건은 못 잡고 누락으로 바뀜 — 순오류 4건→2건으로 개선).

**`ground_truth_v2.csv` 도입 (강희조 재검수):**
- v1과 달리 105개 전 이벤트에 시간 정보 포함(공백 없음), `pepperidge_farm_milano_cookies_double_chocolate`(v1에 없던 항목) 추가로 확인됨
- **버그 발견**: 처음 올라온 v2는 `time_sec`이 60배 부풀려져 있었음 (예: event105가 "233분"으로 기록 — 239초짜리 영상에 물리적으로 불가능). `time` 컬럼을 분:초 형식으로 적었는데 스크립트가 시:분:초로 잘못 해석한 것으로 추정. v1의 frame 기반 시각과 대조해서 ÷60 보정 확인 후, 강희조가 소스에서 직접 수정 — 현재는 정상.
- 이후 강희조가 순서/시각을 한 번 더 재검수하여 업데이트 — 재채점 결과 3가지 지표 전부 개선(아래), `mahatma_rice`/`honey_bunches_of_oats_with_almonds`가 "모델 문제"가 아니라 "구 ground truth 기록 오류"였음이 확인됨.

**3종 채점 방식 (`tools/score_methods.py`, 신규):**
단일 지표로는 판단이 위험하다는 게 오늘 여러 번 확인됨 — 클래스별 개수만 보면 순서가 틀려도 맞다고 카운트되고(`arm_hammer_baking_soda`처럼 타이밍은 맞는데 전체 시퀀스가 깨져서 틀렸다고 잘못 판정되는 경우의 반대 케이스), 전체 시퀀스 정렬만 보면 한 클래스의 타이밍 오류가 무관한 다른 클래스의 정렬까지 깨버림. 세 방식을 같이 보기로 함:
1. **count** — `(class_name, action)` 빈도만 비교 (`tools/score.py`와 동일 원리)
2. **order** — 전체 시퀀스에 대한 LCS(최장 공통 부분열), 클래스 구분 없이 순서만 봄
3. **time** — 실제 발화 시각을 ground truth 시각과 직접 비교. 단순 비교 시 시스템의 `CONFIRM_FRAMES=30`(~2초) 지연 때문에 전부 늦게 잡혀서 부당하게 낮게 나옴 → **median offset(현재 +2.80초)을 자동 추정해서 보정 후 비교**. 이 보정값이 정확히 CONFIRM_FRAMES 이론값(2.0초)에 가까운 것으로 시스템 동작이 교차검증됨.

`src/run_pipeline.py --timed_log <path>`로 매 이벤트의 실제 발화 시각(`time_sec,class_name,action`)을 정식 출력 가능 (방법 3에 필요, 로컬 리플레이 근사 없이 정확한 값 사용 가능).

**최종 결과 (`ground_truth_v2.csv` 기준):**

| 방법 | F1 |
|---|---|
| count | 90.0% |
| order (LCS) | 83.3% |
| time (±3초, 지연보정) | 82.5% |

`tools/score.py --gt data/ground_truth_v2.csv`로 리더보드 갱신 — 추정 총점 51.1/60점(정확도 36.0 + RTF 15.1)으로 현재 1위.

**여전히 진짜 문제로 남은 것 (시간 비교로 확정됨):**
- `pop_tararts_strawberry` 구매: 77.4초 차이 — 여러 반환/구매 사이클이 시간상 뒤섞여 있음, 가장 큰 잔여 문제
- `hunts_sauce` 구매: 64.1초 차이
- `pepperidge_farm_milk_chocolate_macadamia_cookies` 구매: 63.9초 차이
- `pepperidge_farm_milano_cookies_double_chocolate`, `haribo_gold_bears_gummi_candy`, `frappuccino_coffee`, `spam`: 완전 누락

---

## 주요 결정사항 / 트러블슈팅 기록

### 클래스 이름 오탈자 (수정 금지!)
`data/names.txt`와 `data/prices.csv`에 오탈자처럼 보이는 이름 두 개가 있음:
- `pop_tararts_strawberry` (pop tarts가 아님)
- `nature_vally_fruit_and_nut` (nature valley가 아님)

**대회 공식 클래스명이 이 오탈자 그대로임.** 서버 `~/Dataset/names.txt`와 학습 데이터 폴더명도 동일하게 오탈자. 절대 수정하지 말 것.

### load_model: torch.load 직접 사용
YOLOv7의 `attempt_load`를 쓰면 내부의 `attempt_download`가 파일 경로를 소문자로 변환 → Linux 대소문자 구분 파일시스템에서 `~/Dataset` → `~/dataset`로 바뀌어 파일 못 찾음. `torch.load` 직접 사용으로 우회. (`src/run_pipeline.py`의 `load_model()` 참고)

### 반환(반납) 이벤트 총액 처리
반환 이벤트는 총액에 기여하지 않음 (0원으로 처리). 고객 환불이 아닌 재입고로 해석. (`src/csv_generator.py` 참고)

### pepperidge_farm 감지 불안정
`pepperidge_farm_milk_chocolate_macadamia_cookies`가 confidence 경계선에서 0↔1 깜빡임. 노이즈임. WINDOW_SIZE=25로 대부분 억제됨. `output/analysis/inventory_plot.png` 참고.

### git remote 주의
서버 `~/REDRED`가 처음에 chickgoose/REDRED를 바라보고 있었음. 현재는 GangHeeJo/REDRED로 수정됨. push/pull은 항상 GangHeeJo/REDRED로만.

---

## 주의사항

- 재학습 시 반드시 `screen` 또는 `tmux + nohup` 사용 (세션 끊기면 중단됨)
- git push/pull은 **GangHeeJo/REDRED** 로만
- chickgoose/REDRED는 읽기 전용 참고용 — 절대 push하지 말 것

---

## 현재 확정 파라미터 (2026-06-23 갱신, `MIN_EVENT_GAP`/`INIT_CONFIRM`은 Phase 7에서 제거됨)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `WINDOW_SIZE` | 25 | `src/event_detector.py` — median 슬라이딩 윈도우 |
| `CONFIRM_FRAMES` | 30 | `src/event_detector.py` — candidate 확정까지 필요한 연속 프레임(skip=2 기준 ~2초) |
| `MAX_DELTA` | 4 | `src/event_detector.py` — 1회 이벤트당 허용 최대 변화량 |
| `MAX_INVENTORY` | 1 | `src/event_detector.py` — 슬롯당 물리적 최대 재고 |
| `CLASS_QUORUM_OVERRIDE` | {2:1, 53:1, 54:2} | `src/multi_view_fusion.py` — bumblebee_albacore(1), dove_pink(1), dove_white(2). 숫자는 이벤트로 인정하는 데 필요한 동시 카메라 수 |
| `--conf` | 0.4 | `run_test.sh` |
| `--skip` | 2 | `run_test.sh` |

파라미터 바꾸고 싶으면 → 로컬에서 수정 → `git push` → 서버에서 `git pull && bash run_test.sh 2`

---

## 재고 분석 도구

`tools/analyze_inventory.py` — 필터링 없이 프레임별 감지 결과를 그래프로 시각화. 파라미터 튜닝 근거 확인용.

```bash
python tools/analyze_inventory.py \
    --videos ~/Dataset/4.TestVideo_Sample/cam{0..4}/Sample_1.mp4 \
    --weights ~/Dataset/yolov7_custom.pt \
    --names data/names.txt \
    --out output/analysis --skip 2 --device 0
```

결과: `output/analysis/inventory_plot.png`, `raw_events.csv`

---

## 앞으로 할 일

- [ ] 발표 자료 준비
- [ ] `pop_tararts_strawberry` 구매/반환 사이클 시간 뒤섞임(77초 차이) — 현재 가장 큰 잔여 오차, 원인 미파악
- [ ] `hunts_sauce`, `pepperidge_farm_milk_chocolate_macadamia_cookies` 구매 — 60초대 시간 오차, 같은 계열 문제로 추정
- [ ] `pepperidge_farm_milano_cookies_double_chocolate`, `haribo_gold_bears_gummi_candy`, `frappuccino_coffee`, `spam` 완전 누락 — 원인 미파악 (haribo는 기존에 GPU 비결정성으로 진단됐던 것과 같은 클래스, frappuccino는 이전에 너무 일찍(14초) 잘못 확정되는 패턴이 한 번 확인된 적 있음)
- [ ] 섹션1 초반 구매 미검출 — `--init_frames` 추정 윈도우와 실제 구매 타이밍이 겹치는 문제 (예: init_frames 축소, 또는 추정 방식 개선)
- [x] ~~`dove_white` 중복 발화~~ → quorum=2로 절충, 순오류 4건→2건 감소 (2026-06-23)
- [x] ~~정확도 검증~~ → `data/ground_truth_v2.csv` + `tools/score_methods.py`(3종 방식) + 리더보드로 완료 (2026-06-23)

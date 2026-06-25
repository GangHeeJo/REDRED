# REDRED 진행 현황

**팀원:** 조강희 / 정현수 / 박준영  
**서버:** `ssh aicompetition30@147.46.121.38` (학교 내부망/VPN 필요)

---

## 현재 상태 (2026-06-25 최종)

파이프라인 정상 동작 중. `data/ground_truth_v2.csv`(105개 실측 이벤트, **시간 포함**)가 현재 기준 GT — `tools/score_methods.py`로 3가지 방식 동시 채점.

**현재 main 브랜치 = Phase 10 (spam quorum=2) — 브랜치 테스트 후 최고점 유지**

| 항목 | 값 |
|------|-----|
| RTF | 0.7575 (목표 < 1.0, **RTF≤1이면 20점 만점**) |
| F1 (count 참고용) | 92.0% (TP=98 FP=10 FN=7) |
| **F1 (order/LCS — `tools/score.py` 기준)** | **85.4%** (TP=91 FP=17 FN=14) |
| F1 (time, 지연보정 ±3초) | 84.5% |
| **추정 총점 (정확도+RTF, /60)** | **54.2점** (정확도 34.2 + RTF 20.0) |
| 모델 mAP@0.5 | 98.1% (제공 가중치 `yolov7_custom.pt` 사용 중) |
| 제출 파일 | `~/REDRED/output/submission_skip2.csv` |

**채점 기준 (2026-06-25 갱신):**
- 정확도 40점: `score.py`가 **order/LCS F1 × 0.4** 기준으로 변경 (공식 미공개이나 이벤트번호 포함 기준에 더 근접)
- RTF 20점: 대회 공고 기준 **RTF ≤ 1 → 만점(20점)**, RTF > 1 → 상대평가(미공개)
- 기존 `20×(1-RTF/3)` 공식은 잘못된 역산값이었음 — 수정됨

**브랜치 테스트 결과 (2026-06-25):**
| 브랜치 | order F1 | 결과 |
|--------|---------|------|
| Phase 10 baseline (main) | **85.4%** | 기준 |
| fix/tracker-default (max_age=15) | 83.4% | ❌ 악화 — 트래커 코드 main에서 완전 제거 |
| fix/pepperidge-milano-confirm | 84.7% | ❌ 악화 — milano 검출되나 +50초 지연 |
| fix/per-class-conf (bulls_eye) | 85.4% | ❌ 효과 없음 |

⚠️ 단일 F1만으로 판단하지 말 것 — count/order/time 세 지표 같이 보기. 리더보드: `output/leaderboard.html` (브라우저로 열기).

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

### 2026-06-23 | Phase 8 추가 — 리더보드 자동화 + GT 재검수 (강희조+Claude)

**리더보드 웹 자동화:**
- `tools/score.py`: 채점 후 `output/leaderboard.csv` 누적 기록 + `output/leaderboard.html`(다크 테마 웹 대시보드) 자동 생성. 히스토리 항목(CSV 없는 과거 기록)도 dimmed 행으로 표시.
- `src/run_pipeline.py`: 실행 완료 시 `output/run_stats.json` 자동 저장 (RTF, 처리시간, 영상길이 포함)
- `run_test.sh`: 파이프라인 성공 → `score.py` 자동 호출 → `git push` — **서버에서 한 번 돌리면 리더보드가 자동 갱신됨**
- `.gitignore`: `output/leaderboard.csv`, `output/leaderboard.html` 예외 추가 (git 추적 대상)
- RTF 점수 공식 PDF 역추산으로 확정: `rtf_score = 20 × (1 − RTF/3)` — 예시(RTF=0.75→15점) 검증됨

**ground_truth v1 vs v2 비교:**
- 전체 105행 중 다른 행 10개 발견:
  - **순서만 뒤바뀐 것** (내용 동일): 9↔10행, 40↔41행, 101↔102행 — Excel에서 인접 이벤트 순서 재정렬한 것으로 추정
  - **상품 자체가 다른 것**: 25, 53행 — v1의 `nabisco_nilla_wafers` → v2의 `pepperidge_farm_milano_cookies_double_chocolate` (v2가 정확, 재검수 확인)
- prefix 버그 2건(`50.hersheys_symphony`, `6.honey_bunches_of_oats_honey_roasted`) 수정 완료

**대회 평가 기준 재확인 (튜토리얼 PDF 48페이지 직접 확인):**
- 제출 CSV 평가 항목: `품목명, 이벤트번호, 구매/반환여부, 재고수량, 총액`
- **시간(timestamp)은 공식 평가 항목에 없음** — `ground_truth_v2.csv`의 `time_sec`은 내부 분석/채점 전용
- 정량 평가(60점) = 재고인식 정확도 40점 + RTF 20점 (발표 40점 별도)

### 2026-06-23 | Phase 9 — 오류 클래스 전수조사 + quorum 확장 (박준영+Claude)

**오류 클래스 분류 (raw 신호량 기준, `output/debug_frame_counts.csv`):**
- `score_methods.py`의 mismatch 출력이 `[:10]`으로 잘려 있어서 일부 클래스가 안 보였음 — python으로 전체 Counter diff 재계산해서 19개 (class,action) 조합 / 16개 고유 클래스 확인.
- 3그룹으로 분류됨:
  1. **완전 블라인드**(raw=0, 영상 전체에서 단 한 프레임도 감지 안 됨): `campbells_chicken_noodle_soup`, `redbull`, `crystal_hot_sauce`
  2. **희미하게 보임**(raw 16~61프레임, max동시count=1): `haribo_gold_bears_gummi_candy`, `pepperidge_farm_milano_cookies_double_chocolate`, `spam`, `frappuccino_coffee`, `dr_pepper`, `bulls_eye_bbq_sauce_original`
  3. **과다 인식**(raw 86~914프레임, 정상보다 훨씬 많음): `pop_tararts_strawberry`, `hunts_sauce`, `pepperidge_farm_milk_chocolate_macadamia_cookies` — 인식은 잘 되는데 이벤트가 중복 발화. `white_rain_body_wash`/`coca_cola_glass_bottle`도 같은 패턴으로 가짜 이벤트 생성.
- 클래스 속성으론 공통점 없고, GT 발생 "시각"이 공통점 — 오류 이벤트 대부분이 3개의 연속 다중 이벤트 구간(0~23초/40~68초/105~133초, 아이템이 1~3초 간격으로 연속 변화)에 몰림. `WINDOW_SIZE=25`/`CONFIRM_FRAMES=30`이 아이템이 띄엄띄엄 변하는 걸 가정한 파라미터라 이런 구간에서 신호 약한 클래스는 누락, 신호 강한 클래스는 중복발화로 갈림.

**`run_test.sh` 개선:** `--debug_log`/`--timed_log`를 매 실행 기본 포함 + 자동으로 `score_methods.py` 3종 채점까지 실행. (이전엔 둘 다 수동으로 따로 돌려야 해서 채점에 쓴 제출 파일과 진단 로그가 서로 다른 실행 결과인 경우가 있었음 — 재현성 위해 한 실행에서 다 나오게 통일.)

**quorum 확장 (`src/multi_view_fusion.py`):**
- `tools/probe_low_confidence.py`를 `campbells_chicken_noodle_soup`/`redbull`/`crystal_hot_sauce`/`dr_pepper`에 대해 conf=0.05로 재실행(`output/low_conf_probe2.csv`).
- **`redbull`(cam0 단독)/`crystal_hot_sauce`(cam3 위주+가끔 cam4)/`dr_pepper`(cam4→+cam1→+cam0)**: bumblebee_albacore/dove와 동일한 구조적 문제 확인 — 구매 직전까지 끊김없이 화면에 계속 있는데 5캠 중 1~2캠에만 보여서 median 퓨전(과반 필요)이 항상 0으로 만들었던 것. `CLASS_QUORUM_OVERRIDE`에 quorum=1로 추가(`redbull`=15, `crystal_hot_sauce`=39, `dr_pepper`=21).
- **`campbells_chicken_noodle_soup`은 다른 문제**: cam4가 GT 구매 시각(11s) 이후로도 ~100초간 계속 감지 — quorum 문제가 아니라 `campbells_chunky_classic_chicken_noodle`과의 클래스 혼동으로 추정. **의도적으로 quorum override에서 제외**, bbox 위치 확인 필요.
- **결과**: F1 90.0%→**91.5%**, order 83.3%→**84.9%**, time 82.5%→**84.0%**, 추정 총점 50.9→**51.6점**. `redbull`/`crystal_hot_sauce`/`dr_pepper`가 양쪽 채점 방식에서 완전히 사라짐, 다른 클래스 부작용 없음(pop_tararts/hunts_sauce/pepperidge_milk 카운트 그대로).

**남은 문제 (재실행 2회로 변동 여부 구분됨):**
- 두 번 다 동일하게 깨짐(진짜 문제, 재현됨): `bulls_eye_bbq_sauce_original`, `haribo_gold_bears_gummi_candy`, `pepperidge_farm_milano_cookies_double_chocolate`, `spam`, `frappuccino_coffee`, `campbells_chicken_noodle_soup` — 이 중 `bulls_eye_bbq_sauce_original`/`haribo`/`milano`/`spam`은 quorum probe 안 해봤음 (다음 후보), `frappuccino_coffee`는 quorum이 아니라 "너무 일찍(3.6s) 확정"되는 별개 버그(로컬 리플레이로 확인, 실제 구매는 16s).
- 실행마다 다르게 나타남(GPU 비결정성 의심): `nabisco_nilla_wafers`, `white_rain_body_wash`(가짜 반환), `coca_cola_glass_bottle`(가짜 반환) — 우선순위 낮음.

### 2026-06-24 | Phase 10 — probe3 quorum 전수조사 + spam quorum=2 (강희조+Claude)

**probe3 실행 (`output/low_conf_probe3.csv`):**
`bulls_eye_bbq_sauce_original` / `haribo_gold_bears_gummi_candy` / `pepperidge_farm_milano_cookies_double_chocolate` / `spam` 4개 클래스를 conf=0.05, skip=10으로 재추론. 프레임별 동시 카메라 수 분석:

| 클래스 | max 동시 카메라 | mean_conf | 판정 |
|--------|--------------|-----------|------|
| `bulls_eye_bbq_sauce_original` | 5대 | 0.355 | quorum 문제 아님 (5대까지 보임, 다른 원인) |
| `haribo_gold_bears_gummi_candy` | 5대 | 0.739 | quorum 문제 아님 (GPU 비결정성 의심) |
| `pepperidge_farm_milano_cookies_double_chocolate` | 3대 | 0.549 | 경계선 — quorum=2 시도 |
| `spam` | 3대 | 0.421 | 경계선 — quorum=2 시도 |

**`pepperidge_farm_milano` quorum=2 시도 및 취소:**
quorum=2 추가 시 Sub=5 purchase + Sub=5 return으로 5번 중복발화 발생 — dove_white 때와 동일 패턴(2대 신호가 들쭉날쭉해서 count가 1↔0 반복). 즉시 원복. `CLASS_QUORUM_OVERRIDE`에서 제외.

**`spam` quorum=2 확정 (`src/multi_view_fusion.py`, class_id=29):**
quorum=2 추가 후 spam 정상 감지 확인. 중복발화 없음. TP +1 추가.

**결과**: F1 91.5%→**92.0%**, order 84.9%→**85.4%**, time 84.0%→**84.5%**, 추정 총점 51.6→**51.8점**.

**남은 문제:**
- `pepperidge_farm_milano`: quorum=2 불가 (중복발화), 별도 해결책 필요
- `haribo`, `bulls_eye`: quorum 문제 아님, 원인 미파악
- `pop_tararts_strawberry`/`hunts_sauce`/`pepperidge_farm_milk_choc`: 60~77초 타이밍 오차 (이벤트 감지 로직 문제)
- `frappuccino_coffee`: 너무 일찍(3.6s) 확정 (실제 16s)
- `campbells`: chunky 클래스 혼동

### 2026-06-25 | Phase 12~14 — 브랜치 전수 테스트 및 코드 정리 (강희조+Claude)

**채점 방식 갱신 (이번 세션):**
- `tools/score.py`: order/LCS F1 기반으로 채점 전환, RTF≤1=20점 만점으로 수정
- 기존 리더보드 항목 전부 새 기준으로 재계산 반영

**Phase 12 — SORT 트래커 on main:**
- `--use_tracker --tracker_max_age 15` 적용 상태로 서버 실행
- 결과: order F1 **85.4% → 83.4%** (FP 17→18, 악화)
- 확인용 baseline 재실행(tracker 없음): order F1 85.4% 재현 → **트래커가 원인임 확정**
- Phase 11(A6000)에서 85.3%로 거의 동일했던 것과 달리 현재 GPU에서 악화 — GPU 비결정성 or frappuccino confirm=200 유무 차이로 추정

**Phase 13 — fix/pepperidge-milano-confirm:**
- milano quorum=2 + per_class_confirm=150(~10초) 적용
- count F1: 92.0%→93.0% (TP+2: milano purchase/return 이제 검출됨)
- order F1: **85.4%→84.7%** (FP 17→19, 악화)
- 원인: milano는 검출되지만 confirm=150 지연으로 Sub 내 순서가 틀림 (return +50초 지연, 3종 채점 모두서 FP로 처리됨)
- **결론: 버림**

**Phase 14 — fix/per-class-conf (bulls_eye conf=0.2):**
- count/order/time F1 세 지표 모두 Phase 10과 완전 동일 (92.0%/85.4%/84.5%)
- bulls_eye 여전히 FN — conf=0.2로 낮춰도 여전히 미검출
- 원인: threshold 문제가 아니라 fusion quorum 또는 신호 자체 부족 문제
- **결론: 효과 없음, 버림**

**코드 정리:**
- `src/tracker.py` 삭제
- `src/run_pipeline.py`에서 트래커 import/argparse/cam_tracker 로직 전부 제거
- `run_test.sh`에서 `--use_tracker` 제거
- 원격 브랜치 삭제: `fix/pepperidge-milano-confirm`, `fix/tracker-default`, `fix/per-class-conf`

**`pop_tararts_strawberry` debug_log 분석 (GT=1인데 Sub=3인 주원인):**
- `output/debug_frame_counts.csv`에서 신호 분포 확인:
  - 프레임 0~1296 (0~43s): 완전 미검출 (count=0)
  - 프레임 1296~3792 (43~126s): count=1 (중간 gaps 있음)
  - 프레임 3792~ (126s~): 미검출 → GT purchase=127s와 일치
- **근본 원인: 초기 재고 추정 실패.** `--init_frames 30`(0~2초)동안 미검출 → `initial_inventory[pop_tararts]=0`으로 잘못 설정. 이후 프레임 1296에서 첫 검출 시 RETURN 이벤트 발화, 이후 gaps마다 PURCHASE/RETURN 반복 oscillation.
- 가능한 해결책: `--init_inv` 로 pop_tararts 초기 재고를 수동으로 1로 지정, 또는 `--init_frames` 대폭 확대. 단, gaps가 16초 이상이어서 confirm_frames 조정만으론 불가.

### 2026-06-24 | Phase 11 — "이벤트직후 유령반전" 분석, 진단 도구 버그 수정, SORT 트래커 A/B (박준영+Claude)

**진단 도구 버그 발견/수정 (`tools/replay_event_detector.py`):** per-frame 디버그 출력 코드가 `detector._sm_state[cid]`/`._committed[cid]`를 `[]`로 직접 인덱싱했는데, 둘 다 `defaultdict`라 frame 0부터 강제로 키가 생성되며 `_history`까지 조기 활성화됨(실제 파이프라인은 해당 클래스가 처음 감지될 때까지 활성화 안 됨). 그 결과 **`haribo_gold_bears_gummi_candy`가 "반환은 맞고 구매가 유령으로 뜬다"고 잘못 보였음** — 수정 후 재확인하니 실제로는 반환/구매 둘 다 전혀 발화 안 하는 깨끗한 더블-FN(신호가 WINDOW_SIZE/CONFIRM_FRAMES 기준을 넘긴 적이 없음)이었음. **GPU 비결정성 의심은 정정됨** — haribo는 신호 부족 문제, GPU 비결정성과 무관. `[]` 대신 `in` 멤버십 체크로 수정, 커밋됨.

**"이벤트직후 유령반전" 그룹 확정**: `pop_tararts_strawberry`/`hunts_sauce`/`pepperidge_farm_milk_chocolate_macadamia_cookies` (haribo는 제외, 위 정정 참고). 진짜 GT 이벤트는 제때 발화하지만, 이벤트 발생 시 `_history` 클리어 직후 혼잡구간(0~23/40~68/105~133초)의 occlusion flicker로 1~2개 유령 반전 쌍이 곧바로(약 3.7초 만에) 추가 확정됨. **per-class CONFIRM_FRAMES/WINDOW_SIZE를 올리는 방법은 기각** — 로컬 스윕(`output/debug_frame_counts.csv` 기준) 결과 진짜 신호도 똑같이 간헐적이라 기준을 높이면 진짜 이벤트까지 같이 사라짐. **`per_class_cooldown`(이벤트 발생 직후 N프레임 동안 새 candidate 형성을 차단하는 신규 메커니즘)을 구현해서 로컬 테스트 시 효과 확인** — 단, 노이즈 구간이 거의 전체 반환↔구매 간격(70~80초)에 걸쳐 있어서, 완전히 없애려면 **이 영상의 실측 간격에 맞춘 video-specific 값**(80초/70초)이 필요했음. 일반화 위험(다른 영상이면 그 시간 안의 진짜 재거래를 놓침) 때문에 **롤백 결정** (`fix/ghost-event-cooldown` 브랜치 히스토리 참고, 미병합 상태로 보존).

**SORT 트래커(`--use_tracker`) A/B 테스트**: 기존에 구현은 돼 있었지만 `--tracker_max_age` 기본값(3, ~0.2초)이 측정된 occlusion gap(0.4~1.5초)보다 짧아서 거의 효과가 없었을 것으로 추정. A6000 큐에서 0(미사용)/15/25로 비교:

| max_age | RTF | count F1 | order F1 | time F1 |
|---|---|---|---|---|
| 0 (미사용) | 0.770 | 92.0% | 85.4% | 84.5% |
| 15 | 0.756 | **92.9%** | 85.3% | 85.3% |
| 25 | 0.766 | 92.9% | 85.3% | 85.3% |

RTF는 거의 무료(트래커는 CPU 쪽 Kalman/IoU 매칭만 추가, GPU 추론 비용 없음), 15에서 이미 효과가 다 나오고 25는 추가 이득 없음 → **15로 확정**. order/time F1이 트래커 켰을 때 TP/FP/FN까지 완전히 동일해지는 것도 확인됨(끄면 서로 다름) — 중복/유령 이벤트가 줄어 두 채점 방식의 매칭 모호성이 줄었다는 신호로, 노이즈가 아니라 실제 개선으로 판단. **단, pop_tararts 등 "이벤트직후 유령반전"은 트래커로도 그대로 남음** — 트래커는 카메라 내부 occlusion만 버텨주고, 5캠 합치는 fusion 단계의 진동은 못 잡음. `run_test.sh`에 `--use_tracker --tracker_max_age 15` 추가 + 코드 기본값도 15로 변경, `fix/tracker-default` 브랜치로 push.

**frappuccino_coffee 회귀 의심**: 오늘 서버 재추론(A6000)에서 `fix/frappuccino-init`의 `confirm_frames=200` 적용 상태로 돌렸는데 **3번(notracker/max_age15/max_age25) 다 frappuccino 구매가 완전히 미발화(Sub=0)**. 기존엔 "3.6s에 너무 일찍 확정"이 문제였는데, 지금은 진짜 16s 신호도 200프레임(~13초)을 못 버티는 것으로 보임 — GPU 추론 결과가 그날그날 달라질 수 있어서, confirm_frames=200이 그새 너무 보수적인 값이 됐을 가능성. **재검증 필요, 아직 안 함.**

**미병합 브랜치 4개로 늘어남, 통합 필요**: `fix/frappuccino-init`(per_class_confirm 인프라 + frappuccino confirm=200, 위 회귀 의심), `fix/pepperidge-milano-confirm`(per_class_confirm 인프라 중복 구현 + milano quorum=2/confirm=150), `fix/per-class-conf`(bulls_eye conf=0.2, event_detector 안 건드림 — 독립적), `fix/tracker-default`(트래커 기본 활성화, event_detector 안 건드림 — 독립적). **`fix/frappuccino-init`과 `fix/pepperidge-milano-confirm`은 같은 `per_class_confirm` 인프라를 각자 구현해서 그대로 두면 충돌** — 하나로 합친 브랜치로 정리해서 머지하는 작업이 다음 우선순위.

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
| `CLASS_QUORUM_OVERRIDE` | {2:1, 53:1, 54:2, 15:1, 39:1, 21:1, 29:2} | `src/multi_view_fusion.py` — bumblebee_albacore(1), dove_pink(1), dove_white(2), redbull(1), crystal_hot_sauce(1), dr_pepper(1), spam(2). 숫자는 이벤트로 인정하는 데 필요한 동시 카메라 수 |
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
- [ ] `pop_tararts_strawberry` FP×4 (GT=1인데 Sub=3씩) — **원인 파악됨 (2026-06-25)**: `init_frames=30` 동안 미검출 → `initial_inventory=0` 오설정. 프레임 1296~3792에 실제 신호 있으나 0~1296 구간 미검출로 첫 감지 시 RETURN 발화 후 gaps마다 oscillation. 해결 후보: `--init_inv` 수동 지정(pop_tararts=1).
- [ ] `hunts_sauce`, `pepperidge_farm_milk_chocolate_macadamia_cookies` — 60초대 시간 오차. pop_tararts와 동일한 초기재고 실패 패턴으로 추정. debug_log 확인 필요.
- [ ] `bulls_eye_bbq_sauce_original` — conf=0.2 테스트(Phase 14)에서 효과 없음 확인. fusion quorum 또는 신호 자체 부족 원인으로 추정. 미해결.
- [ ] `haribo_gold_bears_gummi_candy` — 반환/구매 둘 다 더블-FN. 신호가 WINDOW_SIZE/CONFIRM_FRAMES를 넘긴 적 없음. 원인 미파악.
- [ ] `pepperidge_farm_milano_cookies_double_chocolate` — confirm=150 테스트(Phase 13)에서 검출은 되나 +50초 지연으로 order F1 악화. confirm 값 재탐색(60~80) 가능하나 우선순위 낮음.
- [ ] `campbells_chicken_noodle_soup` — cam4가 구매(11s) 이후로도 계속 오감지. `campbells_chunky_classic_chicken_noodle`과 혼동 의심.
- [ ] `frappuccino_coffee` — 영상 초반 노이즈로 너무 일찍(3.6s) 확정(실제 구매는 16s). confirm=200은 Phase 11에서 완전 미발화로 역효과. 적정 confirm 값(50~100 범위) 재탐색 필요.
- [x] ~~SORT 트래커~~ → Phase 12에서 order F1 악화(85.4%→83.4%) 확인. `src/tracker.py` 및 관련 코드 main에서 완전 제거 (2026-06-25)
- [x] ~~`fix/per-class-conf` (bulls_eye conf=0.2)~~ → Phase 14에서 효과 없음 확인, 브랜치 삭제 (2026-06-25)
- [x] ~~`fix/pepperidge-milano-confirm`~~ → Phase 13에서 order F1 악화 확인, 브랜치 삭제 (2026-06-25)
- [x] ~~`dove_white` 중복 발화~~ → quorum=2로 절충, 순오류 4건→2건 감소 (2026-06-23)
- [x] ~~정확도 검증~~ → `data/ground_truth_v2.csv` + `tools/score_methods.py`(3종 방식) + 리더보드로 완료 (2026-06-23)
- [x] ~~`redbull`/`crystal_hot_sauce`/`dr_pepper` 완전누락~~ → quorum=1 추가로 해결, F1 90.0%→91.5% (2026-06-23)
- [x] ~~`spam` 완전누락~~ → quorum=2 추가로 해결, F1 91.5%→92.0% (2026-06-24)

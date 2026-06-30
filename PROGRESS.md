# REDRED 진행 현황

**팀원:** 조강희 / 정현수 / 박준영  
**서버:** `ssh aicompetition30@147.46.121.38` (학교 내부망/VPN 필요)

---

## 현재 상태 (2026-06-28 최종)

파이프라인 정상 동작 중. `data/ground_truth_v2.csv`(105개 실측 이벤트, **시간 포함**)가 현재 기준 GT — `tools/score_methods.py`로 3가지 방식 동시 채점.

**현재 최고 성능 = main (Phase 24, 2026-06-27 강희조+Claude, main merge 완료)**  
**발표용 범용 기준선 = test/generic-pipeline (Phase 25, 2026-06-28) — order F1 91.7%**

| 항목 | 값 |
|------|-----|
| RTF | 0.751 (A6000 측정, 목표 < 1.0, **RTF≤1이면 20점 만점**) |
| F1 (count 참고용) | **99.5%** |
| **F1 (order/LCS — `tools/score.py` 기준)** | **98.6%** |
| F1 (time, 지연보정 ±3초) | 98.6% |
| **추정 총점 (정확도+RTF, /60)** | **약 59.4점** (정확도 39.4 + RTF 20.0) |
| 모델 mAP@0.5 | 98.1% (제공 가중치 `yolov7_custom.pt` 사용 중) |
| 제출 파일 | `~/REDRED/output/submission_skip2.csv` |

**채점 기준 (2026-06-25 갱신):**
- 정확도 40점: `score.py`가 **order/LCS F1 × 0.4** 기준으로 변경 (공식 미공개이나 이벤트번호 포함 기준에 더 근접)
- RTF 20점: 대회 공고 기준 **RTF ≤ 1 → 만점(20점)**, RTF > 1 → 상대평가(미공개)
- 기존 `20×(1-RTF/3)` 공식은 잘못된 역산값이었음 — 수정됨

**브랜치 테스트 결과 (2026-06-25):**
| 브랜치 | order F1 | 결과 |
|--------|---------|------|
| Phase 10 baseline | 85.4% | 기준 |
| fix/tracker-default (max_age=15) — 어느 GPU에서 측정했는지에 따라 다름 | 83.4% (특정 GPU) / **85.3%(A6000)** | 처음엔 ❌ 판정해서 브랜치 삭제했으나, A6000에서 재측정하니 baseline과 거의 동급(order -0.1pp, count +0.9pp, RTF 변화 없음) → **Phase 15에서 재도입, main에 직접 반영** |
| fix/pepperidge-milano-confirm | 84.7% | ❌ 악화 — milano 검출되나 +50초 지연. 브랜치 삭제 |
| fix/per-class-conf (bulls_eye) | 85.4% | ❌ 효과 없음. 브랜치 삭제 |
| fix/frappuccino-init / fix/ghost-event-cooldown | (미측정) | ❌ confirm_frames=200이 frappuccino 구매를 완전 미발화로 만드는 회귀 확인(Phase 11) — 둘 다 브랜치 삭제, `replay_event_detector.py` 진단도구 버그수정만 main으로 별도 반영 |

⚠️ 단일 F1만으로 판단하지 말 것 — count/order/time 세 지표 같이 보기. ⚠️ **GPU 종류(A6000 vs MIG)에 따라 같은 코드도 다른 점수가 나올 수 있음** — 브랜치 비교할 땐 어떤 GPU에서 측정했는지 같이 기록할 것. 리더보드: `output/leaderboard.html` (브라우저로 열기).

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

### 2026-06-25 | Phase 12~14 — 채점 기준 갱신 + 브랜치 전수 테스트 (강희조+Claude)

**채점 기준 갱신:**
- `tools/score.py`: 정확도 기준 count F1 → **order/LCS F1**, RTF 공식 → **RTF≤1=20점 만점**
- 리더보드 기존 항목 전부 재계산

**브랜치 테스트 결과:**

| 브랜치 | count F1 | order F1 | 결과 |
|--------|---------|---------|------|
| Phase 10 baseline | 92.0% | 85.4% | 기준 |
| Phase 12: tracker max_age=15 | 92.9% | 83.4%→**85.3%(정정)** | ~~❌ order 악화~~ → `score.py`가 구버전 GT(v1) 쓰던 버그였음(아래 Phase 15 정정 참고). 실제론 baseline과 동급 이상, main에 재도입함 |
| Phase 13: milano quorum=2+confirm=150 | 93.0% | 84.7% | ❌ 검출은 되나 confirm 지연으로 순서 틀림 |
| Phase 14: bulls_eye conf=0.2 | 92.0% | 85.4% | ❌ 효과 없음 (threshold 문제 아님) |

- 브랜치 3개 삭제: `fix/pepperidge-milano-confirm`, `fix/tracker-default`, `fix/per-class-conf`
- `tracker.py` 코드는 복원 유지 (`--use_tracker` 플래그로 선택 활성화 가능, run_test.sh는 off)
- `--tracker_max_age` 기본값 3 → **15** 로 변경

**`pop_tararts_strawberry` 원인 분석:**
- debug_log 확인: 프레임 0~1296(0~43s) 미검출, 1296~3792(43~126s) 신호 있음, 3792~(126s~) 미검출 → GT purchase=127s 일치
- **원인: 초기 재고 추정 실패.** init_frames=30 동안 미검출 → initial_inventory=0 오설정 → 첫 감지 시 RETURN 발화, 이후 16초 gaps마다 oscillation 반복
- 해결 후보: `--init_inv` 수동 지정(pop_tararts=1)

### 2026-06-25 | Phase 15 — 미병합 브랜치 정리, 트래커 GPU 비결정성 확인 후 재도입 (박준영+Claude)

**브랜치 정리**: 어수선하게 쌓인 미병합 브랜치들을 main 하나로 정리하는 작업. `fix/frappuccino-init`/`fix/ghost-event-cooldown`은 둘 다 `confirm_frames=200`이 frappuccino 구매를 완전 미발화시키는 회귀(Phase 11)를 그대로 갖고 있고 건질 만한 추가 가치가 없어서 **삭제**. 단 `fix/ghost-event-cooldown`에만 있던 `tools/replay_event_detector.py`의 defaultdict-peek 버그 수정(haribo 오판 원인, Phase 11 참고)은 **main에 별도로 반영**.

**트래커 재도입 — Phase 12 판정 재검토**: Phase 12에서 "order F1 85.4%→83.4% 악화"로 판정해 `fix/tracker-default`를 삭제했었는데, 그 각주("현재 GPU 한정, A6000은 85.3%")를 다시 검증함. Phase 11에서 A6000 큐로 직접 측정한 결과(count 92.0%→92.9%, order 85.4%→85.3%, time 84.5%→85.3%, RTF 0.770→0.756)는 baseline과 거의 동급이거나 더 나음. `run_test.sh`에 `--use_tracker --tracker_max_age 15` 재추가, main에 직접 반영. (83.4%의 진짜 원인은 GPU가 아니었음 — 아래 정정 참고)

**⚠️ 정정 (같은 날 늦게 발견): 83.4%는 GPU 비결정성이 아니라 `score.py`의 GT 파일 버그였음.** `tools/score.py`의 `GT_PATH` 기본값이 `data/ground_truth.csv`(Phase 8에서 폐기된 v1, 순서 일부 뒤바뀜+행 내용 오류 2건 있던 그 버전)로 남아있었는데, `run_test.sh`가 `score.py`를 호출할 때 `--gt`를 안 줘서 매번 이 구버전으로 채점되고 있었음(`score_methods.py`는 `--gt data/ground_truth_v2.csv`를 명시적으로 받아서 영향 없었음). 같은 `submission_skip2.csv`로 `python tools/score.py --sub output/submission_skip2.csv --gt data/ground_truth_v2.csv`를 돌리니 **정확히 85.3%(TP=90 FP=16 FN=15)**가 나와서 확인됨 — LCS 알고리즘 자체는 두 스크립트가 동일, GT 파일만 다름. **`GT_PATH`를 `ground_truth_v2.csv`로 수정함.** 즉 Phase 12의 트래커 거부 판정과 그 근거였던 "GPU 종류 비결정성" 둘 다 잘못된 진단이었음 — 트래커는 처음부터 안정적으로 baseline과 동급 이상이었음. (참고: `tools/analyze_detections.py`/`compare_to_ground_truth.py`/`diagnose_missing_events.py`는 Phase 7 시절 일회성 진단 스크립트라 여전히 v1을 기본값으로 쓰고 있음 — 지금 쓸 일 있으면 `--gt data/ground_truth_v2.csv` 명시할 것.)

**`feature/camera-weights`(정현수) → `feature/camera-weights-v2`(박준영+Claude)로 재작업 후 main merge 완료 (2026-06-25)**: 원본은 카메라별 동적 weight(좌우 occlusion 감지 + top캠 1.5배) 아이디어는 유효했으나 base `event_detector.py`가 Phase 7 버그 2개를 되돌리고 있어 merge 불가였음 — `compute_cam_weights()`만 최신 main 위로 이식(event_detector.py 무변경, diff 확인됨). 서버 A6000 검증 과정:
  1. 1차(0.5x/1.5x 곱셈): baseline과 **완전히 동일한 점수** — `fuse_weighted_median`이 과반 투표 방식이라 1.5x 정도 곱으로는 median 구성이 거의 안 바뀜.
  2. 2차(가려진 쪽 weight=0으로 완전 제외): 여전히 동일 → 알고보니 `git checkout`이 한 번도 실제로 안 먹혀서 줄곧 main에서 테스트하고 있었음(서버 git이 untracked 파일 충돌로 checkout silently 실패).
  3. 체크아웃 재확인 후 3차 진짜 실행: **count F1 92.9%→93.9%, order F1 85.3%→85.4%, time F1 85.3%→85.4%**, occlusion 감지율 right=8.2%/left=1.9%(`Camera occlusion stats` 로그로 확인). `haribo_gold_bears_gummi_candy`가 **처음으로 발화**(기존엔 신호부족 더블-FN이라 결론 — bumblebee/dove/redbull과 같은 "5캠 중 소수만 보임" 구조적 문제였음이 확인됨). 단 새 haribo 이벤트가 26초 타이밍 오차 있음(별도 과제).
  - 전 지표 순개선, 회귀 없음 → **main에 merge 완료**.

### 2026-06-28 | Phase 25 — 완전 범용 파이프라인 기준선 측정 (강희조+Claude)

**목적:** 정성평가 발표용. 영상 특화 튜닝 전부 제거하고 순수 범용 파이프라인 성능 측정.

**제거한 설정 (`test/generic-pipeline` 브랜치):**
- `CLASS_QUORUM_OVERRIDE = {}` → 기본 weighted median만
- `CLASS_CAM_WHITELIST = {}` → 카메라 필터링 없음
- `_DEFAULT_CAM_WEIGHTS = [1,1,1,1,1]` → cam2 1.5배 사전지식 제거
- `_cam_weight_excluded = set()` → milano 예외 제거

**결과 (서버 A6000, test/generic-pipeline):**

| 지표 | 범용 파이프라인 | Phase 24 (튜닝) | 차이 |
|------|----------------|----------------|------|
| count F1 | 93.6% | 99.5% | -5.9%p |
| order F1 | **91.7%** | **98.6%** | **-6.9%p** |
| time F1 | 89.9% | 98.6% | -8.7%p |
| 추정 총점 | ~56.7점 | 59.4점 | -2.7점 |

**범용에서 낮은 원인:**
- milano/dove_pink: 2대만 보여 median 0↔1 반복 → 과다발화 (각 4회/3회, GT=1회)
- redbull/crystal_hot_sauce/campbells: 5대 과반 구조적 불가 → 아예 미감지
- dove_white/white_rain: 타이밍 오차 22s/10s

**의의:** "범용 91.7% → 영상 분석+튜닝 98.6%" 기여도 정량화 (발표용). **브랜치 보존, main merge 안 함.**

---

### 2026-06-28 | Phase 26 — 최신 논문 기술 적용 시도 (강희조+Claude) [진행 중]

**목적:** 정성평가 발표용 "최신 연구 흐름 적용 시도 + 이유 분석". 도움이 되면 채택, 안 되면 왜 안 되는지 분석.

#### 검토한 논문들

| 논문 | 연도 | 핵심 기술 | 우리 적용 가능성 |
|------|------|-----------|-----------------|
| [ByteTrack: ECCV 2022](https://arxiv.org/abs/2110.06864) | 2022 | 2-stage matching으로 저신뢰도 detection도 track 유지 | ✅ **구현 완료** (test/bytetrack) |
| [Survey: Autonomous Retail (2025)](https://arxiv.org/abs/2503.07997) | 2025 | 무인판매대 기술 조망 (손 감지, BEV, 다중카메라 추적) | 아이디어 참고 |
| [Hand-Object Interaction (2025)](https://arxiv.org/abs/2507.13326) | 2025 | 손-물체 상호작용으로 구매 이벤트 직접 감지 | ❌ egocentric 카메라 전용, 고정 카메라 학습 데이터 없음 |
| [MCBLT: Multi-Camera 3D (2024)](https://arxiv.org/abs/2412.00692) | 2024 | 다중 카메라 homography로 3D 위치 융합 | ❌ 카메라 캘리브레이션 행렬 필요, 환경 미비 |
| [Enhanced Self-Checkout YOLOv10 (2024)](https://arxiv.org/abs/2407.21308) | 2024 | 개선된 감지 모델 + 체크아웃 파이프라인 | ❌ 모델 재학습 필요 (YOLO11 시도에서 이미 실패, Phase 23) |

#### Phase 26-A: ByteTrack (2-stage matching) 적용

**근거:** ByteTrack은 SORT와 달리 low-confidence detection을 Stage 2에서 미매칭 track에 추가 연결.
우리 파이프라인에서 기대 효과:
- `white_rain_body_wash` 21-23s 구간 occlusion → confidence 하락한 detection이 기존 track 유지 → fused count 하락 방지 → FP purchase 제거 기대
- low-confidence(0.4~0.6)로 새 track 생성 안 함 → FP track 억제

**구현 (`src/tracker.py`, `test/bytetrack` 브랜치):**
```python
class ByteSort(Sort):
    # Stage 1: conf >= high_thresh(0.6) → 모든 기존 track 매칭
    # Stage 2: conf < high_thresh → Stage 1 미매칭 track에만 추가 매칭
    # 새 track은 Stage 1 (high-conf) 미매칭만 생성
```
`--tracker_type bytetrack` 플래그 추가. SORT 인터페이스 완전 호환.

**로컬 검증:**
```
FP noise (conf=0.45 x5):
  SORT:      confirmed tracks=1 (FP track 생성됨)
  ByteSort:  confirmed tracks=0 (low-conf → 새 track 생성 안 함)
```

**서버 테스트 커맨드:**
```bash
git fetch && git checkout test/bytetrack && git pull
PYTHONPATH=~/yolov7 python src/run_pipeline.py \
    --videos ~/Dataset/4.TestVideo_Sample/cam{0..4}/Sample_1.mp4 \
    --weights ~/Dataset/yolov7_custom.pt \
    --names data/names.txt --prices data/prices.csv \
    --out output/submission_bytetrack.csv \
    --skip 2 --conf 0.4 --device 0 \
    --use_tracker --tracker_type bytetrack --tracker_max_age 15 \
    --timed_log output/sub_bytetrack_timed.csv
python tools/score_methods.py --gt data/ground_truth_v2.csv \
    --sub output/submission_bytetrack.csv --timed output/sub_bytetrack_timed.csv
```

**결과:** (서버 테스트 후 기입)

#### 구조적으로 적용 불가한 기술 분석 (발표용)

1. **Hand-Object Interaction Detection** — 구매 이벤트를 count 변화가 아닌 "손이 물건 집는 동작"으로 직접 감지. 정확도·실시간성 모두 우월할 수 있으나: (a) 기존 논문들이 egocentric(1인칭) 카메라 기반 — 우리 고정 카메라와 시점 전혀 다름, (b) 고정 카메라 기준 학습 데이터 없음, (c) 추가 모델 추론으로 RTF 크게 저하 예상. **"다음 단계 개선 방향"으로 발표에서 언급 적합.**

2. **Multi-camera 3D Tracking (BEV fusion)** — 5대 카메라를 homography로 Bird's Eye View로 통합해 정확한 위치 기반 count. 우리 weighted median보다 이론적으로 우수하나: (a) 카메라 내/외부 파라미터 캘리브레이션 행렬 필요, (b) 대회 환경에서 캘리브레이션 데이터 미제공. **캘리브레이션이 있었다면 적용 가능했을 것으로 분석.**

3. **YOLO11 / YOLOv10 재학습** — Phase 23에서 이미 시도. YOLO11m 18k장 학습 → order F1 73.4% (기존 98.6% 대비 -25%p). 제공 가중치(`yolov7_custom.pt`, mAP 98.1%)가 20만 장 풀 학습 결과라 18k 재학습으론 경쟁 불가. **"최신 아키텍처 전환 시도, 데이터 규모 문제로 기각" 스토리.**

---

### 2026-06-27 | Phase 24 — per-class 카메라 화이트리스트 (강희조+Claude) [완료, main merge 완료]

**목표:** order F1 100% 시도 (main 기준 96.6%)

**`--per_cam_log` 추가 (`src/run_pipeline.py`):**
퓨전 이전 카메라별 raw 감지 수 저장 옵션. 서버 실행 후 클래스별 카메라 분포 분석.

**per_cam_log 분석 결과:**

| 클래스 | 유효 카메라 | 분석 |
|--------|------------|------|
| campbells (43) | cam0=64fr, cam4=39fr | cam4가 chunky 혼동 주범 → cam0만 사용 |
| milano (42) | cam3=929fr, cam4=492fr, cam0=20fr(노이즈) | cam3+cam4만 |
| dove_white (54) | cam3=614fr, cam2=311fr, cam4=184fr, cam0=1fr | cam3만, quorum=1 |

**`CLASS_CAM_WHITELIST` 구현 (`src/multi_view_fusion.py`):**
```python
CLASS_CAM_WHITELIST = {
    43: [0],     # campbells: cam4 chunky혼동 차단
    42: [3, 4],  # milano: cam0 노이즈 제거, quorum=1
    54: [3],     # dove_white: cam3만, quorum=1
}
```

**결과 (서버 A6000, fix/cam-whitelist):**

| 지표 | Phase 20 (main) | Phase 24 | 변화 |
|------|----------------|---------|------|
| count F1 | 98.6% | 99.5% | +0.9% |
| order F1 | 96.6% | **98.6%** | **+2.0%p** |
| time F1 | 96.6% | 98.6% | +2.0% |
| RTF | 0.751 | 0.749 | 동일 |
| 추정 총점 | 58.6점 | **59.4점** | **+0.8점** |

milan purchase/return 모두 감지됨 ✅, dove_white 타이밍 22.5s 오차 해소 ✅

**남은 문제 (구조적 한계로 판단):**
- `campbells`: initial_inventory=0으로 잘못 추정(cam0가 init_frames=30 window 밖인 raw frame 36부터 감지) → FP return 발생. 모든 quorum/confirm/whitelist 조합 시도했으나 FP return↔purchase 간 트레이드오프로 order F1 변화 없음.
- `white_rain`: 모든 카메라가 영상 끝까지 감지(cam0~4: 60~477s). occlusion 메커니즘이 fused count를 21-23s에 오하락시켜 FP purchase 발생. occlusion 제외 시도 → 오히려 악화(59.4→59.2). confirm=120 시도 → 50s로 과지연(59.4→59.4 유지). 구조적 한계.

**시도 이력:**
1. cam-whitelist 첫 적용 → **59.4 (최고점)**
2. campbells confirm=90 → 59.4 (FP↔FN 상쇄)
3. white_rain confirm=120 → 59.4 (50s 과지연)
4. white_rain occlusion 제외 → **59.2 (악화)** → 즉시 revert

**최종 상태:** 첫 적용 시점 코드로 복원. `fix/cam-whitelist` 브랜치 보존(main merge 미진행).

---

### 2026-06-26 | Phase 23 — YOLO11 학습 및 파이프라인 테스트, 기각 (강희조+Claude)

YOLO11m(ultralytics) 학습 후 파이프라인에 적용. 학습 조건: 증강 제외 18,392장, 50epoch, batch=32, A6000.

**학습 결과 (epoch 50):**
- mAP@0.5: 98.4%, Precision: 95.4%, Recall: 96.1% — YOLOv7(98.1%)과 유사

**파이프라인 결과 (서버 A6000, fix/yolo11 브랜치):**

| 지표 | YOLOv7 main | YOLO11 | 변화 |
|------|------------|--------|------|
| count F1 | 98.6% | 78.0% | **-20.6%p** |
| order F1 | 96.6% | 73.4% | **-23.2%p** |
| RTF | 0.751 | **0.467** | +빠름 |

이벤트 105개 중 72개만 감지 (FN=36). mAP는 비슷하지만 파이프라인 정확도가 심각하게 낮음.

**원인:** 대회 제공 `yolov7_custom.pt`는 주최측이 ~20만장으로 학습한 가중치. 우리가 학습한 YOLO11은 제공된 이미지 중 증강 제외 18,392장만 사용 → 학습 데이터 10배 이상 차이. 동등 비교하려면 20만장으로 YOLO11 재학습 필요하나 시간 대비 효과 불확실. **브랜치 삭제.**

`run_pipeline.py`에 ultralytics 자동감지 로직(load_model fallback)은 fix/yolo11 브랜치에만 존재, main은 YOLOv7 전용 유지.

---

### 2026-06-26 | Phase 22 — CSV 포맷 수정: 총 재고 금액 + UTF-8 BOM (강희조+Claude)

대회 스펙 재확인: "이벤트 발생 후 총 재고 금액"은 상품별 누적 판매액이 아니라 **이벤트 후 전체 진열대 재고 × 단가 합산**이었음.

**변경사항:**

1. `csv_generator.py`: `total_mode="inventory"` 옵션 추가 — 이벤트 발생 후 모든 클래스의 `inventory[c] × price[c]` 합산. 헤더도 `"총액"` → `"총 재고 금액"` 자동 변경.
2. `run_pipeline.py`: `total_mode="per_class"` → `"inventory"` 전환.
3. `run_pipeline.py`: `encoding="utf-8-sig"` 추가 — Excel에서 바로 열면 한글 깨지던 문제 수정.

점수 변화 없음 (채점은 재고 수량/이벤트 순서 기준, 총액은 평가 외 항목).

4. `data/ground_truth_v2.csv`: `total_inventory_krw` 컬럼 추가 — 이벤트별 정답 총 재고 금액(원). 초기 재고는 GT 첫 등장 `before` 기준(19개=1, 41개=0). Event 1 후 62,600원 → Event 105 후 0원(전부 판매).

---

### 2026-06-26 | Phase 21 — milano per_class_confirm=3 시도 및 기각 (강희조+Claude)

WINDOW_SIZE=15 환경에서 milano(class_id=42)에 `per_class_confirm=3` 적용. 9fr 신호가 15-frame window에서 median을 뒤집을 수 있게 됐으므로 confirm을 짧게 줘서 잡는 시도.

**결과 (서버 A6000):**

| 지표 | Phase 20 (main) | Phase 21 | 변화 |
|------|----------------|---------|------|
| count F1 | 98.6% | 99.5% (FN=1, campbells만 남음) | +0.9% ✅ |
| order F1 | **96.6%** | 95.7% | **-0.9%** ❌ |
| time F1 | 96.6% | 96.7% | +0.1% |
| 추정 총점 | **58.6점** | 58.3점 | -0.3 ❌ |

milano는 감지됐으나 **return이 GT=53.0s인데 Sub=113.2s로 60초 늦게 발화** — confirm=3이 53초 시점의 짧은 신호를 못 잡고, 훨씬 나중에 다른 burst에서 발화한 것. 그 결과 LCS 순서에서 FP가 2→4로 증가, order F1 악화. count는 좋아졌으나 총점 기준으로 Phase 20이 더 나음. **브랜치 삭제.**

**결론:** milano는 신호 자체가 짧고 불규칙해서 confirm 값으로 타이밍을 맞추기 어려움. count만 보면 잡히나 순서/타이밍이 틀려 오히려 손해.

### 2026-06-26 | Phase 20 — WINDOW_SIZE 25→15 (강희조+Claude)

`pepperidge_farm_milk_choc` 노이즈 억제를 위해 25로 설정했던 `WINDOW_SIZE`를 **15**로 낮춤. Phase 16 camera-weights로 해당 노이즈가 이미 해결됐으므로 낮춰도 FP 증가 없음.

milano 감지(9fr 신호 → 25-frame window에서 9/25=36%로 median 미달)가 목적이었으나 **milano는 여전히 미감지** — quorum=3(default) 조건 자체가 병목. 대신 기존 이벤트들이 더 빨리 확정되면서 LCS 순서 매칭 1건 개선.

| 지표 | Phase 18 | Phase 20 | 변화 |
|------|---------|---------|------|
| count F1 | 98.6% (FP=0) | 98.6% (FP=0) | 동일 |
| order F1 | 95.7% | **96.6%** | +0.9% |
| time F1 | 96.6% | 96.6% | 동일 |
| 추정 총점 | 58.3점 | **58.6점** | +0.3 |

FP=0 유지, 회귀 없음. 원복 필요 시 `src/event_detector.py`의 `WINDOW_SIZE = 15` → `25` 1줄 변경. **main merge 완료.**

### 2026-06-26 | Phase 19 — 미발화 3건 해결 시도, 전부 기각 (박준영+Claude)

campbells/milano/dove_white/white_rain 4개 클래스 추가 개선 시도. 전부 구조적 한계로 Phase 18 상태가 최선임을 확인.

**campbells_chicken_noodle_soup**: quorum=1 시도 → cam4가 campbells_chunky와 혼동하여 purchase 3.6s 조기발화 + FP return + 117s FP purchase = 이벤트 3개 발생(GT=1). order F1 95.7%→94.3%로 악화. **기각**. bbox 필터 없이는 불가.

**pepperidge_farm_milano_cookies_double_chocolate**:
- quorum=2 + confirm=45: 4이벤트 과다발화, 타이밍 모두 틀림
- quorum=2 + confirm=90: 2이벤트(count 맞음), return 49s 늦음/purchase ±3s 초과. order 악화
- quorum=3(default) + per_class_confirm=9: WINDOW_SIZE=25 때문에 9프레임 run이 median을 못 뒤집음 → 이벤트 0건, Phase 18와 동일
- 근본 원인: WINDOW_SIZE=25가 너무 커서 3-camera 신호(9fr/17fr)를 잡지 못함. quorum=2는 신호가 0↔1 진동해서 타이밍 불가. **기각**. WINDOW_SIZE를 낮추면 해결 가능하나(원래 설정 이유인 pepperidge_farm 노이즈가 Phase 16 camera-weights로 이미 해결됨), 전역 파라미터 변경 리스크 있어 보류.

**dove_white/white_rain 타이밍**: per_class_confirm으로 지연 시도 → 계산값(220fr/105fr)이 Sample_1.mp4 타이밍에 맞춘 하드코딩이라 채점 영상에서 역효과 가능. Phase 11 per_class_cooldown 기각 때와 동일한 이유. **기각**.

**최종 확인**: fix/milano-campbells 브랜치 모든 시도 revert → Phase 18 수치 재현 확인(count 98.6%, order 95.7%, time 96.6%, 58.3점). main merge 예정.

### 2026-06-26 | Phase 18 — bumblebee_albacore quorum 1→2, 타이밍 오차 해결 (박준영+Claude)

Phase 16에서 `bumblebee_albacore`에 quorum=1을 설정했는데, 1대 카메라 신호만으로도 이벤트가 확정되다 보니 타이밍 오차가 생겼음:
- return: Sub=54.1s, GT=59s, diff=**4.9s** (1대가 실제보다 일찍 감지)
- purchase: Sub=119.9s, GT=103s, diff=**16.9s** (1대가 ~14s간 false-positive 발화)

로컬 `debug_frame_counts.csv`로 quorum=2 시뮬레이션:
- return → 62.3s, diff=**3.3s** ✓ (±3s 이내)
- purchase → 105.8s, diff=**2.8s** ✓ (±3s 이내)

`src/multi_view_fusion.py`의 `CLASS_QUORUM_OVERRIDE`에서 bumblebee(id=2) 1→2로 변경. 서버 A6000 검증으로 두 이벤트 모두 time mismatch 목록에서 사라짐.

추가로 `per_class_confirm` 인프라를 `EventDetector`에 재추가 — `fix/frappuccino-init`(Phase 15) 삭제 시 같이 제거됐던 것. 로컬 replay 테스트 시 필요.

**결과 (서버 A6000 검증)**: count F1 **98.6%**(불변), order F1 **93.7%→95.7%**, time F1 **94.7%→96.6%**, RTF 불변, 추정 총점 **57.5→58.3점**. time mismatch 4건→2건(남은 것: dove_white purchase 22.7s, white_rain_body_wash purchase 10.2s). **main에 merge 완료.**

### 2026-06-26 | Phase 17 — 초기재고 추정에도 camera-weights 적용 (박준영+Claude)

`estimate_initial_inventory()`가 `fuse(per_cam)` — 균등weight — 을 쓰고 있어서, 영상 시작 직후 ~1초(init_frames=30) 동안 일부 카메라에서 가려진 클래스가 median=0으로 잘못 집계됨. 그 결과 해당 클래스가 initial_inventory=0(없음)으로 잡히고, 나중에 시스템이 처음으로 안정적으로 감지하는 순간(`WINDOW_SIZE+CONFIRM_FRAMES=55` 처리프레임 = raw frame ~110 = Frame 112)에 `반환(0→1)` 가짜 이벤트가 일제히 발화됨 — 이게 매 실행마다 Frame 112에서 `white_rain_body_wash`/`frappuccino_coffee`/`coca_cola_glass_bottle` 3개가 동시에 뜨던 이유.

`compute_per_class_cam_weights(..., exclude_class_ids=_cam_weight_excluded)`를 초기재고 추정 루프 내 `fuse()` 호출에도 전달. 메인 루프와 동일한 occlusion-aware weight가 init 단계에도 적용되어 첫 ~1초 동안 가려진 클래스도 정확하게 카운트됨.

**결과 (A6000 서버 검증)**: count F1 96.7%→**98.6%**, order F1 91.9%→**93.7%**, time F1 92.8%→**94.7%**, RTF 변화 없음, 추정 총점 56.7→**57.5점**. **FP=0 달성** — 제출물에 가짜 이벤트가 하나도 없음. `bulls_eye_bbq_sauce_original`도 count 불일치에서 사라짐(초기재고에 올바르게 포함됨). 남은 FN: `campbells`(클래스혼동), `milano`(의도적 camera-weight 예외). **main에 merge 완료.**

### 2026-06-26 | Phase 16 — camera-weights를 개별 카메라 단위로 일반화 (박준영+Claude)

**문제**: Phase 15의 camera-weights-v2(좌(0,4)/우(1,3) 그룹 평균 비교)는 haribo는 구제했지만 `pepperidge_farm_milano_cookies_double_chocolate`(probe3: 최대 3대 동시)는 그대로였음 — milano가 occlusion될 때 다른 클래스들은 정상이라 **전역 프레임 평균**에 묻혀 70% 임계값을 못 넘었을 것으로 추정.

**1단계 — 클래스별(per-class) confidence로 분리**: `compute_cam_weights(per_cam_dets, class_id=...)` — 그 클래스 자신의 confidence만으로 좌/우 판단. 로컬 시뮬레이션으로 "다른 클래스는 정상, milano만 가려진" 상황에서 전역평균은 0(틀림)/클래스별은 1(맞음)임을 확인. 서버 결과: **count F1 93.9%→95.7%, order/time F1 85.4%→91.0%**. `frappuccino_coffee` 구매도 처음으로 정확히 검출됨(이전 완전 미발화). 단 milano는 여전히 그대로(↓1 purchase/return) — occlusion 감지율은 8%→27~38%로 대폭 증가(전역평균이 실제 가림을 그만큼 희석시키고 있었음을 보여줌).

**2단계 — 좌/우 그룹 비교를 카메라 5대 개별 비교로 일반화**: milano가 안 풀린 이유는 "그룹 *내부*에서 비대칭으로 가려지는" 패턴(예: 왼쪽1+오른쪽1+top만 보임)이라 그룹 평균 비교 자체가 못 잡았을 것으로 추정. 규칙 변경: 카메라 i가 0인데 나머지 4대 중 N대 이상이 양수면 i 제외.
- N=2로 첫 시도: **order/time F1 85.3%→91.0%**(milano 포함 추정 대상이 더 넓어짐). 그런데 milano가 GT=1인데 Sub=4로 **과다발화** — "정확히 2대만 보임"이 불안정하게 반복돼서 median이 0↔1을 오가며 여러 번 confirm된 것(dove_white quorum=1 때와 동일 패턴). 전체 지표 90.3%로 소폭 악화.
- N=3으로 올려서 재시도: **haribo까지 다시 깨짐**(order/time F1 85.3%로 원복). 원인: "나머지 4대 중 3대 이상"은 전체 5대 중 60%로 이미 과반이라, 균등weight로도 원래 median=1이 나오는 상황 — 이 메커니즘이 개입할 필요 자체가 없어서 N=3은 사실상 무의미한 임계값이었음. **N=2가 과반 미달(40%)을 구제하는 유일한 지점**이라는 게 확인됨.
- **최종**: N=2로 복귀 + milano만 `exclude_class_ids`로 weight 메커니즘에서 예외처리(기본 weight 유지, 깨끗한 미검출로 남김) — `compute_per_class_cam_weights(..., exclude_class_ids={milano_id})`.

**최종 결과 (서버 A6000 검증)**: **count F1 96.7%, order F1 91.9%, time F1 92.8%**, RTF 0.811(변화 없음). 추정 총점 54.2→**56.7점**. haribo는 GT와 완전 일치, milano는 부작용 없이 미검출 유지.

**예상 밖의 보너스**: `pop_tararts_strawberry`/`hunts_sauce`/`pepperidge_farm_milk_chocolate_macadamia_cookies`의 "이벤트직후 유령반전"(Phase 11에서 별개 문제로 분류했던 것)이 count 불일치 목록에서 **완전히 사라짐**. Phase 11 분석대로 이 유령반전의 원인이 "혼잡구간 occlusion으로 인한 프레임별 fusion 신호 불안정"이었다면, camera-weights가 그 occlusion 자체를 직접 보정해주니 유령반전도 같이 줄어드는 게 인과적으로 타당함 — multi-camera occlusion과 단일클래스 유령반전이 사실 같은 근본 원인이었을 가능성. (1회 실행 결과라 재현성 확인은 아직 안 함.)

**main merge 완료.** 남은 잔여 이슈(이번 merge로 영향 없음): `bulls_eye`/`campbells`는 여전히 깨끗한 FN(다른 원인), `coca_cola_glass_bottle`/`frappuccino_coffee`/`white_rain_body_wash`는 가짜 반환 노이즈, `dove_white`/`bumblebee_albacore`는 타이밍 오차(quorum override 대상이라 camera-weights와 무관).

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
| `CLASS_QUORUM_OVERRIDE` | {2:2, 53:1, 54:2, 15:1, 39:1, 21:1, 29:2} | `src/multi_view_fusion.py` — bumblebee_albacore(2), dove_pink(1), dove_white(2), redbull(1), crystal_hot_sauce(1), dr_pepper(1), spam(2). 숫자는 이벤트로 인정하는 데 필요한 동시 카메라 수 |
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

## KD_clean 파이프라인 클래스별 제어 현황 (2026-06-30 핸드오프)

> **작성 배경:** KD 학습 완료(`yolo11m_kd_0630_0036/weights/best.pt`) 후 YOLO11 파이프라인을
> 재보정하면서 YOLOv7용 파라미터를 교체했음. 실행 스크립트: `run_test_kd_clean.sh`.
> 현재 count F1 **96.7%** (TP=104, FP=6, FN=1).

### 제어 레이어 3단계

파이프라인에 클래스별로 개입할 수 있는 포인트가 3곳 있다.

| 레이어 | 파일 | 무엇을 제어 |
|--------|------|------------|
| **L1 CLASS_CAM_WHITELIST** | `src/multi_view_fusion.py` | 클래스별로 어떤 카메라의 투표를 허용할지 |
| **L2 CLASS_QUORUM_OVERRIDE** | `src/multi_view_fusion.py` | whitelist 내 카메라 중 몇 대가 동의해야 count=1로 올릴지 |
| **L3 per_class_confirm** | `src/event_detector.py` | 이벤트 확정까지 대기 프레임을 클래스별로 다르게 설정 |

### 현재 클래스별 설정 (KD_clean 기준, per_cam_log 분석으로 YOLO11 재보정됨)

`output/per_cam_kd_clean.csv` — 각 클래스별 카메라 감지율이 여기 있음.

| class_id | 클래스명 | WHITELIST | QUORUM | 설정 이유 |
|----------|----------|-----------|--------|----------|
| 3  | cholula_hot_sauce | [3, 4] | 2 | cam3(9%), cam4(8.4%) 주도. cam0~2 노이즈 제거 |
| 5  | hersheys_cocoa | [1] | 1 | cam1(3%)만 감지. 1대뿐이라 quorum=1 |
| 8  | hunts_sauce | [0, 3] | 2 | cam0(30%), cam3(26%). 둘 다 동의해야 인정 |
| 14 | hersheys_bar | [3] | 1 | cam3(6.4%)만 감지 |
| 15 | redbull | [0] | 1 | cam0(7.1%)만 감지 |
| 21 | dr_pepper | [4] | 1 | cam4(9.9%) 주도 |
| 23 | bulls_eye_bbq_sauce_original | [3] | 1 | cam3(1.1%)만 감지 |
| 28 | quaker_big_chewy_chocolate_chip | *(없음, 전체)* | 3 | 5대 다 보이지만 중복발화가 있음 → 3대 동의 필요 |
| 38 | palmolive_orange | [3] | 1 | cam3(0.5%)만 감지 |
| 39 | crystal_hot_sauce | [3] | 1 | cam3(3.6%)만 감지 |

**설정 안 된 나머지 클래스:** 전역 기본값 (quorum=2, whitelist 없음, confirm=30프레임).

### 남은 FP 6개 — 원인 분석 및 다음 시도

| 클래스 | FP 유형 | 원인 | 다음 시도 후보 |
|--------|---------|------|--------------|
| **cholula** (id=3) | purchase FP ×1 | cam3가 NMS 이전 단계에서 count=2 이중감지. WHITELIST=[3,4]/quorum=2를 뚫고 fused=2로 올라옴 → initial=2 오추정 | YOLO11 `--iou-thres` 조정 or cam3 단독 whitelist로 교체 후 cam4 fallback 포기. 파이프라인 레벨 근본 해결 어려움 |
| **hersheys_cocoa** (id=5) | return FP ×1 | cam1 감지율 3% → init_frames=30 window(~30프레임)에서 ~40% 확률로 미감지 → initial=0 오추정 → 이후 cam1이 잡을 때 0→1=return FP | **`--n_frames 60` 시도** (서버 CLI 파라미터만 변경, 코드 수정 없음). 감지율 3%이면 60프레임에서 기댓값 1.8회 → 누락 확률 대폭 감소 |
| **hunts_sauce** (id=8) | events FP ×2 | 106~114s 구간에서 짧은 blip 재감지 → 전역 CONFIRM_FRAMES=30이 부족해서 추가 return+purchase 확정됨 | **`per_class_confirm={8: 60}` 추가** (`src/event_detector.py`의 `per_class_confirm` 인프라 기활성화 상태) |
| **quaker_big_chewy** (id=28) | events FP ×2 | hunts_sauce와 동일 blip 패턴. quorum=3으로 올려도 5대 다 보이는 구간이라 blip이 통과됨 | **`per_class_confirm={28: 60}` 추가** |
| **campbells_chicken_noodle_soup** (id=43) | FN ×1 | YOLO11 모델이 전혀 감지 못함 (mAP 기준 보면 있는데 이 영상에서 zero-detection). 파이프라인 개입 불가 | 포기. 모델 재학습 외 방법 없음 |

### 핵심 작업 파일

```
src/multi_view_fusion.py     ← L1(WHITELIST) + L2(QUORUM) 수정
src/event_detector.py        ← L3(per_class_confirm) 수정
run_test_kd_clean.sh         ← KD_clean 실행 + 자동채점 + git push
output/per_cam_kd_clean.csv  ← 카메라별 감지율 (whitelist 재보정 근거)
output/debug_kd_clean_frame_counts.csv  ← 퓨전 후 프레임별 count (blip 확인용)
```

### 다음 작업 순서 (추천)

1. **hersheys_cocoa** — `run_test_kd_clean.sh`에서 `--n_frames` 파라미터를 30→60으로 변경 후 서버 실행 (가장 간단, 코드 수정 없음)
2. **hunts_sauce + quaker** — `src/event_detector.py`의 `EventDetector.__init__` 내 `per_class_confirm` 기본값에 `{8: 60, 28: 60}` 추가, 또는 `run_pipeline.py`에서 CLI로 넘기도록 확장
3. **cholula** — 시도할 수 있지만 NMS 레벨 문제라 파이프라인 수정으로 해결하기 어려울 가능성 높음

---

## 실패한 시도 기록 (논문 기반)

### SeqNMS (2026-06-30) — `feat/seq-nms` 브랜치

**아이디어:** 프레임 간 bbox를 IoU로 연결, 연속 2프레임 이상 나타나지 않으면 confidence 0으로 억제 → FP blip 제거.  
논문: Han et al., "Seq-NMS for Video Object Detection" (2016) — https://arxiv.org/abs/1602.08465  
구현: `src/seq_nms.py` + `run_pipeline.py --seq_nms`. 파라미터: `seq_len=5, min_seq=2, penalty=0.0`

**결과 (YOLO11m KD 기준선 대비):**

| 지표 | YOLO11m 기준선 | SeqNMS |
|------|--------------|--------|
| Count F1 | 97.7% | 94.8% ▼ |
| Order F1 | 91.1% | 91.0% ▼ |

**실패 원인:** YOLO11m이 일부 클래스(dove_white, chewy_dips)를 카메라 1~2대에서 sparse하게 감지 → 연속 2프레임 조건이 정상 감지까지 억제해 FN 증가. FP 억제 효과보다 FN 부작용이 더 컸음.

---

### EMA Smoothing (2026-06-30) — `feat/ema-smoothing` 브랜치

**아이디어:** EventDetector의 sliding median(15프레임)을 EMA(지수이동평균, α=0.3)로 교체 → blip에 덜 반응하고 변화에 더 빠르게 반응.  
구현: `src/event_detector.py use_ema/ema_alpha` + `run_pipeline.py --ema --ema_alpha`

**결과 (YOLO11m KD 기준선 대비):**

| 지표 | YOLO11m 기준선 | EMA α=0.3 |
|------|--------------|-----------|
| Count F1 | 97.7% | 96.7% ▼ |
| Order F1 | 91.1% | 91.0% ▼ |

**실패 원인:** EMA가 이벤트 발생 후 내부값을 reset하지 않아 chewy_dips 두 번째 이벤트를 억제(FN). SeqNMS와 동일한 구조적 문제 — 스무딩 계열 방법은 YOLO11m의 sparse하지만 실제인 감지와 충돌.

**결론:** 시간 축 스무딩 계열(SeqNMS, EMA) 전부 이 파이프라인에 맞지 않음. 기존 sliding median이 이 구조에 최적화돼 있음. 다음 시도: Feature-level KD 재학습 (저녁 예정).

---

### Ghost Detector (2026-06-30) — `feat/ghost-detector` 브랜치

**아이디어:** "Objects Do Not Disappear" (ICCV 2023, arxiv:2308.04770) 기반. 모델이 일시적으로 감지 실패 시 마지막 bbox 위치에 ghost 감지 삽입 → dove_white(54)/milano(42) 타이밍 오차 해결 시도.  
구현: `src/ghost_detector.py` + `run_pipeline.py --ghost --ghost_per_class`. 파라미터: `{"42":700,"54":450}` (각각 47s, 30s ghost)

**결과 (YOLO11m KD 기준선 대비):**

| 지표 | YOLO11m 기준선 | Ghost Detector |
|------|--------------|----------------|
| Count F1 | 97.7% | 91.6% ▼▼ |
| Order F1 | 91.1% | 86.3% ▼▼ |

**실패 원인:** ghost가 "일시적 미감지"와 "실제 이벤트 발생"을 구분하지 못함. 고객이 실제로 물건을 집을 때(115s) YOLO11m이 잠깐 재감지 → ghost timer reset → ghost가 추가 700프레임 더 유지 → 이벤트가 오히려 더 늦게(163s) 발화. mom_to_mom 등 예상치 못한 클래스에도 ghost 중복 발화 발생(FP +6개).

논문 원본의 핵심 기여(학습 기반 위치 예측)를 재학습 없이 구현하면 이 한계가 구조적으로 발생함.

**결론:** 재학습 없이는 적용 불가. Feature KD 재학습 + occlusion augmentation이 근본 해결책.

---

## 앞으로 할 일

- [ ] 발표 자료 준비
- [x] ~~`pop_tararts_strawberry`/`hunts_sauce`/`pepperidge_farm_milk_chocolate_macadamia_cookies` 유령반전~~ → Phase 16 camera-weights merge 후 count 불일치 목록에서 완전히 사라짐(예상 밖 보너스, occlusion이 근본 원인이었을 가능성). **1회 실행만 확인, 재현성 검증 안 함** — 다시 나타나면 Phase 11 분석 참고.
- [x] ~~`bulls_eye_bbq_sauce_original`~~ → Phase 17에서 초기재고 추정 개선으로 자연스럽게 해결(초기 ~1초 occlusion으로 initial_inventory=0 오설정됐다가 이제 정확히 1로 잡힘). 더 이상 count 불일치 목록에 없음 (2026-06-26)
- [x] ~~`haribo_gold_bears_gummi_candy` 더블-FN~~ → Phase 16 camera-weights(per-camera occlusion)로 완전 해결, GT와 정확히 일치 (2026-06-26)
- [x] ~~`pepperidge_farm_milano_cookies_double_chocolate`~~ → Phase 24에서 CLASS_CAM_WHITELIST=[3,4] + quorum=1으로 해결. cam0 노이즈 제거로 order F1 정상 발화 확인 (2026-06-27)
- [x] ~~`dove_white` 타이밍 오차~~ → Phase 24에서 CLASS_CAM_WHITELIST=[3] + quorum=1으로 22.5s→0s 오차 해결 (2026-06-27)
- [ ] `campbells_chicken_noodle_soup` — cam0 whitelist 적용으로 purchase는 살아났으나, initial_inventory=0 오추정(cam0가 raw frame 36부터 감지, init_frames=30 window 밖)으로 FP return 잔존. bbox 필터 없이는 구조적 한계.
- [ ] `white_rain_body_wash` — 모든 카메라가 영상 끝까지 감지, occlusion 메커니즘이 21-23s에 fused count 오하락 → FP purchase. confirm=120(과지연), occlusion 제외(악화) 모두 실패. 구조적 한계.
- [x] ~~`frappuccino_coffee`~~ → Phase 17에서 초기재고 추정 개선으로 해결(initial_inventory=1로 올바르게 시작, 가짜 반환이 사라지고 구매도 정상 발화). 더 이상 count 불일치 목록에 없음 (2026-06-26)
- [x] ~~`feature/camera-weights`~~ → `feature/camera-weights-v2`로 재작업(`compute_cam_weights()`만 이식, weight=0 방식)해서 main에 merge 완료. count F1 92.9%→93.9%, order/time F1 85.3%→85.4%, haribo 더블-FN 해결 (2026-06-25)
- [x] ~~`bumblebee_albacore` 타이밍 오차~~ → Phase 18에서 quorum 1→2로 해결. return diff 4.9s→3.3s, purchase diff 16.9s→2.8s, 둘 다 ±3s 이내 (2026-06-26)
- [ ] `haribo_gold_bears_gummi_candy` 새 purchase 이벤트 타이밍 오차(26초, GT=163s/Sub=139.8s) — camera-weights 적용으로 처음 발화는 됐으나 정확한 시각은 아직 안 맞음.
- [x] ~~SORT 트래커~~ → Phase 12에서 "order F1 악화(85.4%→83.4%)"로 판정해 코드 제거했으나, A6000 큐로 재측정하니 85.3%(거의 동급, count는 오히려 92.0%→92.9% 개선). 83.4%의 진짜 원인은 GPU가 아니라 **`score.py`가 구버전 GT(v1)를 기본값으로 쓰던 버그**였음(`GT_PATH`를 v2로 수정함). Phase 15에서 main에 재도입 확정 (2026-06-25)
- [x] ~~`score.py` GT 버그~~ → `GT_PATH` 기본값이 `data/ground_truth.csv`(v1, 폐기된 구버전)로 남아있어서 `run_test.sh` 자동채점이 매번 잘못된 GT로 계산되고 있었음. `ground_truth_v2.csv`로 수정 (2026-06-25)
- [x] ~~`fix/per-class-conf` (bulls_eye conf=0.2)~~ → Phase 14에서 효과 없음 확인, 브랜치 삭제 (2026-06-25)
- [x] ~~`fix/pepperidge-milano-confirm`~~ → Phase 13에서 order F1 악화 확인, 브랜치 삭제 (2026-06-25)
- [x] ~~`fix/frappuccino-init`/`fix/ghost-event-cooldown`~~ → confirm=200 회귀만 남아있고 건질 게 없어 Phase 15에서 브랜치 삭제, 진단도구 버그수정만 main에 반영 (2026-06-25)
- [x] ~~`dove_white` 중복 발화~~ → quorum=2로 절충, 순오류 4건→2건 감소 (2026-06-23)
- [x] ~~정확도 검증~~ → `data/ground_truth_v2.csv` + `tools/score_methods.py`(3종 방식) + 리더보드로 완료 (2026-06-23)
- [x] ~~`redbull`/`crystal_hot_sauce`/`dr_pepper` 완전누락~~ → quorum=1 추가로 해결, F1 90.0%→91.5% (2026-06-23)
- [x] ~~`spam` 완전누락~~ → quorum=2 추가로 해결, F1 91.5%→92.0% (2026-06-24)

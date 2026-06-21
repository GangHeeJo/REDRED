# REDRED 진행 현황

**팀원:** 조강희 / 정현수 / 박준영  
**서버:** `ssh aicompetition30@147.46.121.38` (학교 내부망/VPN 필요)

---

## 현재 상태 (2026-06-22)

파이프라인 정상 동작 중.

| 항목 | 값 |
|------|-----|
| RTF | 0.742 (목표 < 1.0, 통과) |
| 처리 시간 | 177.2s (영상 길이 239.0s) |
| 감지 이벤트 수 | 112 (파라미터 튜닝 후) |
| 모델 mAP@0.5 | 98.1% (제공 가중치 `yolov7_custom.pt` 사용 중) |
| 제출 파일 | `~/REDRED/output/submission_skip2.csv` |

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

### Phase 1 — 초기 파이프라인 구축 (chickgoose)
- 5카메라 → YOLOv7 → 멀티뷰 퓨전 → 이벤트 감지 → CSV 기본 뼈대 완성
- `torch.load` 직접 사용으로 `attempt_download` 버그 우회 (경로 소문자 변환 문제)
- PyTorch 1.12 호환성 패치: `Upsample.recompute_scale_factor = None`
- `run_test.sh` 추가, PYTHONPATH 포함

### Phase 2 — RTF 최적화 (chickgoose)
- `grab()`/`retrieve()` 방식 도입: skip 프레임에서 H.264 디코딩 없이 위치만 이동
- 5카메라 배치 GPU 추론: 단일 forward pass로 RTF 대폭 절감
- RTF 0.751 달성

### Phase 3 — 증강 파이프라인 개선 (chickgoose)
- chickgoose 레포 기준 `augment/cut_paste_aug.py` 여러 차례 개선:
  - `load_seg_images()`: 폴더명 숫자 prefix에서 class_id 파싱 버그 수정
  - 회전 각도 ±15° → ±5°로 축소 (과도한 회전이 실제 데이터와 괴리)
  - 마스크 halo(경계 번짐) 제거
  - `--no_erasing` 플래그 추가 (erasing이 오히려 품질 저하)
- 5,000장 증강 완료 (`~/Dataset/augmented/`)
- 세그멘테이션 소스: `~/Dataset/3.background_substracted_white/` (클래스별 폴더)

### Phase 4 — 파인튜닝 시도 및 미채택 (chickgoose)
- 원본 가중치 기반 30 epoch 추가 학습
- mAP@0.5: 0.9904 (원본) → 0.9840 — 오히려 하락
- 원인 추정: Cut&Paste 증강 분포가 실제 테스트 영상과 다름
- 결론: **원본 가중치(`yolov7_custom.pt`) 유지**

### Phase 5 — GangHeeJo 레포 동기화 및 코드 정리 (강희조)
- 서버 `~/REDRED` remote를 chickgoose → GangHeeJo로 변경
- chickgoose `pipeline/` 코드를 우리 `src/`로 통합
- 주요 병합 내용:
  - `infer_batch()` 배치 추론 로직
  - `grab()`/`retrieve()` 프레임 스킵 최적화
- CSV 포맷 수정: 셀값 `"재고 수량: N개"` → `"N개"`, `"총액: X"` → `"X"`
- 반환 이벤트 총액 처리: 환불이 아닌 재입고로 해석 → 총액 기여 0원

### Phase 6 — EventDetector 파라미터 튜닝 (강희조, 2026-06-22)
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

## 현재 확정 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `WINDOW_SIZE` | 25 | `src/event_detector.py` |
| `MIN_EVENT_GAP` | 90 | `src/event_detector.py` |
| `MAX_DELTA` | 4 | `src/event_detector.py` |
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
- [ ] 파라미터 추가 튜닝 필요 시: `src/event_detector.py`의 `WINDOW_SIZE`, `MIN_EVENT_GAP` 조정
- [ ] 정확도 검증: 영상 직접 보면서 이벤트 수 수동 확인 (학교 네트워크 필요)

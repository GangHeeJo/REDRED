# CLAUDE.md

딥러닝 기반 무인판매대 상품인식 시스템 경진대회 파이프라인.
5개 카메라 영상에서 YOLOv7로 60개 상품 클래스 인식 → 구매/반환 이벤트 감지 → submission.csv 생성.

**팀명:** REDRED | **팀원:** 조강희/정현수/박준영 (서울대 전기정보공학부) | **GitHub:** GangHeeJo/REDRED

## 폴더 구조

```
REDRED/
├── src/           # 파이프라인 핵심 모듈
│   ├── run_pipeline.py      # 메인 실행 파일
│   ├── event_detector.py    # 구매/반환 이벤트 감지
│   ├── multi_view_fusion.py # 5카메라 weighted median 융합
│   ├── csv_generator.py     # submission.csv 생성
│   └── tracker.py           # SORT 기반 다중카메라 트래커
├── tools/         # 학습 데이터 준비 + 정확도 진단 유틸리티
│   ├── make_split.py        # train/val 90:10 분할
│   ├── make_yaml.py         # custom.yaml 자동 생성
│   ├── cut_paste_aug.py     # 증강 (flip/perspective/erasing/blur/cut&paste)
│   ├── measure_rtf.py       # RTF 측정 (warmup 포함)
│   ├── score.py               # 채점 + 리더보드(output/leaderboard.html) 갱신
│   ├── score_methods.py       # count/order(LCS)/time(지연보정) 3종 동시 채점 — 권장
│   ├── compare_to_ground_truth.py / diagnose_missing_events.py / analyze_detections.py  # 세부 원인 분류
│   ├── replay_event_detector.py  # debug_log를 서버 없이 로컬에서 재생/디버깅
│   ├── check_training_class_counts.py  # 클래스별 학습 이미지 개수
│   └── probe_low_confidence.py   # 낮은 conf로 재추론, 진짜 미검출 vs threshold 문제 구분
├── data/          # 정적 데이터
│   ├── names.txt            # 60개 클래스명
│   ├── prices.csv           # 공식 가격표 (USD+KRW)
│   ├── ground_truth.csv     # v1, 18개 이벤트 frame 공백 — 더 이상 사용 안 함
│   └── ground_truth_v2.csv  # **현재 기준 GT**. 105개 전부 시간 포함, 강희조가 재검수 완료
└── output/        # 생성 파일 (git 제외, 단 분석용 csv는 -f로 강제 추가하기도 함)
    └── submission_skip2.csv # 최종 제출 파일
```

## 파이프라인 실행 (서버에서, 프로젝트 루트 기준)

```bash
python src/run_pipeline.py \
    --videos /path/cam0.mp4 cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 \
    --weights ~/yolov7/runs/train/exp/weights/best.pt \
    --names   data/names.txt \
    --prices  data/prices.csv \
    --out     output/submission.csv \
    --skip 2 \
    --device 0
```

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--skip` | 2 | N프레임마다 처리 (RTF 최적화, 최대 3 권장) |
| `--conf` | 0.4 | 감지 confidence threshold |
| `--use_tracker` | off | SORT 트래커 활성화 플래그 |
| `--init_frames` | 30 | 초기 재고 자동 추정 프레임 수 |
| `--init_inv` | None | 초기 재고 JSON 직접 지정 (`{"0": 5, "1": 3, ...}`) |

## 학습 데이터 준비

```bash
python tools/make_split.py   # train/val 분할
python tools/make_yaml.py    # custom.yaml 생성
python tools/cut_paste_aug.py  # 증강
python tools/measure_rtf.py  # RTF 측정
```

## 서버 정보

- **접속:** `ssh aicompetition30@147.46.121.38` (학교 내부망/VPN 필요)
- **데이터셋:** `/home/aicompetition/Dataset/`
  - `1.competition_trainset` — 학습 이미지+라벨 (~20만 개)
  - `2.backsub_images_100` — 배경 제거 이미지
  - `3.Background_Images` — 배경 이미지
- **학습 확인:** `ls ~/yolov7/runs/train/` → `cat results.txt`
- **학습 실행 시 반드시 tmux + nohup 사용:**
  ```bash
  nohup python train.py ... > train.log 2>&1 &
  tail -f train.log
  ```

## 평가 기준

- 재고 인식 정확도 CSV 제출 (40점)
- RTF 처리 속도 (20점) — RTF = 처리시간 / 영상길이, 낮을수록 유리
- 정성 평가 + 발표 (40점)

## 제출 CSV 포맷

`품목명, 이벤트번호, 구매/반환여부, 재고수량, 총액` — `data/prices.csv` 기준 (USD+KRW 둘 다 포함).

## 정확도 평가 (2026-06-23 추가/갱신)

단일 지표만 보면 오판하기 쉬움(클래스별 개수만 맞으면 순서/시각이 틀려도 맞다고 카운트되거나, 반대로 전체 시퀀스 정렬 하나가 다른 클래스까지 깨뜨림) — **3가지 방식을 같이 보는 `score_methods.py` 사용 권장**:

```bash
python tools/score_methods.py \
    --gt data/ground_truth_v2.csv \
    --sub output/submission_skip2.csv \
    --timed output/submission_skip2_timed.csv   # run_pipeline.py --timed_log로 생성, 없으면 방법3 생략됨

# 리더보드(웹) 갱신은 별도로:
python tools/score.py --gt data/ground_truth_v2.csv --rtf 0.742 --desc "변경 내용"
```

현재 count=90.0% / order=83.3% / time=82.5% F1, 리더보드 추정 총점 51.1/60점. 자세한 버그 수정 히스토리는 `PROGRESS.md`의 "Phase 7"/"Phase 8" 참고.

## 주의사항

- `event_detector.py`: STABLE/CANDIDATE 2-state 머신(UNKNOWN 상태는 버그로 제거됨), sliding window median(25) + MAX_DELTA(4) + CONFIRM_FRAMES(30, ~2초). `MIN_EVENT_GAP`/`INIT_CONFIRM`은 더 이상 존재하지 않음.
- `multi_view_fusion.py`: 기본은 5캠 weighted median(과반 동의 필요), 단 `CLASS_QUORUM_OVERRIDE={2:1, 53:1, 54:2}`(bumblebee_albacore/dove_pink/dove_white)는 더 적은 카메라 동의로도 인정 — 이 3개는 물리적으로 카메라 과반이 동시에 못 보는 위치라 median이 구조적으로 0이 됨. dove_white는 quorum=1(max)에서 노이즈로 중복발화가 나서 quorum=2로 절충함.
- `prices.csv`는 PDF 43페이지 공식 가격표와 동일 확인됨
- Downloads 폴더의 `.py` 파일들은 구버전 — 이 폴더 src/ 기준으로 작업

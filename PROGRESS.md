# REDRED 진행 현황

**팀원:** 조강희 / 정현수 / 박준영  
**서버:** `ssh aicompetition30@147.46.121.38` (학교 내부망/VPN 필요)

---

## 현재 상태 (2026-06-21)

파이프라인 정상 동작 중. 서버에서 `bash run_test.sh 2`로 즉시 실행 가능.

| 항목 | 값 |
|------|-----|
| RTF | 0.744 (목표 < 1.0, 통과) |
| 감지 이벤트 수 | ~246 (파라미터 튜닝 후 재측정 예정) |
| 모델 mAP@0.5 | 98.1% (제공 가중치 `yolov7_custom.pt`) |

---

## 서버에서 실행하는 법

```bash
# 서버 접속
ssh aicompetition30@147.46.121.38

# 최신 코드 받기
cd ~/REDRED
git pull

# 파이프라인 실행 (skip=2 권장)
bash run_test.sh 2

# 결과 확인
cat output/submission_skip2.csv | head -20
```

---

## 최근 변경 사항

### src/run_pipeline.py
- `load_model()`: `torch.load` 직접 사용 (yolov7 `attempt_download` 버그 우회)
- `infer_batch()`: 5카메라 GPU 배치 추론 (chickgoose 코드 머지)
- `grab()`/`retrieve()` 방식으로 스킵 프레임 최적화 (불필요한 디코딩 제거)
- `events_to_csv()`: `include_action=True`, `total_mode="per_class"` 적용

### src/event_detector.py
- `WINDOW_SIZE`: 7 → 9 (안정적인 중앙값 판단)
- `MIN_EVENT_GAP`: 10 → 30 (skip=2 기준 실제 60프레임 ≈ 2초, 오탐 억제)

### src/csv_generator.py
- 셀 포맷 수정: `"재고 수량: N개"` → `"N개"`, `"총액: X"` → `"X"`
- 반환 이벤트는 총액에 기여하지 않음 (재입고로 처리)

### data/names.txt / data/prices.csv
- `pop_tararts_strawberry`, `nature_vally_fruit_and_nut` — 대회 공식 오탈자, 수정하지 말 것

---

## 서버 환경

```
~/REDRED/          ← 이 레포 (GangHeeJo/REDRED)
~/yolov7/          ← YOLOv7 코드
~/Dataset/
  ├── yolov7_custom.pt          ← 제공 가중치 (mAP 98.1%)
  ├── 1.competition_trainset/   ← 학습 이미지+라벨
  └── 4.TestVideo_Sample/
        ├── cam0/Sample_1.mp4
        ├── cam1/Sample_1.mp4
        ...
```

---

## 앞으로 할 일

- [ ] 파라미터 튜닝 후 이벤트 수 재확인 (현재 서버에서 실행 중)
- [ ] 테스트 영상 직접 보면서 오탐/미탐 분석 (학교 네트워크 필요)
- [ ] 필요 시 confidence threshold (`--conf`) 조정
- [ ] 발표 자료 준비

REDRED — 2026 딥러닝 기반 무인판매대 상품인식 인공지능 경진대회
Final_submission README
================================================================

팀명: REDRED
팀원: 박준영 / 강희조 / 정현수 (서울대학교 전기·정보공학부)

이 팀은 두 개의 서로 다른 탐지기 아키텍처로 각각 독립적으로
Count/Order/Time F1 100%를 달성했습니다.

  - rfdetr_sam2/  : RF-DETR + SAM2 파이프라인  ← 메인 제출 모델
  - yolov7/       : YOLOv7 파이프라인          ← 비교/검증 모델

두 모델 모두 이 폴더 안에 실행 가능한 상태로 포함되어 있으며,
테스트셋 결과 파일도 두 모델 각각의 결과를 함께 제출합니다.

----------------------------------------------------------------
0. 폴더 구조
----------------------------------------------------------------
Final_submission/
├── README.txt                (이 파일)
├── rfdetr_sam2/               ← 메인 제출 모델
│   ├── src/rfdetr_native_pipeline.py   (메인 파이프라인)
│   ├── src/infer_rfdetr.py             (RF-DETR 추론 래퍼)
│   ├── src/rfdetr_margin_infer.py      (top-2 class margin 옵션용)
│   ├── config/rfdetr_native_class_config_v2_reinforced.json
│   │                                    (클래스별 튜닝 설정, 코드 수정 없이 조정 가능)
│   ├── tools/score.py, score_methods.py (채점 스크립트)
│   ├── tools/yolo_to_coco.py            (학습 데이터 COCO 변환 — 재현용)
│   ├── tools/sam2_video_label.py        (SAM2 기반 데이터 보강 — 재현용)
│   ├── data/names.txt, prices.csv, ground_truth_v2.csv
│   ├── run_test_rfdetr_native.sh        (실행 스크립트)
│   ├── requirements.txt
│   └── setup_rfdetr_env.sh              (conda 환경 구축 스크립트)
├── yolov7/                    ← 비교/검증 모델
│   ├── src/run_pipeline.py, event_detector.py, multi_view_fusion.py,
│   │       tracker.py, csv_generator.py
│   ├── tools/score.py, score_methods.py
│   ├── data/names.txt, prices.csv, ground_truth_v2.csv
│   ├── run_test.sh              (실행 스크립트)
│   └── requirements.txt
└── weights_file/
    ├── checkpoint_best_total.pth   (RF-DETR — 팀이 SAM2로 보강한 데이터로 자체 학습)
    └── yolov7_custom.pt            (YOLOv7 — 대회 제공 가중치, 원본 그대로 사용)

----------------------------------------------------------------
1. Conda 환경 설명
----------------------------------------------------------------
두 파이프라인은 서로 다른 conda 환경을 사용합니다 (라이브러리 버전 충돌 방지).

  RF-DETR+SAM2 (메인):
    conda create -n rfdetr python=3.10
    conda activate rfdetr
    bash rfdetr_sam2/setup_rfdetr_env.sh
    # 또는: pip install -r rfdetr_sam2/requirements.txt

  YOLOv7:
    conda activate ~/envs/yolov7      (또는 동등한 YOLOv7 환경)
    pip install -r yolov7/requirements.txt
    # PYTHONPATH에 yolov7 리포지토리 경로 포함 필요 (아래 3번 참고)

----------------------------------------------------------------
2. 실행 방법
----------------------------------------------------------------

[RF-DETR+SAM2 — 메인 모델, Darknet/YOLOv7이 아닌 네트워크]
  cd rfdetr_sam2
  conda activate rfdetr
  bash run_test_rfdetr_native.sh 3 0.35 ../weights_file/checkpoint_best_total.pth \
       noisy_or 0 config/rfdetr_native_class_config_v2_reinforced.json

  인자 순서: [skip] [conf] [weights] [fusion_mode] [use_margin] [class_config]
  ※ 가중치 경로를 반드시 ../weights_file/checkpoint_best_total.pth로 지정할 것
    (스크립트 기본값은 원본 repo 구조 기준 상대경로라 이 폴더에서는 다름)
  ※ --videos 인자는 대회 제공 테스트 영상 경로(~/Dataset/4.TestVideo_Sample/...)를
    스크립트 내부에서 참조함. 다른 환경에서 실행 시 run_test_rfdetr_native.sh
    상단의 CAM_DIR 변수를 실제 영상 경로로 수정할 것.

[YOLOv7 — 비교 모델]
  cd yolov7
  conda activate ~/envs/yolov7   (환경명은 서버 설정에 따라 다를 수 있음)
  PYTHONPATH=~/yolov7 bash run_test.sh 2

  ※ run_test.sh는 가중치를 ~/Dataset/yolov7_custom.pt(대회 제공 경로)에서
    직접 읽음 — 이 폴더의 weights_file/yolov7_custom.pt와 동일한 파일이며,
    참고/백업용으로 함께 포함함. 대회 서버가 아닌 다른 환경에서 실행 시
    run_test.sh 상단의 WEIGHTS 변수를 ../weights_file/yolov7_custom.pt로 수정할 것.

----------------------------------------------------------------
3. 결과 저장 경로
----------------------------------------------------------------
두 스크립트 모두 각자 폴더 안의 output/ 디렉토리에 결과를 저장합니다
(스크립트 최초 실행 시 자동 생성됨).

  rfdetr_sam2/output/submission_native_skip3_conf0.35_noisy_or_nomargin_
              rfdetr_native_class_config_v2_reinforced_checkpoint_best_total.csv
  rfdetr_sam2/output/candidate_native.csv, timed_native.csv, debug_frame_counts_native.csv
              (진단/디버그 로그)

  yolov7/output/submission_skip2.csv
  yolov7/output/debug_frame_counts.csv, sub_events_timed.csv (진단/디버그 로그)

각 스크립트는 실행 후 자동으로 tools/score.py, tools/score_methods.py를 호출해
정확도(Count/Order/Time F1)와 RTF를 콘솔에 출력합니다.

----------------------------------------------------------------
4. 두 모델의 실측 성능 (2026-07-10 서버 재검증 완료)
----------------------------------------------------------------
              RF-DETR+SAM2 (메인)    YOLOv7 (비교모델)
  Count F1         100%                  100%
  Order F1         100%                  100%
  Time F1          100%                  100%
  RTF             0.6856                0.763
  mAP          0.9045 (@0.5:0.95)     98.1% (@0.5)

RF-DETR+SAM2는 팀이 SAM2로 대회 테스트 영상에서 직접 도메인 데이터를
추출·보강해 재학습한 자체 가중치를 사용하며, YOLOv7 대비 더 단순한
파이프라인 구조(Noisy-OR 확률융합 1개 수식)로 동일한 정확도와 더 나은
처리속도를 달성했습니다. 자세한 방법론은 발표자료_slides.html /
REDRED_1.pdf 및 최종보고서를 참고해 주세요.

----------------------------------------------------------------
5. 발표 영상
----------------------------------------------------------------
(선택 사항 — 별도 첨부 시 여기 파일명 기재)

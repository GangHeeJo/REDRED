# 무인판매대 비전 연구 논문 분석 보고서
> 작성일: 2026-06-30  
> 분석 대상: 서울대 CAPP 연구실 (이혁재 교수) 논문 시리즈  
> 목적: 무인판매대 비전 시스템 연구에 실질적으로 적용 가능한 기술 정리  
> 이 문서는 다른 Claude Code 세션에 컨텍스트로 그대로 전달 가능하도록 작성됨

---

## 배경 및 연구 맥락

모든 논문은 **서울대학교 CAPP(Computer Architecture and Parallel Processing) 연구실** (이혁재 교수, 이태호, 위두랑가 등)에서 발표한 시리즈 논문이다. 2019년부터 2025년까지 무인판매대/스마트 냉장고 시스템을 단계적으로 구축한 연구 로드맵이 존재한다.

### 연구 로드맵 (연대순)
```
2019 ─ [Paper A] ASMO: 가려진 물체 합성 학습 데이터 자동 생성
2019 ─ [Paper B] 객체 추적으로 재고 입출 판단 (YOLOv3 기반)
2019 ─ [Paper C] 클릭 동작 인식 (CNN 기반, 키오스크 인터랙션)
2020 ─ [Paper D] Hand Occlusion: 손 가려짐 포함 학습 데이터 생성
2020 ─ [Paper E] 손가락 위치 탐색 (YOLOv3 + OpenPose)
2021 ─ [Paper F] ROI Method: 변동 영역만 탐지로 정확도 향상
2022 ─ [Paper G] GP-GAN + Content Loss: 합성 이미지 품질 향상
2023 ─ [Paper H] 멀티카메라 + USF Transform: 노이즈/FP 95% 제거 ← 핵심
2025 ─ [Paper I] LLM(SASRec + LLaMA) 기반 순차 추천 시스템
```

---

## Paper A: 가려진 물체의 합성을 이용한 학습 데이터 생성 방법 (ASMO)
**이주한, 이태호, 김태현, 김진성, 이혁재 | 2019년 대한전자공학회 추계학술대회**

### 핵심 문제
상품이 서로 겹쳐있을 때 CNN 인식률이 급격히 저하됨. 겹쳐진 상황에 대한 학습 데이터를 수동으로 만들기에는 경우의 수가 너무 많음 (상품 수 증가 → 경우의 수 폭발적 증가).

### 제안 방법: ASMO (Automatic Synthetic Merged Object)
1. 단일 상품을 다양한 각도에서 촬영
2. 배경 분할(segmentation)로 상품 영역만 추출 + 자동 라벨 생성
3. 가려진 물체 이미지를 자동 합성:
   - **배치 형식 다양화**: 육각형 배열, 바둑판 배열, 직선 배열
   - 가로/세로 간격 다양화, 크기 표준편차 다양화
4. 다양한 배경 합성 → 최종 학습 데이터

### 실험 결과
- YOLO v3 사용
- 육각형 배열만으로 학습 → 해당 배치에서만 인식, 다른 배열에서 실패
- 바둑판 + 직선 배열 추가 학습 → 일반화 성능 크게 향상
- **핵심 발견: 배치 형식 다양성이 일반화에 결정적**

### 무인판매대 적용 포인트
- **학습 데이터 부족 문제 즉각 해결**: 상품 사진 몇 장만으로 대량 합성 데이터 생성 가능
- **주의사항**: 한 가지 배치 형식만 학습하면 실제 환경에서 무너짐 → 반드시 다양한 배치로 학습
- 최신 발전: **Copy-Paste Augmentation** (Ghiasi et al., CVPR 2021) — 같은 원리의 더 정교한 버전
- 구현 참고: `albumentations` 라이브러리의 `CopyPaste` 증강

### GitHub
- 공개 코드 없음 (학회 단발 논문)
- 유사 구현: [Copy-Paste Augmentation](https://github.com/conradry/copy-paste-aug)

---

## Paper B: 인공지능 기반 재고관리를 위한 객체 추적 알고리즘 구현
**김태현, 이태호, 김진성, 이혁재 | 2019년 대한전자공학회 추계학술대회**

### 핵심 문제
상품을 인식하는 것만으로는 부족 — 상품이 냉장고/판매대 **안으로 들어갔는지, 나갔는지** 판단해야 함.

### 제안 방법
```
연속 프레임에서 YOLOv3로 객체 인식
  → 연속 프레임 간 객체 위치 변화 추적
  → 냉장고/판매대 선반 끝을 기준선(중앙 파랑선)으로 설정
  → 객체가 기준선 완전 통과 → 입고/출고 이벤트 발생
  → 재고 카운트 업데이트
```

**위치 변화 추적 로직:**
- 연속 프레임에서 인식된 객체의 위치 변화가 일정 수준 이하 → 같은 물체로 판단
- 물체의 이동 경로(궤적) 저장
- 객체가 완전히 판매대에 들어가거나 빠져나간 후 일정 기간 미인식 → 첫/마지막 좌표 + 기준선으로 입출 판단

### 카메라 설치 위치 (중요)
- **냉장고/판매대 측면에 광각 카메라** 설치
- 정면이 아닌 측면 → 상품 이동 방향이 명확히 보임

### 실험 결과
| 상품 | AP (%) | Update 정확도 (%) |
|---|---|---|
| Apple | 90.86 | 100 |
| Avocado | 98.79 | 95 |
| PaprikaYellow | 96.01 | 87.5 |
| Pocari | 93.10 | 97.5 |
| PaprikaRed | 92.41 | 70 |
| Coke | 87.46 | 92.5 |
| Sprite | 83.48 | 95 |
| Greenapple | 86.48 | 82.5 |
| Mustard | 87.26 | 97.5 |
| Ketchup | 66.15 | 97.5 |
| **Average** | **88.20** | **91.5** |

- 인식률(AP) 88.2% → 재고 판단 정확도 91.5% (**추적으로 인식 오류를 보완**)
- 각 상품당 20회 반복 입출 실험

### 핵심 통찰
- **인식이 완벽하지 않아도 궤적으로 보완 가능** — 중요한 설계 원칙
- Latency 문제 명시: CNN latency가 크면 빠른 이동 포착 불가 → 프레임레이트가 중요

### 무인판매대 적용 포인트
- **기준선(threshold line)** 개념 → 무인판매대 출입구/선반 끝에 그대로 적용
- 더 정확한 최신 tracker 사용 권장:
  - **ByteTrack**: [github.com/ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)
  - **DeepSORT**: [github.com/nwojke/deep_sort](https://github.com/nwojke/deep_sort)
  - **BoT-SORT**: ByteTrack 후속, 더 정확

---

## Paper C: CNN 기반의 배경 변화에 강인한 클릭 동작 인식
**이태호, 김태현, 김진성, 이혁재 | 2019년 대한전자공학회 추계학술대회**

### 핵심 문제
무인판매대 키오스크에서 화면 터치 없이 손동작으로 입력하는 방법 필요.

### 제안 방법
**가상 키보드 클릭 3단계:**
```
준비 동작: 검지 + 엄지 편 상태 (글쇠를 가리킴)
  ↓
클릭 동작: 검지 + 엄지가 닿는 순간 (입력 발생)
  ↓
완료 동작: 다시 펼침 (입력 완료)
```

**학습 데이터 생성:**
- 배경 3종: Basic bg(단순), Skin bg(피부색), Complex bg(복잡)
- 3만 5천장 학습 데이터
- YOLOv3로 탐지

### 실험 결과
| 배경 | TP | FP | FN | precision | recall | mAP |
|---|---|---|---|---|---|---|
| Basic bg | 1355 | 4 | 4 | 1.0 | 1.0 | 98.69 |
| Skin bg | 3277 | 8 | 8 | 1.0 | 1.0 | 98.70 |
| **Complex bg** | **3267** | **49** | **18** | **0.99** | **0.99** | **98.59** |

복잡한 배경에서도 **98.59% mAP** — 매우 높은 성능.

### 무인판매대 적용 포인트
- **터치리스 키오스크 UI** 구현 가능 → 위생 + 편의
- YOLO 기반이라 상품 인식 파이프라인과 통합 용이
- **단점**: 클릭 이벤트만 인식, 좌표는 Paper E에서 추가

---

## Paper D: CNN 기반의 스마트 냉장고 재고 관리를 위한 움직이는 손안의 상품 인식
**강신우, 이태호, 김진성, 이혁재 | 2020년 대한전자공학회 추계학술대회**

### 핵심 문제
고객이 상품을 손으로 집을 때 손이 상품을 가림 → 인식률 급락. 또한 냉장고 내외부 배경 다양성 문제.

### 제안 방법: Hand Occlusion 자동 학습 데이터 생성
```
상품 단일 촬영
  → 배경 제거 (segmentation)
  → 다양한 배경 합성 (냉장고 내부 + 외부)
  → 손 가려짐 효과: 손 모양/색깔 비슷한 직사각형 박스 합성
  → 자동 labeling 데이터 완성
```

**핵심**: 실제 손을 찍지 않고 직사각형 박스로 근사해도 성능 향상.

### 실험 결과 (15개 상품, YOLOv3)
| 학습 방식 | mAP |
|---|---|
| 가려짐 없는 상품만 | 25.61% |
| 수작업 hand labeling | 54.93% |
| **제안 방법 (자동 생성)** | **65.13%** |

주요 클래스별:
- pringles: 40.56 → 89.35% (+48.79)
- shinramyun: 54.34 → 85.42% (+31.08)
- coke_bottle: 4.54 → 71.64% (+67.1)
- shampoo: 1.25 → 27.51% (효과 제한적 — 배경 다양성 부족 시 역효과)

### 무인판매대 적용 포인트
- **지금 당장 구현 가능**: 상품 사진 몇 장 + 배경 이미지 + 직사각형 occlusion 합성
- 더 발전: 직사각형 대신 **실제 손 segmentation** 이미지 사용 → 더 정확
- 관련 도구: `albumentations`, SAM (Segment Anything Model) for background removal

---

## Paper E: AR/VR 응용을 위한 CNN 기반의 배경 변화에 강인한 검지 손가락 위치 탐색
**이염미, 이태호, 김진성, 이혁재 | 2020년 대한전자공학회 추계학술대회**

### 핵심 문제
클릭 이벤트 인식(Paper C)으로는 부족 — 손가락이 가리키는 정확한 **좌표**도 필요.

### 제안 방법: YOLOv3 + OpenPose 2단계
```
RGB 이미지 입력
  → [Step 1] YOLOv3: 손 영역 Bounding Box 탐지 + 동작 분류 (준비/클릭)
  → [Step 2] OpenPose: 손 21개 관절의 2D 좌표 추정
  → 검지 손가락 좌표 출력
```

### 실험 결과
| 배경 | 인식률 | YOLOv3 FPS | OpenPose FPS |
|---|---|---|---|
| Skin color bg | 91.08% | 19.32 | 5.92 |
| Complex bg | 95.89% | 19.48 | 6.07 |
| **Simple bg** | **97.55%** | **22.86** | **4.24** |

### 특이점 (중요)
- **OpenPose가 병목**: YOLOv3 ~20fps vs OpenPose ~6fps → 전체 시스템 6fps로 제한
- 2020년 기준이라 현재는 더 빠른 대안 존재

### 무인판매대 적용 포인트
- Paper C + Paper E = **완전한 터치리스 키오스크 입력 시스템**
- **현재 권장 대안**: OpenPose 대신 **MediaPipe Hands** 사용
  - 30fps 이상 가능, CPU에서도 동작, 모바일 지원
  - `pip install mediapipe`
  - GitHub: [google/mediapipe](https://github.com/google/mediapipe)

---

## Paper F: 변동 재고 추적 방식을 이용한 CNN 기반의 스마트 냉장고 상품 인식
**위두랑가, 이태호, 김태현, 이혁재 | 2021년 대한전자공학회 하계학술대회**

### 핵심 문제
전체 선반을 매번 탐지하면 겹쳐진 상품들 사이에서 인식률 저하. "변한 것만 인식"하면 겹침 문제 우회 가능.

### 제안 방법: ROI Method
```
문 열림 감지 (Door Open 이벤트)
  → 이벤트 전 이미지 저장 (Before)
  → 상품 이동/추가/제거 발생
  → 이벤트 후 이미지 저장 (After)
  → cv2.absdiff(Before, After) → 차영상 계산
  → 차영상 이진화 → 변동 영역 ROI 생성
  → ROI 영역 내에서만 YOLO 탐지
  → 해당 상품 인식 → 재고 업데이트
```

### 실험 결과 (14개 상품 클래스, YOLOv3)
| 방법 | mAP |
|---|---|
| ASMO (전체 선반 탐지) | 83.21% |
| **ROI Method (변동 영역만 탐지)** | **88.43%** |

클래스별 특이사항:
- Aunt jemima: 0.75 → 1.00 (겹침 심한 상품에서 효과 극대화)
- Baking soda: 0.65 → 0.82
- Crystal hotsauce: 0.84 → 0.64 (오히려 낮아짐 — ROI가 너무 작게 잡힌 경우)

### 구현 방법 (코드 레벨)
```python
import cv2
import numpy as np

def get_roi(before_img, after_img, threshold=30):
    diff = cv2.absdiff(before_img, after_img)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # morphological operations for noise removal
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=2)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # bounding box of all changed regions
    x_min = min(cv2.boundingRect(c)[0] for c in contours)
    y_min = min(cv2.boundingRect(c)[1] for c in contours)
    x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
    y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)
    
    roi = after_img[y_min:y_max, x_min:x_max]
    return roi, (x_min, y_min, x_max, y_max)

# 사용 예시
before = cv2.imread('before_event.jpg')
after = cv2.imread('after_event.jpg')
roi, bbox = get_roi(before, after)

# YOLO로 ROI 영역만 탐지
results = yolo_model(roi)
```

### 무인판매대 적용 포인트
- **구현 난이도 낮음** + **효과 즉각적** → 가장 먼저 적용할 기술
- 트리거 조건 필요: 문 열림/닫힘 센서 OR 모션 감지
- 주의: Crystal hotsauce처럼 ROI가 너무 작으면 역효과 → ROI에 패딩 추가 권장
- **무인판매대에서 직접 사용**: 고객이 상품 집기 전/후 이미지 비교

---

## Paper G: 실감나는 이미지 블렌딩을 위한 GP-GAN과 콘텐츠 손실의 조합
**Truong Thanh Hien, 이태호, 위두랑가, 이혁재 | 2022년 대한전자공학회 하계학술대회**

### 핵심 문제
ASMO 등 합성 데이터 생성 시 상품을 배경에 합성하면 경계가 부자연스럽고 색상이 바뀌는 문제.

### 제안 방법: Combined GP-GAN
```
Source 이미지 (상품) + Target 배경 + Binary Mask
  → [Blending GAN] 자연스러운 경계 생성 (Gaussian-Poisson 방정식)
  → [Content Loss] VGG-16으로 특징 추출 → 색상 보존 강제
  → 결과: 경계 자연스럽고 색상도 원본과 유사하게 유지
```

**Content Loss 수식:**
```
L_cont = Σ (α_l / 2N_l M_l) * Σ_i Σ_k (F_l[x̃_l] ⊙ x_m - F_l[x¹] ⊙ x_m)²_ik
```
- F_l: VGG-16의 l번째 레이어 activation
- x_m: binary mask
- x̃_l: Blending GAN 출력, x¹: 입력 이미지

### 실험 결과
| 방법 | PSNR | User Votes (40명) |
|---|---|---|
| GP-GAN | 16.96 dB | 452 (36%) |
| **Combined GP-GAN (제안)** | **24.21 dB** | **808 (64%)** |

- PSNR 7.25dB 향상 (매우 큰 폭)
- 데이터셋: GS-Home Shopping database (416개 모델 사진)

### 무인판매대 적용 포인트
- **합성 학습 데이터 품질 향상**: ASMO/Hand Occlusion에서 배경 합성 시 이 방법 적용 → 더 자연스러운 데이터
- GS-Home Shopping 데이터(상품 사진)에서 검증 → 무인판매대 상품과 도메인 유사
- **처리 시간**: 512×512 이미지 1장당 약 50초 (NVIDIA RTX 2080) → 실시간 불가, 전처리용
- GitHub 원본 GP-GAN: [github.com/wuhuikai/GP-GAN](https://github.com/wuhuikai/GP-GAN)

---

## Paper H: 멀티카메라를 활용한 딥러닝 기반 무인판매대를 위한 Noise 및 False Positive 이벤트 감소 파이프라인
**위두랑가, 이태호, 이혁재, 김진성 | 2023년 대한전자공학회 하계학술대회** ← **가장 중요한 논문**

### 핵심 문제
1. 상품이 겹쳐있으면 단일 카메라로는 인식률 저하
2. 딥러닝 오인식으로 인한 **False Positive Event** 다수 발생
   - 예: 구매하지 않은 상품이 구매된 것으로 처리
   - 예: 반품하지 않은 상품이 반품 처리됨

### 제안 방법: 4단계 파이프라인
```
[카메라 5대 설치]
    ↓
[Step 1] 각 카메라에서 CNN(YOLO) 기반 상품 탐지
    ↓
[Step 2] 동일 시간대 5개 카메라 결과 Merge
    ↓
[Step 3] USF(Ultrafast Shapelet) Transform + Smoothing
         - 탐지 결과를 시계열로 표현
         - Shapelet 추출로 노이즈 패턴 제거
         - Human Recognizable Latency 기반 smoothing
    ↓
[Step 4] 최종 Invoice 생성 (구매/반환 목록)
```

### USF Transform 상세 설명
- **Shapelet**: 시계열 데이터에서 특정 패턴을 잘 나타내는 작은 부분 시계열
- 탐지 결과를 시계열로 보고, 실제 이벤트(상품 집기/넣기)와 노이즈(일시적 오인식)를 패턴으로 구분
- **Human Recognizable Latency**: 사람이 상품을 집는데 걸리는 최소 시간 → 이보다 짧은 이벤트는 노이즈로 제거
- pyts 라이브러리로 구현 가능: `pip install pyts`

### 실험 결과 (핵심)
| Event | 카메라1 | 카메라2 | 카메라3 | 카메라4 | 카메라5 | 전체합산 | **제안방법** |
|---|---|---|---|---|---|---|---|
| Remove(62) TP | 54 | 47 | 49 | 53 | 55 | 60 | **60** |
| Remove(62) FP | 31 | 27 | 28 | 26 | 34 | 27 | **2** |
| Add(43) TP | 40 | 38 | 36 | 38 | 43 | 43 | **43** |
| Add(43) FP | 19 | 19 | 31 | 25 | 14 | 22 | **0** |
| Total(105) TP | 94 | 85 | 85 | 91 | 98 | 103 | **103** |
| Total(105) FP | 50 | 46 | 59 | 51 | 48 | 49 | **2** |

- **FP: 49 → 2 (95.9% 감소)** — 결정적 성능 향상
- TP는 103/105 (98%) 유지
- 환경: Intel i7-8700 @3.2GHz, 32GB RAM, TITAN Xp GPU

### 무인판매대 적용 포인트
1. **멀티카메라 결과 Merge** → 카메라 한 대로 못 보는 각도 보완
2. **USF Transform** → YOLO가 몇 프레임 오인식해도 시계열 필터로 제거
3. **Human Recognizable Latency** → 상품 집는 시간보다 짧은 이벤트는 노이즈로 처리
4. 구현: `pyts` 라이브러리 + 멀티카메라 결과 merge 로직

```python
from pyts.transformation import ShapeletTransform
import numpy as np

# 탐지 결과를 시계열로 변환
# events: 시간에 따른 탐지 결과 시퀀스 (1=탐지, 0=미탐지)
def apply_usf_smoothing(detection_sequence, min_event_duration=0.3):
    """
    detection_sequence: list of (timestamp, detected_class) 
    min_event_duration: Human Recognizable Latency (초)
    """
    # 짧은 이벤트(노이즈) 제거
    filtered = []
    for event_start, event_end, cls in events:
        if (event_end - event_start) >= min_event_duration:
            filtered.append((event_start, event_end, cls))
    return filtered
```

### GitHub
- 공개 코드 없음. 구성 요소별 참고:
  - USF Transform: [github.com/johannfaouzi/pyts](https://github.com/johannfaouzi/pyts)
  - 관련 이상 탐지 논문 [4]: Da et al., "Anomaly detection framework for unmanned vending machines," Knowledge-Based Systems, Vol. 262, 2023 — 별도 탐색 권장

---

## Paper I: 무인판매대 키오스크를 위한 LLM 기반 순차 추천 시스템
**김솔지, 한정민, 위두랑가, 이태호, 이혁재 | 2025년 대한전자공학회 하계학술대회**

### 핵심 문제
무인판매대는 사용자 데이터 부족 + 상품 설명/리뷰 없음 → 기존 온라인 추천 시스템 직접 적용 불가.

### 제안 방법: LLaRA (LLM-empowered Sequential Recommendation with Augmentation)
```
사용자 음료 구매 시퀀스
  → [SASRec] Item Embedding 학습 (Hidden Size 16, 100 Epoch, Adam)
  → Behavioral Token 변환
  → [LLaMA + LoRA] 자연어로 추천 결과 + 추천 이유 생성
```

- Self-Supervised Learning으로 데이터 부족 극복
- Single Transaction(단일 구매 기록)으로도 추천 가능 (Cold Start 해결)
- 데이터: 9,200명 사용자 × 60개 음료 (합성 데이터), 8:1:1 분할

### 실험 결과
| 모델 | HR@1 |
|---|---|
| SASRec 단독 | 0.134 |
| **LLaRA (제안)** | **0.421** (+214%) |
| LastFM (실제 데이터) | 0.508 |
| Steam (실제 데이터) | 0.472 |

합성 데이터임에도 실제 데이터셋 성능과 유사한 수준.

### 무인판매대 적용 포인트
- 비전 인식 결과(구매 상품 목록) → 추천 시스템 입력으로 연결
- **주의**: 합성 데이터로만 검증, 실제 무인판매대 데이터 필요
- LLaRA GitHub: [github.com/ljy0ustc/LLaRA](https://github.com/ljy0ustc/LLaRA)
- SASRec GitHub: [github.com/kang205/SASRec](https://github.com/kang205/SASRec)

---

## 성능 수치 전체 요약

| 논문 | 주요 지표 | 최고 수치 |
|---|---|---|
| Paper A (ASMO) | 인식 성능 향상 확인 | 수치 미기재 |
| Paper B (객체 추적) | 재고 판단 정확도 | **91.5%** |
| Paper C (클릭 인식) | mAP | **98.70%** |
| Paper D (Hand Occlusion) | mAP | **65.13%** |
| Paper E (손가락 위치) | 인식률 | **97.55%** |
| Paper F (ROI Method) | mAP | **88.43%** |
| Paper G (GP-GAN) | PSNR | **24.21 dB** |
| Paper H (멀티카메라+USF) | FP 감소율 | **95.9% (49→2)** |
| Paper I (LLM 추천) | HR@1 | **0.421** |

**주의**: 모든 수치는 연구실 통제 환경에서 측정. 실제 환경에서는 조명 변화, 다양한 사용자, 동시 접근 등으로 낮아질 수 있음.

---

## 내 연구에 적용할 기술 우선순위

### 즉시 적용 가능 (구현 난이도 낮음)
1. **ROI Method (Paper F)**: `cv2.absdiff` + 이진화 + YOLO → 변동 영역만 탐지
2. **ASMO 데이터 생성 (Paper A)**: 상품 사진 + 배경 합성으로 학습 데이터 대량 생성

### 중기 적용 (구현 난이도 중간)
3. **Hand Occlusion 데이터 (Paper D)**: 손 가려짐 효과 추가한 학습 데이터 생성
4. **ByteTrack/BoT-SORT (Paper B 발전)**: 최신 tracker로 재고 입출 판단 정확도 향상

### 장기/선택 적용
5. **멀티카메라 + USF (Paper H)**: 카메라 추가 시 FP 극적으로 감소
6. **GP-GAN 합성 (Paper G)**: 더 자연스러운 학습 데이터 생성
7. **MediaPipe Hands (Paper C+E 발전)**: 터치리스 키오스크 입력

---

## 핵심 참고 문헌 (추가 탐색 권장)

이 시리즈 논문들이 인용하는 외부 논문 중 중요한 것:

1. **Da et al., "Anomaly detection framework for unmanned vending machines," Knowledge-Based Systems, Vol. 262, 2023** — 이상 탐지 관점의 무인판매대 연구, 별도 분석 권장
2. **Cai et al., "Messytable: Instance association in multiple camera views," ECCV 2020** — 멀티카메라 인스턴스 연결
3. **T.H. Lee & H.J. Lee, "A New Virtual Keyboard with Finger Gesture Recognition for AR/VR devices," HCI 2018** — 클릭 인식 원본 논문

---

## 이 문서를 받은 Claude Code에게

이 문서는 조강희 (ganghee1245@gmail.com)의 **무인판매대 경진대회 연구** 프로젝트를 위해 작성됨.
- 메인 프로젝트 폴더: `Desktop\REDRED`
- 구조: `src/tools/data/output`
- GitHub 연동 완료

분석된 모든 논문은 서울대 CAPP 연구실 (이혁재 교수)의 시리즈 연구로, 무인판매대 비전 시스템 전체를 커버함. 연구자는 이 기술들을 자신의 무인판매대 연구에 실질적으로 적용하려 함. 비전 측면에 집중하고 있으며, 상품 인식 정확도 향상이 주요 목표임.

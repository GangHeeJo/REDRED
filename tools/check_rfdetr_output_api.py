"""
RF-DETR model.predict()가 top-1 class+confidence 말고 전체 클래스 확률분포
(top-2 margin 계산용)까지 주는지 확인. 딱 1프레임만 추론해서 반환 객체 구조 출력.

Usage:
    python tools/check_rfdetr_output_api.py --weights runs/rfdetr/checkpoint_best_total.pth
"""
import argparse
import sys
from pathlib import Path

import cv2
import PIL.Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from infer_rfdetr import load_rfdetr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--video", default=None,
                    help="기본값: ~/Dataset/4.TestVideo_Sample/cam0/Sample_1.mp4")
    p.add_argument("--device", default="0")
    args = p.parse_args()

    video = args.video or str(Path.home() / "Dataset/4.TestVideo_Sample/cam0/Sample_1.mp4")
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device

    print("모델 로딩...")
    model = load_rfdetr(args.weights, num_classes=60, device=device)

    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("프레임 읽기 실패")
        return

    pil_img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = model.predict(pil_img, threshold=0.1)

    print("\n=== result 타입 ===")
    print(type(result))
    print("\n=== result의 속성/메서드 목록 ===")
    print([a for a in dir(result) if not a.startswith("_")])

    print("\n=== 주요 필드 값 (있으면) ===")
    for attr in ["class_id", "confidence", "xyxy", "data", "mask", "tracker_id"]:
        if hasattr(result, attr):
            val = getattr(result, attr)
            print(f"\n--- {attr} ---")
            print(f"type: {type(val)}")
            try:
                print(f"len/shape: {len(val) if val is not None else None}")
                if val is not None and len(val) > 0:
                    print(f"첫 항목: {val[0]}")
                    if hasattr(val, "shape"):
                        print(f"shape: {val.shape}")
            except Exception as e:
                print(f"(길이 확인 실패: {e})")

    if hasattr(result, "data") and result.data:
        print("\n=== result.data 딕셔너리 키 (여기 전체 클래스 확률분포가 있을 수도 있음) ===")
        for k, v in result.data.items():
            print(f"  {k}: type={type(v)}")
            try:
                print(f"    shape/len: {v.shape if hasattr(v, 'shape') else len(v)}")
            except Exception:
                pass


if __name__ == "__main__":
    main()

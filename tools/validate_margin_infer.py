"""
rfdetr_margin_infer.py의 raw forward 재구현이 진짜 model.predict()랑 같은 결과를
내는지 검증. margin 값을 실제 파이프라인에 쓰기 전에 반드시 이거부터 통과해야 함.

Usage:
    python tools/validate_margin_infer.py --weights runs/rfdetr/checkpoint_best_total.pth
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from infer_rfdetr import load_rfdetr, infer_rfdetr
from rfdetr_margin_infer import infer_rfdetr_with_margin


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--video", default=None)
    p.add_argument("--conf", type=float, default=0.35)
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

    frames = [frame, None, None, None, None]

    print("\n=== 기존 infer_rfdetr (predict() 그대로) ===")
    normal = infer_rfdetr(model, frames, args.conf, device)[0] or []
    normal_sorted = sorted(normal, key=lambda d: -d["confidence"])
    for d in normal_sorted[:10]:
        print(f"  cls={d['class_id']:2d} conf={d['confidence']:.4f} bbox={[round(x,1) for x in d['bbox']]}")

    print("\n=== 새 rfdetr_margin_infer (raw forward 재구현 + margin) ===")
    margin = infer_rfdetr_with_margin(model, frames, args.conf, device)[0] or []
    margin_sorted = sorted(margin, key=lambda d: -d["confidence"])
    for d in margin_sorted[:10]:
        print(f"  cls={d['class_id']:2d} conf={d['confidence']:.4f} margin={d['margin']:.4f} "
              f"bbox={[round(x,1) for x in d['bbox']]}")

    print(f"\n검출 개수: 기존={len(normal)}  새로운={len(margin)}")

    print("\n=== 일치 여부 확인 (class_id+confidence 기준 정렬 후 비교) ===")
    if len(normal_sorted) != len(margin_sorted):
        print(f"!! 개수가 다름 ({len(normal_sorted)} vs {len(margin_sorted)}) -- 뭔가 안 맞을 가능성")
    mismatches = 0
    for a, b in zip(normal_sorted, margin_sorted):
        if a["class_id"] != b["class_id"] or abs(a["confidence"] - b["confidence"]) > 1e-3:
            mismatches += 1
            print(f"  !! MISMATCH: 기존(cls={a['class_id']},conf={a['confidence']:.4f}) vs "
                  f"새로운(cls={b['class_id']},conf={b['confidence']:.4f})")
    if mismatches == 0:
        print("전부 일치함 -- margin 값 신뢰 가능")
    else:
        print(f"{mismatches}건 불일치 -- margin 구현에 버그 있음, 쓰면 안 됨")


if __name__ == "__main__":
    main()

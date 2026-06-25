"""
Main pipeline: video → detection → fusion → events → CSV

Usage:
    python src/run_pipeline.py \
        --videos cam0.mp4 cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 \
        --weights ~/yolov7/runs/train/exp/weights/best.pt \
        --names   data/names.txt \
        --prices  data/prices.csv \
        --out     output/submission.csv \
        --conf    0.4 \
        --device  0

Initial inventory options (pick one):
    --init_inv init.json        Load from JSON file {"0": 5, "1": 3, ...}
    --init_frames 30            Auto-detect from first N frames (default, n=30)

RTF is logged automatically.
"""

import argparse
import json
import time
import sys
import os
import cv2
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path

# Allow importing sibling modules
sys.path.insert(0, str(Path(__file__).parent))
from event_detector import EventDetector
from multi_view_fusion import fuse
from csv_generator import load_prices, events_to_csv
from tracker import MultiCameraTracker


def load_names(names_path: str):
    with open(names_path) as f:
        return [line.strip() for line in f if line.strip()]


def load_initial_inventory_from_file(path: str) -> dict:
    """Load initial inventory from JSON: {"0": 5, "1": 3, ...}"""
    with open(path) as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def estimate_initial_inventory(caps, model, nms_fn, n_frames, conf, iou, img_size, device) -> dict:
    """
    Run detection on the first n_frames, fuse per-camera counts each frame,
    then take the per-class median. Rewinds all caps to frame 0 when done.
    """
    counts_history: dict = defaultdict(list)
    for _ in range(n_frames):
        frames = read_frames(caps)
        if all(f is None for f in frames):
            break
        per_cam = infer_batch(model, nms_fn, frames, conf, iou, img_size, device)
        fused = fuse(per_cam)
        for cls_id, cnt in fused.items():
            counts_history[cls_id].append(cnt)

    for cap in caps:
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return {
        cls_id: int(np.median(vals))
        for cls_id, vals in counts_history.items()
        if int(np.median(vals)) > 0
    }


def load_model(weights: str, device: str):
    """Load YOLOv7 directly via torch.load (bypasses attempt_download 버그)."""
    yolov7_root = str(Path.home() / "yolov7")
    if yolov7_root not in sys.path:
        sys.path.insert(0, yolov7_root)
    from utils.general import non_max_suppression
    import torch.nn as nn

    ckpt = torch.load(weights, map_location=device)
    model = (ckpt.get("ema") or ckpt["model"]).float().fuse().eval()
    # PyTorch 1.12+ 호환성 패치
    for m in model.modules():
        if isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None
    return model, non_max_suppression


def _preprocess_single(frame, img_size=640):
    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img).float() / 255.0


def infer_batch(model, nms_fn, frames, conf_thres=0.4, iou_thres=0.45,
                img_size=640, device="cpu"):
    """5개 카메라 프레임을 GPU 한 번에 배치 추론."""
    valid_idx = [i for i, f in enumerate(frames) if f is not None]
    if not valid_idx:
        return [None] * len(frames)

    tensors = [_preprocess_single(frames[i], img_size) for i in valid_idx]
    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        preds = model(batch)[0]
    preds = nms_fn(preds, conf_thres, iou_thres)

    per_cam = [None] * len(frames)
    for out_i, cam_i in enumerate(valid_idx):
        pred = preds[out_i]
        dets = []
        if pred is not None and len(pred):
            for *xyxy, conf, cls in pred.cpu().numpy():
                dets.append({
                    "class_id":   int(cls),
                    "confidence": float(conf),
                    "bbox":       [float(v) for v in xyxy],
                })
        per_cam[cam_i] = dets
    return per_cam


def open_videos(video_paths):
    caps = []
    for p in video_paths:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            print(f"Warning: cannot open {p}")
            caps.append(None)
        else:
            caps.append(cap)
    return caps


def grab_frames(caps):
    """Advance all caps by one frame without decoding. Returns alive status per cap."""
    return [cap.grab() if cap is not None else False for cap in caps]


def retrieve_frames(caps, statuses):
    """Decode frames that were grabbed. Only call after grab_frames()."""
    frames = []
    for cap, ok in zip(caps, statuses):
        if cap is None or not ok:
            frames.append(None)
        else:
            ret, frame = cap.retrieve()
            frames.append(frame if ret else None)
    return frames


def read_frames(caps):
    frames = []
    for cap in caps:
        if cap is None:
            frames.append(None)
            continue
        ret, frame = cap.read()
        frames.append(frame if ret else None)
    return frames


def video_duration(video_paths):
    total = 0.0
    for p in video_paths:
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total = max(total, n / fps)
        cap.release()
    return total


# Camera layout
# 0: 왼쪽 앞  1: 오른쪽 앞  2: 위(top)  3: 오른쪽 뒤  4: 왼쪽 뒤
_LEFT_CAMS  = [0, 4]
_RIGHT_CAMS = [1, 3]
_TOP_CAM    = 2


_occlusion_stats = {"total": 0, "right_blocked": 0, "left_blocked": 0}


def compute_cam_weights(per_cam_dets, class_id=None):
    """
    class_id=None: 프레임 전체 평균 confidence로 occlusion 판단(레거시 동작).
    class_id=<int>: 그 클래스의 confidence만 사용 -- 같은 프레임의 무관한
        클래스들이 occlusion 신호를 희석시키는 문제를 피함. 2026-06-25
        whole-frame 버전으로는 haribo_gold_bears_gummi_candy는 구제됐지만
        pepperidge_farm_milano_cookies_double_chocolate는 그대로였음 --
        milano가 occlusion되는 시점에 다른 클래스들이 정상으로 보이면
        평균에 묻혀서 70% 임계값을 못 넘었을 가능성.
    """
    conf = []
    for dets in per_cam_dets:
        if not dets:
            conf.append(0.0)
            continue
        relevant = dets if class_id is None else [d for d in dets if d["class_id"] == class_id]
        conf.append(sum(d["confidence"] for d in relevant) / len(relevant) if relevant else 0.0)

    left_conf  = (conf[0] + conf[4]) / 2
    right_conf = (conf[1] + conf[3]) / 2

    weights = [1.0, 1.0, 1.5, 1.0, 1.0]  # 위 카메라 기본 1.5배

    # 0.5x/1.5x 곱셈으로는 weighted median의 과반 구성 자체가 안 바뀌어서 효과 없음
    # (2026-06-25 server test: camera-weights-v2 결과가 베이스라인과 완전히 동일했음)
    # -> 가려진 쪽은 weight=0으로 완전히 제외해서 median 투표 구성 자체를 바꿈
    _occlusion_stats["total"] += 1
    if right_conf < left_conf * 0.7:    # 오른쪽 손 가림
        weights[1] = 0.0; weights[3] = 0.0
        _occlusion_stats["right_blocked"] += 1
    elif left_conf < right_conf * 0.7:  # 왼쪽 손 가림
        weights[0] = 0.0; weights[4] = 0.0
        _occlusion_stats["left_blocked"] += 1

    return weights


def compute_per_class_cam_weights(per_cam_dets):
    """프레임에 등장한 클래스마다 따로 occlusion weight 계산 (class_id -> weights)."""
    class_ids = set()
    for dets in per_cam_dets:
        if dets:
            class_ids.update(d["class_id"] for d in dets)
    return {cid: compute_cam_weights(per_cam_dets, class_id=cid) for cid in class_ids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos",   nargs="+", required=True)
    parser.add_argument("--weights",  required=True)
    parser.add_argument("--names",    required=True)
    parser.add_argument("--prices",   required=True)
    parser.add_argument("--out",      default="output/submission.csv")
    parser.add_argument("--conf",     type=float, default=0.4)
    parser.add_argument("--iou",      type=float, default=0.45)
    parser.add_argument("--img_size", type=int,   default=640)
    parser.add_argument("--device",   default="0")
    parser.add_argument("--skip",       type=int,   default=2,
                        help="Process every Nth frame (speed vs accuracy)")
    parser.add_argument("--init_inv",   default=None,
                        help="JSON file with initial inventory {\"class_id\": count}")
    parser.add_argument("--init_frames", type=int, default=30,
                        help="Frames to sample for auto initial inventory (if --init_inv not set)")
    parser.add_argument("--debug_log", default=None,
                        help="CSV path: dump per-frame fused counts (frame_idx,class_id,class_name,count) "
                             "before EventDetector smoothing, for diagnosing missed events")
    parser.add_argument("--timed_log", default=None,
                        help="CSV path: dump (time_sec,class_name,action) for every fired event, "
                             "for time-based scoring (tools/score_methods.py). Not part of the "
                             "official submission format -- diagnostic only.")

    # Tracker 옵션
    parser.add_argument("--use_tracker",      action="store_true",
                        help="SORT 트래커 활성화 (--use_tracker 없으면 기존 카운팅 방식)")
    parser.add_argument("--tracker_max_age",  type=int, default=15,
                        help="트래커: 미감지 허용 최대 프레임 수 (A/B 테스트: 15 최적)")
    parser.add_argument("--tracker_min_hits", type=int, default=3,
                        help="트래커: 확정까지 필요한 연속 감지 횟수")
    parser.add_argument("--tracker_iou",      type=float, default=0.3,
                        help="트래커: 매칭 최소 IoU")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device

    print("Loading model...")
    model, nms_fn = load_model(args.weights, device)

    class_names = load_names(args.names)
    prices      = load_prices(args.prices)

    caps = open_videos(args.videos)

    if args.init_inv:
        initial_inventory = load_initial_inventory_from_file(args.init_inv)
        print(f"Loaded initial inventory: {len(initial_inventory)} classes from {args.init_inv}")
    else:
        print(f"Estimating initial inventory from first {args.init_frames} frames...")
        initial_inventory = estimate_initial_inventory(
            caps, model, nms_fn, args.init_frames,
            args.conf, args.iou, args.img_size, device,
        )
        print(f"Initial inventory: {len(initial_inventory)} classes detected")
        print("Initial inventory detail:", {class_names[k]: v for k, v in initial_inventory.items()})
        if args.debug_log:
            init_dump_path = os.path.splitext(args.debug_log)[0] + "_initial_inventory.json"
            with open(init_dump_path, "w", encoding="utf-8") as f:
                json.dump({class_names[k]: v for k, v in initial_inventory.items()}, f,
                          ensure_ascii=False, indent=2)
            print(f"Initial inventory dumped to {init_dump_path}")

    detector = EventDetector(class_names, initial_counts=initial_inventory)
    vid_len  = video_duration(args.videos)

    fps_cap = cv2.VideoCapture(args.videos[0])
    fps = fps_cap.get(cv2.CAP_PROP_FPS) or 30
    fps_cap.release()

    cam_tracker = None
    if args.use_tracker:
        cam_tracker = MultiCameraTracker(
            n_cameras=len(caps),
            max_age=args.tracker_max_age,
            min_hits=args.tracker_min_hits,
            iou_threshold=args.tracker_iou,
        )
        print(f"SORT 트래커 활성화 (max_age={args.tracker_max_age}, "
              f"min_hits={args.tracker_min_hits}, iou={args.tracker_iou})")
    else:
        print("카운팅 방식 사용 (--use_tracker로 트래커 활성화 가능)")

    debug_writer = None
    debug_file = None
    if args.debug_log:
        import csv as _csv
        debug_file = open(args.debug_log, "w", newline="", encoding="utf-8")
        debug_writer = _csv.writer(debug_file)
        debug_writer.writerow(["frame_idx", "class_id", "class_name", "count"])

    timed_writer = None
    timed_file = None
    if args.timed_log:
        import csv as _csv
        timed_file = open(args.timed_log, "w", newline="", encoding="utf-8")
        timed_writer = _csv.writer(timed_file)
        timed_writer.writerow(["time_sec", "class_name", "action"])

    print(f"Processing {len(caps)} cameras, video length ≈ {vid_len:.1f}s ...")
    t_start = time.time()
    frame_idx = 0

    while True:
        statuses = grab_frames(caps)
        if not any(statuses):
            break

        if frame_idx % args.skip != 0:
            frame_idx += 1
            continue

        frames = retrieve_frames(caps, statuses)
        per_cam_dets = infer_batch(model, nms_fn, frames,
                                   args.conf, args.iou, args.img_size, device)

        if cam_tracker is not None:
            per_cam_dets = cam_tracker.update(per_cam_dets)

        fused_counts = fuse(per_cam_dets, cam_weights=compute_per_class_cam_weights(per_cam_dets))

        if debug_writer is not None:
            for cls_id, cnt in fused_counts.items():
                if cnt > 0:
                    debug_writer.writerow([frame_idx, cls_id, class_names[cls_id], cnt])

        # Convert fused counts back to flat detection list for EventDetector
        flat_dets = [
            {"class_id": cls_id, "confidence": 1.0, "bbox": []}
            for cls_id, cnt in fused_counts.items()
            for _ in range(cnt)
        ]

        new_events = detector.update(flat_dets)
        if new_events:
            for ev in new_events:
                print(f"  [Frame {frame_idx}] {ev.class_name}: {ev.action} "
                      f"({ev.before}→{ev.after})")
                if timed_writer is not None:
                    timed_writer.writerow([round(frame_idx / fps, 2), ev.class_name, ev.action])

        frame_idx += 1

    for cap in caps:
        if cap:
            cap.release()

    if debug_file is not None:
        debug_file.close()
        print(f"Debug log written to {args.debug_log}")

    if timed_file is not None:
        timed_file.close()
        print(f"Timed event log written to {args.timed_log}")

    _s = _occlusion_stats
    print(f"Camera occlusion stats (per class-frame, now per-class instead of whole-frame): "
          f"{_s['total']} class-frame pairs, "
          f"right_blocked={_s['right_blocked']} ({_s['right_blocked']/max(1,_s['total'])*100:.1f}%), "
          f"left_blocked={_s['left_blocked']} ({_s['left_blocked']/max(1,_s['total'])*100:.1f}%)")

    t_end = time.time()
    proc_time = t_end - t_start
    rtf = proc_time / vid_len if vid_len > 0 else float("inf")
    print(f"\nProcessing time: {proc_time:.1f}s  |  Video length: {vid_len:.1f}s  |  RTF: {rtf:.3f}")

    # run_stats.json: score.py 자동 호출에 사용
    stats_path = os.path.join(os.path.dirname(args.out), "run_stats.json")
    with open(stats_path, "w") as f:
        json.dump({"rtf": round(rtf, 4), "proc_time": round(proc_time, 1),
                   "vid_len": round(vid_len, 1), "submission": args.out}, f)

    events_to_csv(
        events=detector.all_events,
        prices=prices,
        out_path=args.out,
        initial_inventory=initial_inventory,
        include_action=True,
        total_mode="per_class",
    )


if __name__ == "__main__":
    main()

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


def estimate_initial_inventory(caps, model, nms_fn, n_frames, conf, iou, img_size, device,
                               quorum=2, min_corroborate=2, no_tuning=False,
                               init_min_detections=1) -> dict:
    """
    초기 재고 추정: n_frames 샘플 중 init_min_detections 회 이상 감지된 클래스만 포함.
    count 값: 감지된 프레임들의 median (0이 포함되지 않는 비율 기반 확정).

    기존 방식(median of all): 감지율 50% 미만 상품은 median=0 → 초기 재고 누락.
    새 방식(감지율 임계값): N프레임 중 K회 이상 감지 → 포함, count는 감지 프레임 median.
    """
    detect_count: dict = defaultdict(int)   # 감지된 프레임 수
    count_values: dict = defaultdict(list)  # 감지된 프레임의 count 값

    sampled = 0
    for _ in range(n_frames):
        frames = read_frames(caps)
        if all(f is None for f in frames):
            break
        per_cam = infer_batch(model, nms_fn, frames, conf, iou, img_size, device)
        if no_tuning:
            cam_weights = None
        else:
            cam_weights = compute_per_class_cam_weights(per_cam, min_corroborate=min_corroborate)
        fused = fuse(per_cam, cam_weights=cam_weights, quorum=quorum)
        for cls_id, cnt in fused.items():
            if cnt > 0:
                detect_count[cls_id] += 1
                count_values[cls_id].append(cnt)
        sampled += 1

    for cap in caps:
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    result = {}
    for cls_id, n_detected in detect_count.items():
        if n_detected < init_min_detections:
            continue
        median_cnt = int(np.median(count_values[cls_id]))
        if median_cnt > 0:
            result[cls_id] = median_cnt
    return result


def load_model(weights: str, device: str):
    """
    YOLOv7(.pt with 'model'/'ema' key) 또는 YOLO11(ultralytics) 자동 감지 후 로드.
    반환: (model, nms_fn)
      - YOLOv7: nms_fn = non_max_suppression
      - YOLO11:  nms_fn = None  (NMS는 ultralytics 내부 처리)
    """
    try:
        ckpt = torch.load(weights, map_location="cpu")
        # ultralytics 체크포인트 구분: 'train_args' 또는 'version' 키 존재 시 YOLO11
        is_ultralytics = isinstance(ckpt, dict) and (
            "train_args" in ckpt or "version" in ckpt
        )
        if not is_ultralytics and isinstance(ckpt, dict) and ("model" in ckpt or "ema" in ckpt):
            # YOLOv7 checkpoint
            yolov7_root = str(Path.home() / "yolov7")
            if yolov7_root not in sys.path:
                sys.path.insert(0, yolov7_root)
            from utils.general import non_max_suppression
            import torch.nn as nn
            model = (ckpt.get("ema") or ckpt["model"]).float().fuse().eval().to(device)
            for m in model.modules():
                if isinstance(m, nn.Upsample):
                    m.recompute_scale_factor = None
            print("Loaded YOLOv7 model")
            return model, non_max_suppression
    except Exception:
        pass

    # YOLO11 (ultralytics)
    from ultralytics import YOLO
    model = YOLO(weights)
    print("Loaded YOLO11 model")
    return model, None


def _preprocess_single(frame, img_size=640):
    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img).float() / 255.0


def infer_batch(model, nms_fn, frames, conf_thres=0.4, iou_thres=0.45,
                img_size=640, device="cpu"):
    """5개 카메라 프레임을 GPU 한 번에 배치 추론. YOLOv7/YOLO11 공용."""
    valid_idx = [i for i, f in enumerate(frames) if f is not None]
    if not valid_idx:
        return [None] * len(frames)

    per_cam = [None] * len(frames)

    if nms_fn is not None:
        # YOLOv7 경로: 직접 전처리 → GPU 배치 → NMS
        tensors = [_preprocess_single(frames[i], img_size) for i in valid_idx]
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            preds = model(batch)[0]
        preds = nms_fn(preds, conf_thres, iou_thres)
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
    else:
        # YOLO11 경로: raw BGR 프레임 그대로 전달 (ultralytics 내부 전처리+NMS)
        frame_list = [frames[i] for i in valid_idx]
        results = model(frame_list, conf=conf_thres, iou=iou_thres,
                        imgsz=img_size, verbose=False)
        for out_i, cam_i in enumerate(valid_idx):
            r = results[out_i]
            dets = []
            if r.boxes is not None and len(r.boxes):
                for xyxy, conf, cls in zip(
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                    r.boxes.cls.cpu().numpy(),
                ):
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

_occlusion_stats = {"total": 0, "cams_excluded": 0}


def compute_cam_weights(per_cam_dets, class_id=None, min_corroborate=2):
    """
    class_id=None: 프레임 전체 평균 confidence로 occlusion 판단(레거시 동작).
    class_id=<int>: 그 클래스의 confidence만 사용 -- 같은 프레임의 무관한
        클래스들이 occlusion 신호를 희석시키는 문제를 피함.

    2026-06-25: 좌(0,4)/우(1,3) 그룹 평균 비교 방식 -> 개별 카메라 단위로 일반화.
    haribo_gold_bears_gummi_candy(한쪽 그룹 전체가 막히는 패턴)는 그룹 비교로도
    구제됐지만 pepperidge_farm_milano_cookies_double_chocolate(probe3: 최대 3대
    동시)는 그룹 평균이 서로 비슷해져서 70% 임계값을 못 넘었을 것으로 추정.

    규칙: 카메라 i의 confidence가 0인데, 나머지 4대 중 MIN_CORROBORATE대 이상이
    양수면 i를 완전히 제외(weight=0). MIN_CORROBORATE=2가 유일하게 의미 있는 값임이
    서버 테스트로 확인됨 -- "나머지 4대 중 N대 corroborate"는 전체 5대 중 N대가
    보고 있다는 뜻인데, N>=3이면 이미 5대 중 과반(60%+)이라 균등weight로도 원래
    median=1이 나옴(이 함수가 개입할 필요가 없는 상황). 즉 N=3으로 올리면 조건은
    트리거되지만 결과가 안 바뀌는 무의미한 임계값이 되어, 의도와 달리 haribo까지
    다시 미검출로 돌아감(과반 미달 40%를 구제하는 유일한 지점은 N=2). N=2 자체의
    부작용(milano 과다발화, 아래 exclude_class_ids 참고)은 임계값이 아니라 클래스
    단위 예외로 처리.
    - bumblebee_albacore/dove/redbull류(CLASS_QUORUM_OVERRIDE 대상, 원래 1~2대만
      보임)는 N=2 조건도 못 채워서 영향 없음(quorum 분기가 weight를 이미 무시하므로
      애초에 무해하긴 함).
    """
    conf = []
    for dets in per_cam_dets:
        if not dets:
            conf.append(0.0)
            continue
        relevant = dets if class_id is None else [d for d in dets if d["class_id"] == class_id]
        conf.append(sum(d["confidence"] for d in relevant) / len(relevant) if relevant else 0.0)

    weights = [1.0, 1.0, 1.5, 1.0, 1.0]  # 위 카메라 기본 1.5배
    n = len(conf)

    _occlusion_stats["total"] += 1
    for i in range(n):
        if conf[i] > 0:
            continue
        others_nonzero = sum(1 for j in range(n) if j != i and conf[j] > 0)
        if others_nonzero >= min_corroborate:
            weights[i] = 0.0
            _occlusion_stats["cams_excluded"] += 1

    return weights


_DEFAULT_CAM_WEIGHTS = [1.0, 1.0, 1.5, 1.0, 1.0]


def compute_per_class_cam_weights(per_cam_dets, min_corroborate=2):
    """
    프레임에 등장한 클래스마다 자동으로 occlusion weight 계산 (class_id -> weights).
    클래스 예외 없이 동일한 규칙 적용.
    """
    class_ids = set()
    for dets in per_cam_dets:
        if dets:
            class_ids.update(d["class_id"] for d in dets)
    return {
        cid: compute_cam_weights(per_cam_dets, class_id=cid, min_corroborate=min_corroborate)
        for cid in class_ids
    }


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
    parser.add_argument("--init_min_detections", type=int, default=1,
                        help="초기재고 포함 최소 감지 횟수: N프레임 중 이 값 이상 감지돼야 재고로 인정 "
                             "(기본=1: 1회만 감지돼도 포함, 이전 방식 median과 달리 희귀 상품도 잡음)")
    parser.add_argument("--debug_log", default=None,
                        help="CSV path: dump per-frame fused counts (frame_idx,class_id,class_name,count) "
                             "before EventDetector smoothing, for diagnosing missed events")
    parser.add_argument("--timed_log", default=None,
                        help="CSV path: dump (time_sec,class_name,action) for every fired event, "
                             "for time-based scoring (tools/score_methods.py). Not part of the "
                             "official submission format -- diagnostic only.")
    parser.add_argument("--per_cam_log", default=None,
                        help="CSV path: dump per-camera raw counts (frame_idx,cam_id,class_id,class_name,count) "
                             "BEFORE fusion, for camera whitelist analysis.")

    # Tracker 옵션
    parser.add_argument("--use_tracker",      action="store_true",
                        help="SORT 트래커 활성화 (--use_tracker 없으면 기존 카운팅 방식)")
    parser.add_argument("--tracker_max_age",  type=int, default=15,
                        help="트래커: 미감지 허용 최대 프레임 수 (A/B 테스트: 15 최적)")
    parser.add_argument("--tracker_min_hits", type=int, default=3,
                        help="트래커: 확정까지 필요한 연속 감지 횟수")
    parser.add_argument("--tracker_iou",       type=float, default=0.3,
                        help="트래커: 매칭 최소 IoU")
    parser.add_argument("--tracker_type",      type=str,   default="bytetrack",
                        help="트래커 종류: sort | bytetrack")
    parser.add_argument("--tracker_high_thresh", type=float, default=0.6,
                        help="ByteSort 전용: stage 1/2 분기 confidence 기준")
    parser.add_argument("--quorum",        type=int, default=2,
                        help="카메라 동의 쿼럼: quorum-th highest vote 적용 (1=단일카메라, 2=2대동의, 3=과반)")
    parser.add_argument("--min_corroborate", type=int, default=2,
                        help="occlusion weight 제외 기준: 이 수 이상의 다른 카메라가 감지해야 제외")
    parser.add_argument("--no_tuning",         action="store_true",
                        help="cam weights 비활성화 (uniform weights, quorum만 적용)")
    # EventDetector 파라미터
    parser.add_argument("--window_size",    type=int, default=15,
                        help="EventDetector 슬라이딩 윈도우 크기 (기본 15프레임)")
    parser.add_argument("--confirm_frames", type=int, default=30,
                        help="이벤트 확정까지 유지돼야 하는 프레임 수 (기본 30, skip=2 → 실제 60프레임≈2초)")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device

    if args.no_tuning:
        print(f"[no_tuning] cam weights disabled, quorum={args.quorum}")

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
            quorum=args.quorum,
            min_corroborate=args.min_corroborate,
            no_tuning=args.no_tuning,
            init_min_detections=args.init_min_detections,
        )
        print(f"Initial inventory: {len(initial_inventory)} classes detected")
        print("Initial inventory detail:", {class_names[k]: v for k, v in initial_inventory.items()})
        if args.debug_log:
            init_dump_path = os.path.splitext(args.debug_log)[0] + "_initial_inventory.json"
            with open(init_dump_path, "w", encoding="utf-8") as f:
                json.dump({class_names[k]: v for k, v in initial_inventory.items()}, f,
                          ensure_ascii=False, indent=2)
            print(f"Initial inventory dumped to {init_dump_path}")

    detector = EventDetector(
        class_names,
        initial_counts  = initial_inventory,
        window_size     = args.window_size,
        confirm_frames  = args.confirm_frames,
    )
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
            tracker_type=args.tracker_type,
            high_thresh=args.tracker_high_thresh,
        )
        print(f"{args.tracker_type.upper()} 트래커 활성화 (max_age={args.tracker_max_age}, "
              f"min_hits={args.tracker_min_hits}, iou={args.tracker_iou}"
              + (f", high_thresh={args.tracker_high_thresh}" if args.tracker_type == "bytetrack" else "")
              + ")")
    else:
        print("카운팅 방식 사용 (--use_tracker로 트래커 활성화 가능)")

    per_cam_writer = None
    per_cam_file = None
    if args.per_cam_log:
        import csv as _csv
        per_cam_file = open(args.per_cam_log, "w", newline="", encoding="utf-8")
        per_cam_writer = _csv.writer(per_cam_file)
        per_cam_writer.writerow(["frame_idx", "cam_id", "class_id", "class_name", "count"])

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

        if per_cam_writer is not None:
            for cam_id, dets in enumerate(per_cam_dets):
                if dets is None:
                    continue
                counts = {}
                for d in dets:
                    counts[d["class_id"]] = counts.get(d["class_id"], 0) + 1
                for cls_id, cnt in counts.items():
                    per_cam_writer.writerow([frame_idx, cam_id, cls_id, class_names[cls_id], cnt])

        if args.no_tuning:
            fused_counts = fuse(per_cam_dets, quorum=args.quorum)
        else:
            fused_counts = fuse(
                per_cam_dets,
                cam_weights=compute_per_class_cam_weights(per_cam_dets, args.min_corroborate),
                quorum=args.quorum,
            )

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

    if per_cam_file is not None:
        per_cam_file.close()
        print(f"Per-camera log written to {args.per_cam_log}")

    if debug_file is not None:
        debug_file.close()
        print(f"Debug log written to {args.debug_log}")

    if timed_file is not None:
        timed_file.close()
        print(f"Timed event log written to {args.timed_log}")

    _s = _occlusion_stats
    print(f"Camera occlusion stats (per-camera, per class-frame): "
          f"{_s['total']} class-frame pairs, "
          f"{_s['cams_excluded']} individual camera-votes excluded "
          f"({_s['cams_excluded']/max(1,_s['total']*5)*100:.1f}% of all camera-votes)")

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
        total_mode="inventory",
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()

"""
YOLO11 Knowledge Distillation Trainer — v2 (feat/kd-occ-aug)

v1 대비 변경사항:
  1. Occlusion augmentation — preprocess_batch에서 GT bbox 위에 랜덤 마스킹
     dove_white/milano처럼 occlusion 구간에서 검출 끊기는 문제를 훈련에서 직접 재현
  2. GT-aligned anchor selection — top-50 naive 대신 GT box 위치 기반 anchor 선택
     실제 물체가 있는 anchor 위치의 예측으로 KL divergence 계산
  3. Weak-class oversampling (RF-DETR 방식 이식) — campbells/dove_white/milano 등
     감지율 낮은 클래스 포함 이미지를 oversample_weak배 복제하여 학습 노출 빈도 강제 증가

KD Loss:
  L = (1-alpha)*L_det  +  alpha * tau^2 * KL(teacher_gt || student_gt_aligned)
"""
import os
import random
import shutil
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss

# RF-DETR yolo_to_coco.py의 동일 클래스 ID 기준 이식
WEAK_CLASS_IDS = {0, 8, 43, 45, 48}       # aunt_jemima, hunts_sauce, campbells, chewy_dips_choc, cheerios
TIMING_CLASS_IDS = {2, 22, 42, 46, 50}    # bumblebee, haribo, milano, chewy_dips_pb, lindt
REINFORCE_CLASS_IDS = WEAK_CLASS_IDS | TIMING_CLASS_IDS


def build_oversampled_yaml(data_yaml: str, oversample_factor: int, tmp_dir: str) -> str:
    """
    data_yaml의 train split에서 REINFORCE_CLASS_IDS 포함 이미지를 oversample_factor배 복제.
    수정된 yaml 경로를 반환. oversample_factor <= 1이면 원본 그대로 반환.
    """
    if oversample_factor <= 1:
        return data_yaml

    import yaml

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    train_val = cfg.get("train", "")
    train_txt = Path(train_val)
    if not train_txt.exists():
        print(f"[oversample] train.txt 없음: {train_txt}, 오버샘플링 스킵")
        return data_yaml

    train_paths = [l.strip() for l in train_txt.read_text().splitlines() if l.strip()]

    def has_reinforce_class(img_path: str) -> bool:
        p = Path(img_path)
        lp = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
        if not lp.exists():
            lp = p.with_suffix(".txt")
        if not lp.exists():
            return False
        with open(lp) as f:
            for line in f:
                parts = line.strip().split()
                if parts and int(parts[0]) in REINFORCE_CLASS_IDS:
                    return True
        return False

    reinforce = [p for p in train_paths if has_reinforce_class(p)]
    extra = reinforce * (oversample_factor - 1)
    all_paths = train_paths + extra
    random.shuffle(all_paths)

    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    new_txt = tmp / "train_oversampled.txt"
    new_txt.write_text("\n".join(all_paths))

    cfg["train"] = str(new_txt)
    new_yaml = tmp / "custom_oversampled.yaml"
    with open(new_yaml, "w") as f:
        yaml.dump(cfg, f)

    print(f"[oversample] reinforce 이미지 {len(reinforce)}장 x{oversample_factor} "
          f"(+{len(extra)}장 복제, 전체 {len(all_paths)}장)")
    return str(new_yaml)


class KDDetectionLoss(v8DetectionLoss):
    def __init__(self, model, soft_label_dir, kd_alpha=0.5, kd_tau=4.0):
        super().__init__(model)
        self.soft_label_dir = Path(soft_label_dir).expanduser()
        self.kd_alpha = kd_alpha
        self.kd_tau = kd_tau

    def _gt_aligned_student_dist(self, pred_scores, batch, b_idx, strides):
        """
        GT bbox 위치에 해당하는 anchor의 class prediction을 뽑아 평균.
        top-50 naive 방식 → 실제 물체가 있는 grid cell의 예측으로 개선.

        pred_scores: [A, nc]
        """
        device = pred_scores.device

        mask = (batch['batch_idx'].cpu() == b_idx)
        if not mask.any():
            topk = min(20, pred_scores.shape[0])
            top_idx = pred_scores.max(dim=1).values.topk(topk).indices
            return pred_scores[top_idx].mean(dim=0)

        gt_bboxes = batch['bboxes'][mask].cpu()  # [N, 4] normalized xywh
        imgsz = batch['img'].shape[-1]

        selected_indices = []
        anchor_start = 0

        for stride in strides:
            grid_size = imgsz // stride
            n_anchors = grid_size * grid_size

            gt_cx = (gt_bboxes[:, 0] * grid_size).long().clamp(0, grid_size - 1)
            gt_cy = (gt_bboxes[:, 1] * grid_size).long().clamp(0, grid_size - 1)
            anchor_idx = (gt_cy * grid_size + gt_cx) + anchor_start
            selected_indices.append(anchor_idx)

            anchor_start += n_anchors

        indices = torch.cat(selected_indices).clamp(0, pred_scores.shape[0] - 1).to(device)
        return pred_scores[indices].mean(dim=0)

    def __call__(self, preds, batch):
        base_loss, loss_items = super().__call__(preds, batch)

        im_files = batch.get("im_file", [])
        if not im_files:
            return base_loss, loss_items

        device = base_loss.device
        batch_size = len(im_files)

        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_all = torch.cat(
            [xi.view(batch_size, self.no, -1) for xi in feats], dim=2
        )  # [B, no, A]
        pred_scores = pred_all[:, self.reg_max * 4:, :].permute(0, 2, 1).sigmoid()  # [B, A, nc]

        imgsz = batch['img'].shape[-1]
        strides = [imgsz // xi.shape[-1] for xi in feats]

        kd_loss = base_loss * 0.0
        n_valid = 0

        for b_idx, im_file in enumerate(im_files):
            stem = Path(im_file).stem
            npy_path = self.soft_label_dir / (stem + ".npy")
            if not npy_path.exists():
                continue

            soft = np.load(str(npy_path))
            if soft.shape[0] == 0:
                continue

            teacher_dist = torch.from_numpy(soft.mean(axis=0)).float().to(device)

            # v2: GT-aligned anchor selection
            student_dist = self._gt_aligned_student_dist(
                pred_scores[b_idx], batch, b_idx, strides
            )

            tau = self.kd_tau
            t_log = torch.log(teacher_dist + 1e-8) / tau
            s_log = torch.log(student_dist + 1e-8) / tau
            t_soft = F.softmax(t_log, dim=0)
            s_lsft = F.log_softmax(s_log, dim=0)

            kl = F.kl_div(s_lsft, t_soft, reduction="sum") * (tau ** 2)
            kd_loss = kd_loss + kl
            n_valid += 1

        if n_valid > 0:
            kd_loss = kd_loss / n_valid

        combined = (1.0 - self.kd_alpha) * base_loss + self.kd_alpha * kd_loss
        return combined, loss_items


class KDTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        from ultralytics.cfg import DEFAULT_CFG
        overrides = dict(overrides or {})
        self.soft_label_dir = Path(overrides.pop("soft_label_dir", "~/Dataset/soft_labels")).expanduser()
        self.kd_alpha = float(overrides.pop("kd_alpha", 0.5))
        self.kd_tau = float(overrides.pop("kd_tau", 4.0))
        self.occ_prob = float(overrides.pop("occ_prob", 0.3))
        super().__init__(cfg=cfg or DEFAULT_CFG, overrides=overrides, _callbacks=_callbacks)

    def preprocess_batch(self, batch):
        """Occlusion augmentation: GT bbox 위에 랜덤 마스킹 추가."""
        batch = super().preprocess_batch(batch)

        if not self.model.training:
            return batch

        imgs = batch['img']             # [B, C, H, W] float [0,1], on device
        bboxes = batch.get('bboxes')
        batch_idx_t = batch.get('batch_idx')

        if bboxes is None or batch_idx_t is None:
            return batch

        B, C, H, W = imgs.shape
        bboxes_cpu = bboxes.cpu() if torch.is_tensor(bboxes) else bboxes
        bidx_cpu = batch_idx_t.cpu() if torch.is_tensor(batch_idx_t) else batch_idx_t

        for b in range(B):
            mask = (bidx_cpu == b)
            if not mask.any():
                continue

            for box in bboxes_cpu[mask]:
                if random.random() > self.occ_prob:
                    continue

                cx, cy, bw, bh = box.tolist()
                x1 = int((cx - bw / 2) * W)
                y1 = int((cy - bh / 2) * H)
                x2 = int((cx + bw / 2) * W)
                y2 = int((cy + bh / 2) * H)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # bbox 내 30~70% 크기의 랜덤 영역을 어두운 노이즈로 가림
                occ_ratio = random.uniform(0.3, 0.7)
                occ_w = max(1, int((x2 - x1) * occ_ratio))
                occ_h = max(1, int((y2 - y1) * occ_ratio))
                ox1 = random.randint(x1, max(x1, x2 - occ_w))
                oy1 = random.randint(y1, max(y1, y2 - occ_h))
                ox2 = min(ox1 + occ_w, W)
                oy2 = min(oy1 + occ_h, H)

                imgs[b, :, oy1:oy2, ox1:ox2] = torch.rand(
                    C, oy2 - oy1, ox2 - ox1, device=imgs.device
                ) * 0.3

        batch['img'] = imgs
        return batch

    def init_criterion(self):
        return KDDetectionLoss(
            self.model,
            soft_label_dir=self.soft_label_dir,
            kd_alpha=self.kd_alpha,
            kd_tau=self.kd_tau,
        )


def train_kd(
    model_path:       str   = "yolo11m.pt",
    data_yaml:        str   = "data/custom.yaml",
    soft_label_dir:   str   = "~/Dataset/soft_labels",
    epochs:           int   = 100,
    batch:            int   = 16,
    imgsz:            int   = 640,
    device:           str   = "0",
    project:          str   = "runs/kd",
    name:             str   = "yolo11m_kd",
    alpha:            float = 0.5,
    tau:              float = 4.0,
    occ_prob:         float = 0.3,
    oversample_weak:  int   = 1,
    resume:           bool  = False,
):
    tmp_dir = None
    if oversample_weak > 1:
        tmp_dir = tempfile.mkdtemp(prefix="kd_oversample_")
        data_yaml = build_oversampled_yaml(data_yaml, oversample_weak, tmp_dir)

    model = YOLO(model_path)
    try:
      model.train(
        trainer=KDTrainer,
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        resume=resume,
        soft_label_dir=soft_label_dir,
        kd_alpha=alpha,
        kd_tau=tau,
        occ_prob=occ_prob,
        lr0=0.01,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        close_mosaic=10,
        augment=True,
      )
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default="yolo11m.pt")
    p.add_argument("--data",           required=True)
    p.add_argument("--soft_label_dir", default="~/Dataset/soft_labels")
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--batch",    type=int,   default=16)
    p.add_argument("--imgsz",    type=int,   default=640)
    p.add_argument("--device",             default="0")
    p.add_argument("--project",            default="runs/kd")
    p.add_argument("--name",               default="yolo11m_kd")
    p.add_argument("--alpha",    type=float, default=0.5)
    p.add_argument("--tau",      type=float, default=4.0)
    p.add_argument("--occ_prob",        type=float, default=0.3)
    p.add_argument("--oversample_weak", type=int,   default=1,
                   help="REINFORCE 클래스 포함 이미지 복제 배수 (1=비활성, RF-DETR는 5 사용)")
    p.add_argument("--resume",   action="store_true")
    args = p.parse_args()
    train_kd(
        model_path=args.model,
        data_yaml=args.data,
        soft_label_dir=args.soft_label_dir,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        alpha=args.alpha,
        tau=args.tau,
        occ_prob=args.occ_prob,
        oversample_weak=args.oversample_weak,
        resume=args.resume,
    )

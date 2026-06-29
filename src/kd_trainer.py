"""
YOLO11 Knowledge Distillation Trainer.

Teacher(YOLOv7) soft label(.npy)을 loss 함수 내부에서 im_file 경로로 직접 로드해
classification loss에 KL divergence를 추가한다. DataLoader 수정 없음.

KD Loss:
  L = (1-alpha)*L_det  +  alpha * tau^2 * KL(teacher_avg || student_avg)

  teacher_avg: 이미지 내 GT box별 soft label의 평균 [nc]
  student_avg: top-K anchor 예측 확률의 평균 [nc]

Usage:
    bash train_kd.sh --skip_softlabel --epochs 100 --batch 16
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss


class KDDetectionLoss(v8DetectionLoss):
    """
    v8DetectionLoss + per-image KL divergence KD term.

    soft label은 batch["im_file"] 경로 stem으로 soft_label_dir에서 로드.
    DataLoader/collate 수정 불필요.
    """

    def __init__(self, model, soft_label_dir, kd_alpha=0.5, kd_tau=4.0):
        super().__init__(model)
        self.soft_label_dir = Path(soft_label_dir).expanduser()
        self.kd_alpha = kd_alpha
        self.kd_tau   = kd_tau

    def __call__(self, preds, batch):
        base_loss, loss_items = super().__call__(preds, batch)

        im_files = batch.get("im_file", [])
        if not im_files:
            return base_loss, loss_items

        device     = base_loss.device
        batch_size = len(im_files)

        # student class predictions: [B, A, nc]
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_all = torch.cat(
            [xi.view(batch_size, self.no, -1) for xi in feats], dim=2
        )  # [B, no, A]
        pred_scores = pred_all[:, self.reg_max * 4:, :].permute(0, 2, 1).sigmoid()  # [B, A, nc]

        kd_loss = base_loss * 0.0  # 0, 그래프에 연결
        n_valid = 0

        for b_idx, im_file in enumerate(im_files):
            stem = Path(im_file).stem
            npy_path = self.soft_label_dir / (stem + ".npy")
            if not npy_path.exists():
                continue

            soft = np.load(str(npy_path))  # [N_gt, nc]
            if soft.shape[0] == 0:
                continue

            # teacher: GT box별 soft label 평균 [nc]
            teacher_dist = torch.from_numpy(soft.mean(axis=0)).float().to(device)

            # student: max confidence 상위 50개 anchor 평균 [nc]
            student_cls = pred_scores[b_idx]                     # [A, nc]
            topk        = min(50, student_cls.shape[0])
            top_idx     = student_cls.max(dim=1).values.topk(topk).indices
            student_dist = student_cls[top_idx].mean(dim=0)       # [nc]

            # temperature softmax
            tau = self.kd_tau
            t_log  = torch.log(teacher_dist + 1e-8) / tau
            s_log  = torch.log(student_dist + 1e-8) / tau
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
    """
    KD 하이퍼파라미터를 overrides dict에서 추출해 KDDetectionLoss를 사용하는 trainer.
    ultralytics는 trainer(overrides=dict, _callbacks=...) 형태로 호출한다.
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        from ultralytics.cfg import DEFAULT_CFG
        overrides = dict(overrides or {})
        self.soft_label_dir = Path(overrides.pop("soft_label_dir", "~/Dataset/soft_labels")).expanduser()
        self.kd_alpha = float(overrides.pop("kd_alpha", 0.5))
        self.kd_tau   = float(overrides.pop("kd_tau",   4.0))
        super().__init__(cfg=cfg or DEFAULT_CFG, overrides=overrides, _callbacks=_callbacks)

    def init_criterion(self):
        return KDDetectionLoss(
            self.model,
            soft_label_dir=self.soft_label_dir,
            kd_alpha=self.kd_alpha,
            kd_tau=self.kd_tau,
        )


def train_kd(
    model_path:     str   = "yolo11m.pt",
    data_yaml:      str   = "data/custom.yaml",
    soft_label_dir: str   = "~/Dataset/soft_labels",
    epochs:         int   = 100,
    batch:          int   = 16,
    imgsz:          int   = 640,
    device:         str   = "0",
    project:        str   = "runs/kd",
    name:           str   = "yolo11m_kd",
    alpha:          float = 0.5,
    tau:            float = 4.0,
    resume:         bool  = False,
):
    model = YOLO(model_path)
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
        lr0=0.01,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        close_mosaic=10,
        augment=True,
    )


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
        resume=args.resume,
    )

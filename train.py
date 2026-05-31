"""
Baseball Detection — Faster R-CNN Training Script
-------------------------------------------------
Step 5 in the pipeline. Runs after prepare_dataset.py.

What this script does
---------------------
    1. Loads train / val COCO splits via BaseballDetectionDataset
    2. Builds a Faster R-CNN (ResNet50 + FPN) with pretrained COCO
       weights, replaces the box predictor head for 2 classes
       (background + baseball), and — because our baseballs are small —
       swaps in a smaller anchor generator
    3. Runs a standard SGD training loop with an LR scheduler
    4. After every epoch: evaluates on the val split using a simple
       IoU-based detection metric (precision / recall / F1 at IoU=0.5)
       and tracks the best checkpoint
    5. Saves model weights + a training-log CSV

Usage
-----
    # with defaults (reads dataset/ produced by prepare_dataset.py)
    python train.py

    # override anything via CLI
    python train.py --epochs 40 --batch-size 4 --lr 0.005

    # resume from the last saved checkpoint (picks up where you left off)
    python train.py --resume --epochs 40

Requirements
------------
    pip install torch torchvision pillow
"""

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

from baseball_dataset import (
    BaseballDetectionDataset,
    collate_fn,
    default_train_transforms,
    default_eval_transforms,
)


# -----------------------------------------------------------------
#  CONFIG
# -----------------------------------------------------------------


@dataclass
class Config:
    dataset_dir: str = "dataset"
    output_dir: str = "runs/run2"

    num_classes: int = 2         # background + baseball
    epochs: int = 40
    batch_size: int = 2
    num_workers: int = 0
    lr: float = 0.003
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_step_size: int = 8
    lr_gamma: float = 0.1

    # Small-box friendly anchors. Our baseballs cluster around 10-50 px
    # even after standardization; the torchvision default starts at 32.
    anchor_sizes: Tuple[Tuple[int, ...], ...] = (
        (4,), (8,), (16,), (32,), (64,)
    )
    anchor_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0)

    # Pretrained weights. Set to False to train from random init (don't).
    pretrained: bool = True

    # Eval threshold for the simple IoU-based precision/recall logger
    eval_iou_threshold: float = 0.5
    # original 0.05
    eval_score_threshold: float = 0.3

    # Resume from last checkpoint instead of starting fresh
    resume: bool = False


# -----------------------------------------------------------------
#  MODEL BUILDER
# -----------------------------------------------------------------


def build_model(cfg: Config) -> torch.nn.Module:
    """Faster R-CNN with ResNet50+FPN, our small-object anchors, and a
    2-class (background + baseball) box predictor."""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if cfg.pretrained else None

    # Custom anchor generator: one size per FPN level, shared aspect ratios
    n_levels = len(cfg.anchor_sizes)
    anchor_generator = AnchorGenerator(
        sizes=cfg.anchor_sizes,
        aspect_ratios=(cfg.anchor_ratios,) * n_levels,
    )

    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        rpn_anchor_generator=anchor_generator,
    )

    # Replace the classification head: default has 91 COCO classes.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.num_classes)

    return model


# -----------------------------------------------------------------
#  RESUME HELPER
# -----------------------------------------------------------------


def find_latest_checkpoint(output_dir: str):
    """
    Scan output_dir for epoch_XXX.pt files and return the path to the
    most recent one, plus the epoch number. Returns (None, 0) if none found.
    """
    if not os.path.isdir(output_dir):
        return None, 0

    checkpoints = sorted([
        f for f in os.listdir(output_dir)
        if f.startswith("epoch_") and f.endswith(".pt")
    ])

    if not checkpoints:
        return None, 0

    latest = checkpoints[-1]
    # Parse epoch number from filename e.g. epoch_020.pt -> 20
    epoch_num = int(latest.replace("epoch_", "").replace(".pt", ""))
    return os.path.join(output_dir, latest), epoch_num


# -----------------------------------------------------------------
#  TRAIN / EVAL LOOPS
# -----------------------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    print_every: int = 20,
) -> Dict[str, float]:
    """Standard torchvision-style detection training loop for one epoch."""
    model.train()
    running = {}
    n_batches = 0

    t0 = time.time()
    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # .detach() removes gradient tracking before converting to float
        for k, v in loss_dict.items():
            running[k] = running.get(k, 0.0) + float(v.detach())
        running["total"] = running.get("total", 0.0) + float(loss.detach())
        n_batches += 1

        if (i + 1) % print_every == 0:
            mean = {k: v / n_batches for k, v in running.items()}
            print(f"  epoch {epoch} batch {i+1}/{len(loader)} "
                  f"loss={mean['total']:.4f}")

    dt = time.time() - t0
    mean = {k: v / max(1, n_batches) for k, v in running.items()}
    mean["epoch_time_s"] = dt
    return mean


def _box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between box sets a[M,4] and b[N,4] (xyxy). Returns [M, N]."""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]))
    return torchvision.ops.box_iou(a, b)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    iou_threshold: float,
    score_threshold: float,
) -> Dict[str, float]:
    """
    Lightweight eval: sweep predictions at one IoU threshold, compute
    precision / recall / F1 at the given score threshold.
    """
    model.eval()
    tp = fp = fn = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            keep = output["scores"] >= score_threshold
            pred_boxes = output["boxes"][keep].cpu()
            gt_boxes = target["boxes"]

            if gt_boxes.numel() == 0 and pred_boxes.numel() == 0:
                continue
            if gt_boxes.numel() == 0:
                fp += pred_boxes.shape[0]
                continue
            if pred_boxes.numel() == 0:
                fn += gt_boxes.shape[0]
                continue

            iou = _box_iou(pred_boxes, gt_boxes)
            matched_gt = set()
            for pi in torch.argsort(-output["scores"][keep]):
                ious_for_pi = iou[pi]
                gi = int(torch.argmax(ious_for_pi))
                if float(ious_for_pi[gi]) >= iou_threshold and gi not in matched_gt:
                    tp += 1
                    matched_gt.add(gi)
                else:
                    fp += 1
            fn += gt_boxes.shape[0] - len(matched_gt)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    }


# -----------------------------------------------------------------
#  MAIN
# -----------------------------------------------------------------


def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    train_ds = BaseballDetectionDataset(
        os.path.join(cfg.dataset_dir, "train"),
        transforms=default_train_transforms(),
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
    )
    val_dir = os.path.join(cfg.dataset_dir, "val")
    val_loader = None
    if os.path.isdir(val_dir) and os.path.isfile(os.path.join(val_dir, "annotations.json")):
        val_ds = BaseballDetectionDataset(val_dir, transforms=default_eval_transforms())
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collate_fn,
        )
    return train_loader, val_loader


def main(cfg: Config) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train images: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"Val images:   {len(val_loader.dataset)}")
    else:
        print("No val split found — training without validation.")

    # Build model — always start from pretrained COCO weights so the
    # architecture is initialized correctly before we load our checkpoint.
    model = build_model(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                    weight_decay=cfg.weight_decay)
    scheduler = StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)

    # Resume: load the latest checkpoint and fast-forward the scheduler
    start_epoch = 1
    best_f1 = -1.0
    best_loss = float("inf")

    if cfg.resume:
        ckpt_path, last_epoch = find_latest_checkpoint(cfg.output_dir)
        if ckpt_path is None:
            print("No checkpoint found — starting from scratch.")
        else:
            print(f"Resuming from: {ckpt_path} (epoch {last_epoch})")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            # Fast-forward the LR scheduler to match where we left off
            for _ in range(last_epoch):
                scheduler.step()
            start_epoch = last_epoch + 1
            print(f"Resuming training from epoch {start_epoch}\n")

    log_path = os.path.join(cfg.output_dir, "training_log.csv")
    log_fields = ["epoch", "total_loss", "loss_classifier", "loss_box_reg",
                  "loss_objectness", "loss_rpn_box_reg", "epoch_time_s",
                  "val_precision", "val_recall", "val_f1"]

    # If resuming, append to existing log. Otherwise start fresh.
    log_mode = "a" if cfg.resume and os.path.isfile(log_path) else "w"
    with open(log_path, log_mode, newline="") as f:
        if log_mode == "w":
            csv.DictWriter(f, fieldnames=log_fields).writeheader()

    for epoch in range(start_epoch, cfg.epochs + 1):
        print(f"\n=== Epoch {epoch}/{cfg.epochs} ===")
        tmean = train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()

        eval_metrics = {}
        if val_loader is not None:
            eval_metrics = evaluate(
                model, val_loader, device,
                iou_threshold=cfg.eval_iou_threshold,
                score_threshold=cfg.eval_score_threshold,
            )
            print(f"  val P={eval_metrics['precision']:.3f} "
                  f"R={eval_metrics['recall']:.3f} "
                  f"F1={eval_metrics['f1']:.3f} "
                  f"(tp={eval_metrics['tp']} fp={eval_metrics['fp']} fn={eval_metrics['fn']})")

        # Log
        row = {
            "epoch": epoch,
            "total_loss": round(tmean.get("total", 0.0), 5),
            "loss_classifier": round(tmean.get("loss_classifier", 0.0), 5),
            "loss_box_reg":    round(tmean.get("loss_box_reg", 0.0), 5),
            "loss_objectness": round(tmean.get("loss_objectness", 0.0), 5),
            "loss_rpn_box_reg":round(tmean.get("loss_rpn_box_reg", 0.0), 5),
            "epoch_time_s":    round(tmean.get("epoch_time_s", 0.0), 2),
            "val_precision":   round(eval_metrics.get("precision", float("nan")), 4),
            "val_recall":      round(eval_metrics.get("recall",    float("nan")), 4),
            "val_f1":          round(eval_metrics.get("f1",        float("nan")), 4),
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow(row)

        # Save checkpoints
        ckpt_path = os.path.join(cfg.output_dir, f"epoch_{epoch:03d}.pt")
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": cfg.__dict__}, ckpt_path)

        # Track best by val F1 (if val available) or by train loss
        if val_loader is not None and eval_metrics.get("f1", -1) > best_f1:
            best_f1 = eval_metrics["f1"]
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best.pt"))
            print(f"  new best val F1={best_f1:.3f} -> best.pt")
        elif val_loader is None and tmean["total"] < best_loss:
            best_loss = tmean["total"]
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best.pt"))
            print(f"  new best train loss={best_loss:.4f} -> best.pt")

    print(f"\nDone. Log: {log_path}")
    print(f"Checkpoints in: {cfg.output_dir}")


# -----------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train Faster R-CNN for baseball detection.")
    p.add_argument("--dataset-dir", default=Config.dataset_dir)
    p.add_argument("--output-dir",  default=Config.output_dir)
    p.add_argument("--epochs",      type=int,   default=Config.epochs)
    p.add_argument("--batch-size",  type=int,   default=Config.batch_size)
    p.add_argument("--num-workers", type=int,   default=Config.num_workers)
    p.add_argument("--lr",          type=float, default=Config.lr)
    p.add_argument("--no-pretrained", action="store_true",
                   help="Disable COCO pretrained weights")
    p.add_argument("--resume", action="store_true",
                   help="Resume training from the latest checkpoint in output-dir")
    a = p.parse_args()
    cfg = Config()
    cfg.dataset_dir = a.dataset_dir
    cfg.output_dir = a.output_dir
    cfg.epochs = a.epochs
    cfg.batch_size = a.batch_size
    cfg.num_workers = a.num_workers
    cfg.lr = a.lr
    cfg.pretrained = not a.no_pretrained
    cfg.resume = a.resume
    return cfg


if __name__ == "__main__":
    main(parse_args())
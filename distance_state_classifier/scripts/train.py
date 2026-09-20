#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from distance_state_classifier.src.config import ensure_dir, get_nested, load_config
from distance_state_classifier.src.dataset import DistanceStateDataset, class_counts
from distance_state_classifier.src.metrics import classification_metrics, confusion_matrix, write_confusion_matrix
from distance_state_classifier.src.model import build_model
from distance_state_classifier.src.transforms import CropBox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the endoscope distance-state classifier.")
    parser.add_argument("--config", default="distance_state_classifier/configs/distance_state_config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable timm pretrained weights even if config enables them.")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_crop(config: dict) -> CropBox | None:
    if not bool(get_nested(config, "image.crop.enabled", False)):
        return None
    return CropBox(
        x_left=int(get_nested(config, "image.crop.x_left", 362)),
        x_right=int(get_nested(config, "image.crop.x_right", 1605)),
        y_top=int(get_nested(config, "image.crop.y_top", 0)),
        y_bottom=int(get_nested(config, "image.crop.y_bottom", 1080)),
    )


def make_dataset(config: dict, split: str, training: bool) -> DistanceStateDataset:
    return DistanceStateDataset(
        csv_path=get_nested(config, f"data.{split}_csv"),
        classes=list(get_nested(config, "data.classes", ["TooFar", "Good", "TooClose"])),
        image_column=str(get_nested(config, "data.image_column", "image_path")),
        label_column=str(get_nested(config, "data.label_column", "label")),
        input_size=int(get_nested(config, "image.input_size", 384)),
        training=training,
        augmentation_cfg=dict(get_nested(config, "augmentation", {})),
        crop=make_crop(config),
    )


def make_class_weights(dataset: DistanceStateDataset, device: torch.device) -> torch.Tensor:
    counts = class_counts(dataset)
    values = np.array([counts[label] for label in dataset.classes], dtype=np.float32)
    weights = values.sum() / np.maximum(values, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def classifier_param_ids(model: nn.Module) -> set[int]:
    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
        if isinstance(classifier, nn.Module):
            return {id(param) for param in classifier.parameters()}
    head = getattr(model, "head", None)
    if isinstance(head, nn.Module):
        return {id(param) for param in head.parameters()}
    return set()


def make_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    weight_decay = float(get_nested(config, "train.weight_decay", 1e-4))
    base_lr = float(get_nested(config, "train.lr", 3e-4))
    head_lr = float(get_nested(config, "train.head_lr", base_lr))
    backbone_lr = float(get_nested(config, "train.backbone_lr", base_lr))
    head_ids = classifier_param_ids(model)

    if head_ids:
        head_params = [param for param in model.parameters() if id(param) in head_ids]
        backbone_params = [param for param in model.parameters() if id(param) not in head_ids]
        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
                {"params": head_params, "lr": head_lr, "name": "head"},
            ],
            weight_decay=weight_decay,
        )

    return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    head_ids = classifier_param_ids(model)
    if not head_ids:
        return
    for param in model.parameters():
        param.requires_grad = (not frozen) or id(param) in head_ids


def progress_line(epoch: int, epochs: int, batch_idx: int, total_batches: int, loss: float) -> None:
    width = 28
    ratio = batch_idx / max(1, total_batches)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(f"\r[train] epoch {epoch}/{epochs} |{bar}| {batch_idx}/{total_batches} loss={loss:.4f}")
    sys.stderr.flush()
    if batch_idx >= total_batches:
        sys.stderr.write("\n")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp_enabled: bool,
    grad_clip_norm: float = 0.0,
    epoch: int = 0,
    epochs: int = 0,
) -> tuple[float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")
    losses = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch_idx, (images, targets, _meta) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)

        if training:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            progress_line(epoch, epochs, batch_idx, len(loader), float(loss.detach().cpu()))

        losses.append(float(loss.detach().cpu()))
        preds = torch.argmax(logits.detach(), dim=1)
        y_true.extend(targets.detach().cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return float(np.mean(losses)) if losses else 0.0, y_true, y_pred


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_metric: float,
    config: dict,
    classes: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "classes": classes,
            "input_size": int(get_nested(config, "image.input_size", 384)),
            "model_name": str(get_nested(config, "model.name", "resnet18_small")),
            "config": config,
        },
        path,
    )


def evaluate_split(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, amp_enabled: bool, num_classes: int) -> dict:
    with torch.no_grad():
        loss, y_true, y_pred = run_epoch(model, loader, criterion, None, device, amp_enabled)
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    metrics = classification_metrics(matrix)
    metrics["loss"] = loss
    metrics["confusion_matrix"] = matrix.tolist()
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config.setdefault("project", {})["output_dir"] = args.output_dir
    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("train", {})["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config.setdefault("train", {})["num_workers"] = args.num_workers
    if args.input_size is not None:
        config.setdefault("image", {})["input_size"] = args.input_size
    if args.model is not None:
        config.setdefault("model", {})["name"] = args.model
    if args.no_pretrained:
        config.setdefault("model", {})["pretrained"] = False
    if args.train_csv is not None:
        config.setdefault("data", {})["train_csv"] = args.train_csv
    if args.val_csv is not None:
        config.setdefault("data", {})["val_csv"] = args.val_csv
    if args.test_csv is not None:
        config.setdefault("data", {})["test_csv"] = args.test_csv

    seed = int(get_nested(config, "train.seed", 42))
    set_seed(seed)
    output_dir = ensure_dir(get_nested(config, "project.output_dir", "distance_state_classifier/runs/default"))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    classes = list(get_nested(config, "data.classes", ["TooFar", "Good", "TooClose"]))
    input_size = int(get_nested(config, "image.input_size", 384))
    batch_size = int(get_nested(config, "train.batch_size", 16))
    num_workers = int(get_nested(config, "train.num_workers", 4))
    epochs = int(get_nested(config, "train.epochs", 60))
    amp_enabled = bool(get_nested(config, "train.amp", True))

    train_dataset = make_dataset(config, "train", training=True)
    val_dataset = make_dataset(config, "val", training=False)
    test_dataset = make_dataset(config, "test", training=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")

    model = build_model(
        name=str(get_nested(config, "model.name", "resnet18_small")),
        num_classes=len(classes),
        dropout=float(get_nested(config, "model.dropout", 0.2)),
        pretrained=bool(get_nested(config, "model.pretrained", False)),
    ).to(device)

    weights = make_class_weights(train_dataset, device) if bool(get_nested(config, "train.use_class_weights", True)) else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    start_epoch = 1
    best_metric = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metric = float(checkpoint.get("best_metric", -1.0))

    print("=" * 72)
    print("Distance-state classifier training")
    print(f"output_dir: {output_dir}")
    print(f"device: {device}")
    print(f"model: {get_nested(config, 'model.name', 'resnet18_small')}")
    print(f"input_size: {input_size}")
    print(f"classes: {classes}")
    print(f"train_counts: {class_counts(train_dataset)}")
    print(f"val_counts: {class_counts(val_dataset)}")
    print(f"test_counts: {class_counts(test_dataset)}")
    if weights is not None:
        print(f"class_weights: {[round(float(x), 4) for x in weights.detach().cpu()]}")
    print("optimizer_lrs:", {group.get("name", str(i)): group["lr"] for i, group in enumerate(optimizer.param_groups)})
    print("=" * 72)

    history = []
    patience = int(get_nested(config, "train.patience", 15))
    save_every = int(get_nested(config, "train.save_every", 10))
    stale_epochs = 0

    freeze_backbone_epochs = int(get_nested(config, "train.freeze_backbone_epochs", 0))
    grad_clip_norm = float(get_nested(config, "train.grad_clip_norm", 0.0))

    for epoch in range(start_epoch, epochs + 1):
        if freeze_backbone_epochs > 0 and classifier_param_ids(model):
            freeze_backbone = epoch <= freeze_backbone_epochs
            set_backbone_frozen(model, frozen=freeze_backbone)
            if epoch == 1:
                print(f"[freeze] timm backbone frozen for first {freeze_backbone_epochs} epochs")
            if epoch == freeze_backbone_epochs + 1:
                print("[freeze] backbone unfrozen")
        started = time.time()
        train_loss, train_true, train_pred = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            amp_enabled,
            grad_clip_norm=grad_clip_norm,
            epoch=epoch,
            epochs=epochs,
        )
        scheduler.step()
        train_matrix = confusion_matrix(train_true, train_pred, len(classes))
        train_metrics = classification_metrics(train_matrix)
        val_metrics = evaluate_split(model, val_loader, criterion, device, amp_enabled, len(classes))
        metric = float(val_metrics["macro_f1"])

        record = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "seconds": time.time() - started,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        }
        history.append(record)
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch, best_metric, config, classes)
        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, epoch, best_metric, config, classes)

        if metric > best_metric:
            best_metric = metric
            stale_epochs = 0
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch, best_metric, config, classes)
            write_confusion_matrix(output_dir / "val_confusion_matrix.csv", np.array(val_metrics["confusion_matrix"]), classes)
            print(f"[best] epoch={epoch} val_macro_f1={best_metric:.4f}")
        else:
            stale_epochs += 1

        (output_dir / "metrics.json").write_text(json.dumps({"history": history, "best_metric": best_metric}, indent=2), encoding="utf-8")
        if stale_epochs >= patience:
            print(f"[early-stop] no val macro-F1 improvement for {patience} epochs")
            break

    best_path = output_dir / "best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate_split(model, test_loader, criterion, device, amp_enabled, len(classes))
    write_confusion_matrix(output_dir / "test_confusion_matrix.csv", np.array(test_metrics["confusion_matrix"]), classes)
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    (output_dir / "class_mapping.json").write_text(json.dumps({label: idx for idx, label in enumerate(classes)}, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Training complete")
    print(f"best_checkpoint: {best_path}")
    print(f"test_accuracy: {test_metrics['accuracy']:.4f}")
    print(f"test_macro_f1: {test_metrics['macro_f1']:.4f}")
    print(f"test_balanced_accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()

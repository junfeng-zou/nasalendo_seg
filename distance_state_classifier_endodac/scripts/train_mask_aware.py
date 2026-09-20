#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
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

from distance_state_classifier_endodac.src.config import ensure_dir, get_nested, load_config, merge_config_overrides
from distance_state_classifier_endodac.src.mask_dataset import MaskAwareDistanceStateDataset
from distance_state_classifier_endodac.src.dataset import class_counts
from distance_state_classifier_endodac.src.depth_encoder_classifier import make_optimizer_param_groups, parameter_counts, set_trainable_scope
from distance_state_classifier_endodac.src.metrics import classification_metrics, confusion_matrix, write_confusion_matrix
from distance_state_classifier_endodac.src.model import build_model
from distance_state_classifier_endodac.src.transforms import CropBox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the mask-aware EndoDAC distance-state classifier.")
    parser.add_argument("--config", default="distance_state_classifier_endodac/configs/distance_state_endodac_maskaware_config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-checkpoint", default=None, help="Initialize compatible model weights only.")
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


def make_dataset(config: dict, split: str, training: bool) -> MaskAwareDistanceStateDataset:
    return MaskAwareDistanceStateDataset(
        csv_path=get_nested(config, f"data.{split}_csv"),
        classes=list(get_nested(config, "data.classes", ["TooFar", "Good", "TooClose"])),
        image_column=str(get_nested(config, "data.image_column", "image_path")),
        label_column=str(get_nested(config, "data.label_column", "label")),
        input_size=int(get_nested(config, "image.input_size", 392)),
        training=training,
        mask_cache_dir=get_nested(config, "mask.cache_dir", "distance_state_classifier_endodac/mask_cache/yolo11s_seg_formal"),
        augmentation_cfg=dict(get_nested(config, "augmentation", {})),
        normalize_cfg=dict(get_nested(config, "image.normalize", {"mode": "imagenet"})),
        crop=make_crop(config),
        missing_mask_policy=str(get_nested(config, "mask.missing_policy", "error")),
    )


def make_class_weights(dataset: MaskAwareDistanceStateDataset, device: torch.device, config: dict) -> torch.Tensor:
    configured = get_nested(config, "train.class_weights", None)
    if configured is not None:
        if isinstance(configured, dict):
            values = np.array([float(configured[label]) for label in dataset.classes], dtype=np.float32)
        else:
            values = np.array(list(configured), dtype=np.float32)
        return torch.tensor(values, dtype=torch.float32, device=device)
    counts = class_counts(dataset)
    values = np.array([counts[label] for label in dataset.classes], dtype=np.float32)
    weights = values.sum() / np.maximum(values, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


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

    for batch_idx, (images, masks, targets, _meta) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled and device.type == "cuda"):
            logits = model(images, masks)
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
        y_true.extend(targets.detach().cpu().tolist())
        y_pred.extend(torch.argmax(logits.detach(), dim=1).cpu().tolist())
    return float(np.mean(losses)) if losses else 0.0, y_true, y_pred


def evaluate_split(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, amp_enabled: bool, classes: list[str]) -> dict:
    with torch.no_grad():
        loss, y_true, y_pred = run_epoch(model, loader, criterion, None, device, amp_enabled)
    matrix = confusion_matrix(y_true, y_pred, len(classes))
    metrics = classification_metrics(matrix, classes)
    metrics["loss"] = loss
    metrics["confusion_matrix"] = matrix.tolist()
    return metrics


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int, best_metric: float, config: dict, classes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_cfg = dict(get_nested(config, "model", {}))
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "classes": classes,
            "input_size": int(get_nested(config, "image.input_size", 392)),
            "model_name": str(model_cfg.get("name", "endodac_mask_aware_classifier")),
            "feature_dim": getattr(model, "feature_dim", model_cfg.get("feature_dim")),
            "config": config,
        },
        path,
    )


def apply_epoch_trainable_scope(model: nn.Module, config: dict, epoch: int) -> dict[str, int]:
    model_cfg = dict(get_nested(config, "model", {}))
    final_scope = str(model_cfg.get("trainable_scope", "adapter"))
    freeze_epochs = int(get_nested(config, "train.freeze_encoder_epochs", 5))
    last_blocks = int(model_cfg.get("last_blocks", 2))
    if freeze_epochs > 0 and epoch <= freeze_epochs:
        return set_trainable_scope(model, "head_only", last_blocks=last_blocks)
    return set_trainable_scope(model, final_scope, last_blocks=last_blocks)


def load_compatible_init(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    current_state = model.state_dict()
    compatible = {key: value for key, value in checkpoint["model_state"].items() if key in current_state and current_state[key].shape == value.shape}
    skipped = len(checkpoint["model_state"]) - len(compatible)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if missing:
        print(f"[init] missing model keys after compatible load: {len(missing)}")
    if unexpected:
        print(f"[init] unexpected model keys after compatible load: {len(unexpected)}")
    if skipped:
        print(f"[init] skipped shape-mismatched or unknown keys: {skipped}")
    print(f"[init] loaded compatible weights from {checkpoint_path}")


def main() -> None:
    args = parse_args()
    config = merge_config_overrides(load_config(args.config), args)
    set_seed(int(get_nested(config, "train.seed", 42)))

    output_dir = ensure_dir(get_nested(config, "project.output_dir", "distance_state_classifier_endodac/runs/maskaware_default"))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    classes = list(get_nested(config, "data.classes", ["TooFar", "Good", "TooClose"]))
    batch_size = int(get_nested(config, "train.batch_size", 8))
    num_workers = int(get_nested(config, "train.num_workers", 0))
    epochs = int(get_nested(config, "train.epochs", 60))
    amp_enabled = bool(get_nested(config, "train.amp", True))

    train_dataset = make_dataset(config, "train", training=True)
    val_dataset = make_dataset(config, "val", training=False)
    test_dataset = make_dataset(config, "test", training=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")

    model_cfg = dict(get_nested(config, "model", {}))
    model = build_model(
        name=str(model_cfg.get("name", "endodac_mask_aware_classifier")),
        num_classes=len(classes),
        dropout=float(model_cfg.get("dropout", 0.25)),
        pretrained=bool(model_cfg.get("pretrained", False)),
        config=model_cfg,
        input_size=int(get_nested(config, "image.input_size", 392)),
    ).to(device)
    if args.init_checkpoint:
        load_compatible_init(model, args.init_checkpoint)

    final_scope = str(model_cfg.get("trainable_scope", "adapter"))
    set_trainable_scope(model, final_scope, last_blocks=int(model_cfg.get("last_blocks", 2)))
    optimizer = torch.optim.AdamW(
        make_optimizer_param_groups(
            model,
            head_lr=float(get_nested(config, "train.head_lr", 3e-4)),
            encoder_lr=float(get_nested(config, "train.encoder_lr", 5e-6)),
            adapter_lr=float(get_nested(config, "train.adapter_lr", 1e-5)),
        ),
        weight_decay=float(get_nested(config, "train.weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    weights = make_class_weights(train_dataset, device, config) if bool(get_nested(config, "train.use_class_weights", True)) else None
    criterion = nn.CrossEntropyLoss(weight=weights)

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
    print("Mask-aware EndoDAC distance-state classifier training")
    print(f"output_dir: {output_dir}")
    print(f"device: {device}")
    print(f"model: {model_cfg.get('name', 'endodac_mask_aware_classifier')}")
    print(f"mask_cache: {get_nested(config, 'mask.cache_dir')}")
    print(f"classes: {classes}")
    print(f"train_counts: {class_counts(train_dataset)}")
    print(f"val_counts: {class_counts(val_dataset)}")
    print(f"test_counts: {class_counts(test_dataset)}")
    print(f"param_counts: {parameter_counts(model)}")
    print("optimizer_lrs:", {group.get("name", str(i)): group["lr"] for i, group in enumerate(optimizer.param_groups)})
    if weights is not None:
        print(f"class_weights: {[round(float(x), 4) for x in weights.detach().cpu()]}")
    print("=" * 72)

    history = []
    patience = int(get_nested(config, "train.patience", 15))
    save_every = int(get_nested(config, "train.save_every", 10))
    grad_clip_norm = float(get_nested(config, "train.grad_clip_norm", 1.0))
    stale_epochs = 0
    last_scope_frozen = None

    for epoch in range(start_epoch, epochs + 1):
        counts = apply_epoch_trainable_scope(model, config, epoch)
        frozen_now = epoch <= int(get_nested(config, "train.freeze_encoder_epochs", 5))
        if frozen_now != last_scope_frozen:
            phase = "head_only" if frozen_now else final_scope
            print(f"[trainable] epoch={epoch} scope={phase} counts={counts}")
            last_scope_frozen = frozen_now

        started = time.time()
        train_loss, train_true, train_pred = run_epoch(model, train_loader, criterion, optimizer, device, amp_enabled, grad_clip_norm, epoch, epochs)
        scheduler.step()
        train_metrics = classification_metrics(confusion_matrix(train_true, train_pred, len(classes)), classes)
        val_metrics = evaluate_split(model, val_loader, criterion, device, amp_enabled, classes)
        metric = float(val_metrics["macro_f1"])

        record = {
            "epoch": epoch,
            "lr": [float(lr) for lr in scheduler.get_last_lr()],
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
        print(f"[epoch {epoch:03d}] train_loss={train_loss:.4f} train_f1={train_metrics['macro_f1']:.4f} val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}")

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
        model.load_state_dict(torch.load(best_path, map_location="cpu")["model_state"])
    test_metrics = evaluate_split(model, test_loader, criterion, device, amp_enabled, classes)
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

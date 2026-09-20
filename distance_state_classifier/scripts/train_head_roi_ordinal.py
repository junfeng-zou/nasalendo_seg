#!/usr/bin/env python3
"""Train ConvNeXt with fixed-scale local input, spatial features and ordered logits."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from distance_state_classifier.src.config import load_config, resolve_path
from distance_state_classifier.src.convnext_roi_ordinal import ConvNeXtROIOrdinal, ordered_log_probabilities, ordinal_loss
from distance_state_classifier.src.transforms import augment_image, resize_and_normalize
from distance_state_classifier.scripts.prepare_head_roi import file_hash, write_json
from distance_state_classifier_endodac.src.metrics import classification_metrics

DEFAULT_CONFIG = "distance_state_classifier/configs/convnext_head_roi_ordinal.yaml"


class HeadRoiDataset(Dataset):
    def __init__(self, records, config, split):
        self.all_rows = [r for r in records if r["split"] == split]
        self.rows = [r for r in self.all_rows if r["geometry"]["valid"]] if split == "train" else self.all_rows
        self.classes, self.config, self.training = config["data"]["classes"], config, split == "train"
        self.class_to_idx = {label: i for i, label in enumerate(self.classes)}
        self._cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        valid = row["geometry"]["valid"]
        if valid:
            if index not in self._cache:
                image = cv2.imread(row["roi_path"])
                if image is None:
                    raise RuntimeError(row["roi_path"])
                self._cache[index] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self._cache[index]
            if self.training and self.config["augmentation"]["enabled"]:
                image = augment_image(image, self.config["augmentation"])
            image = torch.from_numpy(resize_and_normalize(image, self.config["image"]["input_size"]))
        else:
            image = torch.zeros(3, self.config["image"]["input_size"], self.config["image"]["input_size"])
        return image, self.class_to_idx[row["label"]], bool(valid), index


def summarize_predictions(truths, predictions, classes):
    # Last column is abstention due to unavailable localization, not a fourth
    # ground-truth class. Such frames remain in recall/accuracy denominators.
    matrix = np.zeros((len(classes), len(classes) + 1), dtype=np.int64)
    for target, pred in zip(truths, predictions):
        matrix[target, pred] += 1
    metrics = classification_metrics(matrix, classes)
    valid = np.asarray(predictions) < len(classes)
    metrics.update(n=len(truths), coverage=float(valid.mean()) if len(valid) else 0.0,
                   invalid_n=int((~valid).sum()), confusion_matrix=matrix.tolist(),
                   columns=[*classes, "Invalid"],
                   opposite_extreme_errors=sum({a, b} == {0, 2} for a, b in zip(truths, predictions)),
                   ordinal_mae_valid=float(np.abs(np.asarray(truths)[valid] - np.asarray(predictions)[valid]).mean()) if valid.any() else None)
    return metrics


def run_epoch(model, loader, device, weights, optimizer=None, scaler=None, amp=False, clip=1.0, max_steps=0):
    training = optimizer is not None
    model.train(training)
    truths, predictions, records = [], [], []
    loss_sum, valid_n, updates, skipped = 0.0, 0, 0, 0
    for step, (images, targets, valid, indices) in enumerate(loader, 1):
        valid = valid.bool()
        batch_predictions = torch.full_like(targets, 3)
        batch_probs = torch.zeros(len(targets), 3)
        if valid.any():
            x, y = images[valid].to(device), targets[valid].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training), torch.cuda.amp.autocast(enabled=amp):
                logits = model.cumulative_logits(x)
                loss = ordinal_loss(logits, y, weights)
                log_probs = ordered_log_probabilities(logits)
            if not torch.isfinite(loss) or not torch.isfinite(log_probs).all():
                raise FloatingPointError(f"Non-finite outputs at step {step}")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                if torch.isfinite(norm):
                    updates += 1
                elif amp:
                    skipped += 1
                else:
                    raise FloatingPointError("Non-finite gradient without AMP")
                scaler.step(optimizer)
                scaler.update()
            batch_predictions[valid] = log_probs.detach().argmax(-1).cpu()
            batch_probs[valid] = log_probs.detach().exp().cpu()
            loss_sum += float(loss.detach()) * int(valid.sum())
            valid_n += int(valid.sum())
        truths.extend(targets.tolist())
        predictions.extend(batch_predictions.tolist())
        for i, index in enumerate(indices.tolist()):
            row = loader.dataset.rows[index]
            records.append({"sample_id": row["sample_id"], "image_path": row["image_path"], "label": row["label"],
                            "prediction": loader.dataset.classes[batch_predictions[i]] if batch_predictions[i] < 3 else "Invalid",
                            "roi_valid": bool(valid[i]), "probabilities": batch_probs[i].tolist() if valid[i] else None})
        if training and (step % 15 == 0 or step == len(loader)):
            print(f"[batch {step}/{len(loader)}] loss={loss_sum/max(valid_n,1):.4f}", flush=True)
        if max_steps and step >= max_steps:
            break
    if training and not updates:
        raise FloatingPointError("No successful optimizer updates")
    metrics = summarize_predictions(truths, predictions, loader.dataset.classes)
    metrics["loss_valid"] = loss_sum / max(valid_n, 1)
    if training:
        metrics.update(optimizer_updates=updates, amp_skipped=skipped, grad_scale=scaler.get_scale())
    return metrics, records


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def save_checkpoint(path, model, optimizer, scheduler, scaler, config, audit, epoch, best, stale, history):
    payload = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
               "scaler_state": scaler.state_dict(), "config": config, "data_audit": audit, "epoch": epoch,
               "best_metric": best, "stale_epochs": stale, "history": history, "rng_state": rng_state(),
               "classes": config["data"]["classes"], "input_size": config["image"]["input_size"], "model_name": config["model"]["name"]}
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def execute(args, config, output):
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    seed = config["train"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    manifest_path = resolve_path(config["roi"]["cache_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["audit"]["roi_config"] != config["roi"] or manifest["audit"]["input_size"] != config["image"]["input_size"]:
        raise ValueError("ROI cache protocol does not match training config")
    for split, digest in manifest["audit"]["csv_sha256"].items():
        if file_hash(resolve_path(config["data"][f"{split}_csv"])) != digest:
            raise ValueError(f"CSV changed after ROI preparation: {split}")
    for path, digest in manifest["audit"]["source_sha256"].items():
        if file_hash(ROOT / path) != digest:
            raise ValueError(f"ROI implementation changed: {path}")
    audit = copy.deepcopy(manifest["audit"])
    audit.update(manifest_sha256=file_hash(manifest_path),
                 baseline_sha256=file_hash(resolve_path(config["experiment"]["baseline_checkpoint"])),
                 pretrained_sha256=file_hash(resolve_path(config["model"]["pretrained_file"])))
    datasets = [HeadRoiDataset(manifest["records"], config, split) for split in ("train", "val", "test")]
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    cfg, train_cfg = config["model"], config["train"]
    model = ConvNeXtROIOrdinal(cfg["hidden_dim"], cfg["grid_size"], cfg["dropout"], resolve_path(cfg["pretrained_file"])).to(device)
    optimizer = torch.optim.AdamW(model.optimizer_groups(train_cfg["backbone_lr"], train_cfg["head_lr"]), weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    amp = bool(train_cfg["amp"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # Use the original training-label frequencies, including the unavailable ROI,
    # so the class weights do not change due to localization filtering.
    counts = Counter(row["label"] for row in datasets[0].all_rows)
    weights = np.array([len(datasets[0].all_rows) / counts[label] for label in datasets[0].classes], np.float32)
    weights = torch.tensor(weights / weights.mean(), device=device)
    if not train_cfg["use_class_weights"]:
        weights = None
    if train_cfg["num_workers"] != 0:
        raise ValueError("This protocol uses num_workers=0 for reproducible cached loading")
    loaders = [DataLoader(ds, batch_size=train_cfg["batch_size"], shuffle=i == 0, num_workers=0,
                          pin_memory=device.type == "cuda") for i, ds in enumerate(datasets)]
    history, best, stale, start = [], -1.0, 0, 1
    if args.resume:
        checkpoint = torch.load(resolve_path(args.resume), map_location="cpu", weights_only=True)
        if checkpoint["config"] != config or checkpoint["data_audit"] != audit:
            raise ValueError("Resume config/data do not match")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        history, best, stale, start = checkpoint["history"], checkpoint["best_metric"], checkpoint["stale_epochs"], checkpoint["epoch"] + 1
        state = checkpoint["rng_state"]
        random.setstate(state["python"])
        np.random.set_state((state["numpy"][0], np.asarray(state["numpy"][1], dtype=np.uint32), *state["numpy"][2:]))
        torch.set_rng_state(state["torch"])
        if device.type == "cuda" and state["cuda"]:
            torch.cuda.set_rng_state_all(state["cuda"])
    write_json(output / "resolved_config.json", config)
    write_json(output / "data_audit.json", audit)
    runtime = {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "torch": str(torch.__version__), "status": "running", "smoke_steps": args.smoke_steps,
               "class_weights": weights.tolist() if weights is not None else None,
               "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in (
                   Path(__file__).resolve(), ROOT / "distance_state_classifier/src/convnext_roi_ordinal.py")}}
    write_json(output / "runtime.json", runtime)
    print(f"[train] device={device} train/val/test={[len(ds) for ds in datasets]} weights={runtime['class_weights']}", flush=True)
    if args.smoke_steps:
        result, _ = run_epoch(model, loaders[0], device, weights, optimizer, scaler, amp, train_cfg["grad_clip_norm"], args.smoke_steps)
        write_json(output / "smoke_metrics.json", result)
        save_checkpoint(output / "smoke.pt", model, optimizer, scheduler, scaler, config, audit, 0, best, stale, history)
        runtime["status"] = "smoke_complete"
        write_json(output / "runtime.json", runtime)
        print(f"[smoke complete] {result}", flush=True)
        return
    for epoch in range(start, train_cfg["epochs"] + 1):
        if stale >= train_cfg["patience"]:
            break
        model.freeze_backbone(epoch <= train_cfg["freeze_backbone_epochs"])
        begin = time.monotonic()
        print(f"[epoch {epoch}] trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}", flush=True)
        train_metrics, _ = run_epoch(model, loaders[0], device, weights, optimizer, scaler, amp, train_cfg["grad_clip_norm"])
        scheduler.step()
        val_metrics, _ = run_epoch(model, loaders[1], device, weights, amp=amp)
        improved = val_metrics["macro_f1"] > best
        best, stale = (val_metrics["macro_f1"], 0) if improved else (best, stale + 1)
        history.append({"epoch": epoch, "seconds": time.monotonic() - begin, "lr": scheduler.get_last_lr(), "train": train_metrics, "val": val_metrics})
        if improved:
            save_checkpoint(output / "best.pt", model, optimizer, scheduler, scaler, config, audit, epoch, best, stale, history)
            write_json(output / "best_val_metrics.json", val_metrics)
        save_checkpoint(output / "last.pt", model, optimizer, scheduler, scaler, config, audit, epoch, best, stale, history)
        if train_cfg["save_every"] > 0 and epoch % train_cfg["save_every"] == 0:
            save_checkpoint(output / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, scaler, config, audit, epoch, best, stale, history)
        write_json(output / "metrics.json", {"history": history, "best_metric": best, "stale_epochs": stale})
        print(f"[epoch {epoch} complete] seconds={history[-1]['seconds']:.1f} train_f1={train_metrics['macro_f1']:.4f} val_f1={val_metrics['macro_f1']:.4f} best={best:.4f} stale={stale}", flush=True)
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    test_metrics, predictions = run_epoch(model, loaders[2], device, weights, amp=False)
    write_json(output / "test_metrics.json", test_metrics)
    write_json(output / "test_predictions.json", predictions)
    write_json(output / "completion.json", {"status": "complete", "best_epoch": checkpoint["epoch"], "last_epoch": history[-1]["epoch"],
                                           "best_val_macro_f1": best, "test": test_metrics})
    runtime["status"] = "complete"
    write_json(output / "runtime.json", runtime)
    print(f"[complete] best_epoch={checkpoint['epoch']} test={test_metrics}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.threads < 1 or args.smoke_steps < 0 or (args.smoke_steps and args.resume):
        parser.error("Invalid threads/smoke/resume arguments")
    config = load_config(args.config)
    if args.output_dir:
        config["project"]["output_dir"] = args.output_dir
    output = resolve_path(config["project"]["output_dir"])
    if output.resolve() == resolve_path(config["experiment"]["baseline_checkpoint"]).resolve().parent:
        raise ValueError("Cannot overwrite the baseline directory")
    if output.exists() and (list(output.glob("*.pt")) or (output / "metrics.json").exists()) and not args.resume:
        raise FileExistsError(output)
    if args.resume and (resolve_path(args.resume).resolve().parent != output.resolve() or not (output / "best.pt").exists()):
        raise ValueError("Resume needs this run's checkpoint and best.pt")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".training.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    try:
        execute(args, config, output)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

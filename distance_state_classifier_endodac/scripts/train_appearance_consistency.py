#!/usr/bin/env python3
"""Experiment B: original/appearance-paired CE + JS, with fixed A architecture."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from distance_state_classifier_endodac.scripts.train import (
    apply_epoch_trainable_scope, make_class_weights, make_crop, make_dataset, set_seed,
)
from distance_state_classifier_endodac.src.appearance_consistency import (
    AppearancePairDataset, appearance_regions, augment_appearance, paired_objective,
)
from distance_state_classifier_endodac.src.config import get_nested, load_config, resolve_path
from distance_state_classifier_endodac.src.dataset import class_counts
from distance_state_classifier_endodac.src.depth_encoder_classifier import make_optimizer_param_groups, set_trainable_scope
from distance_state_classifier_endodac.src.metrics import classification_metrics, confusion_matrix, write_confusion_matrix
from distance_state_classifier_endodac.src.model import build_model


DEFAULT_CONFIG = "distance_state_classifier_endodac/configs/distance_state_endodac_experiment_b.yaml"


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_dataset(config):
    return AppearancePairDataset(
        csv_path=config["data"]["train_csv"], classes=config["data"]["classes"],
        image_column=config["data"]["image_column"], label_column=config["data"]["label_column"],
        input_size=config["image"]["input_size"], training=True,
        augmentation_cfg=config["augmentation"], normalize_cfg=config["image"]["normalize"],
        crop=make_crop(config), appearance_cfg=config["appearance_consistency"],
    )


def preflight(config, train, val, test, output):
    baseline_path = resolve_path(config["experiment"]["baseline_checkpoint"])
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
    reference = baseline["config"]
    # These fields define the controlled comparison. B only changes the training
    # views and objective. A's checkpoint is recorded, never used to warm-start B.
    for section in ("data", "image", "model", "train", "inference"):
        if config[section] != reference[section]:
            raise ValueError(f"Experiment B must match the saved A config: {section}")
    audit = {"baseline_checkpoint": str(baseline_path), "baseline_sha256": sha256(baseline_path),
             "initialization": "Configured EndoDAC pretrained weights + new classification head; no A warm-start",
             "nominal_input": [392, 392], "encoder_input": config["model"]["encoder"]["image_shape"],
             "splits": {}, "train_empty_masks": [], "train_empty_foreground_interiors": [],
             "train_empty_fov": []}
    seen_paths, seen_videos = set(), set()
    for name, dataset in (("train", train), ("val", val), ("test", test)):
        paths = {str(resolve_path(row[dataset.image_column])) for row in dataset.rows}
        videos = {row["video_id"] for row in dataset.rows}
        if len(paths) != len(dataset) or seen_paths & paths or seen_videos & videos:
            raise ValueError(f"Duplicate image or overlapping video split: {name}")
        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        seen_paths |= paths
        seen_videos |= videos
        csv_path = resolve_path(config["data"][f"{name}_csv"])
        audit["splits"][name] = {"n": len(dataset), "counts": class_counts(dataset),
                                 "videos": sorted(videos), "csv": str(csv_path), "csv_sha256": sha256(csv_path)}
    mask_digest = hashlib.sha256()
    for index in range(len(train)):
        image, mask = train.inputs(index)  # Fails on missing/unreadable/misaligned masks.
        mask_digest.update(mask.tobytes())
        regions = appearance_regions(image, mask, config["appearance_consistency"]["guard_px"])
        sample = train.rows[index].get("sample_id", str(index))
        if not mask.any():
            audit["train_empty_masks"].append(sample)
        if not regions["fg"].any():
            audit["train_empty_foreground_interiors"].append(sample)
        if not regions["fov"].any():
            audit["train_empty_fov"].append(sample)
        if (index + 1) % 200 == 0:
            print(f"[preflight] checked {index + 1}/{len(train)} training images/masks", flush=True)
    audit["resized_train_masks_sha256"] = mask_digest.hexdigest()
    for key in ("pretrained_path", "checkpoint"):
        asset = resolve_path(config["model"]["encoder"][key])
        if asset.is_file():
            audit[key + "_sha256"] = sha256(asset)
    asset = resolve_path(config["model"]["encoder"]["pretrained_path"]) / "depth_anything_vitb14.pth"
    audit["foundation_sha256"] = sha256(asset)
    write_json(output / "data_audit.json", audit)
    preview(output, train, config["appearance_consistency"])
    return audit


def preview(output, dataset, cfg):
    directory = output / "preview"
    directory.mkdir(exist_ok=True)
    rows, counts = [], {}
    for index, row in enumerate(dataset.rows):
        group = (row.get("video_id", ""), row[dataset.label_column])
        if counts.get(group, 0) >= 1:
            continue
        counts[group] = 1
        original, mask = dataset.inputs(index)
        images = [original]
        for seed in (42, 137, 2026):
            image, _ = augment_appearance(original, mask, cfg, np.random.default_rng(seed + index))
            images.append(image)
        overlay = original.copy()
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
        images.append(overlay)
        cells = []
        for variant, image in enumerate(images):
            filename = f"{index:04d}_{variant}.png"
            if not cv2.imwrite(str(directory / filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Failed to write {filename}")
            cells.append(f'<td><img width="220" src="preview/{filename}"></td>')
        rows.append(f'<tr><td>{html.escape(str(group))}<br>{html.escape(row.get("sample_id", ""))}</td>{"".join(cells)}</tr>')
    page = '<!doctype html><meta charset="utf-8"><title>实验 B：训练外观增强</title>'
    page += '<style>body{font-family:sans-serif}td,th{padding:8px;vertical-align:top}img{max-width:18vw}</style>'
    page += '<h1>实验 B：训练外观增强预览</h1><p>仅训练集，每个视频/标签一例。保持 392×392 画布及器械轮廓；黑边和轮廓保护带保持原像素。mask 仅用于训练增强。</p>'
    page += '<table><tr><th>样本</th><th>原图</th><th>增强 1</th><th>增强 2</th><th>增强 3</th><th>mask 轮廓</th></tr>' + ''.join(rows) + '</table>'
    (output / "augmentation_preview.html").write_text(page, encoding="utf-8")


def run_epoch(model, loader, criterion, device, *, optimizer=None, scaler=None,
              consistency_weight=1.0, grad_clip=1.0, amp=False, max_steps=0):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3, dtype=np.float64)
    truths, predictions = [], []
    optimizer_steps, skipped_steps = 0, 0
    for step, (images, targets, _) in enumerate(loader, 1):
        images, targets = images.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.cuda.amp.autocast(enabled=amp):
            if training:
                logits = model(images.flatten(0, 1)).reshape(len(targets), 2, -1)
                loss, ce, js = paired_objective(logits[:, 0], logits[:, 1], targets, criterion, consistency_weight)
                clean_logits = logits[:, 0]
            else:
                clean_logits = model(images)
                loss = ce = criterion(clean_logits, targets)
                js = loss.new_zeros(())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at batch {step}")
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(norm):
                if not amp:
                    raise FloatingPointError(f"Non-finite gradient at batch {step}")
                print(f"[amp] non-finite gradient at batch {step}; GradScaler will skip this step", flush=True)
                skipped_steps += 1
            else:
                optimizer_steps += 1
            scaler.step(optimizer)
            scaler.update()
        totals += np.array([float(loss.detach()), float(ce.detach()), float(js.detach())]) * len(targets)
        truths.extend(targets.cpu().tolist())
        predictions.extend(clean_logits.detach().argmax(-1).cpu().tolist())
        if training and (step % 30 == 0 or step == len(loader)):
            print(f"[batch {step}/{len(loader)}] loss={float(loss.detach()):.4f}", flush=True)
        if max_steps and step >= max_steps:
            break
    metrics = classification_metrics(confusion_matrix(truths, predictions, len(loader.dataset.classes)), loader.dataset.classes)
    metrics.update(zip(("loss", "ce", "js"), (totals / max(len(truths), 1)).tolist()))
    metrics["confusion_matrix"] = confusion_matrix(truths, predictions, len(loader.dataset.classes)).tolist()
    metrics["n"] = len(truths)
    if training:
        metrics.update(optimizer_steps=optimizer_steps, skipped_steps=skipped_steps, grad_scale=scaler.get_scale())
        if optimizer_steps == 0:
            raise FloatingPointError("No optimizer updates completed in this training epoch")
    return metrics


def checkpoint_payload(model, optimizer, scheduler, scaler, config, epoch, best, stale, history, audit):
    numpy_state = np.random.get_state()
    return {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
            "epoch": epoch, "best_metric": best, "stale_epochs": stale, "history": history,
            "config": config, "classes": config["data"]["classes"], "input_size": config["image"]["input_size"],
            "model_name": config["model"]["name"], "backbone": config["model"]["backbone"],
            "feature_dim": model.feature_dim, "pool_type": model.pool_type,
            "normalize_mode": config["image"]["normalize"]["mode"], "data_audit": audit,
            "rng": {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                    "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}


def save_checkpoint(path, payload):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_main(args, config, output):
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    set_seed(config["train"]["seed"])
    train, val, test = pair_dataset(config), make_dataset(config, "val", False), make_dataset(config, "test", False)
    audit = preflight(config, train, val, test, output)
    write_json(output / "resolved_config.json", config)
    if args.preflight_only:
        print(f"[preflight complete] {output / 'augmentation_preview.html'}", flush=True)
        return
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process; no automatic CPU fallback for --device cuda")
    cfg, train_cfg = config["model"], config["train"]
    model = build_model(cfg["name"], len(train.classes), cfg["dropout"], cfg["pretrained"], cfg, config["image"]["input_size"]).to(device)
    set_trainable_scope(model, cfg["trainable_scope"], last_blocks=cfg["last_blocks"])
    groups = make_optimizer_param_groups(model, train_cfg["head_lr"], train_cfg["encoder_lr"], train_cfg["adapter_lr"])
    optimizer = torch.optim.AdamW(groups, weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    amp = bool(train_cfg["amp"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    weights = make_class_weights(train, device, config) if train_cfg["use_class_weights"] else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    if train_cfg["num_workers"] != 0:
        raise ValueError("Use num_workers=0 for the fixed B protocol and reproducible epoch-boundary resume")
    loaders = [DataLoader(ds, batch_size=train_cfg["batch_size"], shuffle=i == 0, num_workers=0,
                          pin_memory=device.type == "cuda") for i, ds in enumerate((train, val, test))]
    history, best, stale, start = [], -1.0, 0, 1
    if args.resume:
        checkpoint = torch.load(resolve_path(args.resume), map_location="cpu", weights_only=True)
        if checkpoint["config"] != config or checkpoint["data_audit"] != audit:
            raise ValueError("Resume config/data differ from checkpoint")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        history, best, stale, start = checkpoint["history"], checkpoint["best_metric"], checkpoint["stale_epochs"], checkpoint["epoch"] + 1
        state = checkpoint["rng"]
        random.setstate(state["python"])
        np.random.set_state((state["numpy"][0], np.array(state["numpy"][1], dtype=np.uint32), *state["numpy"][2:]))
        torch.set_rng_state(state["torch"])
        if device.type == "cuda" and state["cuda"]:
            torch.cuda.set_rng_state_all(state["cuda"])
    runtime = {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "torch": str(torch.__version__), "smoke_steps": args.smoke_steps, "status": "running",
               "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
                   Path(__file__).resolve(), ROOT / "distance_state_classifier_endodac/src/appearance_consistency.py")}}
    write_json(output / "runtime.json", runtime)
    print(f"[train] device={device} paired_batch={train_cfg['batch_size']}x2 epochs={train_cfg['epochs']} output={output}", flush=True)
    if args.smoke_steps:
        # Exercise adapter backward as well as the head. This checkpoint must not
        # be interpreted as a trained B result; no validation/test-set selection.
        smoke = run_epoch(model, loaders[0], criterion, device, optimizer=optimizer, scaler=scaler,
                          consistency_weight=config["appearance_consistency"]["weight"],
                          grad_clip=train_cfg["grad_clip_norm"], amp=amp, max_steps=args.smoke_steps)
        write_json(output / "smoke_metrics.json", smoke)
        save_checkpoint(output / "smoke.pt", checkpoint_payload(model, optimizer, scheduler, scaler, config, 0, best, stale, history, audit))
        runtime["status"] = "smoke_complete"
        write_json(output / "runtime.json", runtime)
        print(f"[smoke complete] {smoke}", flush=True)
        return
    for epoch in range(start, train_cfg["epochs"] + 1):
        if stale >= train_cfg["patience"]:
            break
        counts = apply_epoch_trainable_scope(model, config, epoch)
        started = time.monotonic()
        print(f"[epoch {epoch}] parameters={counts}", flush=True)
        training = run_epoch(model, loaders[0], criterion, device, optimizer=optimizer, scaler=scaler,
                             consistency_weight=config["appearance_consistency"]["weight"],
                             grad_clip=train_cfg["grad_clip_norm"], amp=amp)
        scheduler.step()
        validation = run_epoch(model, loaders[1], criterion, device, amp=amp)
        improved = validation["macro_f1"] > best
        best, stale = (validation["macro_f1"], 0) if improved else (best, stale + 1)
        history.append({"epoch": epoch, "seconds": time.monotonic() - started,
                        "lr": scheduler.get_last_lr(), "train": training, "val": validation})
        payload = checkpoint_payload(model, optimizer, scheduler, scaler, config, epoch, best, stale, history, audit)
        if improved:
            save_checkpoint(output / "best.pt", payload)
            write_confusion_matrix(output / "val_confusion_matrix.csv", np.array(validation["confusion_matrix"]), train.classes)
        save_checkpoint(output / "last.pt", payload)
        if train_cfg["save_every"] > 0 and epoch % train_cfg["save_every"] == 0:
            save_checkpoint(output / f"epoch_{epoch:03d}.pt", payload)
        write_json(output / "metrics.json", {"history": history, "best_metric": best, "stale_epochs": stale})
        print(f"[epoch {epoch} complete] seconds={history[-1]['seconds']:.1f} train_ce={training['ce']:.4f} train_js={training['js']:.4f} val_f1={validation['macro_f1']:.4f} best={best:.4f} stale={stale}", flush=True)
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    # Only now evaluate the test set; it never selects epochs or hyperparameters.
    test_metrics = run_epoch(model, loaders[2], criterion, device, amp=False)
    write_json(output / "test_metrics.json", test_metrics)
    write_confusion_matrix(output / "test_confusion_matrix.csv", np.array(test_metrics["confusion_matrix"]), train.classes)
    write_json(output / "class_mapping.json", train.class_to_idx)
    write_json(output / "completion.json", {"status": "complete", "best_epoch": checkpoint["epoch"],
                                           "last_epoch": history[-1]["epoch"], "best_val_macro_f1": best,
                                           "test": test_metrics})
    runtime["status"] = "complete"
    write_json(output / "runtime.json", runtime)
    print(f"[complete] best_epoch={checkpoint['epoch']} test={test_metrics}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.threads < 1 or args.smoke_steps < 0 or (args.resume and (args.smoke_steps or args.preflight_only)):
        parser.error("Invalid threads/smoke/resume combination")
    config = load_config(args.config)
    if args.output_dir:
        config["project"]["output_dir"] = args.output_dir
    output = resolve_path(config["project"]["output_dir"])
    baseline_dir = resolve_path(config["experiment"]["baseline_checkpoint"]).parent
    if output.resolve() == baseline_dir.resolve():
        raise ValueError("Cannot write experiment B into the baseline directory")
    if output.exists() and (list(output.glob("*.pt")) or (output / "metrics.json").exists()) and not args.resume:
        raise FileExistsError(f"Training artifacts already exist; use a new output directory or --resume: {output}")
    if args.resume and (resolve_path(args.resume).resolve().parent != output.resolve() or not (output / "best.pt").exists()):
        raise ValueError("Resume must use a checkpoint from this run directory, with its best.pt present")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".training.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        train_main(args, config, output)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare fixed-FOV head candidates, audit coverage and render train previews."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from distance_state_classifier.src.config import load_config, resolve_path
from distance_state_classifier.src.head_roi import crop_with_padding, estimate_fov, locate_roi
from distance_state_classifier_endodac.src.mask_utils import mask_cache_path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="distance_state_classifier/configs/convnext_head_roi_ordinal.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    cfg, data = config["roi"], config["data"]
    output = resolve_path(cfg["cache_dir"])
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    rows, groups, split_videos, paths_seen = [], defaultdict(list), {}, set()
    csv_hashes = {}
    for split in ("train", "val", "test"):
        path = resolve_path(data[f"{split}_csv"])
        csv_hashes[split] = file_hash(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        split_videos[split] = set()
        for row in source_rows:
            if row[data["label_column"]] not in data["classes"]:
                raise ValueError(f"Unexpected label in {path}")
            image_path = resolve_path(row[data["image_column"]]).resolve()
            if str(image_path) in paths_seen:
                raise ValueError(f"Repeated image: {image_path}")
            paths_seen.add(str(image_path))
            row = {**row, "split": split, "image_path": str(image_path)}
            rows.append(row)
            groups[row["video_id"]].append(row)
            split_videos[split].add(row["video_id"])
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_videos[first] & split_videos[second]:
            raise ValueError(f"Video overlap: {first}/{second}")
    calibration = {}
    for video, items in groups.items():
        initial = sorted(items, key=lambda row: int(row["frame_index"]))[:cfg["calibration_frames"]]
        estimates = []
        for row in initial:
            image = cv2.imread(row["image_path"])
            if image is None:
                raise RuntimeError(row["image_path"])
            estimates.append(estimate_fov(image))
        if len({tuple(item["shape"]) for item in estimates}) != 1:
            raise ValueError(f"Changing image dimensions: {video}")
        calibration[video] = {"center": np.median([e["center"] for e in estimates], axis=0).tolist(),
                              "diameter": float(np.median([e["diameter"] for e in estimates])),
                              "shape": estimates[0]["shape"], "calibration_sample_ids": [r["sample_id"] for r in initial]}
    preview_entries, preview_counts = [], Counter()
    (output / "preview").mkdir()
    records = []
    split_indices = Counter()
    for row in rows:
        split = row["split"]
        index = split_indices[split]
        split_indices[split] += 1
        image = cv2.imread(row["image_path"])
        if image is None:
            raise RuntimeError(row["image_path"])
        mask_path = mask_cache_path(cfg["mask_cache"], row["image_path"])
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != image.shape[:2]:
            raise ValueError(f"Invalid cached mask: {mask_path}")
        geometry = locate_roi(mask, calibration[row["video_id"]], cfg)
        roi = crop_with_padding(image, geometry["box"]) if "box" in geometry else np.zeros((384, 384, 3), np.uint8)
        relative = Path("rois") / split / f"{index:04d}.png"
        destination = output / relative
        if geometry["valid"]:
            destination.parent.mkdir(parents=True, exist_ok=True)
            resized = cv2.resize(roi, (config["image"]["input_size"],) * 2, interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(destination), resized):
                raise RuntimeError(destination)
        record = {**row, "mask_path": str(mask_path), "roi_path": str(destination) if geometry["valid"] else "",
                  "geometry": geometry, "image_sha256": file_hash(row["image_path"]), "mask_sha256": file_hash(mask_path)}
        records.append(record)
        group = (row["video_id"], row[data["label_column"]])
        # Geometry parameters are inspected on training data only, never chosen
        # using validation/test images or classification results.
        if split == "train" and (preview_counts[group] < 2 or not geometry["valid"]):
            preview_counts[group] += 1
            full = image.copy()
            contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(full, contours, -1, (0, 210, 0), 2)
            if "box" in geometry:
                x1, y1, x2, y2 = geometry["box"]
                cv2.rectangle(full, (x1, y1), (x2, y2), (0, 180, 255), 3)
                cv2.circle(full, tuple(np.rint(geometry["tip"]).astype(int)), 7, (255, 0, 255), -1)
            full_path, local_path = f"preview/{index:04d}_full.jpg", f"preview/{index:04d}_local.jpg"
            cv2.imwrite(str(output / full_path), cv2.resize(full, (520, round(520 * full.shape[0] / full.shape[1]))))
            cv2.imwrite(str(output / local_path), cv2.resize(roi, (384, 384)))
            preview_entries.append(f'<tr><td>{html.escape(str(group))}<br>{html.escape(row["sample_id"])}<br>{html.escape(json.dumps(geometry, ensure_ascii=False))}</td><td><img width="520" src="{full_path}"></td><td><img width="384" src="{local_path}"></td></tr>')
        if len(records) % 200 == 0:
            print(f"[prepare] {len(records)}/{len(rows)}", flush=True)
    audit = {"csv_sha256": csv_hashes, "roi_config": cfg, "input_size": config["image"]["input_size"],
             "calibration": calibration, "splits": {},
             "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in (
                 ROOT / "distance_state_classifier/src/head_roi.py", ROOT / "inference/realtime_improved_tip.py")},
             "localization": "Independent current-frame heuristic using cached instrument masks; no manual head ground truth"}
    for split in split_videos:
        selected = [r for r in records if r["split"] == split]
        audit["splits"][split] = {"total": len(selected), "valid": sum(r["geometry"]["valid"] for r in selected),
                                   "videos": sorted(split_videos[split]),
                                   "reasons": dict(Counter(r["geometry"]["reason"] for r in selected)),
                                   "by_class": {label: {"total": sum(r["label"] == label for r in selected),
                                                       "valid": sum(r["label"] == label and r["geometry"]["valid"] for r in selected)} for label in data["classes"]}}
    write_json(output / "audit.json", audit)
    write_json(output / "manifest.json", {"records": records, "audit": audit})
    page = '<!doctype html><meta charset="utf-8"><title>ConvNeXt 头部 ROI 审查</title><style>body{font-family:sans-serif}td{max-width:340px;padding:10px;vertical-align:top;overflow-wrap:anywhere}</style>'
    page += '<h1>训练集头部候选 ROI</h1><p>绿色：已有器械 mask；黄色：固定视野比例窗口；紫色：几何尖端。此预览不等于人工头部分割真值。</p>'
    page += '<table><tr><th>样本与定位信息</th><th>原图及 ROI</th><th>局部输入</th></tr>' + ''.join(preview_entries) + '</table>'
    (output / "preview.html").write_text(page, encoding="utf-8")
    print(json.dumps(audit["splits"], ensure_ascii=False, indent=2), flush=True)
    print(f"[complete] {output / 'preview.html'}", flush=True)


if __name__ == "__main__":
    main()

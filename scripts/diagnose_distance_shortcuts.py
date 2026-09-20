#!/usr/bin/env python3
"""Paired, offline appearance interventions for the distance-state classifiers.

No training or robot interface is used. Images and checkpoints are read-only.
Predictions use raw single-frame logits, without temporal smoothing.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CHECKPOINTS = [
    "distance_state_classifier/runs/final_convnext_tiny_384_v2/best.pt",
    "distance_state_classifier_endodac/runs/endodac_encoder_multiscale_adapter_reweight_soft_392/best.pt",
]
VARIANTS = {
    "original": "原图",
    "bg_desaturate": "仅背景去色",
    "bg_chroma_plus": "仅背景色相 +",
    "bg_chroma_minus": "仅背景色相 −",
    "fg_desaturate": "仅器械内部去色",
    "fg_chroma_plus": "仅器械内部色相 +",
    "fg_chroma_minus": "仅器械内部色相 −",
    "all_desaturate": "有效视野去色",
    "bg_texture_blur": "背景纹理模糊（强干预）",
}


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_rows(rows, limit, seed):
    """Round-robin sampling across video/label groups for a small smoke run."""
    if limit <= 0 or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("video_id", ""), row.get("label", ""))].append(row)
    keys = sorted(groups)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < limit:
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                if len(selected) == limit:
                    break
    return selected


def cached_mask_path(cache, image_path):
    # Match the existing mask_cache_path convention exactly (absolute image path).
    key = hashlib.sha1(resolve(image_path).as_posix().encode("utf-8")).hexdigest()
    return resolve(cache) / key[:2] / f"{key}.png"


def intervention_regions(image, mask, guard_px=5):
    """Protect the instrument silhouette and black border from regional edits.

    The FOV estimate is a heuristic: fill the largest non-black outer contour.
    Its extent and the segmentation must be visually checked in the gallery.
    """
    foreground = mask > 0
    if not foreground.any():
        raise ValueError("empty instrument mask")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours((gray > 8).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fov = np.zeros(gray.shape, np.uint8)
    if not contours:
        raise ValueError("no visible FOV")
    cv2.drawContours(fov, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * guard_px + 1, 2 * guard_px + 1))
    fov_inner = cv2.erode(fov, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    fg_inner = cv2.erode(foreground.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    fg_outer = cv2.dilate(foreground.astype(np.uint8), kernel) > 0
    return {"fg": fg_inner & fov_inner, "bg": ~fg_outer & fov_inner, "fov": fov_inner}


def appearance_interventions(image, regions, hue_degrees=30.0, blur_sigma=3.0):
    """Change Lab chroma, keeping Lab L before conversion back to BGR.

    RGB gamut clipping/quantization can slightly change reconstructed luminance.
    Unselected pixels are copied exactly. No spatial resampling is performed.
    """
    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    gray_lab = lab.copy()
    gray_lab[:, :, 1:] = 0
    chroma_versions = {"desaturate": gray_lab}
    for name, angle in [("chroma_plus", hue_degrees), ("chroma_minus", -hue_degrees)]:
        theta = math.radians(angle)
        modified = lab.copy()
        a, b = lab[:, :, 1], lab[:, :, 2]
        modified[:, :, 1] = math.cos(theta) * a - math.sin(theta) * b
        modified[:, :, 2] = math.sin(theta) * a + math.cos(theta) * b
        chroma_versions[name] = modified
    converted = {
        key: np.rint(np.clip(cv2.cvtColor(value, cv2.COLOR_LAB2BGR), 0, 1) * 255).astype(np.uint8)
        for key, value in chroma_versions.items()
    }
    variants = {"original": image.copy()}
    for prefix in ["bg", "fg"]:
        for operation, replacement in converted.items():
            out = image.copy()
            region = regions[prefix]
            out[region] = replacement[region]
            variants[f"{prefix}_{operation}"] = out
    out = image.copy()
    out[regions["fov"]] = converted["desaturate"][regions["fov"]]
    variants["all_desaturate"] = out
    # Normalized convolution prevents instrument pixels leaking into the blur.
    weight = regions["bg"].astype(np.float32)
    denominator = cv2.GaussianBlur(weight, (0, 0), blur_sigma)
    numerator = cv2.GaussianBlur(image.astype(np.float32) * weight[:, :, None], (0, 0), blur_sigma)
    blurred = np.rint(np.clip(numerator / np.maximum(denominator[:, :, None], 1e-6), 0, 255)).astype(np.uint8)
    out = image.copy()
    out[regions["bg"]] = blurred[regions["bg"]]
    variants["bg_texture_blur"] = out
    return {name: variants[name] for name in VARIANTS}


def probability_distance(original, changed):
    """Total variation distance in [0, 1]."""
    return float(np.abs(np.asarray(original) - np.asarray(changed)).sum() / 2.0)


def classification_metrics(rows, classes):
    labeled = [r for r in rows if r["label"] in classes]
    if not labeled:
        return {"labeled_n": 0, "accuracy": None, "macro_f1": None, "confusion_matrix": None}
    index = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for row in labeled:
        matrix[index[row["label"]], index[row["prediction"]]] += 1
    f1 = []
    for i in range(len(classes)):
        denom = matrix[i].sum() + matrix[:, i].sum()
        f1.append(float(2 * matrix[i, i] / denom) if denom else 0.0)
    return {"labeled_n": len(labeled), "accuracy": float(np.trace(matrix) / matrix.sum()),
            "macro_f1": float(np.mean(f1)), "confusion_matrix": matrix.tolist()}


def summarize(rows, classes):
    result = {}
    original_metrics = classification_metrics([r for r in rows if r["variant"] == "original"], classes)
    for variant in VARIANTS:
        subset = [r for r in rows if r["variant"] == variant]
        # Tiny masks can have no foreground left after boundary erosion.
        valid = [r for r in subset if r["intervention_valid"]]
        if not valid:
            result[variant] = {"n": 0, "unavailable_n": len(subset)}
            continue
        metrics = classification_metrics(valid, classes)
        correct = [r for r in valid if r["original_prediction"] == r["label"] and r["label"] in classes]
        result[variant] = {
            "n": len(valid), "unavailable_n": len(subset) - len(valid), **metrics,
            "flip_rate": float(np.mean([r["flipped"] for r in valid])),
            "mean_probability_tv": float(np.mean([r["probability_tv"] for r in valid])),
            "mean_changed_pixel_fraction": float(np.mean([r["changed_pixel_fraction"] for r in valid])),
            "mean_pixel_delta_255": float(np.mean([r["mean_pixel_delta_255"] for r in valid])),
            "correct_to_wrong_rate": float(np.mean([r["prediction"] != r["label"] for r in correct])) if correct else None,
            "original_correct_n": len(correct),
            "opposite_extreme_flip_rate": float(np.mean([r["opposite_extreme_flip"] for r in valid])),
            "accuracy_delta": None,
        }
        if metrics["accuracy"] is not None:
            # Paired original accuracy on exactly the same valid subset.
            paired_original = np.mean([r["original_prediction"] == r["label"] for r in valid if r["label"] in classes])
            result[variant]["accuracy_delta"] = float(metrics["accuracy"] - paired_original)
    return {"class_order": classes, "original_metrics": original_metrics, "variants": result}


class ModelRunner:
    def __init__(self, checkpoint_path, device, batch_size):
        import torch
        self.torch = torch
        self.path = resolve(checkpoint_path)
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        self.config = copy.deepcopy(checkpoint.get("config", {}))
        if not self.config:
            raise ValueError(f"Checkpoint has no training config: {self.path}")
        name = checkpoint.get("model_name") or self.config["model"]["name"]
        self.classes = checkpoint.get("classes") or self.config["data"]["classes"]
        self.input_size = int(checkpoint.get("input_size") or self.config["image"]["input_size"])
        self.mask_aware = bool(self.config["model"].get("mask_aware", {}).get("enabled", False)) or "mask_aware" in name
        self.is_endodac = "endodac" in name or "depth_encoder" in name
        if self.is_endodac:
            from distance_state_classifier_endodac.src.model import build_model
            from distance_state_classifier_endodac.src.transforms import resize_and_normalize
            model_config = copy.deepcopy(self.config["model"])
            # The trained checkpoint already contains the complete model state.
            model_config["pretrained_checkpoint"] = None
            model_config.setdefault("encoder", {})["checkpoint"] = None
            self.model = build_model(name, len(self.classes), config=model_config, input_size=self.input_size, pretrained=False)
            self.prepare = lambda rgb: resize_and_normalize(rgb, self.input_size, self.config["image"].get("normalize"))
        else:
            from distance_state_classifier.src.model import build_model
            from distance_state_classifier.src.transforms import resize_and_normalize
            self.model = build_model(name, len(self.classes), dropout=float(self.config["model"].get("dropout", 0.2)), pretrained=False)
            self.prepare = lambda rgb: resize_and_normalize(rgb, self.input_size)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.batch_size = batch_size
        self.key = self.path.parent.name
        self.metadata = {"checkpoint": str(self.path), "model_name": name, "mask_aware": self.mask_aware,
                         "input_size": self.input_size, "classes": self.classes, "data": self.config.get("data"),
                         "image": self.config.get("image"), "device": str(self.device)}

    def predict(self, images, mask):
        torch = self.torch
        results = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start:start + self.batch_size]
            arrays = [self.prepare(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in batch]
            tensor = torch.from_numpy(np.stack(arrays)).to(self.device)
            with torch.inference_mode():
                if self.mask_aware:
                    m = cv2.resize((mask > 0).astype(np.float32), (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
                    masks = torch.from_numpy(m[None, None]).expand(len(batch), -1, -1, -1).to(self.device)
                    logits = self.model(tensor, masks)
                else:
                    logits = self.model(tensor)
                results.extend(torch.softmax(logits, dim=1).cpu().numpy())
        return results


def load_sample(row, args, crop_config=None):
    image_path = resolve(row["image_path"])
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"unreadable image: {image_path}")
    mask_path = resolve(row["mask_path"]) if row.get("mask_path") else cached_mask_path(args.mask_cache, image_path)
    if not mask_path.is_file():
        raise ValueError(f"missing mask: {mask_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != image.shape[:2]:
        raise ValueError(f"unreadable or mismatched mask: {mask_path}")
    if args.apply_crop:
        from distance_state_classifier.src.transforms import CropBox, apply_crop
        crop = CropBox(**{k: crop_config[k] for k in ["x_left", "x_right", "y_top", "y_bottom"] if k in crop_config})
        image, mask = apply_crop(image, crop), apply_crop(mask, crop)
    regions = intervention_regions(image, mask, args.guard_px)
    return image, mask, regions, mask_path


def save_gallery(output, sample_rows, prediction_rows, scores, args, crop):
    selected = sorted(scores, key=lambda i: scores[i], reverse=True)[:args.save_examples]
    fragments = []
    for i in selected:
        row = sample_rows[i]
        image, mask, regions, _ = load_sample(row, args, crop)
        variants = appearance_interventions(image, regions, args.hue_degrees, args.blur_sigma)
        overlay = image.copy()
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([0, 220, 0])).astype(np.uint8)
        cv2.drawContours(overlay, cv2.findContours(regions["bg"].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255, 180, 0), 1)
        variants = {"mask_overlay": overlay, **variants}
        cards = []
        for variant, img in variants.items():
            relative = Path("examples") / f"{i:04d}_{variant}.jpg"
            (output / relative.parent).mkdir(exist_ok=True)
            scale = min(1.0, 400 / img.shape[1])
            thumb = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)))
            if not cv2.imwrite(str(output / relative), thumb):
                raise RuntimeError("Failed to write gallery image")
            lines = []
            for record in prediction_rows:
                if record["sample_index"] == i and record["variant"] == variant:
                    lines.append(f"{html.escape(record['model'])}: <b>{record['prediction']}</b> "
                                 f"p={record['confidence']:.3f}, TV={record['probability_tv']:.3f} "
                                 f"{'⚠ 翻转' if record['flipped'] else ''}")
            title = "mask：绿色为器械，青色为背景干预边界" if variant == "mask_overlay" else VARIANTS[variant]
            cards.append(f'<article><h4>{title}</h4><img src="{relative.as_posix()}"><p>{"<br>".join(lines)}</p></article>')
        fragments.append(f'<section><h2>{html.escape(row.get("sample_id") or str(i))} · 标签 {html.escape(row.get("label", "未标注"))}</h2>'
                         f'<p>{html.escape(row["image_path"])}</p><div class="grid">{"".join(cards)}</div></section>')
    document = ('<!doctype html><meta charset="utf-8"><title>距离分类器干预诊断</title>'
                '<style>body{font:15px sans-serif;margin:24px;background:#f3f4f6}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}'
                'article{background:white;padding:12px;overflow-wrap:anywhere}img{width:100%}section{margin-bottom:36px}p{overflow-wrap:anywhere}</style>'
                '<h1>配对外观干预：按最大概率变化排序</h1><p>推理使用原始分辨率图像；此处图片仅为缩略图。所有变体使用同一原始 mask，未重新分割。'
                '先检查 mask 与干预区域。预测翻转说明对该干预敏感，不等于证明依赖背景，更不能代表临床效果。</p>' + ''.join(fragments))
    (output / "gallery.html").write_text(document, encoding="utf-8")


def save_report(output, summaries, runners, source_csv, skipped):
    lines = ["# 距离分类器外观干预诊断", "", f"输入 CSV：`{source_csv}`", f"跳过样本：{len(skipped)}；原因见 `skipped.csv`。", "",
             "使用原始单帧分类概率，不使用时间平滑。区域干预使用固定原始 mask，仅诊断分类器；不评价分割器的跨域鲁棒性。", "",
             "所有翻转率均相对于同一帧原图。TV = 0.5 × Σ|p_干预 − p_原图|，范围 0–1。原本正确→错误率只在原图分类正确的样本上计算。", "",
             "色彩干预在 Lab 中保持 L 后转换回 RGB，色域裁剪可造成小量明度变化；背景模糊可能破坏有用细节，是强干预。自动 FOV 与 mask 必须在 gallery.html 中复核。", ""]
    for runner in runners:
        lines.extend([f"## {runner.key}", "", f"权重：`{runner.path}`", f"实际模型：`{runner.metadata['model_name']}`；mask-aware：`{runner.mask_aware}`。", "",
                      "| 干预 | n | 翻转率 | 平均 TV | 准确率 | macro-F1 | 原本正确→错误 |", "|---|---:|---:|---:|---:|---:|---:|"])
        def pct(x):
            return "—" if x is None else f"{100*x:.1f}%"
        for variant, info in summaries[runner.key]["overall"]["variants"].items():
            if not info["n"]:
                lines.append(f"| {VARIANTS[variant]} | 0 | — | — | — | — | — |")
                continue
            f1 = "—" if info["macro_f1"] is None else f"{info['macro_f1']:.3f}"
            lines.append(f"| {VARIANTS[variant]} | {info['n']} | {pct(info['flip_rate'])} | {info['mean_probability_tv']:.3f} | {pct(info['accuracy'])} | {f1} | {pct(info['correct_to_wrong_rate'])} |")
        lines.append("")
    lines.extend(["## 如何解释", "", "- 先排除 mask 不准、有效视野误识别，以及干预明显不自然的样本。", "- 温和背景变色就造成多帧预测变化，比只在背景强模糊下变化更能提示外观依赖。", "- 只对强干预敏感，可能来自新分布或有用信息丢失，不能直接归因为捷径。", "- 检查 summary.json 的像素变化量；金属近灰色时色相旋转可能几乎没有效果。", "- 未改变形状/尺度，不代表所有强干预一定保持原标签；结合 gallery.html 人工判断。", "- 样本来自少量视频，帧不独立；小样本翻转率仅供诊断。临床验证应按手术/病例留出。", "- 假模干预结果不能证明临床失效的原因。应用相同脚本检查人工标注的临床 CSV。", "- mask-only、背景整块置黑、器械擦除不是本次主干预，避免把破坏语义误当作外观变化。", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", help="Repeat to compare checkpoints; defaults to two existing models.")
    parser.add_argument("--csv", default="auto_labeling_project/data/final_dataset/test_labels.csv")
    parser.add_argument("--mask-cache", default="distance_state_classifier_endodac/mask_cache/yolo11s_seg_formal")
    parser.add_argument("--output-dir", default=None, help="Must not already exist.")
    parser.add_argument("--max-samples", type=int, default=0, help="0: all; otherwise sample by video/label groups.")
    parser.add_argument("--save-examples", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guard-px", type=int, default=5)
    parser.add_argument("--hue-degrees", type=float, default=30)
    parser.add_argument("--blur-sigma", type=float, default=3)
    parser.add_argument("--apply-crop", action="store_true", help="Only for raw frames; do not use for frames_cropped.")
    args = parser.parse_args()
    if min(args.max_samples, args.save_examples, args.guard_px) < 0 or min(args.batch_size, args.threads, args.blur_sigma) <= 0:
        parser.error("sample/example/guard counts must be nonnegative; batch/threads/sigma must be positive")
    if not 0 < args.hue_degrees <= 180:
        parser.error("--hue-degrees must be in (0, 180]")
    return args


def main():
    import torch
    args = parse_args()
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    source_csv = resolve(args.csv)
    sample_rows = select_rows(read_csv(source_csv), args.max_samples, args.seed)
    if not sample_rows or "image_path" not in sample_rows[0]:
        raise ValueError("CSV must contain at least one image_path row")
    output = resolve(args.output_dir or f"results/distance_shortcut_diagnostics_{datetime.now():%Y%m%d_%H%M%S}")
    output.mkdir(parents=True, exist_ok=False)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    runners = []
    for checkpoint in args.checkpoint or DEFAULT_CHECKPOINTS:
        print(f"Loading {checkpoint} on {device}", flush=True)
        runners.append(ModelRunner(checkpoint, device, args.batch_size))
    if len({r.key for r in runners}) != len(runners):
        raise ValueError("Checkpoint parent directory names must be distinct")
    crop = runners[0].config.get("image", {}).get("crop", {})
    if args.apply_crop and any(r.config.get("image", {}).get("crop", {}) != crop for r in runners):
        raise ValueError("Compared checkpoints have different raw-frame crops; run them separately")
    records, skipped, scores = [], [], {}
    for i, row in enumerate(sample_rows):
        try:
            image, mask, regions, mask_path = load_sample(row, args, crop)
        except (ValueError, KeyError) as exc:
            skipped.append({"sample_index": i, "image_path": row.get("image_path", ""), "reason": str(exc)})
            print(f"Skipping {i}: {exc}", flush=True)
            continue
        variants = appearance_interventions(image, regions, args.hue_degrees, args.blur_sigma)
        scores[i] = 0.0
        for runner in runners:
            predictions = runner.predict(list(variants.values()), mask)
            original = predictions[0]
            original_label = runner.classes[int(np.argmax(original))]
            for (variant, modified), probs in zip(variants.items(), predictions):
                label = runner.classes[int(np.argmax(probs))]
                delta = np.abs(modified.astype(np.float32) - image.astype(np.float32))
                prefix = variant.split("_")[0]
                region = regions.get(prefix, regions["fov"])
                valid = bool(region.any())
                tv = probability_distance(original, probs)
                record = {"sample_index": i, "sample_id": row.get("sample_id", ""), "image_path": str(resolve(row["image_path"])),
                          "mask_path": str(mask_path), "video_id": row.get("video_id", ""), "label": row.get("label", ""),
                          "model": runner.key, "variant": variant, "intervention_valid": valid,
                          "prediction": label, "confidence": float(max(probs)), "original_prediction": original_label,
                          "original_confidence": float(max(original)),
                          "flipped": label != original_label, "probability_tv": tv,
                          "opposite_extreme_flip": {label, original_label} == {"TooFar", "TooClose"},
                          "region_pixel_fraction": float(region.mean()),
                          "changed_pixel_fraction": float(np.any(delta > 0, axis=2).mean()),
                          "mean_pixel_delta_255": float(delta[region].mean()) if valid else 0.0}
                record.update({f"p_{c}": float(p) for c, p in zip(runner.classes, probs)})
                records.append(record)
                if valid:
                    scores[i] = max(scores[i], tv)
        print(f"[{i+1}/{len(sample_rows)}] {row.get('sample_id') or row['image_path']}", flush=True)
    write_csv(output / "predictions.csv", records)
    write_csv(output / "skipped.csv", skipped)
    if not records:
        raise RuntimeError(f"No valid samples; inspect {output / 'skipped.csv'}")
    summaries = {}
    for runner in runners:
        subset = [r for r in records if r["model"] == runner.key]
        summaries[runner.key] = {"metadata": runner.metadata, "overall": summarize(subset, runner.classes),
                                 "by_video": {video: summarize([r for r in subset if r["video_id"] == video], runner.classes)
                                              for video in sorted({r["video_id"] for r in subset})}}
    payload = {"source_csv": str(source_csv), "args": vars(args), "sampled_n": len(sample_rows),
               "included_n": len(scores), "skipped_n": len(skipped), "models": summaries}
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    save_report(output, summaries, runners, source_csv, skipped)
    save_gallery(output, sample_rows, records, scores, args, crop)
    print(f"Report: {output / 'report.md'}\nGallery: {output / 'gallery.html'}", flush=True)


if __name__ == "__main__":
    main()

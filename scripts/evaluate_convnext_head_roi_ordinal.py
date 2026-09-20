#!/usr/bin/env python3
"""Paired evaluation of original ConvNeXt and the completed head ROI experiment."""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnose_distance_shortcuts import ModelRunner, appearance_interventions, intervention_regions, probability_distance, write_csv, summarize
from distance_state_classifier.src.config import resolve_path
from distance_state_classifier.src.head_roi import crop_with_padding
from distance_state_classifier.src.head_roi_predictor import HeadRoiOrdinalPredictor
from distance_state_classifier.scripts.prepare_head_roi import file_hash, write_json
from distance_state_classifier.scripts.train_head_roi_ordinal import summarize_predictions


COLOR_VARIANTS = ["bg_desaturate", "bg_chroma_plus", "bg_chroma_minus", "fg_desaturate", "fg_chroma_plus", "fg_chroma_minus", "all_desaturate"]


def plot_results(output, history, metrics, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    epochs = [r["epoch"] for r in history]
    for split in ("train", "val"):
        axes[0].plot(epochs, [r[split]["macro_f1"] for r in history], label=split)
    axes[0].set(xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1), title="ROI + spatial + ordinal")
    axes[0].legend()
    x = np.arange(3)
    for i, key in enumerate(("A", "ROI_ordinal")):
        axes[1].bar(x + (i-.5)*.36, [r["recall"] for r in metrics[key]["per_class"]], .36, label=key)
    axes[1].set(xticks=x, xticklabels=["TooFar", "Good", "TooClose"], ylim=(0, 1.05), ylabel="Recall", title="Original test images")
    axes[1].legend()
    for i, key in enumerate(("A", "ROI_ordinal")):
        values = [np.mean([summaries[key]["variants"][v][metric] for v in COLOR_VARIANTS])
                  for metric in ("flip_rate", "mean_probability_tv")]
        axes[2].bar(np.arange(2) + (i-.5)*.36, values, .36, label=key)
    axes[2].set(xticks=[0, 1], xticklabels=["Class flip rate", "Probability TV"],
                ylabel="Mean (lower is better)", title="Seven color interventions")
    axes[2].legend()
    fig.savefig(output / "overview.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="distance_state_classifier/runs/convnext_head_roi_ordinal_20260907")
    parser.add_argument("--output-dir", default="results/convnext_head_roi_ordinal_comparison_20260907")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    run, output = resolve_path(args.run_dir), resolve_path(args.output_dir)
    complete = json.loads((run / "completion.json").read_text())
    if complete["status"] != "complete":
        raise RuntimeError("Training must finish before test comparison")
    config = json.loads((run / "resolved_config.json").read_text())
    manifest = json.loads((resolve_path(config["roi"]["cache_dir"]) / "manifest.json").read_text())
    source_rows = [r for r in manifest["records"] if r["split"] == "test"]
    output.mkdir(parents=True, exist_ok=False)
    baseline = ModelRunner(config["experiment"]["baseline_checkpoint"], args.device, 8)
    predictor = HeadRoiOrdinalPredictor(run / "best.pt", args.device)
    gap = float(torch.nn.functional.softplus(predictor.model.raw_gap.detach().float()).cpu() + 1e-4)
    ordinal_limits = {"learned_gap": gap, "maximum_good_probability": math.tanh(gap / 4)}
    classes = predictor.classes
    if baseline.classes != classes:
        raise ValueError("A and new model have different class order")
    records, invalid, scores = [], [], {}
    truth, a_predictions, b_predictions = [], [], []
    original_probs = {"A": [], "ROI_ordinal": []}
    for index, row in enumerate(source_rows):
        if file_hash(row["image_path"]) != row["image_sha256"] or file_hash(row["mask_path"]) != row["mask_sha256"]:
            raise ValueError(f"Source image/mask changed after ROI preparation: {row['sample_id']}")
        image, mask = cv2.imread(row["image_path"]), cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            raise ValueError(row["sample_id"])
        regions = intervention_regions(image, mask, 5)
        variants = appearance_interventions(image, regions, 30.0, 3.0)
        a_probs = baseline.predict(list(variants.values()), mask)
        b_probs = predictor.probabilities_for_fixed_roi(list(variants.values()), row["geometry"])
        truth.append(classes.index(row["label"]))
        a_predictions.append(int(np.argmax(a_probs[0])))
        b_predictions.append(int(np.argmax(b_probs[0])) if b_probs[0] is not None else 3)
        original_probs["A"].append(a_probs[0].tolist())
        original_probs["ROI_ordinal"].append(b_probs[0].tolist() if b_probs[0] is not None else None)
        if not row["geometry"]["valid"]:
            invalid.append({"sample_id": row["sample_id"], "reason": row["geometry"]["reason"]})
            continue
        scores[index] = 0.0
        for key, probabilities in (("A", a_probs), ("ROI_ordinal", b_probs)):
            original = probabilities[0]
            original_label = classes[int(np.argmax(original))]
            for (variant, modified), probs in zip(variants.items(), probabilities):
                prediction = classes[int(np.argmax(probs))]
                region = regions.get(variant.split("_")[0], regions["fov"])
                delta = np.abs(modified.astype(np.float32) - image.astype(np.float32))
                tv = probability_distance(original, probs)
                record = {"sample_index": index, "sample_id": row["sample_id"], "image_path": row["image_path"],
                          "video_id": row["video_id"], "label": row["label"], "model": key, "variant": variant,
                          "prediction": prediction, "confidence": float(max(probs)), "original_prediction": original_label,
                          "intervention_valid": bool(region.any()), "flipped": prediction != original_label,
                          "probability_tv": tv, "opposite_extreme_flip": {prediction, original_label} == {"TooFar", "TooClose"},
                          "changed_pixel_fraction": float(np.any(delta > 0, -1).mean()),
                          "mean_pixel_delta_255": float(delta[region].mean()) if region.any() else 0.0}
                record.update({f"p_{c}": float(p) for c, p in zip(classes, probs)})
                records.append(record)
                scores[index] = max(scores[index], tv)
    metrics = {"A": summarize_predictions(truth, a_predictions, classes), "ROI_ordinal": summarize_predictions(truth, b_predictions, classes)}
    if metrics["ROI_ordinal"]["confusion_matrix"] != complete["test"]["confusion_matrix"]:
        raise ValueError("Live preprocessing differs from cached-ROI test evaluation")
    summaries = {key: summarize([r for r in records if r["model"] == key], classes) for key in ("A", "ROI_ordinal")}
    threshold = config["inference"]["confidence_threshold"]
    accepted = {}
    for key, probs in original_probs.items():
        selected = [i for i, p in enumerate(probs) if p is not None and max(p) >= threshold]
        accepted[key] = {"threshold": threshold, "accepted_n": len(selected), "coverage": len(selected)/len(truth),
                         "accuracy_accepted": float(np.mean([int(np.argmax(probs[i])) == truth[i] for i in selected])) if selected else None,
                         "predicted_counts": dict(Counter(classes[int(np.argmax(probs[i]))] for i in selected))}
    write_csv(output / "predictions.csv", records)
    write_csv(output / "unavailable_rois.csv", invalid)
    payload = {"classes": classes, "original_all_samples": metrics, "appearance_on_common_valid_rois": summaries,
               "thresholded": accepted, "ordinal_probability_limits": ordinal_limits,
               "included_test_n": len(truth), "invalid_roi_n": len(invalid),
               "protocol": "Masks, per-video FOV and ROI boxes held fixed across paired appearance interventions"}
    write_json(output / "summary.json", payload)
    write_json(output / "checkpoint_hashes.json", {"A": file_hash(resolve_path(config["experiment"]["baseline_checkpoint"])), "ROI_ordinal": file_hash(run / "best.pt")})
    history = json.loads((run / "metrics.json").read_text())["history"]
    plot_results(output, history, metrics, summaries)
    percent = lambda v: f"{v*100:.2f}%"
    lines = ["# ConvNeXt 头部 ROI、尺度保留与有序分类实验", "",
             f"完成 {complete['last_epoch']} 轮训练，最佳权重来自第 {complete['best_epoch']} 轮；验证集 macro-F1={complete['best_val_macro_f1']:.4f}。",
             "模型选择仅依据验证集；这是一轮固定配置的组合实验，未按测试结果调整参数。", "",
             "## 原图结果", "", "| 指标 | 原 ConvNeXt A | ROI＋空间特征＋有序头 |", "|---|---:|---:|"]
    for label, metric in (("全部测试帧 accuracy", "accuracy"), ("Macro-F1", "macro_f1"), ("定位可用率", "coverage")):
        lines.append(f"| {label} | {percent(metrics['A'][metric])} | {percent(metrics['ROI_ordinal'][metric])} |")
    for i, label in enumerate(classes):
        lines.append(f"| {label} recall | {percent(metrics['A']['per_class'][i]['recall'])} | {percent(metrics['ROI_ordinal']['per_class'][i]['recall'])} |")
    lines += [f"| 两端状态直接误判数 | {metrics['A']['opposite_extreme_errors']} | {metrics['ROI_ordinal']['opposite_extreme_errors']} |", "",
              "![训练曲线与比较](overview.png)", "", "## 颜色干预", "",
              "| 指标（7 种颜色干预等权平均） | A | 新模型 |", "|---|---:|---:|"]
    for label, metric in (("Accuracy", "accuracy"), ("Macro-F1", "macro_f1"), ("翻转率 ↓", "flip_rate"), ("概率 TV ↓", "mean_probability_tv")):
        values = [np.mean([summaries[key]["variants"][v][metric] for v in COLOR_VARIANTS]) for key in ("A", "ROI_ordinal")]
        lines.append(f"| {label} | {values[0]:.4f} | {values[1]:.4f} |")
    lines += ["", "颜色干预使用固定的原始 mask、FOV 和 ROI，测量的是分类器对外观变化的敏感性；尚未包含分割器在颜色变化下的误差。", "",
              "## 混淆矩阵", "", "行：TooFar、Good、TooClose；列：TooFar、Good、TooClose、Invalid。定位失败保留在总样本数中。", ""]
    for key in ("A", "ROI_ordinal"):
        lines += [key, "", "```text", *[str(row) for row in metrics[key]["confusion_matrix"]], "```", ""]
    lines += ["## 当前 0.55 置信度阈值", "", "以下单独报告阈值过滤，主表仍为未过滤的类别预测。不同分类头的概率校准可能不同。", ""]
    for key, info in accepted.items():
        accuracy = percent(info["accuracy_accepted"]) if info["accuracy_accepted"] is not None else "N/A"
        lines.append(f"- {key}：接受 {info['accepted_n']}/{len(truth)} 帧，接受帧 accuracy={accuracy}；接受类别数量 {info['predicted_counts']}。")
    lines += ["", f"最佳权重的阈值间隔 g={gap:.6f}。当前参数化满足 max P(Good)=tanh(g/4)={ordinal_limits['maximum_good_probability']:.6f}；该上界由分类头直接决定。",
              "若上界低于接受阈值，即使 Good 是 argmax 预测，也无法通过置信度过滤。后续需要在验证集上检查概率校准与拒识规则，不能从测试集挑选阈值。", "",
              "## 结果解释", "",
              "原图分类质量、干预后的类别翻转和概率变化幅度需要合并判断。不同分类头的概率锐度不同，较小的 TV 不能单独证明泛化改善；类别翻转也不等价于新增错误，应结合干预后的准确率。",
              "若需要区分各改动的贡献，后续应固定输入窗口与训练划分，对普通分类头和有序头做消融，并人工核查候选窗口是否覆盖抓钳头部。", "",
              "## 局限与解释", "",
              "- 头部位置来自既有分割与几何尖端启发式，属于头部候选窗口，没有人工头部标注验证的定位准确率。",
              "- ROI 大小相对于每个视频的固定 FOV 直径设定，未按器械框宽度调整；越界先填充，再统一缩放。",
              "- 一张训练帧和一张验证帧定位不可用；训练排除该帧，验证将其计为 Invalid。完整统计见 ROI 审计。",
              "- 新模型保留 stride 16/32 的 2×2 空间分区；输出共享分数与有序阈值，以累积二分类损失训练。输出有序不保证图像尺度与预测严格单调，也不保证零跨级错误。",
              "- 相较历史 A，还关闭了几何增强并更换分类头；这是组合方案对照，不能分离三个改动各自的贡献。",
              "- 只有一个随机种子、一个假模测试视频；TooFar 仅 10 帧，不能据此证明真实手术泛化。",
              "- 推理需要器械 mask 与固定 FOV 标定；现有实时控制程序未接入此实验模型。", "",
              "## 文件", "", "- [测试图与局部输入](gallery.html)", "- [逐图预测](predictions.csv)",
              f"- [头部 ROI 训练预览]({resolve_path(config['roi']['cache_dir']) / 'preview.html'})",
              f"- [最佳权重]({run / 'best.pt'})", f"- [实验说明]({ROOT / 'docs/convnext_head_roi_ordinal_experiment.md'})", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    examples = output / "examples"; examples.mkdir()
    candidates = sorted(scores, key=scores.get, reverse=True)[:12]
    for i, (target, a, b) in enumerate(zip(truth, a_predictions, b_predictions)):
        if a == target and b != target and i in scores and i not in candidates:
            candidates.append(i)
            if len(candidates) >= 18:
                break
    cards = []
    for i in candidates:
        row = source_rows[i]
        image = cv2.imread(row["image_path"])
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        variants = appearance_interventions(image, intervention_regions(image, mask, 5), 30, 3)
        cells = []
        for variant in ("original", "bg_desaturate", "fg_desaturate", "all_desaturate"):
            image = variants[variant]
            roi = crop_with_padding(image, row["geometry"]["box"])
            full, local = f"examples/{i:03d}_{variant}_full.jpg", f"examples/{i:03d}_{variant}_roi.jpg"
            cv2.imwrite(str(output / full), cv2.resize(image, (320, round(320 * image.shape[0]/image.shape[1]))))
            cv2.imwrite(str(output / local), cv2.resize(roi, (240, 240)))
            labels = [f"{r['model']}: {r['prediction']} ({r['confidence']:.3f})" for r in records if r["sample_index"] == i and r["variant"] == variant]
            cells.append(f'<td>{variant}<br><img width="280" src="{full}"><br><img width="240" src="{local}"><p>{"<br>".join(labels)}</p></td>')
        cards.append(f'<h2>{html.escape(row["sample_id"])} · 标签 {row["label"]}</h2><table><tr>{"".join(cells)}</tr></table>')
    (output / "gallery.html").write_text('<!doctype html><meta charset="utf-8"><title>ConvNeXt ROI 有序分类比较</title><style>body{font-family:sans-serif}td{vertical-align:top;padding:10px}</style><h1>原图 / 局部输入与外观干预</h1>' + ''.join(cards), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"[complete] {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()

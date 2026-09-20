#!/usr/bin/env python3
"""Summarize a completed B run and its paired A/B diagnostic results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from distance_state_classifier_endodac.src.config import resolve_path


COLOR_VARIANTS = ["bg_desaturate", "bg_chroma_plus", "bg_chroma_minus", "fg_desaturate",
                  "fg_chroma_plus", "fg_chroma_minus", "all_desaturate"]
NAMES = {"original": "原图", "bg_desaturate": "背景去色", "bg_chroma_plus": "背景色度 +30°",
         "bg_chroma_minus": "背景色度 −30°", "fg_desaturate": "器械内部去色",
         "fg_chroma_plus": "器械色度 +30°", "fg_chroma_minus": "器械色度 −30°",
         "all_desaturate": "有效视野去色", "bg_texture_blur": "背景模糊（压力测试）"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scalar_metrics(overall):
    matrix = np.asarray(overall["original_metrics"]["confusion_matrix"])
    variants = overall["variants"]
    recalls = matrix.diagonal() / np.maximum(matrix.sum(1), 1)
    return {"accuracy": overall["original_metrics"]["accuracy"],
            "macro_f1": overall["original_metrics"]["macro_f1"],
            "recalls": recalls.tolist(), "support": matrix.sum(1).tolist(),
            "precisions": (matrix.diagonal() / np.maximum(matrix.sum(0), 1)).tolist(),
            "predicted_counts": matrix.sum(0).tolist(),
            "mean_color_tv": float(np.mean([variants[name]["mean_probability_tv"] for name in COLOR_VARIANTS])),
            "mean_color_flip": float(np.mean([variants[name]["flip_rate"] for name in COLOR_VARIANTS])),
            "mean_color_accuracy": float(np.mean([variants[name]["accuracy"] for name in COLOR_VARIANTS])),
            "mean_color_macro_f1": float(np.mean([variants[name]["macro_f1"] for name in COLOR_VARIANTS]))}


def render_plots(output, history, a, b, class_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train"]["macro_f1"] for row in history], label="Train (clean view)")
    axes[0].plot(epochs, [row["val"]["macro_f1"] for row in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1), title="B training history")
    axes[0].legend()
    x = np.arange(len(class_names))
    axes[1].bar(x - .18, a["recalls"], .36, label="A")
    axes[1].bar(x + .18, b["recalls"], .36, label="B")
    axes[1].set(xticks=x, xticklabels=class_names, ylabel="Recall", ylim=(0, 1.05), title="Original test images")
    axes[1].legend()
    axes[2].bar([0, 1], [a["mean_color_tv"], b["mean_color_tv"]], color=["C0", "C1"])
    axes[2].set(xticks=[0, 1], xticklabels=["A", "B"], ylabel="Mean probability TV", title="Seven color interventions")
    fig.savefig(output / "experiment_b_overview.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", default="results/distance_experiment_b_comparison_20260907")
    parser.add_argument("--run-dir", default="distance_state_classifier_endodac/runs/endodac_experiment_b_appearance_consistency_20260907")
    args = parser.parse_args()
    output, run = resolve_path(args.comparison_dir), resolve_path(args.run_dir)
    complete, training = load(run / "completion.json"), load(run / "metrics.json")
    summary, config = load(output / "summary.json"), load(run / "resolved_config.json")
    if complete["status"] != "complete" or summary["skipped_n"] != 0:
        raise ValueError("Need a completed training run and a diagnostic set with no skipped samples")
    a_key = Path(config["experiment"]["baseline_checkpoint"]).parent.name
    b_key = run.name
    a_overall, b_overall = [summary["models"][key]["overall"] for key in (a_key, b_key)]
    if a_overall["class_order"] != b_overall["class_order"]:
        raise ValueError("Class order mismatch")
    if not np.array_equal(b_overall["original_metrics"]["confusion_matrix"], complete["test"]["confusion_matrix"]):
        raise ValueError("B diagnostic original confusion matrix differs from completed test evaluation")
    a, b = scalar_metrics(a_overall), scalar_metrics(b_overall)
    if a["support"] != b["support"]:
        raise ValueError("A/B test supports differ")
    classes = a_overall["class_order"]
    percent = lambda value: f"{100 * value:.2f}%"
    lines = ["# 实验 B 结果：外观增强与预测一致性", "",
             f"正式训练完成：共 {complete['last_epoch']} 轮，最佳权重来自第 {complete['best_epoch']} 轮，验证集 macro-F1={complete['best_val_macro_f1']:.4f}。",
             "最佳权重仅按原图验证集 macro-F1 选择；测试集未用于选择 epoch。", "",
             f"A/B 使用同一批 {summary['included_n']} 张测试图，跳过 0 张。两者均只输入 RGB、使用原始单帧预测，没有时间平滑。", "",
             "## 原图与颜色敏感性", "", "| 指标 | A | B |", "|---|---:|---:|"]
    for label, key in (("原图 accuracy", "accuracy"), ("原图 macro-F1", "macro_f1")):
        lines.append(f"| {label} | {percent(a[key])} | {percent(b[key])} |")
    for i, label in enumerate(classes):
        lines.append(f"| {label} 召回率（{a['support'][i]} 张） | {percent(a['recalls'][i])} | {percent(b['recalls'][i])} |")
    lines.append(f"| TooFar 精确率 | {percent(a['precisions'][0])} | {percent(b['precisions'][0])} |")
    for label, key in (("7 种颜色干预平均 accuracy", "mean_color_accuracy"),
                       ("7 种颜色干预平均 macro-F1", "mean_color_macro_f1"),
                       ("7 种颜色干预平均翻转率 ↓", "mean_color_flip")):
        lines.append(f"| {label} | {percent(a[key])} | {percent(b[key])} |")
    lines += [f"| 7 种颜色干预平均概率 TV ↓ | {a['mean_color_tv']:.4f} | {b['mean_color_tv']:.4f} |", "",
              "平均值对 7 种颜色干预等权；背景模糊另列。翻转率和 TV 降低必须结合准确率与类别召回解读，不能单独作为有效性证据。", "",
              "![实验 B 曲线与比较](experiment_b_overview.png)", "", "## 各项干预", "",
              "| 输入 | A accuracy | B accuracy | A 翻转率 | B 翻转率 | A TV | B TV |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ["original", *COLOR_VARIANTS, "bg_texture_blur"]:
        x, y = a_overall["variants"][name], b_overall["variants"][name]
        lines.append(f"| {NAMES[name]} | {percent(x['accuracy'])} | {percent(y['accuracy'])} | {percent(x['flip_rate'])} | {percent(y['flip_rate'])} | {x['mean_probability_tv']:.4f} | {y['mean_probability_tv']:.4f} |")
    lines += ["", "## 原图混淆矩阵", "", "行是真实类别，列是预测类别；顺序为 TooFar、Good、TooClose。", ""]
    for name, overall in (("A", a_overall), ("B", b_overall)):
        lines += [name, "", "```text", *[str(row) for row in overall["original_metrics"]["confusion_matrix"]], "```", ""]
    lines += [f"预测数量（TooFar、Good、TooClose）：A={a['predicted_counts']}，B={b['predicted_counts']}。", "",
              "## 本轮结果解读", ""]
    ma, mb = [np.asarray(x["original_metrics"]["confusion_matrix"]) for x in (a_overall, b_overall)]
    if b["macro_f1"] < a["macro_f1"]:
        lines += ["本轮 B 的原图 macro-F1 低于 A，不能据此替换 A。即使颜色稳定性有所改善，也需要一起考虑分类性能的代价。", ""]
    lines += [f"Good → TooFar：A 为 {ma[1, 0]}/{ma[1].sum()}，B 为 {mb[1, 0]}/{mb[1].sum()}。"
              f"原图两端状态直接误判（TooFar ↔ TooClose）：A 为 {ma[0, 2] + ma[2, 0]} 张，B 为 {mb[0, 2] + mb[2, 0]} 张。", ""]
    with (output / "predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        originals = [row for row in csv.DictReader(handle) if row["model"] == b_key and row["variant"] == "original"]
    threshold = config["inference"]["confidence_threshold"]
    for row in originals:
        if {row["label"], row["prediction"]} == {"TooFar", "TooClose"}:
            confidence = float(row["confidence"])
            lines += [f"具体样本：[{row['sample_id']}]({row['image_path']})，标签 {row['label']}，原始预测 {row['prediction']}，置信度 {confidence:.3f}。"
                      + (f"低于当前 {threshold:.2f} 阈值，现有预测器的单帧 state 会为 Invalid；此处报告的是阈值过滤前的 argmax。" if confidence < threshold else ""), ""]
    direction = "下降" if b["mean_color_tv"] < a["mean_color_tv"] else "上升或持平"
    lines += [f"7 种颜色干预的平均概率 TV {direction}：{a['mean_color_tv']:.4f} → {b['mean_color_tv']:.4f}。"
              "这描述的是扰动前后预测的变化，不等于预测正确率。", "",
              "## 结论边界", "",
              "- 这是固定划分、单个随机种子下的完整 B 结果。A/B 改动包含增强方式与一致性损失两部分，不能单独归因于其中一部分。",
              "- TooFar 只有 10 张测试图，一张对应召回率 10 个百分点；不能把小幅差异当成稳定收益。",
              "- 颜色干预与训练增强属于相关扰动；该评估只支持对这类干预的结论，真实手术视频仍需独立验证。",
              "- 不由此修改机器人控制或替换线上默认权重。", "",
              "## 文件", "",
              "- [逐图干预展示](gallery.html)", "- [完整诊断报告](report.md)", "- [逐图概率与预测](predictions.csv)",
              f"- [B 最佳权重]({(run / 'best.pt').as_posix()})",
              f"- [增强预览]({(run / 'augmentation_preview.html').as_posix()})",
              f"- [实验协议]({(ROOT / 'docs/distance_experiment_b.md').as_posix()})", ""]
    (output / "experiment_b_report.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "ab_metrics.json").write_text(json.dumps({"A": a, "B": b, "best_epoch": complete["best_epoch"]}, indent=2), encoding="utf-8")
    render_plots(output, training["history"], a, b, classes)
    print(output / "experiment_b_report.md")


if __name__ == "__main__":
    main()

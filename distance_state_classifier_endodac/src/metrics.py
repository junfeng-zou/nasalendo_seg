from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(y_true, y_pred):
        matrix[int(target), int(pred)] += 1
    return matrix


def classification_metrics(matrix: np.ndarray, classes: list[str] | None = None) -> dict:
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    per_class = []
    f1_values = []
    recalls = []
    for idx in range(matrix.shape[0]):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - matrix[idx, idx])
        fn = float(matrix[idx, :].sum() - matrix[idx, idx])
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        item = {"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[idx, :].sum())}
        if classes:
            item["label"] = classes[idx]
        per_class.append(item)
        f1_values.append(f1)
        recalls.append(recall)
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "per_class": per_class,
    }


def write_confusion_matrix(path: Path, matrix: np.ndarray, classes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *classes])
        for label, row in zip(classes, matrix.tolist()):
            writer.writerow([label, *row])

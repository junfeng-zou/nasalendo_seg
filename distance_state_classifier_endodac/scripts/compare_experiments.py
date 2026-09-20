#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize distance-state experiment metrics.")
    parser.add_argument("runs", nargs="+", help="Run directories that contain test_metrics.json.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def label_metric(metrics: dict, label: str, key: str) -> float | None:
    for item in metrics.get("per_class", []):
        if item.get("label") == label:
            value = item.get(key)
            return float(value) if value is not None else None
    return None


def main() -> None:
    args = parse_args()
    rows = []
    for run in args.runs:
        run_path = Path(run)
        metrics_path = run_path / "test_metrics.json"
        if not metrics_path.exists():
            rows.append({"run": run, "error": f"missing {metrics_path}"})
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": run,
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "tooclose_recall": label_metric(metrics, "TooClose", "recall"),
                "tooclose_f1": label_metric(metrics, "TooClose", "f1"),
                "confusion_matrix": metrics.get("confusion_matrix"),
            }
        )

    print(json.dumps(rows, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

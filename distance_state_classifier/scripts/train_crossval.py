#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leave-one-video-out training over generated fold CSVs.")
    parser.add_argument("--config", default="distance_state_classifier/configs/distance_state_config.yaml")
    parser.add_argument("--splits-dir", default="distance_state_classifier/splits/leave_one_video_out")
    parser.add_argument("--output-dir", default="distance_state_classifier/runs/crossval_convnext_tiny_384")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only-fold", default=None, help="Run one test video fold only, e.g. bend_data2.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits_dir = Path(args.splits_dir)
    if not splits_dir.is_absolute():
        splits_dir = REPO_ROOT / splits_dir
    summary_path = splits_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing split summary: {summary_path}. Run make_video_splits.py first.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    folds = summary.get("folds", {})
    results = {}

    for test_video, fold in folds.items():
        if args.only_fold and test_video != args.only_fold:
            continue
        fold_dir = Path(fold["fold_dir"])
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
        run_dir = output_dir / f"test_{test_video}"
        cmd = [
            sys.executable,
            "distance_state_classifier/scripts/train.py",
            "--config",
            args.config,
            "--output-dir",
            run_dir.as_posix(),
            "--train-csv",
            (fold_dir / "train_labels.csv").as_posix(),
            "--val-csv",
            (fold_dir / "val_labels.csv").as_posix(),
            "--test-csv",
            (fold_dir / "test_labels.csv").as_posix(),
            "--device",
            args.device,
        ]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
        if args.num_workers is not None:
            cmd += ["--num-workers", str(args.num_workers)]
        print("[crossval] running", " ".join(cmd))
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        metrics_path = run_dir / "test_metrics.json"
        if metrics_path.exists():
            results[test_video] = json.loads(metrics_path.read_text(encoding="utf-8"))

    if results:
        macro_f1 = [float(item["macro_f1"]) for item in results.values()]
        accuracy = [float(item["accuracy"]) for item in results.values()]
        aggregate = {
            "folds": results,
            "mean_macro_f1": sum(macro_f1) / len(macro_f1),
            "mean_accuracy": sum(accuracy) / len(accuracy),
        }
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "crossval_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"[crossval] wrote summary: {output_dir / 'crossval_summary.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from distance_state_classifier.src.config import get_nested, load_config, resolve_path


CSV_FIELDS = [
    "sample_id",
    "frame_path",
    "image_path",
    "label",
    "confidence",
    "video_id",
    "frame_index",
    "source_frame_index",
    "time_sec",
    "use_for_training",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create leave-one-video-out CSV splits for distance-state training.")
    parser.add_argument("--config", default="distance_state_classifier/configs/distance_state_config.yaml")
    parser.add_argument("--input", default="auto_labeling_project/data/filtered_labels/labels_reviewed.jsonl")
    parser.add_argument("--output-dir", default="distance_state_classifier/splits/leave_one_video_out")
    parser.add_argument("--val-video", default=None, help="Optional fixed validation video. Default: next video after test video.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def label_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("label", ""))
        counts[label] = counts.get(label, 0) + 1
    return counts


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    classes = set(get_nested(config, "data.classes", ["TooFar", "Good", "TooClose"]))
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    rows = [
        row for row in read_jsonl(input_path)
        if row.get("use_for_training") and row.get("label") in classes
    ]
    if not rows:
        raise SystemExit(f"No usable rows found in {input_path}")

    videos = sorted({str(row.get("video_id", "")) for row in rows})
    summary = {
        "input": input_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "videos": videos,
        "folds": {},
    }

    for idx, test_video in enumerate(videos):
        if args.val_video:
            val_video = args.val_video
            if val_video == test_video:
                val_video = videos[(idx + 1) % len(videos)]
        else:
            val_video = videos[(idx + 1) % len(videos)]

        fold_rows = {
            "train": [row for row in rows if row.get("video_id") not in {test_video, val_video}],
            "val": [row for row in rows if row.get("video_id") == val_video],
            "test": [row for row in rows if row.get("video_id") == test_video],
        }
        fold_dir = output_dir / f"test_{test_video}"
        for split, split_rows in fold_rows.items():
            write_csv(fold_dir / f"{split}_labels.csv", split_rows)
        summary["folds"][test_video] = {
            "fold_dir": fold_dir.as_posix(),
            "val_video": val_video,
            "counts": {split: len(split_rows) for split, split_rows in fold_rows.items()},
            "label_counts": {split: label_counts(split_rows) for split, split_rows in fold_rows.items()},
        }
        print(f"[split] test={test_video} val={val_video} dir={fold_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[split] wrote summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

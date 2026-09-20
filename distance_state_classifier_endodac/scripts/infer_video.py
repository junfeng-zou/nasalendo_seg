#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from distance_state_classifier_endodac.src.config import load_config, resolve_path
from distance_state_classifier_endodac.src.predictor import DistanceStatePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EndoDAC distance-state inference on a video.")
    parser.add_argument("--config", default="distance_state_classifier_endodac/configs/distance_state_endodac_multiscale_adapter_reweight_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--apply-crop", action="store_true", help="Apply raw-frame crop before inference.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    predictor = DistanceStatePredictor(args.checkpoint, config, device=args.device)

    video_path = resolve_path(args.video)
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    stride = max(1, int(args.stride))
    frame_index = 0
    written = 0
    started = time.time()
    with output_path.open("w", encoding="utf-8") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue
            if args.max_frames is not None and written >= args.max_frames:
                break

            result = predictor.predict_array(frame, apply_raw_crop=args.apply_crop)
            row = {
                "frame_index": frame_index,
                "time_sec": frame_index / fps if fps > 0 else None,
                "raw_label": result.raw_label,
                "raw_confidence": result.raw_confidence,
                "state": result.state,
                "smoothed_state": result.smoothed_state,
                "probabilities": result.probabilities,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written % 30 == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(f"[infer] frames={written} speed={written / elapsed:.2f} fps state={result.smoothed_state}")
            frame_index += 1
    cap.release()
    print(f"[infer] wrote {written} predictions: {output_path}")


if __name__ == "__main__":
    main()

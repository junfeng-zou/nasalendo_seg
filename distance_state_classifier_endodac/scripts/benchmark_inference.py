#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from distance_state_classifier_endodac.src.config import load_config, resolve_path
from distance_state_classifier_endodac.src.predictor import DistanceStatePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark single-frame geometry-aware classifier latency.")
    parser.add_argument("--config", default="distance_state_classifier_endodac/configs/distance_state_endodac_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", default=None, help="Optional image path. Uses a zero image if omitted.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--apply-crop", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((q / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    predictor = DistanceStatePredictor(args.checkpoint, config, device=args.device)
    if args.image:
        image = cv2.imread(str(resolve_path(args.image)), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {args.image}")
    else:
        image = torch.zeros(480, 640, 3, dtype=torch.uint8).numpy()

    for _ in range(max(0, args.warmup)):
        predictor.predict_array(image, apply_raw_crop=args.apply_crop)
    if predictor.device.type == "cuda":
        torch.cuda.synchronize(predictor.device)
        torch.cuda.reset_peak_memory_stats(predictor.device)

    latencies = []
    for _ in range(max(1, args.iters)):
        started = time.perf_counter()
        predictor.predict_array(image, apply_raw_crop=args.apply_crop)
        if predictor.device.type == "cuda":
            torch.cuda.synchronize(predictor.device)
        latencies.append((time.perf_counter() - started) * 1000.0)

    avg_ms = statistics.mean(latencies)
    gpu_memory_mb = 0.0
    if predictor.device.type == "cuda":
        gpu_memory_mb = torch.cuda.max_memory_allocated(predictor.device) / (1024.0 * 1024.0)
    result = {
        "model_name": str(config.get("model", {}).get("name", "endodac_encoder_classifier")),
        "input_size": predictor.input_size,
        "device": str(predictor.device),
        "avg_latency_ms": avg_ms,
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "fps": 1000.0 / avg_ms if avg_ms > 0 else 0.0,
        "gpu_memory_mb": gpu_memory_mb,
        "iters": max(1, args.iters),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

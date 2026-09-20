#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from distance_state_classifier.src.config import load_config
from distance_state_classifier.src.predictor import DistanceStatePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distance-state inference on one image.")
    parser.add_argument("--config", default="distance_state_classifier/configs/distance_state_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--apply-crop", action="store_true", help="Apply raw-frame crop before inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    predictor = DistanceStatePredictor(args.checkpoint, config, device=args.device)
    result = predictor.predict_image(args.image, apply_raw_crop=args.apply_crop)
    print(
        json.dumps(
            {
                "image": args.image,
                "raw_label": result.raw_label,
                "raw_confidence": result.raw_confidence,
                "state": result.state,
                "smoothed_state": result.smoothed_state,
                "probabilities": result.probabilities,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

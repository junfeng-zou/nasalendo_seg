#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from distance_state_classifier_endodac.src.config import get_nested, load_config, resolve_path
from distance_state_classifier_endodac.src.dataset import read_label_csv
from distance_state_classifier_endodac.src.mask_utils import mask_cache_path, write_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cached instrument masks with a YOLO segmentation model.")
    parser.add_argument("--config", default="distance_state_classifier_endodac/configs/distance_state_endodac_maskaware_config.yaml")
    parser.add_argument("--model", default=None, help="YOLO segmentation checkpoint path.")
    parser.add_argument("--output-dir", default=None, help="Mask cache directory. Defaults to config mask.cache_dir.")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None, help="YOLO device, e.g. 0, cuda:0, or cpu.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int, default=None, help="Debug limit.")
    return parser.parse_args()


def collect_image_paths(config: dict) -> list[Path]:
    classes = set(get_nested(config, "data.classes", ["TooFar", "Good", "TooClose"]))
    image_column = str(get_nested(config, "data.image_column", "image_path"))
    label_column = str(get_nested(config, "data.label_column", "label"))
    paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        csv_path = get_nested(config, f"data.{split}_csv")
        for row in read_label_csv(csv_path):
            if row.get(label_column) not in classes:
                continue
            image_path = resolve_path(row[image_column])
            paths[image_path.as_posix()] = image_path
    return list(paths.values())


def union_result_masks(result, image_shape: tuple[int, int], class_ids: set[int] | None = None) -> np.ndarray:
    h, w = image_shape
    union = np.zeros((h, w), dtype=np.uint8)
    if getattr(result, "masks", None) is None or result.masks is None:
        return union
    masks = result.masks.data
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    else:
        masks = np.asarray(masks)

    keep = range(len(masks))
    if class_ids is not None and getattr(result, "boxes", None) is not None and result.boxes is not None:
        cls = result.boxes.cls
        if hasattr(cls, "detach"):
            cls = cls.detach().cpu().numpy()
        keep = [idx for idx, value in enumerate(cls.tolist()) if int(value) in class_ids]

    for idx in keep:
        resized = cv2.resize(masks[idx], (w, h), interpolation=cv2.INTER_NEAREST)
        union[resized > 0.5] = 255
    return union


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_path = resolve_path(args.model or get_nested(config, "mask.yolo_model", "runs/segment/runs/yolo11s_seg_formal/weights/best.pt"))
    cache_dir = resolve_path(args.output_dir or get_nested(config, "mask.cache_dir", "distance_state_classifier_endodac/mask_cache/yolo11s_seg_formal"))
    conf = float(args.conf if args.conf is not None else get_nested(config, "mask.conf", 0.25))
    iou = float(args.iou if args.iou is not None else get_nested(config, "mask.iou", 0.7))
    imgsz = int(args.imgsz if args.imgsz is not None else get_nested(config, "mask.imgsz", 1024))
    device = str(args.device if args.device is not None else get_nested(config, "mask.device", "0"))
    class_ids_cfg = get_nested(config, "mask.class_ids", None)
    class_ids = {int(x) for x in class_ids_cfg} if class_ids_cfg else None

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is required to generate masks. Install it in the nasalendo_seg environment.") from exc

    image_paths = collect_image_paths(config)
    if args.max_images is not None:
        image_paths = image_paths[: max(0, int(args.max_images))]
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Generate YOLO mask cache")
    print(f"model: {model_path}")
    print(f"cache_dir: {cache_dir}")
    print(f"images: {len(image_paths)}")
    print(f"conf={conf} iou={iou} imgsz={imgsz} device={device}")
    print("=" * 72)

    model = YOLO(str(model_path))
    started = time.time()
    written = 0
    skipped = 0
    empty = 0
    for idx, image_path in enumerate(image_paths, start=1):
        out_path = mask_cache_path(cache_dir, image_path)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        results = model.predict(source=image, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)
        mask = union_result_masks(results[0], image.shape[:2], class_ids=class_ids)
        if np.count_nonzero(mask) == 0:
            empty += 1
        write_mask(out_path, mask)
        written += 1
        if idx % 50 == 0 or idx == len(image_paths):
            elapsed = max(time.time() - started, 1e-6)
            print(f"[mask] {idx}/{len(image_paths)} written={written} skipped={skipped} empty={empty} speed={idx / elapsed:.2f}/s")

    summary = {
        "model": model_path.as_posix(),
        "cache_dir": cache_dir.as_posix(),
        "images": len(image_paths),
        "written": written,
        "skipped": skipped,
        "empty": empty,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "device": device,
    }
    (cache_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[mask] wrote summary: {cache_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

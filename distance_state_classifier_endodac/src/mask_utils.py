from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .config import resolve_path


def mask_cache_path(cache_dir: str | Path, image_path: str | Path) -> Path:
    image = resolve_path(image_path)
    key = hashlib.sha1(image.as_posix().encode("utf-8")).hexdigest()
    root = resolve_path(cache_dir)
    return root / key[:2] / f"{key}.png"


def read_mask(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    mask_path = resolve_path(path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        if shape is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        return np.zeros(shape, dtype=np.uint8)
    if shape is not None and mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary = (mask > 0).astype(np.uint8) * 255
    if not cv2.imwrite(str(output_path), binary):
        raise RuntimeError(f"Failed to write mask: {output_path}")

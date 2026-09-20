from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import cv2
import torch
from torch.utils.data import Dataset

from .config import resolve_path
from .transforms import CropBox, apply_crop, augment_image, resize_and_normalize


def read_label_csv(path: str | Path) -> list[dict]:
    csv_path = resolve_path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class DistanceStateDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        classes: Sequence[str],
        image_column: str,
        label_column: str,
        input_size: int,
        training: bool,
        augmentation_cfg: dict | None = None,
        crop: CropBox | None = None,
    ) -> None:
        self.rows = read_label_csv(csv_path)
        self.classes = list(classes)
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        self.image_column = image_column
        self.label_column = label_column
        self.input_size = int(input_size)
        self.training = bool(training)
        self.augmentation_cfg = augmentation_cfg or {}
        self.crop = crop

        self.rows = [row for row in self.rows if row.get(label_column) in self.class_to_idx]
        if not self.rows:
            raise RuntimeError(f"No usable rows found in {csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        row = self.rows[index]
        image_path = resolve_path(row[self.image_column])
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = apply_crop(image_rgb, self.crop)
        if self.training and self.augmentation_cfg.get("enabled", False):
            image_rgb = augment_image(image_rgb, self.augmentation_cfg)
        x = resize_and_normalize(image_rgb, self.input_size)
        y = self.class_to_idx[row[self.label_column]]
        meta = {
            "sample_id": row.get("sample_id", ""),
            "image_path": image_path.as_posix(),
            "label": row[self.label_column],
            "video_id": row.get("video_id", ""),
            "frame_index": row.get("frame_index", ""),
        }
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), meta


def class_counts(dataset: DistanceStateDataset) -> dict[str, int]:
    counts = {label: 0 for label in dataset.classes}
    for row in dataset.rows:
        counts[row[dataset.label_column]] += 1
    return counts

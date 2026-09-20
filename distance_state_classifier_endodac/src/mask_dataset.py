from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import torch

from .config import resolve_path
from .dataset import DistanceStateDataset
from .mask_utils import mask_cache_path, read_mask
from .transforms import CropBox, apply_crop, augment_image, resize_and_normalize


class MaskAwareDistanceStateDataset(DistanceStateDataset):
    def __init__(
        self,
        csv_path: str | Path,
        classes: Sequence[str],
        image_column: str,
        label_column: str,
        input_size: int,
        training: bool,
        mask_cache_dir: str | Path,
        augmentation_cfg: dict | None = None,
        normalize_cfg: dict | None = None,
        crop: CropBox | None = None,
        missing_mask_policy: str = "zeros",
    ) -> None:
        super().__init__(
            csv_path=csv_path,
            classes=classes,
            image_column=image_column,
            label_column=label_column,
            input_size=input_size,
            training=training,
            augmentation_cfg=augmentation_cfg,
            normalize_cfg=normalize_cfg,
            crop=crop,
        )
        self.mask_cache_dir = mask_cache_dir
        self.missing_mask_policy = str(missing_mask_policy)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        row = self.rows[index]
        image_path = resolve_path(row[self.image_column])
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask_path = mask_cache_path(self.mask_cache_dir, image_path)
        if mask_path.exists() or self.missing_mask_policy == "zeros":
            mask = read_mask(mask_path, shape=image_rgb.shape[:2])
        else:
            raise RuntimeError(f"Missing cached mask for image: {image_path}")

        image_rgb = apply_crop(image_rgb, self.crop)
        mask = apply_crop(mask, self.crop)
        if self.training and self.augmentation_cfg.get("enabled", False):
            # Keep augmentation mask-safe: color/noise/blur do not change geometry.
            image_rgb = augment_image(image_rgb, {**self.augmentation_cfg, "rotate_deg": 0, "translate_frac": 0, "scale_frac": 0, "horizontal_flip_prob": 0})

        x = resize_and_normalize(image_rgb, self.input_size, self.normalize_cfg)
        mask_resized = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        mask_tensor = (mask_resized > 0).astype("float32")[None, :, :]
        y = self.class_to_idx[row[self.label_column]]
        meta = {
            "sample_id": row.get("sample_id", ""),
            "image_path": image_path.as_posix(),
            "mask_path": mask_path.as_posix(),
            "label": row[self.label_column],
            "video_id": row.get("video_id", ""),
            "frame_index": row.get("frame_index", ""),
        }
        return torch.from_numpy(x), torch.from_numpy(mask_tensor), torch.tensor(y, dtype=torch.long), meta

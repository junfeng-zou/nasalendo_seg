"""Scale-preserving paired views for experiment B (masks are training-only)."""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import resolve_path
from .dataset import DistanceStateDataset
from .mask_utils import mask_cache_path
from .transforms import apply_crop, resize_and_normalize


def appearance_regions(image_rgb: np.ndarray, mask: np.ndarray, guard_px: int = 2) -> dict:
    if mask.shape != image_rgb.shape[:2]:
        raise ValueError("Image and mask shapes must match")
    if guard_px < 1:
        raise ValueError("guard_px must be positive")
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours((gray > 8).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fov = np.zeros(mask.shape, np.uint8)
    if contours:
        cv2.drawContours(fov, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * guard_px + 1, 2 * guard_px + 1))
    inside = cv2.erode(fov, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    foreground = (mask > 0).astype(np.uint8)
    inner = cv2.erode(foreground, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    outer = cv2.dilate(foreground, kernel) > 0
    # An empty mask must not turn the entire image into a "known background".
    # Such samples retain their clean view and label, and receive no intervention.
    valid = bool(foreground.any() and inside.any())
    return {"fg": inner & inside & valid, "bg": ~outer & inside & valid, "fov": inside}


def augment_appearance(image_rgb: np.ndarray, mask: np.ndarray, cfg: dict,
                       rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Edit chroma and mildly edit luminance; never warp, crop, blur or occlude.

    Protected boundary pixels and the black FOV border are copied exactly.
    Lab L is unchanged during chroma edits, although RGB gamut conversion can
    alter the reconstructed luminance. The image and mask are never mutated.
    """
    regions = appearance_regions(image_rgb, mask, int(cfg.get("guard_px", 2)))
    out = image_rgb.copy()
    info = {"mask_nonempty": bool(np.any(mask)), "fg_pixels": int(regions["fg"].sum()),
            "bg_pixels": int(regions["bg"].sum()), "fg_edited": False, "bg_edited": False}
    if not info["mask_nonempty"] or rng.random() < float(cfg.get("identity_prob", 0.1)):
        return out, info
    lab = cv2.cvtColor(image_rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    changed = lab.copy()
    selected = np.zeros(mask.shape, bool)
    for region_name in ("fg", "bg"):
        region = regions[region_name]
        if not region.any() or rng.random() >= float(cfg.get(f"{region_name}_prob", 0.8)):
            continue
        angle = math.radians(rng.uniform(-float(cfg.get("chroma_degrees", 35)), float(cfg.get("chroma_degrees", 35))))
        scale = rng.uniform(*cfg.get("chroma_scale", [0.5, 1.4]))
        if rng.random() < float(cfg.get("desaturate_prob", 0.2)):
            scale = 0.0
        a, b = lab[:, :, 1][region], lab[:, :, 2][region]
        changed[:, :, 1][region] = scale * (math.cos(angle) * a - math.sin(angle) * b)
        changed[:, :, 2][region] = scale * (math.sin(angle) * a + math.cos(angle) * b)
        selected |= region
        info[f"{region_name}_edited"] = True
    # A shared, small luminance adjustment avoids artificial brightness jumps
    # between foreground and background. The protected boundary is still fixed.
    if rng.random() < float(cfg.get("luminance_prob", 0.3)):
        valid = regions["fg"] | regions["bg"]
        contrast = rng.uniform(*cfg.get("luminance_contrast", [0.95, 1.05]))
        offset = rng.uniform(-float(cfg.get("luminance_offset", 3)), float(cfg.get("luminance_offset", 3)))
        changed[:, :, 0][valid] = np.clip((lab[:, :, 0][valid] - 50) * contrast + 50 + offset, 0, 100)
        selected |= valid
    converted = np.rint(np.clip(cv2.cvtColor(changed, cv2.COLOR_LAB2RGB), 0, 1) * 255).astype(np.uint8)
    out[selected] = converted[selected]
    return out, info


def jensen_shannon_logits(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Mean symmetric JS divergence, in nats, with gradients through both views."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Expected matching [batch, classes] logits")
    log_p, log_q = F.log_softmax(first.float(), dim=-1), F.log_softmax(second.float(), dim=-1)
    log_m = torch.logaddexp(log_p, log_q) - math.log(2)
    divergence = 0.5 * ((log_p.exp() * (log_p - log_m)).sum(-1)
                        + (log_q.exp() * (log_q - log_m)).sum(-1))
    return divergence.mean().clamp_min(0)


def paired_objective(clean_logits, augmented_logits, targets, criterion, consistency_weight):
    ce = 0.5 * (criterion(clean_logits, targets) + criterion(augmented_logits, targets))
    js = jensen_shannon_logits(clean_logits, augmented_logits)
    return ce + float(consistency_weight) * js, ce, js


class AppearancePairDataset(DistanceStateDataset):
    """The clean branch exactly follows the existing 392-square preprocessing."""

    def __init__(self, *args, appearance_cfg: dict, **kwargs):
        super().__init__(*args, **kwargs)
        if self.augmentation_cfg.get("enabled", False):
            raise ValueError("Disable legacy augmentation for experiment B; it includes scale changes")
        self.appearance_cfg = appearance_cfg
        self.cache_dir = appearance_cfg["mask_cache"]
        self.cache_inputs = bool(appearance_cfg.get("cache_inputs", True))
        self._input_cache = {}

    def inputs(self, index):
        if index in self._input_cache:
            return self._input_cache[index]
        row = self.rows[index]
        path = resolve_path(row[self.image_column])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        mask_path = mask_cache_path(self.cache_dir, path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if mask is None:
            raise RuntimeError(f"Missing or unreadable cached mask: {mask_path} ({path})")
        if mask.shape != image.shape[:2]:
            raise ValueError(f"Cached mask shape mismatch: {mask_path}")
        image = apply_crop(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), self.crop)
        mask = apply_crop(mask, self.crop)
        shape = (self.input_size, self.input_size)
        image = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, shape, interpolation=cv2.INTER_NEAREST)
        if self.cache_inputs:
            self._input_cache[index] = (image, mask)
        return image, mask

    def __getitem__(self, index):
        image, mask = self.inputs(index)
        # Worker seeds are initialized by DataLoader. numpy's global stream is
        # seeded by the trainer and saved in checkpoints for epoch-boundary resume.
        rng = np.random.default_rng(int(np.random.randint(0, 2**32)))
        augmented, info = augment_appearance(image, mask, self.appearance_cfg, rng)
        clean = resize_and_normalize(image, self.input_size, self.normalize_cfg)
        changed = resize_and_normalize(augmented, self.input_size, self.normalize_cfg)
        row = self.rows[index]
        label = self.class_to_idx[row[self.label_column]]
        return torch.from_numpy(np.stack([clean, changed])), torch.tensor(label, dtype=torch.long), info

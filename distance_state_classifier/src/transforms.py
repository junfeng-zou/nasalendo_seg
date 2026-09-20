from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class CropBox:
    x_left: int
    x_right: int
    y_top: int = 0
    y_bottom: int | None = None


def apply_crop(image: np.ndarray, crop: CropBox | None) -> np.ndarray:
    if crop is None:
        return image
    h, w = image.shape[:2]
    y_bottom = crop.y_bottom if crop.y_bottom is not None else h
    x1 = max(0, min(w, crop.x_left))
    x2 = max(x1 + 1, min(w, crop.x_right))
    y1 = max(0, min(h, crop.y_top))
    y2 = max(y1 + 1, min(h, y_bottom))
    return image[y1:y2, x1:x2]


def resize_and_normalize(image_rgb: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


def _jitter_color(image: np.ndarray, cfg: dict) -> np.ndarray:
    out = image.astype(np.float32)
    brightness = float(cfg.get("brightness", 0.0))
    contrast = float(cfg.get("contrast", 0.0))
    if brightness > 0:
        factor = 1.0 + random.uniform(-brightness, brightness)
        out *= factor
    if contrast > 0:
        factor = 1.0 + random.uniform(-contrast, contrast)
        mean = out.mean(axis=(0, 1), keepdims=True)
        out = (out - mean) * factor + mean
    out = np.clip(out, 0, 255).astype(np.uint8)

    saturation = float(cfg.get("saturation", 0.0))
    hue = float(cfg.get("hue", 0.0))
    if saturation > 0 or hue > 0:
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
        if hue > 0:
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-hue, hue) * 180.0) % 180.0
        if saturation > 0:
            hsv[:, :, 1] *= 1.0 + random.uniform(-saturation, saturation)
        hsv[:, :, 1:] = np.clip(hsv[:, :, 1:], 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


def _affine(image: np.ndarray, cfg: dict) -> np.ndarray:
    rotate_deg = float(cfg.get("rotate_deg", 0.0))
    translate_frac = float(cfg.get("translate_frac", 0.0))
    scale_frac = float(cfg.get("scale_frac", 0.0))
    if rotate_deg <= 0 and translate_frac <= 0 and scale_frac <= 0:
        return image
    h, w = image.shape[:2]
    angle = random.uniform(-rotate_deg, rotate_deg)
    scale = 1.0 + random.uniform(-scale_frac, scale_frac)
    tx = random.uniform(-translate_frac, translate_frac) * w
    ty = random.uniform(-translate_frac, translate_frac) * h
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[:, 2] += [tx, ty]
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def augment_image(image_rgb: np.ndarray, cfg: dict) -> np.ndarray:
    image = image_rgb
    if random.random() < float(cfg.get("horizontal_flip_prob", 0.0)):
        image = cv2.flip(image, 1)
    image = _affine(image, cfg)
    image = _jitter_color(image, cfg)

    if random.random() < float(cfg.get("blur_prob", 0.0)):
        k = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (k, k), 0)
    if random.random() < float(cfg.get("noise_prob", 0.0)):
        sigma = random.uniform(2.0, 8.0)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image

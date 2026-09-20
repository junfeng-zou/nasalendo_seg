"""Shared preprocessing, frozen backbone and equal-capacity heads for A/B/C."""
from __future__ import annotations

import math
import cv2
import numpy as np
import torch
from torch import nn

from .head_roi import crop_with_padding
from .transforms import IMAGENET_MEAN, IMAGENET_STD


def parse_head_box(annotation, width, height):
    shapes = annotation.get('shapes', [])
    if not shapes:
        return None
    if len(shapes) != 1:
        raise ValueError('Expected one target head box per image')
    shape = shapes[0]
    points = np.asarray(shape.get('points', []), dtype=np.float64)
    if shape.get('label') != 'forceps_head' or shape.get('shape_type') != 'rectangle':
        raise ValueError('Expected a forceps_head rectangle')
    if points.shape not in ((2, 2), (4, 2)) or not np.isfinite(points).all():
        raise ValueError('Invalid rectangle points')
    x1, y1 = points.min(0)
    x2, y2 = points.max(0)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('Invalid head box bounds')
    if len(points) == 4:
        expected = {(x1, y1), (x2, y1), (x2, y2), (x1, y2)}
        if set(map(tuple, points)) != expected:
            raise ValueError('Expected an axis-aligned rectangle')
    return [float(x1), float(y1), float(x2), float(y2)]


def expanded_box(box, factor=1.2):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * factor, (y2 - y1) * factor
    return [math.floor(cx - w / 2), math.floor(cy - h / 2),
            math.ceil(cx + w / 2), math.ceil(cy + h / 2)]


def letterbox(image, size=384):
    h, w = image.shape[:2]
    ratio = min(size / w, size / h)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
    result = np.zeros((size, size, 3), dtype=np.uint8)
    x, y = (size - nw) // 2, (size - nh) // 2
    result[y:y + nh, x:x + nw] = resized
    return result


def prepare_inputs(image, box, fov_diameter, size=384, context_factor=1.2):
    if not np.isfinite(fov_diameter) or fov_diameter <= 0:
        raise ValueError('Invalid FOV diameter')
    local = crop_with_padding(image, expanded_box(box, context_factor))
    scales = np.asarray([(box[2] - box[0]) / fov_diameter,
                         (box[3] - box[1]) / fov_diameter], dtype=np.float32)
    return letterbox(image, size), letterbox(local, size), scales


def image_tensor(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return torch.from_numpy(np.ascontiguousarray(((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)))


def fit_scale_normalizer(scales, train_mask):
    training = np.asarray(scales, dtype=np.float32)[np.asarray(train_mask, dtype=bool)]
    if len(training) == 0:
        raise ValueError('No training samples for normalization')
    return training.mean(0), np.maximum(training.std(0), 1e-6)


def make_backbone(pretrained_file=None):
    import timm
    backbone = timm.create_model('convnext_tiny', pretrained=False, num_classes=1000)
    if pretrained_file is not None:
        from safetensors.torch import load_file
        backbone.load_state_dict(load_file(str(pretrained_file)), strict=True)
    backbone.reset_classifier(0)
    backbone.requires_grad_(False)
    return backbone.eval()


class FusionHead(nn.Module):
    """Same parameters for all arms; A/B receive two constant zeros, C scales."""
    def __init__(self, hidden=128, dropout=.2):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.classifier = nn.Sequential(nn.Linear(770, hidden), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(hidden, 3))

    def forward(self, visual, scales):
        return self.classifier(torch.cat([self.norm(visual), scales], dim=-1))

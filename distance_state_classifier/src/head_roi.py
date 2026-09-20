"""Label-independent tip-guided ROI geometry; never normalizes by tool size."""
from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np


@lru_cache(maxsize=1)
def tip_geometry():
    # Import functions only. The live camera/robot main() is never executed.
    path = Path(__file__).resolve().parents[2] / "inference/realtime_improved_tip.py"
    name = "_nasal_head_roi_tip_geometry"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def estimate_fov(image):
    """Estimate the optical aperture from non-black pixels, excluding image edges.

    The first fixed number of images per video are combined by median during
    preparation. An ellipse supports the existing cropped/partly clipped FOV.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours((gray > 8).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("no visible FOV")
    points = max(contours, key=cv2.contourArea)[:, 0, :]
    keep = (points[:, 0] > 2) & (points[:, 0] < w - 3) & (points[:, 1] > 2) & (points[:, 1] < h - 3)
    points = points[keep]
    if len(points) < 20:
        raise ValueError("insufficient optical boundary for FOV calibration")
    center, axes, angle = cv2.fitEllipse(points[:, None, :].astype(np.float32))
    diameter = float(np.sqrt(axes[0] * axes[1]))
    if not (0 <= center[0] < w and 0 <= center[1] < h and .5 * min(h, w) < diameter < 1.8 * max(h, w)):
        raise ValueError("implausible FOV ellipse")
    return {"center": list(center), "diameter": diameter, "axes": list(axes), "angle": float(angle), "shape": [h, w]}


def locate_roi(mask, fov, cfg):
    """Independent current-frame localization, with no stale temporal holds.

    Cached binary masks have no YOLO score. The returned score describes only
    geometry, and is not a calibrated probability of correct head localization.
    """
    if list(mask.shape) != fov["shape"]:
        raise ValueError("Mask/FOV coordinate systems differ")
    geometry = tip_geometry()
    cleaned = geometry._largest_clean_component(mask, 1)
    if cleaned is None or np.count_nonzero(cleaned) < cfg["min_mask_pixels"]:
        return {"valid": False, "reason": "empty_or_tiny_mask"}
    fraction = float(np.count_nonzero(cleaned) / max(np.count_nonzero(mask), 1))
    if fraction < cfg["min_component_fraction"]:
        return {"valid": False, "reason": "ambiguous_components", "component_fraction": fraction}
    observation, _ = geometry.extract_improved_tip(
        cleaned, detection_confidence=1.0, fov_center=np.asarray(fov["center"]),
        base_radial_band=30.0 * fov["diameter"] / 1200.0,
    )
    if observation is None:
        return {"valid": False, "reason": "no_tip"}
    point, direction = observation.tip, observation.direction
    h, w = mask.shape
    if not (np.isfinite(point).all() and np.isfinite(direction).all() and 0 <= point[0] < w and 0 <= point[1] < h):
        return {"valid": False, "reason": "tip_outside_image"}
    side = max(2, int(round(cfg["side_fraction"] * fov["diameter"])))
    center = point - cfg["backshift_fraction"] * side * direction
    x1, y1 = np.rint(center - side / 2).astype(int)
    box = [int(x1), int(y1), int(x1 + side), int(y1 + side)]
    visible = max(0, min(w, box[2]) - max(0, box[0])) * max(0, min(h, box[3]) - max(0, box[1])) / side**2
    valid = observation.confidence >= cfg["min_geometry_score"] and visible >= cfg["min_visible_fraction"]
    return {"valid": bool(valid), "reason": "ok" if valid else "weak_geometry_or_clipped_roi", "box": box,
            "tip": point.tolist(), "direction": direction.tolist(), "geometry_score": observation.confidence,
            "mode": observation.mode, "visible_fraction": visible, "component_fraction": fraction,
            "fov_diameter": fov["diameter"], "side": side}


def crop_with_padding(image, box):
    """Preserve the requested canvas even where the box crosses image edges."""
    x1, y1, x2, y2 = [int(v) for v in box]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI must have positive area")
    out = np.zeros((y2 - y1, x2 - x1, *image.shape[2:]), dtype=image.dtype)
    h, w = image.shape[:2]
    left, top, right, bottom = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if right > left and bottom > top:
        out[top - y1:bottom - y1, left - x1:right - x1] = image[top:bottom, left:right]
    return out

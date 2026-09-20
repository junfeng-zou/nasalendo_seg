from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import get_nested, resolve_path
from .model import build_model
from .transforms import CropBox, apply_crop, resize_and_normalize


@dataclass
class Prediction:
    raw_label: str
    raw_confidence: float
    state: str
    smoothed_state: str
    probabilities: dict[str, float]


class StateSmoother:
    def __init__(self, window: int = 5, min_state_count: int = 3, prefer_tooclose: bool = True) -> None:
        self.history: deque[str] = deque(maxlen=max(1, int(window)))
        self.min_state_count = max(1, int(min_state_count))
        self.prefer_tooclose = bool(prefer_tooclose)
        self.current_state = "Invalid"

    def update(self, state: str) -> str:
        self.history.append(state)
        counts = {item: list(self.history).count(item) for item in set(self.history)}
        if self.prefer_tooclose and counts.get("TooClose", 0) >= self.min_state_count:
            self.current_state = "TooClose"
            return self.current_state
        best_state, best_count = max(counts.items(), key=lambda item: item[1])
        if best_count >= self.min_state_count:
            self.current_state = best_state
        return self.current_state


class DistanceStatePredictor:
    def __init__(self, checkpoint_path: str | Path, config: dict | None = None, device: str | None = None) -> None:
        checkpoint = torch.load(resolve_path(checkpoint_path), map_location="cpu")
        self.config = checkpoint.get("config") or config or {}
        self.classes = checkpoint.get("classes") or list(get_nested(self.config, "data.classes", ["TooFar", "Good", "TooClose"]))
        self.input_size = int(checkpoint.get("input_size") or get_nested(self.config, "image.input_size", 392))
        self.normalize_cfg = dict(get_nested(self.config, "image.normalize", {"mode": "imagenet"}))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model_cfg = dict(get_nested(self.config, "model", {}))
        model_name = str(checkpoint.get("model_name") or model_cfg.get("name", "endodac_encoder_classifier"))
        self.mask_aware = model_name.lower().strip() in {"endodac_mask_aware_classifier", "depth_encoder_mask_aware_classifier"} or bool(
            get_nested(model_cfg, "mask_aware.enabled", False)
        )
        self.model = build_model(
            name=model_name,
            num_classes=len(self.classes),
            dropout=float(model_cfg.get("dropout", 0.2)),
            pretrained=False,
            config=model_cfg,
            input_size=self.input_size,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()
        self.confidence_threshold = float(get_nested(self.config, "inference.confidence_threshold", 0.55))
        self.crop = CropBox(
            x_left=int(get_nested(self.config, "image.crop.x_left", 362)),
            x_right=int(get_nested(self.config, "image.crop.x_right", 1605)),
            y_top=int(get_nested(self.config, "image.crop.y_top", 0)),
            y_bottom=int(get_nested(self.config, "image.crop.y_bottom", 1080)),
        )
        self.smoother = StateSmoother(
            window=int(get_nested(self.config, "inference.smoothing_window", 5)),
            min_state_count=int(get_nested(self.config, "inference.min_state_count", 3)),
            prefer_tooclose=bool(get_nested(self.config, "inference.prefer_tooclose", True)),
        )

    def _prepare_mask_tensor(self, mask: np.ndarray, shape: tuple[int, int], apply_raw_crop: bool) -> torch.Tensor:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[:2] != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        if apply_raw_crop:
            mask = apply_crop(mask, self.crop)
        mask_resized = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        mask_arr = (mask_resized > 0).astype("float32")[None, None, :, :]
        return torch.from_numpy(mask_arr).to(self.device)

    def predict_array(self, image_bgr: np.ndarray, apply_raw_crop: bool = False, mask: np.ndarray | None = None) -> Prediction:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        raw_shape = image_rgb.shape[:2]
        if apply_raw_crop:
            image_rgb = apply_crop(image_rgb, self.crop)
        tensor = torch.from_numpy(resize_and_normalize(image_rgb, self.input_size, self.normalize_cfg)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            if self.mask_aware:
                if mask is None:
                    raise ValueError("This checkpoint is mask-aware and requires an instrument mask for inference.")
                mask_tensor = self._prepare_mask_tensor(mask, raw_shape, apply_raw_crop=apply_raw_crop)
                logits = self.model(tensor, mask_tensor)
            else:
                logits = self.model(tensor)
            probs_tensor = torch.softmax(logits, dim=1)[0].detach().cpu()
        probs = {label: float(probs_tensor[idx]) for idx, label in enumerate(self.classes)}
        pred_idx = int(torch.argmax(probs_tensor).item())
        raw_label = self.classes[pred_idx]
        confidence = float(probs_tensor[pred_idx])
        state = raw_label if confidence >= self.confidence_threshold else "Invalid"
        smoothed = self.smoother.update(state)
        return Prediction(raw_label=raw_label, raw_confidence=confidence, state=state, smoothed_state=smoothed, probabilities=probs)

    def predict_image(self, image_path: str | Path, apply_raw_crop: bool = False, mask_path: str | Path | None = None) -> Prediction:
        path = resolve_path(image_path)
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        mask = None
        if mask_path is not None:
            mask = cv2.imread(str(resolve_path(mask_path)), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Failed to read mask: {mask_path}")
        return self.predict_array(image_bgr, apply_raw_crop=apply_raw_crop, mask=mask)

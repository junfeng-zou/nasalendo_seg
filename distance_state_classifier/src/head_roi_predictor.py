"""Offline inference for the ROI/ordinal experiment, including unavailable ROIs."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .config import resolve_path
from .convnext_roi_ordinal import ConvNeXtROIOrdinal
from .head_roi import crop_with_padding, locate_roi
from .transforms import resize_and_normalize


class HeadRoiOrdinalPredictor:
    def __init__(self, checkpoint_path, device="cpu"):
        checkpoint = torch.load(resolve_path(checkpoint_path), map_location="cpu", weights_only=True)
        if checkpoint["model_name"] != "convnext_head_roi_ordinal":
            raise ValueError("Expected a convnext_head_roi_ordinal checkpoint")
        self.config, self.classes = checkpoint["config"], checkpoint["classes"]
        cfg = self.config["model"]
        self.model = ConvNeXtROIOrdinal(cfg["hidden_dim"], cfg["grid_size"], cfg["dropout"])
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    def probabilities_for_fixed_roi(self, images, geometry, batch_size=8):
        """For paired appearance tests, keep the original-frame localization fixed."""
        if not geometry["valid"]:
            return [None] * len(images)
        results = []
        for start in range(0, len(images), batch_size):
            arrays = [resize_and_normalize(cv2.cvtColor(crop_with_padding(image, geometry["box"]), cv2.COLOR_BGR2RGB),
                                           self.config["image"]["input_size"]) for image in images[start:start + batch_size]]
            batch = torch.from_numpy(np.stack(arrays)).to(self.device)
            with torch.inference_mode():
                results.extend(self.model(batch).exp().cpu().numpy())
        return results

    def predict(self, image_bgr, instrument_mask, fov):
        """fov is calibrated once per video, in the same coordinates as the image."""
        geometry = locate_roi(instrument_mask, fov, self.config["roi"])
        if not geometry["valid"]:
            return {"raw_label": "Invalid", "state": "Invalid", "confidence": None, "probabilities": None, "geometry": geometry}
        probabilities = self.probabilities_for_fixed_roi([image_bgr], geometry)[0]
        label = self.classes[int(np.argmax(probabilities))]
        confidence = float(max(probabilities))
        return {"raw_label": label, "state": label if confidence >= self.config["inference"]["confidence_threshold"] else "Invalid",
                "confidence": confidence, "probabilities": dict(zip(self.classes, probabilities.tolist())), "geometry": geometry}

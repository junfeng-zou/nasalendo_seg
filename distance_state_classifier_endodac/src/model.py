from __future__ import annotations

import torch
from torch import nn

from .depth_encoder_classifier import build_depth_encoder_classifier


def build_model(
    name: str,
    num_classes: int,
    dropout: float = 0.2,
    pretrained: bool = False,
    config: dict | None = None,
    input_size: int = 392,
) -> nn.Module:
    key = name.lower().strip()
    model_cfg = dict(config or {})
    model_cfg.setdefault("name", key)
    if key in {
        "endodac_encoder_classifier",
        "endodac_mask_aware_classifier",
        "depthanything_v2_encoder_classifier",
        "depth_encoder_classifier",
        "depth_encoder_mask_aware_classifier",
    }:
        return build_depth_encoder_classifier(
            config=model_cfg,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
            input_size=input_size,
        )
    if key.startswith("timm:"):
        model_name = key.split(":", 1)[1]
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("timm is required for timm model names.") from exc
        return timm.create_model(model_name, pretrained=bool(pretrained), num_classes=num_classes, drop_rate=float(dropout))
    if key in {"linear_probe_smoke", "tiny"}:
        from .depth_encoder_classifier import SimpleGeometryEncoder, infer_feature_dim, DepthEncoderDistanceClassifier

        encoder = SimpleGeometryEncoder(feature_dim=int(model_cfg.get("feature_dim") or 256))
        feature_dim = infer_feature_dim(encoder, input_size=input_size)
        return DepthEncoderDistanceClassifier(encoder, feature_dim, num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unsupported model name: {name}")


def load_model_state(model: nn.Module, checkpoint: dict) -> None:
    state = checkpoint["model_state"]
    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(state, strict=False)

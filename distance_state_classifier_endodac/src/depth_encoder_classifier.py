from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path
from typing import Any

import torch.nn.functional as F
import torch
from torch import nn

from .config import resolve_path


class SimpleGeometryEncoder(nn.Module):
    """Small CNN encoder used for local smoke tests when no depth encoder is installed."""

    def __init__(self, feature_dim: int = 384) -> None:
        super().__init__()
        channels = [3, 48, 96, 192, int(feature_dim)]
        layers: list[nn.Module] = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                ]
            )
        self.features = nn.Sequential(*layers)
        self.feature_dim = int(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class OfficialEndoDACEncoder(nn.Module):
    """Adapter that exposes EndoDAC's adapted ViT encoder features without running the depth head."""

    def __init__(
        self,
        repo_path: str | Path,
        pretrained_path: str | Path,
        checkpoint_path: str | Path | None = None,
        backbone_size: str = "base",
        lora_rank: int = 4,
        lora_type: str = "dvlora",
        image_shape: tuple[int, int] = (224, 280),
        residual_block_indexes: list[int] | None = None,
        include_cls_token: bool = True,
        feature_mode: str = "patch",
    ) -> None:
        super().__init__()
        repo = resolve_path(repo_path)
        if not repo.exists():
            raise RuntimeError(f"EndoDAC repo path does not exist: {repo}")
        repo_str = repo.as_posix()
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        try:
            from models.endodac.endodac import endodac
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import official EndoDAC code. Install missing packages in the nasalendo_seg environment "
                "or switch model.encoder.backend back to simple."
            ) from exc

        self.image_shape = tuple(int(v) for v in image_shape)
        self.feature_mode = str(feature_mode)
        self.model = endodac(
            backbone_size=str(backbone_size),
            r=int(lora_rank),
            lora_type=str(lora_type),
            image_shape=self.image_shape,
            pretrained_path=resolve_path(pretrained_path).as_posix(),
            residual_block_indexes=list(residual_block_indexes or [2, 5, 8, 11]),
            include_cls_token=bool(include_cls_token),
        )
        if checkpoint_path:
            state = _load_checkpoint_state(checkpoint_path)
            model_state = self.model.state_dict()
            filtered = {key: value for key, value in state.items() if key in model_state and model_state[key].shape == value.shape}
            missing, unexpected = self.model.load_state_dict(filtered, strict=False)
            if missing:
                warnings.warn(f"Missing EndoDAC checkpoint keys after filtered load: {len(missing)}", RuntimeWarning)
            if unexpected:
                warnings.warn(f"Unexpected EndoDAC checkpoint keys after filtered load: {len(unexpected)}", RuntimeWarning)

        self.feature_dim = int(self.model.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.image_shape, mode="bilinear", align_corners=True)
        features = self.model.encoder.get_intermediate_layers(x, 4, return_class_token=True)
        if self.feature_mode == "multiscale_tokens":
            return [item[0] if isinstance(item, tuple) else item for item in features]
        if self.feature_mode == "multiscale_mean":
            pooled = []
            for item in features:
                patch_tokens = item[0] if isinstance(item, tuple) else item
                pooled.append(patch_tokens.mean(dim=1))
            return torch.cat(pooled, dim=1)
        if self.feature_mode == "multiscale_cls_mean":
            pooled = []
            for item in features:
                if isinstance(item, tuple):
                    patch_tokens, cls_token = item
                    pooled.extend([cls_token, patch_tokens.mean(dim=1)])
                else:
                    pooled.append(item.mean(dim=1))
            return torch.cat(pooled, dim=1)
        last = features[-1]
        if isinstance(last, tuple):
            patch_tokens, cls_token = last
            if self.feature_mode == "cls":
                return cls_token
            if self.feature_mode == "concat_cls":
                return torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
            return patch_tokens
        return last


class DepthEncoderDistanceClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        num_classes: int = 3,
        dropout: float = 0.2,
        pool_type: str = "mean",
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.feature_dim = int(feature_dim)
        self.pool_type = str(pool_type)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def _select_feature(self, features: Any) -> torch.Tensor:
        if isinstance(features, dict):
            if "last_hidden_state" in features:
                return features["last_hidden_state"]
            if "features" in features:
                return self._select_feature(features["features"])
            if not features:
                raise RuntimeError("Encoder returned an empty feature dict")
            return self._select_feature(next(reversed(features.values())))
        if isinstance(features, (list, tuple)):
            if not features:
                raise RuntimeError("Encoder returned an empty feature sequence")
            return self._select_feature(features[-1])
        if not isinstance(features, torch.Tensor):
            raise RuntimeError(f"Unsupported encoder output type: {type(features)!r}")
        return features

    def pool_features(self, features: Any) -> torch.Tensor:
        x = self._select_feature(features)
        if x.ndim == 4:
            return x.mean(dim=(2, 3))
        if x.ndim == 3:
            if self.pool_type == "cls":
                return x[:, 0]
            if self.pool_type == "patch_mean" and x.shape[1] > 1:
                return x[:, 1:].mean(dim=1)
            return x.mean(dim=1)
        if x.ndim == 2:
            return x
        raise RuntimeError(f"Unsupported feature shape: {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool_features(self.encoder(x)))

    def get_classifier(self) -> nn.Module:
        return self.classifier


class MaskAwareDepthEncoderDistanceClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        num_classes: int = 3,
        dropout: float = 0.2,
        hidden_dim: int = 512,
        dilation_px: int = 10,
        use_global: bool = True,
        use_raw_mask: bool = True,
        use_dilated_mask: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.feature_dim = int(feature_dim)
        self.dilation_px = max(0, int(dilation_px))
        self.use_global = bool(use_global)
        self.use_raw_mask = bool(use_raw_mask)
        self.use_dilated_mask = bool(use_dilated_mask)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def get_classifier(self) -> nn.Module:
        return self.classifier

    def _feature_grid(self, num_tokens: int) -> tuple[int, int]:
        image_shape = getattr(self.encoder, "image_shape", None)
        if image_shape is not None:
            h, w = int(image_shape[0]) // 14, int(image_shape[1]) // 14
            if h * w == num_tokens:
                return h, w
        side = int(num_tokens**0.5)
        if side * side == num_tokens:
            return side, side
        return 1, num_tokens

    def _dilate_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if self.dilation_px <= 0:
            return mask
        kernel = 2 * self.dilation_px + 1
        return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=self.dilation_px)

    def _weighted_pool(self, tokens: torch.Tensor, mask: torch.Tensor, global_feature: torch.Tensor) -> torch.Tensor:
        grid_h, grid_w = self._feature_grid(tokens.shape[1])
        weights = F.interpolate(mask, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
        weights = weights.flatten(1).clamp(0.0, 1.0).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1e-6)
        pooled = (tokens * weights).sum(dim=1) / denom
        has_mask = (weights.sum(dim=1) > 1e-5).expand_as(pooled)
        return torch.where(has_mask, pooled, global_feature)

    def _pool_token_features(self, features: Any, mask: torch.Tensor) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            feature_list = [features]
        elif isinstance(features, (list, tuple)):
            feature_list = list(features)
        else:
            raise RuntimeError(f"Unsupported mask-aware encoder output type: {type(features)!r}")

        raw_mask = mask.float().clamp(0.0, 1.0)
        dilated_mask = self._dilate_mask(raw_mask)
        pooled: list[torch.Tensor] = []
        for item in feature_list:
            tokens = item[0] if isinstance(item, tuple) else item
            if tokens.ndim != 3:
                raise RuntimeError(f"Mask-aware classifier expects ViT tokens [B,N,C], got {tuple(tokens.shape)}")
            global_feature = tokens.mean(dim=1)
            if self.use_global:
                pooled.append(global_feature)
            if self.use_raw_mask:
                pooled.append(self._weighted_pool(tokens, raw_mask, global_feature))
            if self.use_dilated_mask:
                pooled.append(self._weighted_pool(tokens, dilated_mask, global_feature))
        return torch.cat(pooled, dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(self._pool_token_features(features, mask))


def _load_checkpoint_state(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(resolve_path(path), map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("encoder_state", "encoder", "model_state", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict):
        return checkpoint
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def _strip_prefixes(state: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        out[new_key] = value
    return out


def load_checkpoint_into_encoder(encoder: nn.Module, checkpoint_path: str | Path | None, strict: bool = False) -> None:
    if not checkpoint_path:
        return
    state = _load_checkpoint_state(checkpoint_path)
    state = _strip_prefixes(state, ("module.", "encoder.", "model.encoder."))
    missing, unexpected = encoder.load_state_dict(state, strict=bool(strict))
    if missing:
        warnings.warn(f"Missing encoder checkpoint keys: {len(missing)}", RuntimeWarning)
    if unexpected:
        warnings.warn(f"Unexpected encoder checkpoint keys: {len(unexpected)}", RuntimeWarning)


def load_timm_encoder(model_name: str, pretrained: bool = True, checkpoint_path: str | Path | None = None) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is required for timm-based depth encoders. Install timm or use backend: simple.") from exc
    encoder = timm.create_model(model_name, pretrained=bool(pretrained), num_classes=0, global_pool="")
    load_checkpoint_into_encoder(encoder, checkpoint_path, strict=False)
    return encoder


def load_python_encoder(config: dict, pretrained: bool = True) -> nn.Module:
    module_name = config.get("module")
    class_name = config.get("class")
    if not module_name or not class_name:
        raise ValueError("python encoder backend requires model.encoder.module and model.encoder.class")
    module = importlib.import_module(str(module_name))
    cls = getattr(module, str(class_name))
    kwargs = dict(config.get("kwargs", {}))
    if "pretrained" not in kwargs:
        kwargs["pretrained"] = bool(pretrained)
    encoder = cls(**kwargs)
    if not isinstance(encoder, nn.Module):
        raise TypeError(f"{module_name}.{class_name} did not create an nn.Module")
    load_checkpoint_into_encoder(encoder, config.get("checkpoint"), strict=bool(config.get("strict_checkpoint", False)))
    return encoder


def _simple_feature_dim(config: dict, encoder_cfg: dict) -> int:
    value = encoder_cfg.get("simple_feature_dim")
    if value in (None, "null", "auto"):
        value = config.get("feature_dim")
    if value in (None, "null", "auto"):
        value = 384
    return int(value)


def load_endodac_encoder(config: dict, pretrained: bool = True) -> nn.Module:
    encoder_cfg = dict(config.get("encoder", {}))
    backend = str(encoder_cfg.get("backend", "timm")).lower()
    if backend in {"official", "official_endodac"}:
        return OfficialEndoDACEncoder(
            repo_path=encoder_cfg.get("repo_path", "distance_state_classifier_endodac/third_party/EndoDAC"),
            pretrained_path=encoder_cfg.get("pretrained_path", "distance_state_classifier_endodac/third_party/EndoDAC/pretrained_model"),
            checkpoint_path=encoder_cfg.get("checkpoint") or config.get("pretrained_checkpoint"),
            backbone_size=str(encoder_cfg.get("backbone_size", "base")),
            lora_rank=int(encoder_cfg.get("lora_rank", 4)),
            lora_type=str(encoder_cfg.get("lora_type", "dvlora")),
            image_shape=tuple(encoder_cfg.get("image_shape", [224, 280])),
            residual_block_indexes=list(encoder_cfg.get("residual_block_indexes", [2, 5, 8, 11])),
            include_cls_token=bool(encoder_cfg.get("include_cls_token", True)),
            feature_mode=str(encoder_cfg.get("feature_mode", "patch")),
        )
    if backend == "python":
        return load_python_encoder(encoder_cfg, pretrained=pretrained)
    if backend == "simple":
        feature_dim = _simple_feature_dim(config, encoder_cfg)
        encoder = SimpleGeometryEncoder(feature_dim=feature_dim)
        load_checkpoint_into_encoder(encoder, encoder_cfg.get("checkpoint") or config.get("pretrained_checkpoint"), strict=False)
        return encoder
    timm_name = str(encoder_cfg.get("timm_model_name", "vit_small_patch14_dinov2.lvd142m"))
    return load_timm_encoder(timm_name, pretrained=pretrained, checkpoint_path=encoder_cfg.get("checkpoint") or config.get("pretrained_checkpoint"))


def load_depthanything_v2_encoder(config: dict, pretrained: bool = True) -> nn.Module:
    encoder_cfg = dict(config.get("encoder", {}))
    backend = str(encoder_cfg.get("backend", "timm")).lower()
    if backend == "python":
        return load_python_encoder(encoder_cfg, pretrained=pretrained)
    if backend == "simple":
        feature_dim = _simple_feature_dim(config, encoder_cfg)
        encoder = SimpleGeometryEncoder(feature_dim=feature_dim)
        load_checkpoint_into_encoder(encoder, encoder_cfg.get("checkpoint") or config.get("pretrained_checkpoint"), strict=False)
        return encoder
    timm_name = str(encoder_cfg.get("timm_model_name", "vit_small_patch14_dinov2.lvd142m"))
    return load_timm_encoder(timm_name, pretrained=pretrained, checkpoint_path=encoder_cfg.get("checkpoint") or config.get("pretrained_checkpoint"))


def infer_feature_dim(encoder: nn.Module, input_size: int = 392, device: torch.device | str = "cpu", pool_type: str = "mean") -> int:
    was_training = encoder.training
    encoder.eval()
    device = torch.device(device)
    probe = torch.zeros(1, 3, int(input_size), int(input_size), device=device)
    helper = DepthEncoderDistanceClassifier(encoder, feature_dim=1, pool_type=pool_type)
    with torch.no_grad():
        pooled = helper.pool_features(encoder(probe))
    if was_training:
        encoder.train()
    return int(pooled.shape[-1])


def infer_mask_feature_dim(
    encoder: nn.Module,
    input_size: int = 392,
    device: torch.device | str = "cpu",
    dilation_px: int = 10,
    use_global: bool = True,
    use_raw_mask: bool = True,
    use_dilated_mask: bool = True,
) -> int:
    was_training = encoder.training
    encoder.eval()
    device = torch.device(device)
    probe = torch.zeros(1, 3, int(input_size), int(input_size), device=device)
    mask = torch.ones(1, 1, int(input_size), int(input_size), device=device)
    helper = MaskAwareDepthEncoderDistanceClassifier(
        encoder=encoder,
        feature_dim=1,
        dilation_px=dilation_px,
        use_global=use_global,
        use_raw_mask=use_raw_mask,
        use_dilated_mask=use_dilated_mask,
    )
    with torch.no_grad():
        pooled = helper._pool_token_features(encoder(probe), mask)
    if was_training:
        encoder.train()
    return int(pooled.shape[-1])


def parameter_counts(model: nn.Module) -> dict[str, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def set_trainable_scope(model: nn.Module, scope: str = "head_only", last_blocks: int = 2) -> dict[str, int]:
    scope = str(scope or "head_only").lower()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    if scope in {"head_only", "frozen", "classifier"}:
        return parameter_counts(model)
    if scope in {"full", "all"}:
        for param in model.parameters():
            param.requires_grad = True
        return parameter_counts(model)
    if scope in {"adapter", "lora", "neck"}:
        keywords = ("adapter", "lora", "neck") if scope == "adapter" else (scope,)
        for name, param in model.encoder.named_parameters():
            if any(keyword in name.lower() for keyword in keywords):
                param.requires_grad = True
        return parameter_counts(model)
    if scope in {"last_blocks", "last"}:
        children = list(model.encoder.children())
        if children:
            for child in children[-max(1, int(last_blocks)) :]:
                for param in child.parameters():
                    param.requires_grad = True
        else:
            named = list(model.encoder.named_parameters())
            for _name, param in named[-max(1, int(last_blocks)) :]:
                param.requires_grad = True
        return parameter_counts(model)
    raise ValueError(f"Unsupported trainable scope: {scope}")


def make_optimizer_param_groups(model: nn.Module, head_lr: float, encoder_lr: float, adapter_lr: float | None = None) -> list[dict]:
    head_params = [param for param in model.classifier.parameters() if param.requires_grad]
    adapter_params = []
    encoder_params = []
    for name, param in model.encoder.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name.lower() for key in ("adapter", "lora", "neck")):
            adapter_params.append(param)
        else:
            encoder_params.append(param)
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": float(encoder_lr), "name": "encoder"})
    if adapter_params:
        groups.append({"params": adapter_params, "lr": float(adapter_lr if adapter_lr is not None else encoder_lr), "name": "adapter"})
    if head_params:
        groups.append({"params": head_params, "lr": float(head_lr), "name": "head"})
    return groups


def build_depth_encoder_classifier(config: dict, num_classes: int, dropout: float, pretrained: bool, input_size: int) -> DepthEncoderDistanceClassifier:
    model_name = str(config.get("name", "depth_encoder_classifier")).lower()
    backbone = str(config.get("backbone", "")).lower()
    mask_aware = bool(config.get("mask_aware", {}).get("enabled", False)) or model_name in {"endodac_mask_aware_classifier", "depth_encoder_mask_aware_classifier"}
    if mask_aware:
        config = dict(config)
        encoder_cfg = dict(config.get("encoder", {}))
        encoder_cfg["feature_mode"] = str(encoder_cfg.get("feature_mode", "multiscale_tokens"))
        config["encoder"] = encoder_cfg

    if model_name in {"endodac_encoder_classifier", "endodac_mask_aware_classifier"} or backbone == "endodac":
        encoder = load_endodac_encoder(config, pretrained=pretrained)
    elif model_name == "depthanything_v2_encoder_classifier" or backbone in {"depthanything", "depthanything_v2"}:
        encoder = load_depthanything_v2_encoder(config, pretrained=pretrained)
    else:
        encoder_cfg = dict(config.get("encoder", {}))
        backend = str(encoder_cfg.get("backend", "simple")).lower()
        if backend == "timm":
            encoder = load_timm_encoder(str(encoder_cfg.get("timm_model_name")), pretrained=pretrained, checkpoint_path=encoder_cfg.get("checkpoint"))
        elif backend == "python":
            encoder = load_python_encoder(encoder_cfg, pretrained=pretrained)
        else:
            encoder = SimpleGeometryEncoder(feature_dim=_simple_feature_dim(config, encoder_cfg))

    if mask_aware:
        mask_cfg = dict(config.get("mask_aware", {}))
        dilation_px = int(mask_cfg.get("dilation_px", 10))
        use_global = bool(mask_cfg.get("use_global", True))
        use_raw_mask = bool(mask_cfg.get("use_raw_mask", True))
        use_dilated_mask = bool(mask_cfg.get("use_dilated_mask", True))
        feature_dim = config.get("feature_dim")
        device = next(encoder.parameters()).device
        if feature_dim in (None, "null", "auto"):
            feature_dim = infer_mask_feature_dim(
                encoder,
                input_size=int(input_size),
                device=device,
                dilation_px=dilation_px,
                use_global=use_global,
                use_raw_mask=use_raw_mask,
                use_dilated_mask=use_dilated_mask,
            )
        return MaskAwareDepthEncoderDistanceClassifier(
            encoder=encoder,
            feature_dim=int(feature_dim),
            num_classes=int(num_classes),
            dropout=float(dropout),
            hidden_dim=int(config.get("hidden_dim", 512)),
            dilation_px=dilation_px,
            use_global=use_global,
            use_raw_mask=use_raw_mask,
            use_dilated_mask=use_dilated_mask,
        )

    pool_type = str(config.get("pool_type", "mean"))
    feature_dim = config.get("feature_dim")
    device = next(encoder.parameters()).device
    if feature_dim in (None, "null", "auto"):
        feature_dim = infer_feature_dim(encoder, input_size=int(input_size), device=device, pool_type=pool_type)
    return DepthEncoderDistanceClassifier(
        encoder=encoder,
        feature_dim=int(feature_dim),
        num_classes=int(num_classes),
        dropout=float(dropout),
        pool_type=pool_type,
        hidden_dim=int(config.get("hidden_dim", 256)),
    )

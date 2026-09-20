"""ConvNeXt spatial features + ordered cumulative logits for three states."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def ordered_log_probabilities(logits):
    """Stable log P(Y=k) from z0 >= z1, where sigmoid(zk) = P(Y>k)."""
    z0, z1 = logits.float().unbind(-1)
    # q0-q1 = sigmoid(z0)*sigmoid(-z1)*(1-exp(z1-z0)).
    middle = F.logsigmoid(z0) + F.logsigmoid(-z1) + torch.log(-torch.expm1(z1 - z0))
    return torch.stack([F.logsigmoid(-z0), middle, F.logsigmoid(z1)], -1)


def ordinal_loss(logits, labels, class_weights=None):
    targets = (labels[:, None] > torch.arange(2, device=labels.device)).float()
    losses = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none").mean(-1)
    if class_weights is None:
        return losses.mean()
    weights = class_weights[labels]
    return (losses * weights).sum() / weights.sum()


class ConvNeXtROIOrdinal(nn.Module):
    def __init__(self, hidden_dim=256, grid_size=2, dropout=.2, pretrained_file=None):
        super().__init__()
        import timm
        backbone = timm.create_model("convnext_tiny", pretrained=False, num_classes=1000)
        if pretrained_file:
            from safetensors.torch import load_file
            backbone.load_state_dict(load_file(str(Path(pretrained_file))), strict=True)
        # Only the feature-producing modules are kept; no unused ImageNet head.
        self.stem, self.stages = backbone.stem, backbone.stages
        self.grid_size = int(grid_size)
        feature_dim = (384 + 768) * self.grid_size**2
        self.head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        # A gap below 2*log(2) would give Good no argmax decision region at all.
        # Start with a nonempty middle interval before learning the thresholds.
        self.raw_gap = nn.Parameter(torch.tensor(math.log(math.expm1(2.0))))

    def cumulative_logits(self, image):
        x = self.stem(image)
        spatial_features = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index in (2, 3):
                spatial_features.append(F.adaptive_avg_pool2d(x, self.grid_size).flatten(1))
        score = self.head(torch.cat(spatial_features, 1)).float()
        gap = F.softplus(self.raw_gap.float()) + 1e-4
        thresholds = torch.stack([-gap / 2, gap / 2])
        return score - thresholds[None, :]

    def forward(self, image):
        return ordered_log_probabilities(self.cumulative_logits(image))

    def freeze_backbone(self, frozen):
        for module in (self.stem, self.stages):
            for parameter in module.parameters():
                parameter.requires_grad_(not frozen)

    def optimizer_groups(self, backbone_lr, head_lr):
        return [{"name": "backbone", "lr": backbone_lr, "params": list(self.stem.parameters()) + list(self.stages.parameters())},
                {"name": "ordinal_head", "lr": head_lr, "params": list(self.head.parameters()) + [self.raw_gap]}]

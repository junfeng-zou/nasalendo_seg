# Geometry-Aware Distance State Classifier

This module adapts EndoDAC encoder features for three-class instrument distance-state classification. See [reproduction requirements](../docs/REPRODUCIBILITY.md) for external dependencies.

The model keeps the same task definition as the baseline classifier:

```text
0: TooFar
1: Good
2: TooClose
```

## Main Entry Points

```bash
python distance_state_classifier_endodac/scripts/train.py \
  --config distance_state_classifier_endodac/configs/distance_state_endodac_config.yaml

python distance_state_classifier_endodac/scripts/infer_image.py \
  --checkpoint distance_state_classifier_endodac/runs/endodac_encoder_392/best.pt \
  --image path/to/frame.png

python distance_state_classifier_endodac/scripts/benchmark_inference.py \
  --checkpoint distance_state_classifier_endodac/runs/endodac_encoder_392/best.pt
```

## Encoder Backends

The default config uses `model.encoder.backend: official` and expects the downloaded EndoDAC repo and weights under this folder:

```text
distance_state_classifier_endodac/
  third_party/EndoDAC/
    pretrained_model/depth_anything_vitb14.pth
  weights/endodac/EndoDAC_fullmodel/
    depth_model.pth
```

Switch `model.encoder.backend` to `simple` only for dependency-free smoke tests.

To use a local EndoDAC implementation, set:

```yaml
model:
  encoder:
    backend: python
    module: your_endodac_module
    class: YourEndoDACEncoderClass
    checkpoint: path/to/endodac_encoder.pth
    kwargs: {}
```

To use a timm/DINOv2-style encoder when `timm` is installed:

```yaml
model:
  encoder:
    backend: timm
    timm_model_name: vit_small_patch14_dinov2.lvd142m
```

The depth decoder is intentionally not used. The classifier consumes encoder features directly and pools CNN feature maps, ViT tokens, or already-pooled vectors.

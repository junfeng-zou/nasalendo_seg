# Configuration entry points

`data.yaml` configures the original single-class YOLO instrument-segmentation dataset. Set its dataset root to your local copy before training.

Distance-state configurations are maintained with their implementations:

- [RGB and ROI classifiers](../distance_state_classifier/configs/)
- [EndoDAC encoder classifiers](../distance_state_classifier_endodac/configs/)

Data paths and pretrained checkpoints refer to separately supplied local assets. See [reproduction requirements](../docs/REPRODUCIBILITY.md).

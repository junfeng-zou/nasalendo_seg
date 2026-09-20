# Local model weights

Weights are excluded from Git. Supply task-trained checkpoints to the inference commands.

The frozen-head and mixed-data experiments expect `convnext_tiny.in12k_ft_in1k.safetensors` in this directory. See [pretrained-model requirements](../docs/REPRODUCIBILITY.md#pretrained-models-and-dependencies). Generic ImageNet weights initialize training; they do not replace a trained distance-state checkpoint.

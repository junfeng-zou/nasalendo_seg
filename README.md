# Endoscopic Instrument Segmentation and Distance-State Classification

Research code for instrument perception in nasal endoscopy: segmentation, single-frame distance-state classification, and controlled tests of appearance dependence.

The classification target is **TooFar / Good / TooClose**. These labels represent discrete viewing-distance states, not metric depth. Some inference pipelines also return `Invalid` when confidence or localization is insufficient.

## Research components

| Component | Implementation | Purpose |
|---|---|---|
| RGB distance classification | [distance_state_classifier](distance_state_classifier/README.md) | Training, video splits, cross-validation, image/video/camera inference |
| Depth-pretrained encoder classification | [distance_state_classifier_endodac](distance_state_classifier_endodac/README.md) | Adapt EndoDAC encoder features; compare pooling and appearance-consistency objectives |
| Head localization and scale | [manual-head experiment](docs/convnext_manual_head_abc_experiment.md) | Compare full-frame, head ROI, and ROI with explicit scale under matched controls |
| Appearance shortcut diagnostics | [diagnostic protocol](docs/distance_shortcut_diagnostics.md) | Paired foreground/background interventions with protected geometry |
| Mixed phantom/clinical experiments | [RGB](docs/mixed_fullframe_distance_experiment.md), [RGB + mask](docs/mixed_rgbmask_distance_experiment.md) | Evaluate distance classification with case-separated clinical validation |
| Segmentation and tip geometry | [inference](inference/README.md) | YOLO segmentation, optical field of view, tip localization, and feature streaming |

Project-specific work includes the experiment protocols, ROI/scale comparisons, diagnostic interventions, evaluation accounting for missed detections, and runtime integration. ConvNeXt, YOLO and EndoDAC are third-party methods; see [dependencies and attribution](docs/REPRODUCIBILITY.md).

## Selected observations

**Manual-head study:** on 94 training and 34 validation images, mean macro-F1 over three seeds was 74.14% for full-frame RGB and 82.05% for head ROI. Adding explicit normalized box scale did not change predicted classes. This is a small, same-video development split with manual localization, not an independent clinical test. [Protocol and results](results/convnext_manual_head_abc_20260917/report.md).

**Automatic localization:** with the fixed seed-42 classifier, replacing manual boxes with detector boxes reduced accuracy from 82.35% to 70.59%; missed detections remain in the denominator. [Replacement study](results/forceps_head_replacement_bc_20260918/report.md).

**Mixed-data study:** clinical validation accuracy was 75.90% for RGB and 77.71% for RGB + predicted mask. The latter improved Good/TooClose recall but reduced TooFar recall. These are single-seed results on four validation cases used for model selection, not evidence of established clinical generalization. [RGB report](results/convnext_mixed_fullframe_20260919/report.md) · [RGB + mask report](results/convnext_mixed_rgbmask_20260920/report.md).

Results across these studies use different splits and training protocols and should not be ranked as a single benchmark.

## Getting started

Use Python 3.10 and install a compatible PyTorch/torchvision pair for your hardware, followed by:

```bash
python -m pip install -r requirements-distance.txt
python -m unittest discover -s tests -v
python distance_state_classifier/scripts/train.py --help
python scripts/diagnose_distance_shortcuts.py --help
```

The tests use synthetic fixtures and do not require patient images or downloaded model weights. Full training and checkpoint inference require separately supplied data and weights. See [reproduction requirements](docs/REPRODUCIBILITY.md) for the tested environment, CSV schema, checkpoint locations and command examples.

## Data and scope

Raw images, videos, clinical annotations, per-frame records and trained weights are not distributed here. Public reports retain aggregate findings and limitations; local image galleries are omitted. Experimental classification and feature-streaming code does not establish a validated autonomous robotic control system.

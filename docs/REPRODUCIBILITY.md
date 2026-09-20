# Reproduction requirements

## Environment

The public test suite was checked using Python 3.10.19, PyTorch 2.1.2+cu121, torchvision 0.16.2+cu121, NumPy 1.26.4, timm 1.0.27, safetensors 0.7.0 and Ultralytics 8.4.17. These describe the development environment; a fresh environment installation and full model retraining have not been validated for this release.

Install PyTorch and torchvision appropriate for your hardware, then `python -m pip install -r requirements-distance.txt`. OpenCV must support GUI windows only if camera display or interactive review is needed. The public unit tests run on CPU without clinical data or pretrained weights:

```bash
python -m unittest discover -s tests -v
```

## Input data

Run commands from the repository root. Basic classifiers read CSV paths from their YAML configurations. The minimum CSV fields are:

```csv
image_path,label
/path/to/image_001.png,TooFar
/path/to/image_002.png,Good
/path/to/image_003.png,TooClose
```

Paths may be absolute or relative to the repository root. Split generation and grouped evaluation additionally use fields such as `video_id`, `sample_id`, `frame_index` and, for clinical experiments, case/clip identifiers. These example rows describe a schema, not supplied images. Keep subjects/cases or videos separated according to the intended evaluation protocol. Do not substitute random frame splits for case-generalization claims.

Edit `data.train_csv`, `data.val_csv` and `data.test_csv` in the relevant config to point to your own data. The default paths document the original experiment layout; the private manifests are not included.

```bash
python distance_state_classifier/scripts/train.py --config distance_state_classifier/configs/distance_state_config.yaml
python distance_state_classifier/scripts/infer_image.py --checkpoint /path/to/best.pt --image /path/to/image.png
```

## Pretrained models and dependencies

- RGB backbones use [timm](https://github.com/huggingface/pytorch-image-models) implementations, including ConvNeXt. The baseline supports timm's pretrained-weight loading.
- ROI A/B/C and mixed-data scripts expect the timm `convnext_tiny.in12k_ft_in1k` safetensors checkpoint at `weights/convnext_tiny.in12k_ft_in1k.safetensors`. Obtain the matching checkpoint separately. ROI configurations expose `pretrained_file`; mixed experiments define `PRETRAIN` in `scripts/train_mixed_fullframe_distance.py`. Checkpoint hashes are recorded by the experiment scripts.
- Detection and segmentation use [Ultralytics](https://github.com/ultralytics/ultralytics). Supply the relevant task-trained YOLO checkpoints; pretrained generic detection weights do not reproduce surgical segmentation or head detection.
- The official encoder backend requires [EndoDAC](https://github.com/BeileiCui/EndoDAC), its dependencies and pretrained weights. Place the upstream checkout under `distance_state_classifier_endodac/third_party/EndoDAC/` and configure checkpoint paths in the supplied YAML. This repository provides the classification adapters, not the upstream EndoDAC implementation. The `simple` backend is for smoke tests and does not reproduce EndoDAC results.

Upstream projects and pretrained weights retain their respective attribution and license requirements. No third-party source trees or weights are bundled in this release.

## Appearance diagnostics

The diagnostic script compares paired interventions against the original image. It requires a classifier checkpoint, evaluation CSV and a matching cached binary mask per image:

```bash
python scripts/diagnose_distance_shortcuts.py \
  --checkpoint /path/to/best.pt \
  --csv /path/to/test_labels.csv \
  --mask-cache /path/to/mask_cache \
  --output-dir results/shortcut_diagnostics \
  --max-samples 12 --save-examples 0
```

Generate masks with `distance_state_classifier_endodac/scripts/generate_yolo_mask_cache.py`; inspect `--help` for its configuration. Cache keys depend on resolved image paths, so copied datasets generally need regenerated caches. `--save-examples 0` avoids creating image examples. Existing output directories are rejected to protect experiment records.

## Experiment-specific inputs

The dated mixed-data and head-detection scripts preserve the original experiment protocol, including expected manifest fields and sample-count assertions. They are research reproduction scripts, not a generic dataset API. Adapt data preparation and the documented protocol deliberately for a new dataset. The mixed full-frame experiment requires the phantom CSVs and the clinical keyframe manifest; head experiments additionally require head annotations or detector checkpoints. Full reproduction of the reported clinical results is unavailable without these private inputs.

The general classifier training/inference entry points and the synthetic test suite are the supported starting points for a new checkout. Reported experiment results were not regenerated during this publication cleanup.

Clinical import paths are configurable through `NASAL_CLINICAL_ROOT` (default `datasets/clinical_source`) and `NASAL_KEYFRAME_EXPORT` (default `datasets/clinical_keyframe_export`). These must contain the original export layouts expected by the scripts. Head detector preparation reads its initialization from `weights/yolo11s.pt`.

# Data preparation and distance-state experiments

The original extraction, annotation, conversion and dataset-splitting tools remain in this directory.

The distance-state additions cover:

- `diagnose_distance_shortcuts.py`: paired foreground/background appearance interventions.
- `train_mixed_*_distance.py`, `infer_mixed_*_distance.py`: full-frame, head and RGB+mask experiments.
- `check_mixed_*_result.py`: preprocessing/checkpoint replay verification.
- `evaluate_convnext_head_roi_ordinal.py`: ROI ordinal evaluation and intervention comparison.
- `prepare_forceps_head_annotation.py`, `train_forceps_head_detector.py`: head-annotation preparation and detector training.
- `evaluate_forceps_head_replacement.py`, `evaluate_head_replacement_bc.py`: manual-to-automatic localization comparisons, including missed detections.
- `run_head_mask_experiment.py`, `evaluate_head_mask_experiment.py`: head-mask experiments.
- `validate_clinical_*.py`: offline evaluation of clinical clips/keyframes.

General classifier training, split generation, cross-validation and camera/video inference live under [distance_state_classifier/scripts](../distance_state_classifier/scripts/). Read [reproduction requirements](../docs/REPRODUCIBILITY.md) before running dated experiment scripts: they expect private manifests and preserve fixed experimental protocols.

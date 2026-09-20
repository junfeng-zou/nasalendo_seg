from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    data["_config_path"] = config_path.as_posix()
    return data


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def ensure_dir(value: str | Path) -> Path:
    path = resolve_path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def merge_config_overrides(config: dict[str, Any], args: Any) -> dict[str, Any]:
    if getattr(args, "output_dir", None):
        config.setdefault("project", {})["output_dir"] = args.output_dir
    if getattr(args, "epochs", None) is not None:
        config.setdefault("train", {})["epochs"] = args.epochs
    if getattr(args, "batch_size", None) is not None:
        config.setdefault("train", {})["batch_size"] = args.batch_size
    if getattr(args, "num_workers", None) is not None:
        config.setdefault("train", {})["num_workers"] = args.num_workers
    if getattr(args, "input_size", None) is not None:
        config.setdefault("image", {})["input_size"] = args.input_size
    if getattr(args, "model", None) is not None:
        config.setdefault("model", {})["name"] = args.model
    if getattr(args, "train_csv", None) is not None:
        config.setdefault("data", {})["train_csv"] = args.train_csv
    if getattr(args, "val_csv", None) is not None:
        config.setdefault("data", {})["val_csv"] = args.val_csv
    if getattr(args, "test_csv", None) is not None:
        config.setdefault("data", {})["test_csv"] = args.test_csv
    if getattr(args, "no_pretrained", False):
        config.setdefault("model", {})["pretrained"] = False
    return config

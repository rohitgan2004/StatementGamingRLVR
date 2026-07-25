"""Config loading with single-level ``extends`` and dotted overrides."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve(path: str) -> str:
    if os.path.exists(path):
        return path
    cand = os.path.join(CONFIG_DIR, path)
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"config not found: {path}")


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config, resolving a single ``extends: base.yaml`` reference."""
    resolved = _resolve(path)
    with open(resolved) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(parent)
        cfg = _deep_merge(base, cfg)
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply dotted-key overrides, e.g. {'training.steps': 50}."""
    out = copy.deepcopy(cfg)
    for dotted, val in overrides.items():
        node = out
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = val
    return out

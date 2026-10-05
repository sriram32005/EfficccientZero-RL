"""Tiny YAML config system with dot access, `_base_` inheritance and CLI overrides."""
from __future__ import annotations

import ast
import os

import yaml


class DotDict(dict):
    """dict that also supports attribute access (cfg.train.batch_size)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - debugging aid
            raise AttributeError(f"No config key '{name}'. Available: {list(self.keys())}") from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


def _to_dot(obj):
    if isinstance(obj, dict):
        return DotDict({k: _to_dot(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_dot(v) for v in obj]
    return obj


def deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_raw(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        base_cfg = _load_raw(os.path.join(os.path.dirname(os.path.abspath(path)), base))
        cfg = deep_update(base_cfg, cfg)
    return cfg


def parse_overrides(pairs):
    """`--set train.lr=0.05 env.game=Breakout` -> nested dict."""
    out: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' must look like key.sub=value")
        key, raw = pair.split("=", 1)
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def load_config(path: str, overrides=None) -> DotDict:
    cfg = _load_raw(path)
    if overrides:
        deep_update(cfg, overrides)
    cfg["_config_path"] = os.path.abspath(path)
    return _to_dot(cfg)


def save_config(cfg: DotDict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(json_safe(cfg), f, sort_keys=False)


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj

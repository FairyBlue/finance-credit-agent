"""Small deterministic YAML inheritance loader for experiment configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_experiment_config(
    path: str | Path,
    *,
    stack: tuple[Path, ...],
    sources: list[Path],
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if config_path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, config_path))
        raise ValueError(f"config inheritance cycle: {cycle}")
    sources.append(config_path)
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"experiment config must be a mapping: {config_path}")
    parents = document.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or not all(isinstance(parent, str) for parent in parents):
        raise ValueError(f"extends must be a string or list of strings: {config_path}")
    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = (config_path.parent / parent).resolve()
        merged = _deep_merge(
            merged,
            _load_experiment_config(
                parent_path,
                stack=(*stack, config_path),
                sources=sources,
            ),
        )
    return _deep_merge(merged, document)


def load_experiment_config_with_sources(
    path: str | Path,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Load a merged config and return every YAML source in resolution order."""

    sources: list[Path] = []
    config = _load_experiment_config(path, stack=(), sources=sources)
    return config, tuple(dict.fromkeys(sources))


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config, _ = load_experiment_config_with_sources(path)
    return config

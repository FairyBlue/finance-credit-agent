"""Build Agent-R1 parquet rows from a human-reviewed temporal episode manifest."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.agents.config import load_experiment_config

SPLITS = ("train", "valid", "test")


def _number(config: dict[str, Any], section: str, key: str) -> float:
    value = config[section][key]
    if not isinstance(value, (int, float)):
        raise ValueError(f"{section}.{key} must replace TODO(human) with a number")
    return float(value)


def _symbols(config: dict[str, Any]) -> list[str]:
    symbols = config["data"]["universe"]["symbols"]
    if (
        not isinstance(symbols, list)
        or not symbols
        or not all(isinstance(item, str) for item in symbols)
    ):
        raise ValueError("data.universe.symbols must be a frozen non-empty ticker list")
    return [symbol.strip().upper() for symbol in symbols]


def _timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or "TODO" in value.upper():
        raise ValueError(f"{field} must be a timezone-aware ISO timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed


def prepare_episode_datasets(
    *, config_path: str | Path, manifest_path: str | Path, repo_root: str | Path
) -> dict[str, Path]:
    """Validate strict temporal splits and write Agent-R1-compatible parquet rows."""

    root = Path(repo_root).expanduser().resolve()
    config = load_experiment_config(config_path)
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("episode manifest must be a version: 1 mapping")
    split_documents = manifest.get("splits")
    if not isinstance(split_documents, dict):
        raise ValueError("episode manifest must contain a splits mapping")

    symbols = _symbols(config)
    data = config["data"]
    execution = config["execution"]
    environment_common = {
        "symbols": symbols,
        "initial_cash": float(execution["initial_cash"]),
        "commission_bps": float(execution["commission_bps"]),
        "slippage_bps": float(execution["slippage_bps"]),
        "drawdown_coefficient": _number(config, "reward", "drawdown_coefficient"),
        "turnover_coefficient": _number(config, "reward", "turnover_coefficient"),
        "finance_epsilon": _number(config, "reward", "epsilon"),
        "risk_lambda": _number(config, "reward", "risk_lambda"),
        "turnover_limit": _number(config, "reward", "turnover_limit"),
        "position_limit": _number(config, "reward", "position_limit"),
        "storage_root": str(root / data["storage_root"]),
        "data_version": str(data["version"]),
        "market_timezone": str(data["availability"]["timezone"]),
        "market_close_time": str(data["availability"]["market_close_time"]),
    }
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    bounds: dict[str, tuple[datetime, datetime]] = {}
    for split in SPLITS:
        episodes = split_documents.get(split)
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f"manifest split {split!r} must contain episodes")
        rows: list[dict[str, Any]] = []
        all_dates: list[datetime] = []
        for index, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                raise ValueError(f"{split}[{index}] must be a mapping")
            episode_id = str(episode.get("episode_id", ""))
            if not episode_id or "TODO" in episode_id.upper():
                raise ValueError(f"{split}[{index}].episode_id must be frozen")
            raw_dates = episode.get("episode_dates")
            raw_open_times = episode.get("episode_open_times")
            benchmark = episode.get("benchmark_closes")
            if not isinstance(raw_dates, list) or len(raw_dates) < 2:
                raise ValueError(f"{episode_id} needs at least two episode_dates")
            dates = [
                _timestamp(value, field=f"{episode_id}.episode_dates") for value in raw_dates
            ]
            if any(
                current >= following
                for current, following in zip(dates, dates[1:], strict=False)
            ):
                raise ValueError(f"{episode_id} dates must be strictly increasing")
            if not isinstance(raw_open_times, list) or len(raw_open_times) != len(dates):
                raise ValueError(f"{episode_id} episode_open_times must align with dates")
            open_times = [
                _timestamp(value, field=f"{episode_id}.episode_open_times")
                for value in raw_open_times
            ]
            if any(
                open_time >= close_time
                for open_time, close_time in zip(open_times, dates, strict=True)
            ):
                raise ValueError(f"{episode_id} opens must precede their session closes")
            if any(
                previous_close >= next_open
                for previous_close, next_open in zip(
                    dates[:-1], open_times[1:], strict=True
                )
            ):
                raise ValueError(f"{episode_id} next opens must follow prior decision closes")
            if not isinstance(benchmark, list) or len(benchmark) != len(dates):
                raise ValueError(f"{episode_id} benchmark_closes must align with dates")
            benchmark_values = [float(value) for value in benchmark]
            if any(value <= 0.0 for value in benchmark_values):
                raise ValueError(f"{episode_id} benchmark closes must be positive")
            all_dates.extend(dates)
            env_kwargs = {
                **environment_common,
                "episode_dates": [value.isoformat() for value in dates],
                "episode_open_times": [value.isoformat() for value in open_times],
                "benchmark_closes": benchmark_values,
            }
            rows.append(
                {
                    "data_source": f"stockbench/{split}",
                    "agent_name": "trading",
                    "prompt": [
                        {
                            "role": "user",
                            "content": "Manage the portfolio using only point-in-time tools.",
                        }
                    ],
                    "reward_model": {"style": "rule", "ground_truth": 0.0},
                    "extra_info": {"episode_id": episode_id, "split": split, "index": index},
                    "env_kwargs": json.dumps(env_kwargs, sort_keys=True),
                }
            )
        bounds[split] = (min(all_dates), max(all_dates))
        rows_by_split[split] = rows

    if not (
        bounds["train"][1] < bounds["valid"][0]
        and bounds["valid"][1] < bounds["test"][0]
    ):
        raise ValueError("manifest must satisfy strict train -> valid -> test chronology")

    outputs: dict[str, Path] = {}
    for split in SPLITS:
        output_key = f"{split}_file"
        output = (root / str(data["processed"][output_key])).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_by_split[split]).to_parquet(output, index=False)
        outputs[split] = output
    return outputs

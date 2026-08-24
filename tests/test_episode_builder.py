"""Audited episode manifest conversion into Agent-R1 parquet rows."""

import json
from pathlib import Path

import pandas as pd
import yaml

from src.data.episodes import prepare_episode_datasets


def test_episode_builder_writes_strict_temporal_env_kwargs(tmp_path: Path) -> None:
    config = {
        "data": {
            "version": "stockbench-tree:test-sha",
            "storage_root": "data/stockbench/storage",
            "processed": {
                "train_file": "data/processed/train.parquet",
                "valid_file": "data/processed/valid.parquet",
                "test_file": "data/processed/test.parquet",
            },
            "universe": {"symbols": ["AAA"]},
            "availability": {
                "timezone": "America/New_York",
                "market_close_time": "16:00:00",
            },
        },
        "execution": {"initial_cash": 1000.0, "commission_bps": 10.0, "slippage_bps": 5.0},
        "reward": {
            "drawdown_coefficient": 0.1,
            "turnover_coefficient": 0.1,
            "epsilon": 0.001,
            "risk_lambda": 0.5,
            "turnover_limit": 1.0,
            "position_limit": 1.0,
        },
    }
    episode_days = {
        "train": ("2025-01-02", "2025-01-03"),
        "valid": ("2025-02-03", "2025-02-04"),
        "test": ("2025-03-03", "2025-03-04"),
    }
    manifest = {"version": 1, "splits": {}}
    for split, days in episode_days.items():
        manifest["splits"][split] = [
            {
                "episode_id": f"{split}-001",
                "episode_dates": [f"{day}T21:00:00+00:00" for day in days],
                "episode_open_times": [f"{day}T14:30:00+00:00" for day in days],
                "benchmark_closes": [100.0, 101.0],
            }
        ]
    config_path = tmp_path / "config.yaml"
    manifest_path = tmp_path / "manifest.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    outputs = prepare_episode_datasets(
        config_path=config_path,
        manifest_path=manifest_path,
        repo_root=tmp_path,
    )

    assert set(outputs) == {"train", "valid", "test"}
    train = pd.read_parquet(outputs["train"])
    env_kwargs = json.loads(train.loc[0, "env_kwargs"])
    assert env_kwargs["episode_open_times"][1] == "2025-01-03T14:30:00+00:00"
    assert env_kwargs["data_version"] == "stockbench-tree:test-sha"
    assert train.loc[0, "agent_name"] == "trading"

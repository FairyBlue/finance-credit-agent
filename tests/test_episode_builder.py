"""Audited episode manifest conversion into Agent-R1 parquet rows."""

import json
import subprocess
import sys
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from src.agents.config import load_experiment_config
from src.data.episodes import (
    benchmark_series_sha256,
    build_formal_episode_manifest_draft,
    build_smoke_episode_manifest_draft,
    freeze_episode_benchmarks,
    load_episode_manifest,
    prepare_episode_datasets,
    resolve_formal_episode_sessions,
    validate_episode_manifest_metadata,
    verify_episode_benchmarks,
)
from src.data.stockbench import dataframe_content_sha256
from src.env import ExecutionPolicy
from src.evaluation import buy_and_hold_equal_weight_targets, replay_target_weights

ROOT = Path(__file__).resolve().parents[1]


def test_formal_sampling_resolves_frozen_21_session_windows_and_full_oos() -> None:
    config = load_experiment_config(ROOT / "configs/base.yaml")

    resolved = resolve_formal_episode_sessions(config)

    assert len(resolved["train"]) == 441
    assert len(resolved["valid"]) == 60
    assert len(resolved["test"]) == 1
    assert {len(episode) for episode in resolved["train"]} == {21}
    assert {len(episode) for episode in resolved["valid"]} == {21}
    assert len(resolved["test"][0]) == 83
    assert resolved["train"][0][0] == "2023-01-03"
    assert resolved["train"][-1][-1] == "2024-10-31"
    assert resolved["valid"][0][0] == "2024-11-01"
    assert resolved["valid"][-1][-1] == "2025-02-28"
    assert resolved["test"][0][0] == "2025-03-03"
    assert resolved["test"][0][-1] == "2025-06-30"


def test_formal_manifest_draft_is_deterministic_and_sampling_bound(
    tmp_path: Path,
) -> None:
    output = build_formal_episode_manifest_draft(
        config_path=ROOT / "configs/base.yaml",
        output_path=tmp_path / "formal-draft.yaml",
    )
    manifest = yaml.safe_load(output.read_text(encoding="utf-8"))
    config = load_experiment_config(ROOT / "configs/base.yaml")

    assert manifest["status"] == "generated_pending_collaborator_review"
    assert len(manifest["splits"]["train"]) == 441
    assert len(manifest["splits"]["valid"]) == 60
    assert len(manifest["splits"]["test"]) == 1
    assert len(manifest["splits"]["test"][0]["episode_dates"]) == 83
    validate_episode_manifest_metadata(config, manifest, verify_benchmark_series=False)

    tampered = deepcopy(manifest)
    tampered["splits"]["valid"][0]["episode_dates"][0] = (
        tampered["splits"]["train"][0]["episode_dates"][0]
    )
    with pytest.raises(ValueError, match="dates do not match frozen sampling"):
        validate_episode_manifest_metadata(
            config, tampered, verify_benchmark_series=False
        )


def test_formal_manifest_cli_is_importable_outside_repository(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_formal_manifest.py"), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--output" in completed.stdout


def test_pending_formal_manifest_cannot_be_used_as_reviewed_input(
    tmp_path: Path,
) -> None:
    path = build_formal_episode_manifest_draft(
        config_path=ROOT / "configs/base.yaml",
        output_path=tmp_path / "formal-candidate.yaml",
    )
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    for episodes in manifest["splits"].values():
        for episode in episodes:
            episode["benchmark_closes"] = [1.0] * len(episode["episode_dates"])
    manifest["benchmark"]["series_sha256"] = benchmark_series_sha256(
        manifest["splits"]
    )
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    config = load_experiment_config(ROOT / "configs/base.yaml")

    with pytest.raises(ValueError, match="must have status reviewed"):
        load_episode_manifest(config=config, manifest_path=path)

    manifest["status"] = "reviewed"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    assert load_episode_manifest(config=config, manifest_path=path)["status"] == "reviewed"


def test_smoke_manifest_draft_is_history_only_and_provenance_tampering_fails(
    tmp_path: Path,
) -> None:
    output = build_smoke_episode_manifest_draft(
        config_path=ROOT / "configs/smoke_1p7b.yaml",
        output_path=tmp_path / "smoke-draft.yaml",
    )
    manifest = yaml.safe_load(output.read_text(encoding="utf-8"))
    config = load_experiment_config(ROOT / "configs/smoke_1p7b.yaml")

    assert set(manifest["data_snapshot"]) == {"history"}
    assert manifest["split_windows"]["test"]["final_oos"] is False
    assert len(manifest["splits"]["train"]) == 4
    assert len(manifest["splits"]["valid"]) == len(manifest["splits"]["test"]) == 1
    validate_episode_manifest_metadata(config, manifest, verify_benchmark_series=False)

    tampered = deepcopy(manifest)
    tampered["splits"]["test"][0]["scientific_use_allowed"] = True
    with pytest.raises(ValueError, match="test episode provenance"):
        validate_episode_manifest_metadata(config, tampered, verify_benchmark_series=False)


def _config() -> dict:
    return {
        "data": {
            "sources": {
                "history": {
                    "provider": "fixture-history",
                    "provider_version": "1",
                    "retrieved_at_utc": "2025-03-05T00:00:00+00:00",
                    "input_manifest_sha256": "a" * 64,
                    "files": 24,
                    "sessions_per_symbol": 24,
                    "version": "yfinance-snapshot:history-sha",
                    "manifest_root": "data/history/storage",
                    "storage_root": "data/history/storage",
                    "input_window": {"start": "2025-01-02", "end": "2025-02-04"},
                },
                "stockbench": {
                    "repo_commit": "b" * 40,
                    "storage_tree": "c" * 40,
                    "input_manifest_sha256": "d" * 64,
                    "files": 2,
                    "sessions_per_symbol": 2,
                    "version": "stockbench-oos:test-sha",
                    "manifest_root": "data/stockbench",
                    "storage_root": "data/stockbench/storage",
                    "input_window": {"start": "2025-03-03", "end": "2025-03-04"},
                },
            },
            "split_source": {
                "train": "history",
                "valid": "history",
                "test": "stockbench",
            },
            "windows": {
                "train": {"start": "2025-01-02", "end": "2025-01-03"},
                "valid": {"start": "2025-02-03", "end": "2025-02-04"},
                "test": {
                    "start": "2025-03-03",
                    "end": "2025-03-04",
                    "indivisible_oos": True,
                },
            },
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
            "retrieval": {
                "news_enabled": False,
                "allowed_modalities": [
                    "market_data",
                    "pit_fundamentals",
                    "causal_technical_indicators",
                ],
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
            "benchmark": "buy_and_hold_equal_weight_20_tickers",
            "benchmark_accounting_engine": "shared_portfolio_ledger",
            "benchmark_execution": "next_open_with_configured_costs",
        },
        "training": {
            "max_agent_steps": 256,
            "reserved_information_steps_per_session": 2,
        },
    }


def _manifest() -> dict:
    episode_days = {
        "train": ("2025-01-02", "2025-01-03"),
        "valid": ("2025-02-03", "2025-02-04"),
        "test": ("2025-03-03", "2025-03-04"),
    }
    manifest: dict[str, Any] = {
        "version": 1,
        "data_snapshot": {
            "history": {
                "provider": "fixture-history",
                "provider_version": "1",
                "retrieved_at_utc": "2025-03-05T00:00:00+00:00",
                "input_manifest_sha256": "a" * 64,
                "files": 24,
            },
            "stockbench": {
                "repo_commit": "b" * 40,
                "storage_tree": "c" * 40,
                "input_manifest_sha256": "d" * 64,
                "files": 2,
            },
        },
        "split_windows": {
            "train": {
                "source": "history",
                "first_session": "2025-01-02",
                "last_session": "2025-01-03",
            },
            "valid": {
                "source": "history",
                "first_session": "2025-02-03",
                "last_session": "2025-02-04",
            },
            "test": {
                "source": "stockbench",
                "first_session": "2025-03-03",
                "last_session": "2025-03-04",
                "indivisible_oos": True,
            },
        },
        "universe": ["AAA"],
        "benchmark": {
            "definition": "buy_and_hold_equal_weight_20_tickers",
            "accounting_engine": "shared_portfolio_ledger",
            "execution": "next_open_with_configured_costs",
        },
        "modalities": {
            "news_enabled": False,
            "allowed_retrieval": [
                "market_data",
                "pit_fundamentals",
                "causal_technical_indicators",
            ],
        },
        "splits": {},
    }
    for split, days in episode_days.items():
        manifest["splits"][split] = [
            {
                "episode_id": f"{split}-001",
                "episode_dates": [f"{day}T21:00:00+00:00" for day in days],
                "episode_open_times": [f"{day}T14:30:00+00:00" for day in days],
                "benchmark_closes": [100.0, 101.0],
            }
        ]
    manifest["benchmark"]["series_sha256"] = benchmark_series_sha256(manifest["splits"])
    return manifest


def _write_inputs(tmp_path: Path, *, config: dict, manifest: dict) -> tuple[Path, Path]:
    config_path = tmp_path / "config.yaml"
    manifest_path = tmp_path / "manifest.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return config_path, manifest_path


def test_episode_builder_writes_strict_temporal_env_kwargs(tmp_path: Path) -> None:
    config_path, manifest_path = _write_inputs(
        tmp_path,
        config=_config(),
        manifest=_manifest(),
    )

    outputs = prepare_episode_datasets(
        config_path=config_path,
        manifest_path=manifest_path,
        repo_root=tmp_path,
    )

    assert set(outputs) == {"train", "valid", "test"}
    train = pd.read_parquet(outputs["train"])
    train_env = json.loads(train.loc[0, "env_kwargs"])
    test = pd.read_parquet(outputs["test"])
    test_env = json.loads(test.loc[0, "env_kwargs"])
    assert train_env["episode_open_times"][1] == "2025-01-03T14:30:00+00:00"
    assert train_env["data_version"] == "yfinance-snapshot:history-sha"
    assert train_env["storage_root"] == str(tmp_path / "data/history/storage")
    assert train_env["data_start"] == "2025-01-02"
    assert train_env["data_end"] == "2025-02-04"
    assert test_env["data_version"] == "stockbench-oos:test-sha"
    assert test_env["storage_root"] == str(tmp_path / "data/stockbench/storage")
    assert test_env["data_start"] == "2025-03-03"
    assert test_env["data_end"] == "2025-03-04"
    assert train.loc[0, "agent_name"] == "trading"


def test_episode_builder_rejects_an_incomplete_indivisible_oos_window(tmp_path: Path) -> None:
    config = _config()
    config["data"]["windows"]["test"]["end"] = "2025-03-05"
    manifest = _manifest()
    manifest["split_windows"]["test"]["last_session"] = "2025-03-05"
    config_path, manifest_path = _write_inputs(
        tmp_path,
        config=config,
        manifest=manifest,
    )

    with pytest.raises(
        ValueError,
        match="test split must cover every XNYS session; missing: 2025-03-05",
    ):
        prepare_episode_datasets(
            config_path=config_path,
            manifest_path=manifest_path,
            repo_root=tmp_path,
        )


def test_episode_builder_rejects_dates_outside_a_frozen_split_window(tmp_path: Path) -> None:
    manifest = _manifest()
    valid = manifest["splits"]["valid"][0]
    valid["episode_dates"][0] = "2025-01-31T21:00:00+00:00"
    valid["episode_open_times"][0] = "2025-01-31T14:30:00+00:00"
    config_path, manifest_path = _write_inputs(
        tmp_path,
        config=_config(),
        manifest=manifest,
    )

    with pytest.raises(
        ValueError,
        match=("valid split date 2025-01-31 is outside its frozen window 2025-02-03..2025-02-04"),
    ):
        prepare_episode_datasets(
            config_path=config_path,
            manifest_path=manifest_path,
            repo_root=tmp_path,
        )


def test_episode_builder_rejects_stockbench_as_a_training_source(tmp_path: Path) -> None:
    config = _config()
    config["data"]["split_source"]["train"] = "stockbench"
    config_path, manifest_path = _write_inputs(
        tmp_path,
        config=config,
        manifest=_manifest(),
    )

    with pytest.raises(
        ValueError,
        match="split sources are frozen as train=history, valid=history, test=stockbench",
    ):
        prepare_episode_datasets(
            config_path=config_path,
            manifest_path=manifest_path,
            repo_root=tmp_path,
        )


def test_episode_builder_rejects_fragmenting_the_indivisible_oos_episode(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    duplicate = dict(manifest["splits"]["test"][0])
    duplicate["episode_id"] = "test-002"
    manifest["splits"]["test"].append(duplicate)
    manifest["benchmark"]["series_sha256"] = benchmark_series_sha256(manifest["splits"])
    config_path, manifest_path = _write_inputs(
        tmp_path,
        config=_config(),
        manifest=manifest,
    )

    with pytest.raises(ValueError, match="indivisible OOS requires exactly one test episode"):
        prepare_episode_datasets(
            config_path=config_path,
            manifest_path=manifest_path,
            repo_root=tmp_path,
        )


def test_episode_builder_rejects_an_episode_longer_than_the_agent_loop(tmp_path: Path) -> None:
    config = _config()
    config["training"]["max_agent_steps"] = 0
    config_path, manifest_path = _write_inputs(
        tmp_path,
        config=config,
        manifest=_manifest(),
    )

    with pytest.raises(ValueError, match="requires 3 budgeted agent steps.*maximum is 0"):
        prepare_episode_datasets(
            config_path=config_path,
            manifest_path=manifest_path,
            repo_root=tmp_path,
        )


def test_episode_builder_rejects_non_calendar_open_or_close(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["splits"]["train"][0]["episode_open_times"][0] = "2025-01-02T14:31:00+00:00"
    config_path, manifest_path = _write_inputs(
        tmp_path,
        config=_config(),
        manifest=manifest,
    )

    with pytest.raises(ValueError, match="must equal XNYS session open"):
        prepare_episode_datasets(
            config_path=config_path,
            manifest_path=manifest_path,
            repo_root=tmp_path,
        )


def test_episode_builder_enforces_configured_smoke_episode_count(tmp_path: Path) -> None:
    config = _config()
    config["training"]["episodes"] = 4
    config_path, manifest_path = _write_inputs(
        tmp_path,
        config=config,
        manifest=_manifest(),
    )

    with pytest.raises(ValueError, match="train manifest episode count must equal"):
        prepare_episode_datasets(
            config_path=config_path,
            manifest_path=manifest_path,
            repo_root=tmp_path,
        )


def test_episode_builder_rejects_benchmark_nav_changed_after_audit(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["splits"]["test"][0]["benchmark_closes"][1] = 999.0
    config_path, manifest_path = _write_inputs(
        tmp_path,
        config=_config(),
        manifest=manifest,
    )

    with pytest.raises(ValueError, match="benchmark series SHA-256 mismatch"):
        prepare_episode_datasets(
            config_path=config_path,
            manifest_path=manifest_path,
            repo_root=tmp_path,
        )


def test_episode_builder_rejects_manifest_metadata_that_differs_from_config(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    manifest["universe"] = ["WRONG"]
    config_path, manifest_path = _write_inputs(
        tmp_path,
        config=_config(),
        manifest=manifest,
    )

    with pytest.raises(ValueError, match="manifest universe does not match"):
        prepare_episode_datasets(
            config_path=config_path,
            manifest_path=manifest_path,
            repo_root=tmp_path,
        )


def test_benchmark_freezer_uses_shared_equal_weight_next_open_ledger(tmp_path: Path) -> None:
    config = _config()
    manifest = _manifest()
    config_path, draft_path = _write_inputs(
        tmp_path,
        config=config,
        manifest=manifest,
    )
    for split, episodes in manifest["splits"].items():
        source_name = config["data"]["split_source"][split]
        storage = tmp_path / config["data"]["sources"][source_name]["storage_root"]
        day_root = storage / "parquet/AAA/day"
        day_root.mkdir(parents=True, exist_ok=True)
        for offset, timestamp in enumerate(episodes[0]["episode_dates"]):
            session = date.fromisoformat(timestamp[:10])
            frame = pd.DataFrame(
                {
                    "date": [session],
                    "open": [100.0 + 10.0 * offset],
                    "high": [113.0],
                    "low": [99.0],
                    "close": [105.0 + 7.0 * offset],
                    "volume": [1000.0],
                }
            )
            path = day_root / f"{session.isoformat()}.parquet"
            frame.to_parquet(path, index=False)
            path.with_suffix(".parquet.sha256").write_text(
                dataframe_content_sha256(
                    frame,
                    sort_by=[column for column in frame.columns if column != "date"],
                ),
                encoding="utf-8",
            )

    output = tmp_path / "reviewed.yaml"
    freeze_episode_benchmarks(
        config_path=config_path,
        draft_manifest_path=draft_path,
        output_path=output,
        repo_root=tmp_path,
    )
    frozen = yaml.safe_load(output.read_text(encoding="utf-8"))
    expected = replay_target_weights(
        session_close_times=[
            datetime(2025, 1, 2, 21, 0, tzinfo=timezone.utc),
            datetime(2025, 1, 3, 21, 0, tzinfo=timezone.utc),
        ],
        session_open_times=[
            datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc),
            datetime(2025, 1, 3, 14, 30, tzinfo=timezone.utc),
        ],
        open_prices=[{"AAA": 100.0}, {"AAA": 110.0}],
        close_prices=[{"AAA": 105.0}, {"AAA": 112.0}],
        target_weights_by_decision=buy_and_hold_equal_weight_targets(
            symbols=["AAA"], decision_count=1
        ),
        initial_cash=1000.0,
        execution_policy=ExecutionPolicy(commission_bps=10.0, slippage_bps=5.0),
    )

    assert frozen["splits"]["train"][0]["benchmark_closes"] == pytest.approx(expected.equity_curve)
    assert frozen["benchmark"]["series_sha256"] == benchmark_series_sha256(frozen["splits"])
    assert verify_episode_benchmarks(
        config=config,
        manifest_path=output,
        repo_root=tmp_path,
    ) == benchmark_series_sha256(frozen["splits"])

    frozen["splits"]["train"][0]["benchmark_closes"][1] = 999.0
    frozen["benchmark"]["series_sha256"] = benchmark_series_sha256(frozen["splits"])
    tampered = tmp_path / "tampered-reviewed.yaml"
    tampered.write_text(yaml.safe_dump(frozen), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match shared-ledger replay"):
        verify_episode_benchmarks(
            config=config,
            manifest_path=tampered,
            repo_root=tmp_path,
        )

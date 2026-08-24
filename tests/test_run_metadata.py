"""Run metadata must identify the exact two-source data snapshot."""

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import yaml

import src.agents.launcher as launcher_module
from src.agents.config import load_experiment_config
from src.agents.launcher import (
    build_upstream_command,
    verify_configured_source_manifests,
    verify_configured_t2_gate,
    verify_processed_dataset_bindings,
)
from src.agents.metadata import RunHashes, composite_data_version, validate_composite_data_version
from src.data.episodes import benchmark_series_sha256
from src.data.integrity import audit_raw_daily_manifest


def test_composite_data_version_is_deterministic_and_source_named() -> None:
    assert composite_data_version(
        history_manifest_sha256=(
            "2612e707446b1c8bcd73ac6583b84558d3c4eddbaa9fc20adf7023c3c8a76c90"
        ),
        stockbench_manifest_sha256=(
            "8b48fc7d50f4ecdbc49e920eb83d72df02b9a56d4365bce7f0db6bb4a342bd1f"
        ),
    ) == (
        "composite-raw-manifest-sha256:"
        "fb674aacaa8c7f6b6992994688733da28cb9e665209af70b778da0ce5688bbb1"
    )


def test_configured_data_version_must_match_both_source_manifests() -> None:
    with pytest.raises(RuntimeError, match="data.version does not match its source manifests"):
        validate_composite_data_version(
            configured="composite-raw-manifest-sha256:" + "0" * 64,
            history_manifest_sha256="a" * 64,
            stockbench_manifest_sha256="b" * 64,
        )


def _source_config(tmp_path: Path) -> dict:
    sources: dict[str, dict[str, object]] = {}
    for name, nested_storage in (("history", False), ("stockbench", True)):
        manifest_root = tmp_path / name
        storage_root = manifest_root / "storage" if nested_storage else manifest_root
        day_root = storage_root / "parquet" / "AAA" / "day"
        day_root.mkdir(parents=True)
        (day_root / "2025-01-02.parquet").write_bytes(name.encode())
        audit = audit_raw_daily_manifest(
            manifest_root=manifest_root,
            storage_root=storage_root,
            symbols=["AAA"],
            start=date(2025, 1, 2),
            end=date(2025, 1, 2),
        )
        sources[name] = {
            "manifest_root": str(manifest_root.relative_to(tmp_path)),
            "storage_root": str(storage_root.relative_to(tmp_path)),
            "input_window": {"start": "2025-01-02", "end": "2025-01-02"},
            "input_manifest_sha256": audit.manifest_sha256,
            "files": 1,
            "sessions_per_symbol": 1,
            "version": f"{name}-version",
        }
    return {
        "data": {
            "sources": sources,
            "universe": {"symbols": ["AAA"]},
            "availability": {"exchange_calendar": "XNYS"},
        }
    }


def test_launcher_preflight_recomputes_each_configured_source_manifest(tmp_path: Path) -> None:
    config = _source_config(tmp_path)

    audits = verify_configured_source_manifests(config, repo_root=tmp_path)

    assert set(audits) == {"history", "stockbench"}
    assert audits["history"].file_count == 1


def test_launcher_preflight_rejects_data_changed_after_configuration(tmp_path: Path) -> None:
    config = _source_config(tmp_path)
    (tmp_path / "stockbench/storage/parquet/AAA/day/2025-01-02.parquet").write_bytes(
        b"tampered"
    )

    with pytest.raises(RuntimeError, match="raw manifest SHA-256 mismatch"):
        verify_configured_source_manifests(config, repo_root=tmp_path)


def test_agent_loop_uses_the_explicit_episode_horizon_not_legacy_max_turns() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/smoke_1p7b.yaml")
    command = build_upstream_command(
        config,
        RunHashes(
            agent_r1_commit="a" * 40,
            repo_commit="b" * 40,
            data_version="fixture",
            repo_dirty=False,
        ),
    )

    assert "actor_rollout_ref.rollout.agent.max_steps=32" in command


def test_main_training_is_blocked_when_real_t2_gate_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failed = SimpleNamespace(
        passed=False,
        criteria={
            "unshifted_above_random": False,
            "significant_drop_after_shift": False,
            "shifted_ci_contains_random": True,
        },
        result=SimpleNamespace(aligned_accuracy=0.53, shifted_accuracy=0.52),
    )
    monkeypatch.setattr(
        launcher_module,
        "run_configured_temporal_leakage_probe",
        lambda **_kwargs: failed,
    )
    config: dict[str, Any] = {
        "run": {"smoke": False},
        "evaluation": {
            "shifted_label_probe": {"require_before_main_training": True}
        },
    }

    with pytest.raises(RuntimeError, match="main training is blocked"):
        verify_configured_t2_gate(config, repo_root=tmp_path)

    config["run"]["smoke"] = True
    verify_configured_t2_gate(config, repo_root=tmp_path)


def _processed_config(tmp_path: Path) -> dict:
    config = _source_config(tmp_path)
    config["data"]["sources"]["history"]["input_window"] = {
        "start": "2025-01-02",
        "end": "2025-02-04",
    }
    config["data"]["sources"]["stockbench"]["input_window"] = {
        "start": "2025-03-03",
        "end": "2025-03-04",
    }
    config["data"]["availability"].update(
        {"timezone": "America/New_York", "market_close_time": "16:00:00"}
    )
    config["data"].update(
        {
            "episode_manifest": "processed/reviewed.yaml",
            "processed": {
                "train_file": "processed/train.parquet",
                "valid_file": "processed/valid.parquet",
            },
            "split_source": {"train": "history", "valid": "history", "test": "stockbench"},
            "windows": {
                "train": {"start": "2025-01-02", "end": "2025-01-03"},
                "valid": {"start": "2025-02-03", "end": "2025-02-04"},
                "test": {
                    "start": "2025-03-03",
                    "end": "2025-03-04",
                    "indivisible_oos": True,
                },
            },
            "retrieval": {
                "news_enabled": False,
                "allowed_modalities": [
                    "market_data",
                    "pit_fundamentals",
                    "causal_technical_indicators",
                ],
            },
        }
    )
    config["execution"] = {
        "initial_cash": 1000.0,
        "commission_bps": 10.0,
        "slippage_bps": 5.0,
    }
    config["reward"] = {
        "drawdown_coefficient": 0.1,
        "turnover_coefficient": 0.2,
        "epsilon": 0.001,
        "risk_lambda": 0.5,
        "turnover_limit": 1.0,
        "position_limit": 1.0,
        "benchmark": "buy_and_hold_equal_weight_20_tickers",
        "benchmark_accounting_engine": "shared_portfolio_ledger",
        "benchmark_execution": "next_open_with_configured_costs",
    }
    config["training"] = {
        "max_agent_steps": 1,
        "reserved_information_steps_per_session": 0,
    }
    episode_days = {
        "train": ("2025-01-02", "2025-01-03"),
        "valid": ("2025-02-03", "2025-02-04"),
        "test": ("2025-03-03", "2025-03-04"),
    }
    splits = {
        split: [
            {
                "episode_id": f"{split}-001",
                "episode_dates": [f"{day}T21:00:00+00:00" for day in days],
                "episode_open_times": [f"{day}T14:30:00+00:00" for day in days],
                "benchmark_closes": [100.0, 101.0],
            }
        ]
        for split, days in episode_days.items()
    }
    manifest = {
        "version": 1,
        "data_snapshot": {
            name: {
                "input_manifest_sha256": source["input_manifest_sha256"],
                "files": source["files"],
            }
            for name, source in config["data"]["sources"].items()
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
            "series_sha256": benchmark_series_sha256(splits),
        },
        "modalities": {
            "news_enabled": False,
            "allowed_retrieval": config["data"]["retrieval"]["allowed_modalities"],
        },
        "splits": splits,
    }
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "reviewed.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    for split in ("train", "valid"):
        source = config["data"]["sources"]["history"]
        episode = splits[split][0]
        env_kwargs = {
            "symbols": ["AAA"],
            "manifest_root": str(tmp_path / source["manifest_root"]),
            "storage_root": str(tmp_path / source["storage_root"]),
            "data_version": source["version"],
            "data_start": "2025-01-02",
            "data_end": "2025-02-04",
            "input_manifest_sha256": source["input_manifest_sha256"],
            "input_file_count": source["files"],
            "input_sessions_per_symbol": source["sessions_per_symbol"],
            "initial_cash": 1000.0,
            "commission_bps": 10.0,
            "slippage_bps": 5.0,
            "drawdown_coefficient": 0.1,
            "turnover_coefficient": 0.2,
            "finance_epsilon": 0.001,
            "risk_lambda": 0.5,
            "turnover_limit": 1.0,
            "position_limit": 1.0,
            "market_timezone": "America/New_York",
            "exchange_calendar": "XNYS",
            "market_close_time": "16:00:00",
            "episode_dates": episode["episode_dates"],
            "episode_open_times": episode["episode_open_times"],
            "benchmark_closes": episode["benchmark_closes"],
        }
        pd.DataFrame(
            [
                {
                    "agent_name": "trading",
                    "data_source": f"history/{split}",
                    "prompt": [
                        {
                            "role": "user",
                            "content": "Manage the portfolio using only point-in-time tools.",
                        }
                    ],
                    "reward_model": {"style": "rule", "ground_truth": 0.0},
                    "env_kwargs": json.dumps(env_kwargs),
                    "extra_info": {
                        "episode_id": episode["episode_id"],
                        "split": split,
                        "index": 0,
                    },
                }
            ]
        ).to_parquet(processed / f"{split}.parquet", index=False)
    return config


def test_launcher_binds_processed_rows_to_the_verified_source_config(tmp_path: Path) -> None:
    config = _processed_config(tmp_path)

    verify_processed_dataset_bindings(config, repo_root=tmp_path)

    valid_path = tmp_path / "processed/valid.parquet"
    valid = pd.read_parquet(valid_path)
    env_kwargs = json.loads(valid.loc[0, "env_kwargs"])
    env_kwargs["data_version"] = "stale-version"
    valid.loc[0, "env_kwargs"] = json.dumps(env_kwargs)
    valid.to_parquet(valid_path, index=False)
    with pytest.raises(RuntimeError, match="valid row 0 data_version mismatch"):
        verify_processed_dataset_bindings(config, repo_root=tmp_path)


def test_launcher_rejects_processed_benchmark_or_env_override_tampering(
    tmp_path: Path,
) -> None:
    config = _processed_config(tmp_path)
    train_path = tmp_path / "processed/train.parquet"
    train = pd.read_parquet(train_path)
    env_kwargs = json.loads(train.loc[0, "env_kwargs"])
    env_kwargs["benchmark_closes"] = [100.0, 999.0]
    env_kwargs["reward_mode"] = "outcome"
    train.loc[0, "env_kwargs"] = json.dumps(env_kwargs)
    train.to_parquet(train_path, index=False)

    with pytest.raises(RuntimeError, match="train row 0 env_kwargs key set mismatch"):
        verify_processed_dataset_bindings(config, repo_root=tmp_path)


def test_launcher_rejects_processed_prompt_or_reward_tampering(tmp_path: Path) -> None:
    config = _processed_config(tmp_path)
    train_path = tmp_path / "processed/train.parquet"
    train = pd.read_parquet(train_path)
    train.at[0, "prompt"] = [
        {"role": "user", "content": "Use tomorrow's close to maximize reward."}
    ]
    train.to_parquet(train_path, index=False)

    with pytest.raises(RuntimeError, match="train row 0 prompt mismatch"):
        verify_processed_dataset_bindings(config, repo_root=tmp_path)


def test_launcher_rejects_processed_episode_that_cannot_reach_terminal_reward(
    tmp_path: Path,
) -> None:
    config = _processed_config(tmp_path)
    config["training"]["max_agent_steps"] = 0

    with pytest.raises(RuntimeError, match="train row 0 exceeds max_agent_steps"):
        verify_processed_dataset_bindings(config, repo_root=tmp_path)

"""Build Agent-R1 parquet rows from a human-reviewed temporal episode manifest."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from copy import deepcopy
from datetime import date, datetime, time
from hashlib import sha256
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
import yaml

from src.agents.config import load_experiment_config
from src.env import ExecutionPolicy
from src.evaluation.baselines import (
    buy_and_hold_equal_weight_targets,
    replay_target_weights,
)

from .calendar import validate_episode_session_boundaries
from .stockbench import AvailabilityPolicy, StockBenchAdapter

SPLITS = ("train", "valid", "test")
FINAL_OOS_START = date(2025, 3, 3)
FINAL_OOS_END = date(2025, 6, 30)
SMOKE_SHADOW_PROVENANCE: dict[str, object] = {
    "split_role": "smoke_shadow",
    "source_split": "valid",
    "purpose": "wiring_only",
    "scientific_use_allowed": False,
    "checkpoint_selection_allowed": False,
    "final_oos": False,
}


def resolve_smoke_episode_sessions(
    config: dict[str, Any],
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Resolve the approved G1 session slices and fail on calendar drift."""

    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("smoke config data must be a mapping")
    sampling = data.get("episode_sampling")
    windows = data.get("windows")
    if not isinstance(sampling, dict) or not isinstance(windows, dict):
        raise ValueError("smoke episode_sampling and windows must be frozen mappings")
    calendar_name = str(sampling.get("calendar", ""))
    calendar = xcals.get_calendar(calendar_name)

    def sessions(window_name: str) -> tuple[str, ...]:
        window = windows.get(window_name)
        if not isinstance(window, dict):
            raise ValueError(f"data.windows.{window_name} must be a mapping")
        return tuple(
            value.strftime("%Y-%m-%d")
            for value in calendar.sessions_in_range(str(window["start"]), str(window["end"]))
        )

    train_policy = sampling.get("train")
    valid_policy = sampling.get("valid")
    test_policy = sampling.get("test")
    if not all(isinstance(policy, dict) for policy in (train_policy, valid_policy, test_policy)):
        raise ValueError("smoke train/valid/test sampling policies must be mappings")
    assert isinstance(train_policy, dict)
    assert isinstance(valid_policy, dict)
    assert isinstance(test_policy, dict)

    train_count = int(train_policy["session_count"])
    episode_count = int(train_policy["episodes"])
    sessions_per_episode = int(train_policy["sessions_per_episode"])
    if train_count != episode_count * sessions_per_episode:
        raise ValueError("smoke train session count must equal episodes * sessions_per_episode")
    train_dates = sessions("train")[:train_count]
    if len(train_dates) != train_count:
        raise ValueError("smoke train window does not contain the approved session count")
    train_episodes = tuple(
        tuple(train_dates[index : index + sessions_per_episode])
        for index in range(0, train_count, sessions_per_episode)
    )

    valid_dates = sessions("valid")
    valid_count = int(valid_policy["session_count"])
    valid_episode = tuple(valid_dates[:valid_count])
    if len(valid_episode) != valid_count:
        raise ValueError("smoke valid window does not contain the approved session count")

    start_index = int(test_policy["start_index"])
    stop_index = int(test_policy["stop_index"])
    shadow = tuple(valid_dates[start_index:stop_index])
    if len(shadow) != int(test_policy["session_count"]):
        raise ValueError("smoke shadow window does not contain the approved session count")
    expected_bounds = (str(test_policy["expected_start"]), str(test_policy["expected_end"]))
    actual_bounds = (shadow[0], shadow[-1])
    if actual_bounds != expected_bounds:
        raise ValueError(
            "smoke shadow calendar drift: "
            f"expected {expected_bounds[0]}..{expected_bounds[1]}, "
            f"got {actual_bounds[0]}..{actual_bounds[1]}"
        )
    test_window = windows.get("test")
    if (
        not isinstance(test_window, dict)
        or (str(test_window.get("start")), str(test_window.get("end"))) != expected_bounds
    ):
        raise ValueError("data.windows.test must equal the approved Valid shadow bounds")

    return {
        "train": train_episodes,
        "valid": (valid_episode,),
        "test": (shadow,),
    }


def build_smoke_episode_manifest_draft(*, config_path: str | Path, output_path: str | Path) -> Path:
    """Create the deterministic, history-only G1 manifest awaiting human review."""

    config = load_experiment_config(config_path)
    verify_smoke_final_oos_isolation(config)
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite smoke manifest draft: {output}")

    data = config["data"]
    history = data["sources"]["history"]
    sampling = data["episode_sampling"]
    calendar = xcals.get_calendar(str(sampling["calendar"]))
    resolved = resolve_smoke_episode_sessions(config)

    def episode(split: str, index: int, session_days: tuple[str, ...]) -> dict[str, object]:
        labels = [pd.Timestamp(day) for day in session_days]
        document: dict[str, object] = {
            "episode_id": f"smoke-{split}-{index + 1:03d}",
            "episode_dates": [calendar.session_close(label).isoformat() for label in labels],
            "episode_open_times": [calendar.session_open(label).isoformat() for label in labels],
        }
        if split == "test":
            document.update(SMOKE_SHADOW_PROVENANCE)
        return document

    split_windows: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        window = data["windows"][split]
        recorded: dict[str, object] = {
            "source": "history",
            "first_session": str(window["start"]),
            "last_session": str(window["end"]),
        }
        if split == "test":
            recorded["indivisible_oos"] = False
            recorded.update(SMOKE_SHADOW_PROVENANCE)
        split_windows[split] = recorded

    manifest: dict[str, object] = {
        "version": 1,
        "status": "pending_human_review",
        "review_focus": [
            "episode_sessions_match_the_frozen_XNYS_calendar",
            "split_adjustment_restoration_is_correct",
            "no_StockBench_or_Final_OOS_source_is_present",
        ],
        "data_snapshot": {
            "history": {
                key: history[key]
                for key in (
                    "provider",
                    "provider_version",
                    "retrieved_at_utc",
                    "input_manifest_sha256",
                    "files",
                )
            }
        },
        "split_windows": split_windows,
        "universe": data["universe"]["symbols"],
        "benchmark": {
            "definition": config["reward"]["benchmark"],
            "accounting_engine": config["reward"]["benchmark_accounting_engine"],
            "execution": config["reward"]["benchmark_execution"],
        },
        "modalities": {
            "news_enabled": data["retrieval"]["news_enabled"],
            "allowed_retrieval": data["retrieval"]["allowed_modalities"],
        },
        "splits": {
            split: [
                episode(split, index, session_days)
                for index, session_days in enumerate(resolved[split])
            ]
            for split in SPLITS
        },
    }
    validate_episode_manifest_metadata(config, manifest, verify_benchmark_series=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return output


def verify_smoke_final_oos_isolation(config: dict[str, Any]) -> dict[str, object]:
    """Fail closed unless the G1 test split is the approved history-backed shadow."""

    run = config.get("run")
    if not isinstance(run, dict) or run.get("smoke") is not True:
        raise ValueError("Final OOS isolation check requires a declared smoke run")
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("smoke config data must be a mapping")
    sources = data.get("sources")
    split_source = data.get("split_source")
    if not isinstance(sources, dict) or sources.get("stockbench") is not None:
        raise ValueError("smoke resolved config must disable the StockBench Final OOS source")
    expected_sources = {"train": "history", "valid": "history", "test": "history"}
    if (
        not isinstance(split_source, dict)
        or {split: split_source.get(split) for split in SPLITS} != expected_sources
    ):
        raise ValueError("smoke train, valid, and shadow test must all use history")

    windows = data.get("windows")
    sampling = data.get("episode_sampling")
    if not isinstance(windows, dict) or not isinstance(sampling, dict):
        raise ValueError("smoke windows and episode_sampling must be frozen mappings")
    test_window = windows.get("test")
    test_policy = sampling.get("test")
    if not isinstance(test_window, dict) or not isinstance(test_policy, dict):
        raise ValueError("smoke shadow test window and policy must be mappings")
    for key, expected in SMOKE_SHADOW_PROVENANCE.items():
        if test_window.get(key) != expected or test_policy.get(key) != expected:
            raise ValueError(f"smoke shadow provenance {key} must be {expected!r}")

    sessions = resolve_smoke_episode_sessions(config)
    shadow = sessions["test"][0]
    for rendered in shadow:
        session = date.fromisoformat(rendered)
        if FINAL_OOS_START <= session <= FINAL_OOS_END:
            raise ValueError(f"smoke date enters Final OOS: {rendered}")

    processed = data.get("processed")
    if not isinstance(processed, dict):
        raise ValueError("smoke processed paths must be a mapping")
    for label, raw_path in processed.items():
        rendered = str(raw_path).casefold()
        if "stockbench" in rendered or "final_oos" in rendered:
            raise ValueError(f"smoke processed path {label} references Final OOS")

    return {
        "final_oos_accessed": False,
        "shadow_start": shadow[0],
        "shadow_end": shadow[-1],
        "shadow_sessions": len(shadow),
        "source": "history",
    }


def benchmark_series_sha256(split_documents: object) -> str:
    """Hash audited benchmark NAV values with split and episode identities."""

    if not isinstance(split_documents, dict):
        raise ValueError("benchmark series requires a splits mapping")
    records: list[dict[str, object]] = []
    for split in SPLITS:
        episodes = split_documents.get(split)
        if not isinstance(episodes, list):
            raise ValueError(f"benchmark series requires manifest split {split}")
        for episode in episodes:
            if not isinstance(episode, dict):
                raise ValueError(f"benchmark series {split} episode must be a mapping")
            episode_id = episode.get("episode_id")
            values = episode.get("benchmark_closes")
            if not isinstance(episode_id, str) or not isinstance(values, list):
                raise ValueError("benchmark series requires episode_id and benchmark_closes")
            numeric = [float(value) for value in values]
            if any(not math.isfinite(value) or value <= 0.0 for value in numeric):
                raise ValueError("benchmark NAV values must be finite and positive")
            records.append({"benchmark_nav": numeric, "episode_id": episode_id, "split": split})
    canonical = json.dumps(
        records,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256(canonical).hexdigest()


def validate_episode_manifest_metadata(
    config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    verify_benchmark_series: bool = True,
) -> None:
    """Bind a reviewed episode manifest to the immutable experiment config."""

    data = config["data"]
    is_smoke = bool(config.get("run", {}).get("smoke"))
    if is_smoke:
        verify_smoke_final_oos_isolation(config)
    snapshots = manifest.get("data_snapshot")
    if not isinstance(snapshots, dict):
        raise ValueError("manifest data_snapshot must be a frozen mapping")
    source_keys = {
        "history": (
            "provider",
            "provider_version",
            "retrieved_at_utc",
            "input_manifest_sha256",
            "files",
        ),
        "stockbench": (
            "repo_commit",
            "storage_tree",
            "input_manifest_sha256",
            "files",
        ),
    }
    active_sources = {str(data["split_source"][split]) for split in SPLITS}
    if is_smoke and "stockbench" in snapshots:
        raise ValueError("smoke manifest must not contain a StockBench snapshot")
    for source_name in sorted(active_sources):
        keys = source_keys.get(source_name)
        if keys is None:
            raise ValueError(f"unsupported manifest source: {source_name}")
        recorded = snapshots.get(source_name)
        configured = data["sources"][source_name]
        if not isinstance(recorded, dict):
            raise ValueError(f"manifest data_snapshot.{source_name} must be frozen")
        for key in keys:
            if recorded.get(key) != configured.get(key):
                raise ValueError(f"manifest {source_name}.{key} does not match the frozen config")

    windows = manifest.get("split_windows")
    if not isinstance(windows, dict):
        raise ValueError("manifest split_windows must be a frozen mapping")
    for split in SPLITS:
        recorded = windows.get(split)
        configured = data["windows"][split]
        if not isinstance(recorded, dict):
            raise ValueError(f"manifest split_windows.{split} must be frozen")
        expected = {
            "source": data["split_source"][split],
            "first_session": str(configured["start"]),
            "last_session": str(configured["end"]),
        }
        if split == "test":
            expected["indivisible_oos"] = configured.get("indivisible_oos")
            if is_smoke:
                expected.update(SMOKE_SHADOW_PROVENANCE)
        if any(recorded.get(key) != value for key, value in expected.items()):
            raise ValueError(f"manifest split_windows.{split} does not match the frozen config")

    if is_smoke:
        split_documents = manifest.get("splits")
        if not isinstance(split_documents, dict):
            raise ValueError("smoke manifest splits must be a frozen mapping")
        for split in SPLITS:
            episodes = split_documents.get(split)
            if not isinstance(episodes, list) or not episodes:
                raise ValueError(f"smoke manifest split {split} must contain episodes")
            for episode in episodes:
                if not isinstance(episode, dict):
                    raise ValueError(f"smoke manifest split {split} episode must be a mapping")
                if split == "test" and any(
                    episode.get(key) != expected
                    for key, expected in SMOKE_SHADOW_PROVENANCE.items()
                ):
                    raise ValueError("smoke test episode provenance is not frozen")
                raw_dates = episode.get("episode_dates")
                if not isinstance(raw_dates, list):
                    raise ValueError(f"smoke manifest split {split} dates must be frozen")
                for raw_date in raw_dates:
                    session = _timestamp(raw_date, field=f"smoke {split} episode_dates").date()
                    if FINAL_OOS_START <= session <= FINAL_OOS_END:
                        raise ValueError(f"smoke manifest enters Final OOS: {session.isoformat()}")

    if manifest.get("universe") != data["universe"]["symbols"]:
        raise ValueError("manifest universe does not match the frozen config")
    benchmark = manifest.get("benchmark")
    reward = config["reward"]
    if not isinstance(benchmark, dict) or any(
        benchmark.get(manifest_key) != reward[config_key]
        for manifest_key, config_key in (
            ("definition", "benchmark"),
            ("accounting_engine", "benchmark_accounting_engine"),
            ("execution", "benchmark_execution"),
        )
    ):
        raise ValueError("manifest benchmark definition does not match the frozen config")
    if verify_benchmark_series:
        actual_series_hash = benchmark_series_sha256(manifest.get("splits"))
        if benchmark.get("series_sha256") != actual_series_hash:
            raise ValueError(
                "benchmark series SHA-256 mismatch: "
                f"expected {benchmark.get('series_sha256')}, got {actual_series_hash}"
            )
    modalities = manifest.get("modalities")
    retrieval = data["retrieval"]
    if not isinstance(modalities, dict) or (
        modalities.get("news_enabled") != retrieval["news_enabled"]
        or modalities.get("allowed_retrieval") != retrieval["allowed_modalities"]
    ):
        raise ValueError("manifest modalities do not match the frozen config")


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


def load_episode_manifest(*, config: dict[str, Any], manifest_path: str | Path) -> dict[str, Any]:
    """Load a versioned manifest and verify all frozen metadata and NAV bytes."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"missing reviewed episode manifest: {manifest_file}")
    manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("episode manifest must be a version: 1 mapping")
    if not isinstance(manifest.get("splits"), dict):
        raise ValueError("episode manifest must contain a splits mapping")
    validate_episode_manifest_metadata(config, manifest)
    return manifest


def _exact_session_prices(
    *,
    adapter: StockBenchAdapter,
    as_of: datetime,
    symbols: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    expected_date = as_of.date().isoformat()
    opens: dict[str, float] = {}
    closes: dict[str, float] = {}
    for observation in adapter.get_observations(as_of):
        if observation.kind != "market" or observation.symbol not in symbols:
            continue
        if observation.payload.get("date") != expected_date:
            continue
        if observation.symbol in opens:
            raise ValueError(f"duplicate benchmark bar for {observation.symbol} on {expected_date}")
        opens[observation.symbol] = float(observation.payload["open"])
        closes[observation.symbol] = float(observation.payload["close"])
    missing = [symbol for symbol in symbols if symbol not in opens]
    if missing:
        raise ValueError(f"missing benchmark bars on {expected_date}: {missing}")
    return opens, closes


def _populate_episode_benchmarks(
    *, config: dict[str, Any], manifest: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    split_documents = manifest.get("splits")
    if not isinstance(split_documents, dict):
        raise ValueError("episode manifest must contain a splits mapping")

    data = config["data"]
    symbols = _symbols(config)
    availability = data["availability"]
    policy = AvailabilityPolicy(
        str(availability["timezone"]),
        time.fromisoformat(str(availability["market_close_time"])),
        str(availability.get("exchange_calendar", "XNYS")),
    )
    adapters: dict[str, StockBenchAdapter] = {}

    def source_adapter(source_name: str) -> StockBenchAdapter:
        existing = adapters.get(source_name)
        if existing is not None:
            return existing
        source = data["sources"][source_name]
        window = source["input_window"]
        storage_root = Path(str(source["storage_root"]))
        if not storage_root.is_absolute():
            storage_root = repo_root / storage_root
        created = StockBenchAdapter.from_storage(
            storage_root,
            symbols=symbols,
            data_version=str(source["version"]),
            availability_policy=policy,
            start_date=date.fromisoformat(str(window["start"])),
            end_date=date.fromisoformat(str(window["end"])),
        )
        adapters[source_name] = created
        return created

    execution = config["execution"]
    execution_policy = ExecutionPolicy(
        commission_bps=float(execution["commission_bps"]),
        slippage_bps=float(execution["slippage_bps"]),
    )
    for split in SPLITS:
        episodes = split_documents.get(split)
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f"manifest split {split!r} must contain episodes")
        source_name = str(data["split_source"][split])
        adapter = source_adapter(source_name)
        for index, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                raise ValueError(f"{split}[{index}] must be a mapping")
            raw_dates = episode.get("episode_dates")
            raw_opens = episode.get("episode_open_times")
            if not isinstance(raw_dates, list) or not isinstance(raw_opens, list):
                raise ValueError(f"{split}[{index}] must freeze episode dates and opens")
            dates = [
                _timestamp(value, field=f"{split}[{index}].episode_dates") for value in raw_dates
            ]
            open_times = [
                _timestamp(value, field=f"{split}[{index}].episode_open_times")
                for value in raw_opens
            ]
            if len(dates) < 2 or len(open_times) != len(dates):
                raise ValueError(f"{split}[{index}] dates and opens must align")
            validate_episode_session_boundaries(
                session_closes=dates,
                session_opens=open_times,
                exchange_calendar=str(availability.get("exchange_calendar", "XNYS")),
            )
            price_pairs = [
                _exact_session_prices(adapter=adapter, as_of=close_at, symbols=symbols)
                for close_at in dates
            ]
            replay = replay_target_weights(
                session_close_times=dates,
                session_open_times=open_times,
                open_prices=[pair[0] for pair in price_pairs],
                close_prices=[pair[1] for pair in price_pairs],
                target_weights_by_decision=buy_and_hold_equal_weight_targets(
                    symbols=symbols,
                    decision_count=len(dates) - 1,
                ),
                initial_cash=float(execution["initial_cash"]),
                execution_policy=execution_policy,
            )
            episode["benchmark_closes"] = list(replay.equity_curve)
    return split_documents


def freeze_episode_benchmarks(
    *,
    config_path: str | Path,
    draft_manifest_path: str | Path,
    output_path: str | Path,
    repo_root: str | Path,
) -> Path:
    """Generate equal-weight buy-and-hold NAV with the shared next-open ledger."""

    root = Path(repo_root).expanduser().resolve()
    config = load_experiment_config(config_path)
    draft_path = Path(draft_manifest_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark manifest: {output}")
    manifest = yaml.safe_load(draft_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("episode manifest must be a version: 1 mapping")
    validate_episode_manifest_metadata(
        config,
        manifest,
        verify_benchmark_series=False,
    )
    split_documents = _populate_episode_benchmarks(
        config=config,
        manifest=manifest,
        repo_root=root,
    )

    benchmark = manifest.get("benchmark")
    assert isinstance(benchmark, dict)
    benchmark["series_sha256"] = benchmark_series_sha256(split_documents)
    validate_episode_manifest_metadata(config, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return output


def verify_episode_benchmarks(
    *,
    config: dict[str, Any],
    manifest_path: str | Path,
    repo_root: str | Path,
) -> str:
    """Recompute reviewed benchmark NAV from raw bars before starting training."""

    manifest = load_episode_manifest(config=config, manifest_path=manifest_path)
    recomputed = deepcopy(manifest)
    split_documents = _populate_episode_benchmarks(
        config=config,
        manifest=recomputed,
        repo_root=Path(repo_root).expanduser().resolve(),
    )
    actual_sha256 = benchmark_series_sha256(split_documents)
    benchmark = manifest.get("benchmark")
    assert isinstance(benchmark, dict)
    if benchmark.get("series_sha256") != actual_sha256:
        raise RuntimeError(
            "reviewed benchmark NAV does not match shared-ledger replay: "
            f"expected {benchmark.get('series_sha256')}, got {actual_sha256}"
        )
    return actual_sha256


def prepare_episode_datasets(
    *, config_path: str | Path, manifest_path: str | Path, repo_root: str | Path
) -> dict[str, Path]:
    """Validate strict temporal splits and write Agent-R1-compatible parquet rows."""

    root = Path(repo_root).expanduser().resolve()
    config = load_experiment_config(config_path)
    symbols = _symbols(config)
    data = config["data"]
    sources = data.get("sources")
    split_source = data.get("split_source")
    if not isinstance(sources, dict) or not isinstance(split_source, dict):
        raise ValueError("data.sources and data.split_source must be frozen mappings")
    is_smoke = bool(config.get("run", {}).get("smoke"))
    expected_split_source = {
        "train": "history",
        "valid": "history",
        "test": "history" if is_smoke else "stockbench",
    }
    if {split: split_source.get(split) for split in SPLITS} != expected_split_source:
        if is_smoke:
            raise ValueError(
                "smoke split sources are frozen as train=history, valid=history, test=history"
            )
        raise ValueError(
            "split sources are frozen as train=history, valid=history, test=stockbench"
        )
    manifest = load_episode_manifest(config=config, manifest_path=manifest_path)
    split_documents = manifest.get("splits")
    assert isinstance(split_documents, dict)  # Guaranteed by load_episode_manifest.
    execution = config["execution"]
    max_agent_steps = int(config["training"]["max_agent_steps"])
    reserved_information_steps = int(config["training"]["reserved_information_steps_per_session"])
    if reserved_information_steps < 0:
        raise ValueError("training.reserved_information_steps_per_session must be non-negative")
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
        "market_timezone": str(data["availability"]["timezone"]),
        "exchange_calendar": str(data["availability"].get("exchange_calendar", "XNYS")),
        "market_close_time": str(data["availability"]["market_close_time"]),
    }
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    bounds: dict[str, tuple[datetime, datetime]] = {}
    session_dates_by_split: dict[str, set[date]] = {}
    market_zone = ZoneInfo(str(data["availability"]["timezone"]))
    for split in SPLITS:
        source_name = split_source.get(split)
        source = sources.get(source_name) if isinstance(source_name, str) else None
        if not isinstance(source, dict):
            raise ValueError(f"data source for split {split!r} must be frozen")
        manifest_root = source.get("manifest_root")
        storage_root = source.get("storage_root")
        data_version = source.get("version")
        input_manifest_sha256 = source.get("input_manifest_sha256")
        input_file_count = source.get("files")
        input_sessions_per_symbol = source.get("sessions_per_symbol")
        input_window = source.get("input_window")
        if (
            not isinstance(manifest_root, str)
            or not manifest_root
            or "TODO" in manifest_root.upper()
        ):
            raise ValueError(f"{source_name}.manifest_root must be frozen")
        if not isinstance(storage_root, str) or not storage_root or "TODO" in storage_root.upper():
            raise ValueError(f"{source_name}.storage_root must be frozen")
        if not isinstance(data_version, str) or not data_version or "TODO" in data_version.upper():
            raise ValueError(f"{source_name}.version must be an immutable identifier")
        if (
            not isinstance(input_manifest_sha256, str)
            or len(input_manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in input_manifest_sha256)
        ):
            raise ValueError(f"{source_name}.input_manifest_sha256 must be lowercase SHA-256")
        if (
            isinstance(input_file_count, bool)
            or not isinstance(input_file_count, int)
            or input_file_count < 1
        ):
            raise ValueError(f"{source_name}.files must be a positive integer")
        if (
            isinstance(input_sessions_per_symbol, bool)
            or not isinstance(input_sessions_per_symbol, int)
            or input_sessions_per_symbol < 1
        ):
            raise ValueError(f"{source_name}.sessions_per_symbol must be a positive integer")
        if not isinstance(input_window, dict):
            raise ValueError(f"{source_name}.input_window must be frozen")
        data_start = date.fromisoformat(str(input_window["start"]))
        data_end = date.fromisoformat(str(input_window["end"]))
        if data_start > data_end:
            raise ValueError(f"{source_name}.input_window start must not follow end")
        resolved_manifest_root = Path(manifest_root).expanduser()
        if not resolved_manifest_root.is_absolute():
            resolved_manifest_root = root / resolved_manifest_root
        resolved_storage_root = Path(storage_root).expanduser()
        if not resolved_storage_root.is_absolute():
            resolved_storage_root = root / resolved_storage_root
        episodes = split_documents.get(split)
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f"manifest split {split!r} must contain episodes")
        configured_episodes = config["training"].get("episodes")
        if split == "train" and configured_episodes is not None:
            if isinstance(configured_episodes, bool) or not isinstance(configured_episodes, int):
                raise ValueError("training.episodes must be a positive integer")
            if configured_episodes < 1 or len(episodes) != configured_episodes:
                raise ValueError(
                    "train manifest episode count must equal training.episodes "
                    f"({configured_episodes})"
                )
        if (
            split == "test"
            and data.get("windows", {}).get("test", {}).get("indivisible_oos") is True
            and len(episodes) != 1
        ):
            raise ValueError("indivisible OOS requires exactly one test episode")
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
            dates = [_timestamp(value, field=f"{episode_id}.episode_dates") for value in raw_dates]
            minimum_steps = len(dates) - 1
            budgeted_steps = minimum_steps * (1 + reserved_information_steps)
            if split in {"train", "valid"} and budgeted_steps > max_agent_steps:
                raise ValueError(
                    f"{episode_id} requires {budgeted_steps} budgeted agent steps "
                    f"({minimum_steps} advancing + {reserved_information_steps} information "
                    "steps/session), "
                    f"configured maximum is {max_agent_steps}"
                )
            if any(
                current >= following for current, following in zip(dates, dates[1:], strict=False)
            ):
                raise ValueError(f"{episode_id} dates must be strictly increasing")
            if not isinstance(raw_open_times, list) or len(raw_open_times) != len(dates):
                raise ValueError(f"{episode_id} episode_open_times must align with dates")
            open_times = [
                _timestamp(value, field=f"{episode_id}.episode_open_times")
                for value in raw_open_times
            ]
            validate_episode_session_boundaries(
                session_closes=dates,
                session_opens=open_times,
                exchange_calendar=str(data["availability"].get("exchange_calendar", "XNYS")),
            )
            if any(
                open_time >= close_time
                for open_time, close_time in zip(open_times, dates, strict=True)
            ):
                raise ValueError(f"{episode_id} opens must precede their session closes")
            if any(
                previous_close >= next_open
                for previous_close, next_open in zip(dates[:-1], open_times[1:], strict=True)
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
                "manifest_root": str(resolved_manifest_root.resolve()),
                "storage_root": str(resolved_storage_root.resolve()),
                "data_version": data_version,
                "data_start": data_start.isoformat(),
                "data_end": data_end.isoformat(),
                "input_manifest_sha256": input_manifest_sha256,
                "input_file_count": input_file_count,
                "input_sessions_per_symbol": input_sessions_per_symbol,
                "episode_dates": [value.isoformat() for value in dates],
                "episode_open_times": [value.isoformat() for value in open_times],
                "benchmark_closes": benchmark_values,
            }
            extra_info: dict[str, object] = {
                "episode_id": episode_id,
                "split": split,
                "index": index,
            }
            if is_smoke and split == "test":
                env_kwargs["provenance"] = dict(SMOKE_SHADOW_PROVENANCE)
                extra_info.update(SMOKE_SHADOW_PROVENANCE)
            rows.append(
                {
                    "data_source": f"{source_name}/{split}",
                    "agent_name": "trading",
                    "prompt": [
                        {
                            "role": "user",
                            "content": "Manage the portfolio using only point-in-time tools.",
                        }
                    ],
                    "reward_model": {"style": "rule", "ground_truth": 0.0},
                    "extra_info": extra_info,
                    "env_kwargs": json.dumps(env_kwargs, sort_keys=True),
                }
            )
        bounds[split] = (min(all_dates), max(all_dates))
        session_dates_by_split[split] = {
            timestamp.astimezone(market_zone).date() for timestamp in all_dates
        }
        rows_by_split[split] = rows

    if not (bounds["train"][1] < bounds["valid"][0] and bounds["valid"][1] < bounds["test"][0]):
        raise ValueError("manifest must satisfy strict train -> valid -> test chronology")

    windows = data.get("windows")
    if not isinstance(windows, dict):
        raise ValueError("data.windows must be a frozen mapping")
    for split in SPLITS:
        window = windows.get(split)
        if not isinstance(window, dict):
            raise ValueError(f"data.windows.{split} must be a frozen mapping")
        start = date.fromisoformat(str(window["start"]))
        end = date.fromisoformat(str(window["end"]))
        outside = sorted(
            session for session in session_dates_by_split[split] if not start <= session <= end
        )
        if outside:
            raise ValueError(
                f"{split} split date {outside[0].isoformat()} is outside its frozen window "
                f"{start.isoformat()}..{end.isoformat()}"
            )

    test_window = windows["test"]
    if isinstance(test_window, dict) and test_window.get("indivisible_oos") is True:
        start = date.fromisoformat(str(test_window["start"]))
        end = date.fromisoformat(str(test_window["end"]))
        calendar_name = str(data["availability"].get("exchange_calendar", "XNYS"))
        calendar = xcals.get_calendar(calendar_name)
        expected = {
            timestamp.date()
            for timestamp in calendar.sessions_in_range(start.isoformat(), end.isoformat())
        }
        actual = session_dates_by_split["test"]
        missing = sorted(expected.difference(actual))
        if missing:
            rendered = ", ".join(value.isoformat() for value in missing)
            raise ValueError(
                f"test split must cover every {calendar_name} session; missing: {rendered}"
            )
        unexpected = sorted(actual.difference(expected))
        if unexpected:
            rendered = ", ".join(value.isoformat() for value in unexpected)
            raise ValueError(f"test split contains dates outside the frozen OOS window: {rendered}")

    outputs: dict[str, Path] = {}
    for split in SPLITS:
        output_key = f"{split}_file"
        output = (root / str(data["processed"][output_key])).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_by_split[split]).to_parquet(output, index=False)
        outputs[split] = output
    return outputs

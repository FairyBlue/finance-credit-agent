"""Prepare and verify the authorized M2 screening inputs without opening formal holdouts."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from copy import deepcopy
from datetime import date, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import yaml

from src.agents.config import load_experiment_config
from src.env import ExecutionPolicy
from src.rewards.calibration import CalibrationMarketData, _replay
from src.rewards.finance import RiskComponents
from src.rewards.screening import (
    SCORER_ID,
    calibrate_screening_reward,
    canonical_sha256,
    load_screening_calibration,
    score_benchmark_scaled,
    screening_payload_sha256,
    screening_proxy_targets,
)

from .calendar import validate_episode_session_boundaries
from .integrity import RawManifestAudit, audit_raw_daily_manifest
from .stockbench import dataframe_content_sha256

SCREEN_TRAIN_INDICES = tuple(range(0, 125, 4))
SCREEN_DEV_INDICES = tuple(range(209, 441, 21))
TRAIN_START = date(2023, 1, 3)
TRAIN_END = date(2024, 10, 31)
DATASET_ROLES = ("train", "dev", "pilot_train", "pilot_eval")


def _resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _json_bytes(document: Any) -> bytes:
    return (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _write_identical_or_new(path: Path, content: bytes) -> None:
    """Publish atomically, preserving a previously different artifact for review."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite a different screening artifact: {path}")
        return
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.chmod(temporary, 0o644)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(
                    f"different screening artifact already exists: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def select_screening_episodes(
    *, manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Select exact Train indices using metadata only; never inspect other split outcomes."""

    if manifest.get("version") != 1 or manifest.get("status") != "reviewed":
        raise ValueError("screening requires the reviewed version-1 episode manifest")
    data = config["data"]
    if data["split_source"]["train"] != "history":
        raise ValueError("screening is restricted to history/Train")
    train_window = data["windows"]["train"]
    if (str(train_window["start"]), str(train_window["end"])) != (
        TRAIN_START.isoformat(),
        TRAIN_END.isoformat(),
    ):
        raise ValueError("screening requires the frozen historical Train window")
    if manifest.get("universe") != data["universe"]["symbols"]:
        raise ValueError("screening manifest universe differs from the frozen config")
    for key, reward_key in (
        ("definition", "benchmark"),
        ("accounting_engine", "benchmark_accounting_engine"),
        ("execution", "benchmark_execution"),
    ):
        if manifest.get("benchmark", {}).get(key) != config["reward"][reward_key]:
            raise ValueError("screening benchmark definition differs from the frozen config")
    history = data["sources"]["history"]
    if (
        manifest.get("data_snapshot", {}).get("history", {}).get("input_manifest_sha256")
        != (history["input_manifest_sha256"])
    ):
        raise ValueError("reviewed manifest history identity differs from the config")
    train = manifest.get("splits", {}).get("train")
    if not isinstance(train, list) or len(train) != 441:
        raise ValueError("screening selection requires all 441 original Train episodes")
    calendar_name = str(data["availability"]["exchange_calendar"])
    calendar = xcals.get_calendar(calendar_name)
    labels = tuple(calendar.sessions_in_range(TRAIN_START.isoformat(), TRAIN_END.isoformat()))
    selected: dict[str, list[dict[str, Any]]] = {}
    selected_dates: dict[str, set[str]] = {}
    for role, indices in (("train", SCREEN_TRAIN_INDICES), ("dev", SCREEN_DEV_INDICES)):
        episodes = []
        dates_for_role: set[str] = set()
        for index in indices:
            episode = deepcopy(train[index])
            if episode.get("episode_id") != f"formal-train-{index + 1:04d}":
                raise ValueError("original Train episode IDs/order have drifted")
            expected_labels = labels[index : index + 21]
            expected_closes = [
                calendar.session_close(label).isoformat() for label in expected_labels
            ]
            expected_opens = [calendar.session_open(label).isoformat() for label in expected_labels]
            if episode.get("episode_dates") != expected_closes or (
                episode.get("episode_open_times") != expected_opens
            ):
                raise ValueError(
                    f"screening episode sessions have drifted: {episode['episode_id']}"
                )
            benchmark = episode.get("benchmark_closes")
            if (
                not isinstance(benchmark, list)
                or len(benchmark) != 21
                or any(
                    not math.isfinite(float(value)) or float(value) <= 0.0 for value in benchmark
                )
            ):
                raise ValueError("screening benchmark must contain 21 finite positive NAVs")
            dates = {value[:10] for value in expected_closes}
            if role == "dev" and dates_for_role.intersection(dates):
                raise ValueError("screen-dev episode sessions must not overlap")
            dates_for_role.update(dates)
            episodes.append(episode)
        selected[role] = episodes
        selected_dates[role] = dates_for_role
    if selected_dates["train"].intersection(selected_dates["dev"]):
        raise ValueError("screen-train and screen-dev sessions must not overlap")
    if len(selected_dates["train"]) != 145 or len(selected_dates["dev"]) != 252:
        raise ValueError("screening unique-session counts have drifted")
    return selected


def _audit_train_source(
    config: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], RawManifestAudit]:
    history = config["data"]["sources"]["history"]
    storage = _resolve(history["storage_root"], root)
    manifest_root = _resolve(history["manifest_root"], root)
    metadata_path = storage / "manifest.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    if (
        metadata.get("input_manifest_sha256") != history["input_manifest_sha256"]
        or metadata.get("file_count") != history["files"]
        or metadata.get("symbols") != config["data"]["universe"]["symbols"]
        or metadata.get("window_start") != history["input_window"]["start"]
        or metadata.get("window_end") != history["input_window"]["end"]
    ):
        raise ValueError("history metadata does not match its declared frozen source identity")
    audit = audit_raw_daily_manifest(
        manifest_root=manifest_root,
        storage_root=storage,
        symbols=config["data"]["universe"]["symbols"],
        start=TRAIN_START,
        end=TRAIN_END,
        exchange_calendar=config["data"]["availability"]["exchange_calendar"],
    )
    if audit.file_count != 9220 or set(audit.sessions_per_symbol.values()) != {461}:
        raise ValueError("Train-only raw snapshot must contain 20 x 461 daily files")
    source = {
        "manifest_root": str(manifest_root),
        "storage_root": str(storage),
        "data_start": TRAIN_START.isoformat(),
        "data_end": TRAIN_END.isoformat(),
        "input_manifest_sha256": audit.manifest_sha256,
        "input_file_count": audit.file_count,
        "input_sessions_per_symbol": 461,
        "data_version": f"history-train-only:raw-manifest-sha256:{audit.manifest_sha256}",
        "original_history_declared_sha256": history["input_manifest_sha256"],
        "original_history_metadata_sha256": sha256(metadata_bytes).hexdigest(),
        "original_full_history_raw_hash_reverified": False,
        "raw_audit_scope": "history/Train only; formal Valid and Final OOS files never opened",
    }
    return source, audit


def _load_selected_markets(
    *,
    episodes: list[dict[str, Any]],
    source: Mapping[str, Any],
    raw_manifest_bytes: bytes,
    symbols: list[str],
    exchange_calendar: str,
    role: str,
) -> dict[str, CalibrationMarketData]:
    """Decode only the selected calibration sessions, checking both raw and logical hashes."""

    root = Path(source["manifest_root"])
    storage = Path(source["storage_root"])
    hashes = {}
    for line in raw_manifest_bytes.decode().splitlines():
        relative, digest = line.split("\t", 1)
        hashes[(root / relative).resolve()] = digest
    if role not in {"train", "dev"}:
        raise ValueError("selected market decoding requires an explicit train/dev role")
    indices = SCREEN_TRAIN_INDICES if role == "train" else SCREEN_DEV_INDICES
    allowed_ids = {f"formal-train-{index + 1:04d}" for index in indices}
    if (
        len(episodes) != len(indices)
        or {episode["episode_id"] for episode in episodes} != allowed_ids
    ):
        raise ValueError(f"{role} market decoding requires exactly its selected episode IDs")
    price_cache: dict[date, tuple[dict[str, float], dict[str, float]]] = {}
    markets = {}
    for episode in episodes:
        closes = tuple(datetime.fromisoformat(value) for value in episode["episode_dates"])
        opens = tuple(datetime.fromisoformat(value) for value in episode["episode_open_times"])
        validate_episode_session_boundaries(
            session_closes=closes,
            session_opens=opens,
            exchange_calendar=exchange_calendar,
        )
        sessions = tuple(value.date() for value in closes)
        bounds = (
            (TRAIN_START, date(2023, 8, 1)) if role == "train" else (date(2023, 11, 1), TRAIN_END)
        )
        if any(not bounds[0] <= day <= bounds[1] for day in sessions):
            raise ValueError(f"market decoding attempted a day outside screen-{role}")
        for session in sessions:
            if session in price_cache:
                continue
            open_row, close_row = {}, {}
            for symbol in symbols:
                path = (storage / "parquet" / symbol / "day" / f"{session}.parquet").resolve()
                expected_hash = hashes.get(path)
                if expected_hash is None:
                    raise ValueError("calibration bar is absent from the Train-only raw manifest")
                raw = path.read_bytes()
                if sha256(raw).hexdigest() != expected_hash:
                    raise ValueError(f"calibration raw daily-bar SHA-256 mismatch: {path}")
                frame = pd.read_parquet(BytesIO(raw))
                if len(frame) != 1 or not {
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                }.issubset(frame.columns):
                    raise ValueError(f"invalid calibration daily-bar schema: {path}")
                sidecar = path.with_suffix(".parquet.sha256").read_text().strip().split()[0]
                logical = dataframe_content_sha256(
                    frame, sort_by=[column for column in frame.columns if column != "date"]
                )
                if logical != sidecar:
                    raise ValueError(f"calibration logical daily-bar checksum mismatch: {path}")
                row = frame.iloc[0]
                if pd.Timestamp(row["date"]).date() != session:
                    raise ValueError("calibration daily-bar partition date mismatch")
                open_row[symbol], close_row[symbol] = float(row["open"]), float(row["close"])
            price_cache[session] = (open_row, close_row)
        markets[episode["episode_id"]] = CalibrationMarketData(
            symbols=tuple(symbols),
            sessions=sessions,
            session_open_times=opens,
            session_close_times=closes,
            open_prices=tuple(price_cache[day][0] for day in sessions),
            close_prices=tuple(price_cache[day][1] for day in sessions),
            train_start=sessions[0],
            train_end=sessions[-1],
            data_version=source["data_version"],
            source_manifest_sha256=source["input_manifest_sha256"],
        )
    return markets


def load_selected_calibration_markets(
    *,
    episodes: list[dict[str, Any]],
    source: Mapping[str, Any],
    raw_manifest_bytes: bytes,
    symbols: list[str],
    exchange_calendar: str,
) -> dict[str, CalibrationMarketData]:
    """Decode only the 32 screen-train markets; this entry cannot load screen-dev prices."""

    return _load_selected_markets(
        episodes=episodes,
        source=source,
        raw_manifest_bytes=raw_manifest_bytes,
        symbols=symbols,
        exchange_calendar=exchange_calendar,
        role="train",
    )


def _environment_common(
    *, config: Mapping[str, Any], source: Mapping[str, Any], calibration: Mapping[str, Any]
) -> dict[str, Any]:
    execution, availability = config["execution"], config["data"]["availability"]
    reward = calibration["reward"]
    return {
        **{
            key: source[key]
            for key in (
                "manifest_root",
                "storage_root",
                "data_version",
                "data_start",
                "data_end",
                "input_manifest_sha256",
                "input_file_count",
                "input_sessions_per_symbol",
            )
        },
        "symbols": config["data"]["universe"]["symbols"],
        "initial_cash": float(execution["initial_cash"]),
        "commission_bps": float(execution["commission_bps"]),
        "slippage_bps": float(execution["slippage_bps"]),
        "drawdown_coefficient": 0.0,
        "turnover_coefficient": 0.0,
        "risk_lambda": reward["risk_lambda"],
        "turnover_limit": 1.0,
        "position_limit": 0.30,
        "screening": True,
        "finance_scorer_id": SCORER_ID,
        "finance_volatility_floor": reward["volatility_floor"],
        "screening_calibration_sha256": calibration["hashes"]["payload_sha256"],
        "market_timezone": availability["timezone"],
        "exchange_calendar": availability["exchange_calendar"],
        "market_close_time": availability["market_close_time"],
    }


def _dataset_rows(
    *,
    role: str,
    episodes: list[dict[str, Any]],
    indices: list[int],
    environment_common: Mapping[str, Any],
) -> list[dict[str, Any]]:
    split = "train" if role in {"train", "pilot_train"} else "valid"
    screening_role = {"train": "screen_train", "dev": "screen_dev"}.get(role, role)
    return [
        {
            "data_source": f"history/{screening_role}",
            "agent_name": "trading",
            "prompt": [
                {"role": "user", "content": "Manage the portfolio using only point-in-time tools."}
            ],
            "reward_model": {"style": "rule", "ground_truth": 0.0},
            "extra_info": {
                "episode_id": episode["episode_id"],
                "split": split,
                "index": row_index,
                "source_split": "train",
                "source_index": source_index,
                "screening_role": screening_role,
                "scientific_use": "screening",
            },
            "env_kwargs": json.dumps(
                {
                    **environment_common,
                    **{
                        key: episode[key]
                        for key in ("episode_dates", "episode_open_times", "benchmark_closes")
                    },
                },
                sort_keys=True,
                allow_nan=False,
            ),
        }
        for row_index, (episode, source_index) in enumerate(zip(episodes, indices, strict=True))
    ]


def prepare_screening_datasets(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config_path, manifest_path, output = (
        _resolve(config_path, root),
        _resolve(manifest_path, root),
        _resolve(output_dir, root),
    )
    config = load_experiment_config(config_path)
    original_bytes = manifest_path.read_bytes()
    original = yaml.safe_load(original_bytes)
    selected = select_screening_episodes(manifest=original, config=config)
    source, audit = _audit_train_source(config, root)
    markets = load_selected_calibration_markets(
        episodes=selected["train"],
        source=source,
        raw_manifest_bytes=audit.manifest_bytes,
        symbols=config["data"]["universe"]["symbols"],
        exchange_calendar=config["data"]["availability"]["exchange_calendar"],
    )
    execution = config["execution"]
    calibration = calibrate_screening_reward(
        markets=markets,
        expected_benchmarks={
            row["episode_id"]: row["benchmark_closes"] for row in selected["train"]
        },
        initial_cash=float(execution["initial_cash"]),
        execution_policy=ExecutionPolicy(
            float(execution["commission_bps"]), float(execution["slippage_bps"])
        ),
        source={
            **source,
            "original_episode_manifest_sha256": sha256(original_bytes).hexdigest(),
            "calibration_selection_sha256": canonical_sha256(selected["train"]),
            "calibration_unique_session_count": 145,
            "calibration_price_decode_start": "2023-01-03",
            "calibration_price_decode_end": "2023-08-01",
            "formal_valid_prices_accessed": False,
            "final_oos_prices_accessed": False,
        },
        legacy_reward={
            "epsilon": float(config["reward"]["epsilon"]),
            "risk_lambda": float(config["reward"]["risk_lambda"]),
        },
    )
    calibration_bytes = _json_bytes(calibration)
    _write_identical_or_new(output / "reward_calibration.json", calibration_bytes)
    _write_identical_or_new(output / "train_raw_manifest.tsv", audit.manifest_bytes)
    environment = _environment_common(config=config, source=source, calibration=calibration)
    datasets = {}
    for role in DATASET_ROLES:
        episodes = selected["dev"] if role == "dev" else selected["train"]
        indices = list(SCREEN_DEV_INDICES if role == "dev" else SCREEN_TRAIN_INDICES)
        if role.startswith("pilot_"):
            episodes, indices = episodes[:2], indices[:2]
        rows = _dataset_rows(
            role=role, episodes=episodes, indices=indices, environment_common=environment
        )
        buffer = BytesIO()
        pq.write_table(pa.Table.from_pylist(rows), buffer)
        content = buffer.getvalue()
        filename = f"{role}.parquet"
        _write_identical_or_new(output / filename, content)
        datasets[role] = {
            "path": filename,
            "sha256": sha256(content).hexdigest(),
            "row_count": len(rows),
            "episode_ids": [row["episode_id"] for row in episodes],
            "source_indices": indices,
            "rows_sha256": canonical_sha256(rows),
        }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": "m2_screening_v2",
        "scorer_id": SCORER_ID,
        "config": {
            "path": str(config_path),
            "sha256": sha256(config_path.read_bytes()).hexdigest(),
        },
        "original_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(original_bytes).hexdigest(),
        },
        "source": source,
        "selected_episodes": selected,
        "selection": {
            "train_indices": list(SCREEN_TRAIN_INDICES),
            "dev_indices": list(SCREEN_DEV_INDICES),
            "train_unique_session_count": 145,
            "dev_unique_session_count": 252,
        },
        "calibration": {
            "path": "reward_calibration.json",
            "sha256": sha256(calibration_bytes).hexdigest(),
            "payload_sha256": calibration["hashes"]["payload_sha256"],
        },
        "raw_manifest": {"path": "train_raw_manifest.tsv", "sha256": audit.manifest_sha256},
        "datasets": datasets,
        "historical_exposure": {
            "old_full_train_calibration_included_screen_dev": True,
            "prior_formal_valid_wiring_smoke": True,
            "screen_dev_is_untouched_holdout": False,
        },
        "hashes": {},
    }
    manifest["hashes"]["payload_sha256"] = screening_payload_sha256(manifest)
    _write_identical_or_new(output / "manifest.json", _json_bytes(manifest))
    return validate_prepared_screening(output, root)


def validate_prepared_screening(
    output_dir: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Verify source bytes and regenerate exact rows; extra env kwargs fail closed."""

    root = Path(repo_root).resolve()
    output = _resolve(output_dir, root)
    document = json.loads((output / "manifest.json").read_bytes())
    if document.get("protocol_id") != "m2_screening_v2" or document.get("schema_version") != 1:
        raise ValueError("unsupported prepared screening manifest")
    if document.get("hashes", {}).get("payload_sha256") != screening_payload_sha256(document):
        raise ValueError("prepared screening manifest payload_sha256 mismatch")
    for name in ("config", "original_manifest"):
        descriptor = document[name]
        if sha256(Path(descriptor["path"]).read_bytes()).hexdigest() != descriptor["sha256"]:
            raise ValueError(f"prepared screening {name} source bytes have drifted")
    config = load_experiment_config(document["config"]["path"])
    original = yaml.safe_load(Path(document["original_manifest"]["path"]).read_bytes())
    expected_selection = select_screening_episodes(manifest=original, config=config)
    if document["selected_episodes"] != expected_selection:
        raise ValueError("prepared screening selected episodes have drifted")
    source, audit = _audit_train_source(config, root)
    if document["source"] != source:
        raise ValueError("prepared screening Train-only source bytes have drifted")
    for name in ("calibration", "raw_manifest"):
        descriptor = document[name]
        path = output / descriptor["path"]
        if (
            not path.resolve().is_relative_to(output)
            or sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]
        ):
            raise ValueError(f"prepared screening {name} artifact has drifted")
    if (output / document["raw_manifest"]["path"]).read_bytes() != audit.manifest_bytes:
        raise ValueError("prepared screening per-file raw manifest has drifted")
    calibration = load_screening_calibration(output / document["calibration"]["path"])
    if calibration["hashes"]["payload_sha256"] != document["calibration"]["payload_sha256"]:
        raise ValueError("prepared screening calibration identity mismatch")
    if calibration["source"]["calibration_selection_sha256"] != canonical_sha256(
        expected_selection["train"]
    ):
        raise ValueError("calibration does not bind the selected screen-train inputs")
    if calibration["source"]["input_manifest_sha256"] != audit.manifest_sha256:
        raise ValueError("calibration does not bind the verified Train raw snapshot")
    if set(document["datasets"]) != set(DATASET_ROLES):
        raise ValueError("prepared screening requires exactly four dataset artifacts")
    environment = _environment_common(config=config, source=source, calibration=calibration)
    for role in DATASET_ROLES:
        descriptor = document["datasets"][role]
        path = output / descriptor["path"]
        if (
            not path.resolve().is_relative_to(output)
            or sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]
        ):
            raise ValueError(f"prepared screening {role} parquet bytes have drifted")
        episodes = expected_selection["dev"] if role == "dev" else expected_selection["train"]
        indices = list(SCREEN_DEV_INDICES if role == "dev" else SCREEN_TRAIN_INDICES)
        if role.startswith("pilot_"):
            episodes, indices = episodes[:2], indices[:2]
        expected_rows = _dataset_rows(
            role=role, episodes=episodes, indices=indices, environment_common=environment
        )
        rows = pq.read_table(path).to_pylist()
        if rows != expected_rows or descriptor["rows_sha256"] != canonical_sha256(expected_rows):
            raise ValueError(f"prepared screening {role} row/env_kwargs contract has drifted")
        if (
            descriptor["row_count"] != len(expected_rows)
            or descriptor["source_indices"] != indices
            or descriptor["episode_ids"] != [episode["episode_id"] for episode in episodes]
        ):
            raise ValueError(f"prepared screening {role} dataset metadata has drifted")
    return document


def evaluate_screening_baselines(
    prepared_dir: str | Path,
    repo_root: str | Path,
) -> list[dict[str, Any]]:
    """Evaluate the 24 CPU dev baselines only when the final readout explicitly calls this.

    Preparation/validation never call this function or decode screen-dev prices.
    """

    import numpy as np

    root = Path(repo_root).resolve()
    output = _resolve(prepared_dir, root)
    manifest = validate_prepared_screening(output, root)
    config = load_experiment_config(manifest["config"]["path"])
    calibration = load_screening_calibration(output / manifest["calibration"]["path"])
    episodes = manifest["selected_episodes"]["dev"]
    markets = _load_selected_markets(
        episodes=episodes,
        source=manifest["source"],
        raw_manifest_bytes=(output / manifest["raw_manifest"]["path"]).read_bytes(),
        symbols=config["data"]["universe"]["symbols"],
        exchange_calendar=config["data"]["availability"]["exchange_calendar"],
        role="dev",
    )
    execution, reward = config["execution"], calibration["reward"]
    policy = ExecutionPolicy(float(execution["commission_bps"]), float(execution["slippage_bps"]))
    results = []
    for episode in episodes:
        market = markets[episode["episode_id"]]
        benchmark = np.asarray(episode["benchmark_closes"], dtype=np.float64)
        benchmark_return = float(benchmark[-1] / benchmark[0] - 1.0)
        benchmark_volatility = float(np.std(np.diff(benchmark) / benchmark[:-1], ddof=0))
        proxies = dict(screening_proxy_targets(market.symbols))
        for baseline, proxy in (("cash", "cash"), ("buy_and_hold", "equal_weight_100")):
            replay = _replay(
                market=market,
                start=0,
                targets=proxies[proxy],
                initial_cash=float(execution["initial_cash"]),
                execution_policy=policy,
            )
            if baseline == "buy_and_hold" and not np.allclose(
                replay.equity_curve, benchmark, rtol=1e-9, atol=0.0
            ):
                raise ValueError("dev buy-and-hold baseline must reproduce the reviewed benchmark")
            risk = RiskComponents(
                max(replay.drawdowns),
                max(0.0, max(replay.daily_bilateral_turnovers) - 1.0),
                max(0.0, max(replay.maximum_position_weights) - 0.30),
            )
            net_return = replay.equity_curve[-1] / replay.equity_curve[0] - 1.0
            score = score_benchmark_scaled(
                episode_return=net_return,
                benchmark_return=benchmark_return,
                benchmark_volatility=benchmark_volatility,
                volatility_floor=reward["volatility_floor"],
                risk_lambda=reward["risk_lambda"],
                risk=risk,
            )
            results.append(
                {
                    "episode_id": episode["episode_id"],
                    "baseline": baseline,
                    "terminal_score": score.advantage,
                    "raw_net_return": net_return,
                    "benchmark_return": benchmark_return,
                    "benchmark_volatility": benchmark_volatility,
                    "max_drawdown": risk.max_drawdown,
                    "turnover_excess": risk.turnover_excess,
                    "position_excess": risk.position_excess,
                    "risk_penalty": risk.penalty,
                    "turnover": math.fsum(replay.daily_bilateral_turnovers),
                    "max_daily_turnover": max(replay.daily_bilateral_turnovers),
                    "transaction_cost": math.fsum(replay.transaction_costs),
                    "agent_volatility": float(
                        np.std(
                            np.diff(np.asarray(replay.equity_curve))
                            / np.asarray(replay.equity_curve)[:-1],
                            ddof=0,
                        )
                    ),
                    "scorer_id": SCORER_ID,
                    "calibration_sha256": calibration["hashes"]["payload_sha256"],
                }
            )
    return results

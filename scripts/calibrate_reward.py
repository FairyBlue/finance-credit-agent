#!/usr/bin/env python3
"""Compute the unsigned reward freeze table from the verified history train window."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.config import load_experiment_config  # noqa: E402
from src.data import verify_raw_daily_manifest  # noqa: E402
from src.env import ExecutionPolicy  # noqa: E402
from src.rewards import (  # noqa: E402
    CalibrationMarketData,
    build_pending_calibration_document,
    calibrate_reward_parameters,
    verify_calibration_document,
)


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required(mapping: dict[str, Any], key: str, *, name: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{name}.{key} is required")
    return mapping[key]


def _resolved(path: object, *, repo_root: Path) -> Path:
    candidate = Path(str(path)).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()


def _load_verified_train_market(
    *,
    config: dict[str, Any],
    repo_root: Path,
) -> CalibrationMarketData:
    data = _mapping(_required(config, "data", name="config"), name="data")
    split_source = _mapping(
        _required(data, "split_source", name="data"),
        name="data.split_source",
    )
    if _required(split_source, "train", name="data.split_source") != "history":
        raise ValueError("reward calibration is restricted to the history train source")
    sources = _mapping(_required(data, "sources", name="data"), name="data.sources")
    history = _mapping(
        _required(sources, "history", name="data.sources"),
        name="data.sources.history",
    )
    windows = _mapping(_required(data, "windows", name="data"), name="data.windows")
    train = _mapping(_required(windows, "train", name="data.windows"), name="data.windows.train")
    input_window = _mapping(
        _required(history, "input_window", name="data.sources.history"),
        name="data.sources.history.input_window",
    )
    train_start = pd.Timestamp(_required(train, "start", name="data.windows.train")).date()
    train_end = pd.Timestamp(_required(train, "end", name="data.windows.train")).date()
    input_start = pd.Timestamp(
        _required(input_window, "start", name="data.sources.history.input_window")
    ).date()
    input_end = pd.Timestamp(
        _required(input_window, "end", name="data.sources.history.input_window")
    ).date()
    if not input_start <= train_start <= train_end <= input_end:
        raise ValueError("train window must be wholly inside the frozen history input window")

    universe = _mapping(_required(data, "universe", name="data"), name="data.universe")
    raw_symbols = _required(universe, "symbols", name="data.universe")
    if not isinstance(raw_symbols, list) or not all(isinstance(item, str) for item in raw_symbols):
        raise ValueError("data.universe.symbols must be an ordered string list")
    symbols = tuple(raw_symbols)
    availability = _mapping(
        _required(data, "availability", name="data"),
        name="data.availability",
    )
    exchange_calendar = str(_required(availability, "exchange_calendar", name="data.availability"))
    manifest_root = _resolved(
        _required(history, "manifest_root", name="data.sources.history"),
        repo_root=repo_root,
    )
    storage_root = _resolved(
        _required(history, "storage_root", name="data.sources.history"),
        repo_root=repo_root,
    )
    expected_manifest_sha256 = str(
        _required(history, "input_manifest_sha256", name="data.sources.history")
    )
    expected_files = int(_required(history, "files", name="data.sources.history"))
    expected_sessions = int(_required(history, "sessions_per_symbol", name="data.sources.history"))
    audit = verify_raw_daily_manifest(
        manifest_root=manifest_root,
        storage_root=storage_root,
        symbols=symbols,
        start=input_start,
        end=input_end,
        expected_sha256=expected_manifest_sha256,
        expected_file_count=expected_files,
        expected_sessions_per_symbol=expected_sessions,
        exchange_calendar=exchange_calendar,
    )
    expected_raw_sha256: dict[Path, str] = {}
    for line in audit.manifest_bytes.decode("utf-8").splitlines():
        relative, digest = line.split("\t", maxsplit=1)
        expected_raw_sha256[(manifest_root / relative).resolve()] = digest

    calendar = xcals.get_calendar(exchange_calendar)
    session_labels = tuple(
        calendar.sessions_in_range(train_start.isoformat(), train_end.isoformat())
    )
    sessions = tuple(label.date() for label in session_labels)
    if not sessions or sessions[0] != train_start or sessions[-1] != train_end:
        raise ValueError("train boundaries must each be complete exchange sessions")
    open_prices: list[dict[str, float]] = []
    close_prices: list[dict[str, float]] = []
    required_columns = {"date", "open", "high", "low", "close", "volume"}
    for session in sessions:
        open_row: dict[str, float] = {}
        close_row: dict[str, float] = {}
        for symbol in symbols:
            path = (
                storage_root / "parquet" / symbol / "day" / f"{session.isoformat()}.parquet"
            ).resolve()
            expected_digest = expected_raw_sha256.get(path)
            if expected_digest is None:
                raise RuntimeError(f"train daily bar is absent from the frozen manifest: {path}")
            raw_bytes = path.read_bytes()
            actual_digest = sha256(raw_bytes).hexdigest()
            if actual_digest != expected_digest:
                raise RuntimeError(
                    f"train daily-bar SHA-256 mismatch: expected {expected_digest}, "
                    f"got {actual_digest}: {path}"
                )
            frame = pd.read_parquet(BytesIO(raw_bytes))
            if len(frame) != 1:
                raise ValueError(f"train daily partition must contain exactly one row: {path}")
            missing = required_columns.difference(frame.columns)
            if missing:
                raise ValueError(f"train daily partition is missing {sorted(missing)}: {path}")
            row = frame.iloc[0]
            if pd.Timestamp(row["date"]).date() != session:
                raise ValueError(f"train partition date does not match its path: {path}")
            open_price = float(row["open"])
            close_price = float(row["close"])
            if any(not math.isfinite(value) or value <= 0.0 for value in (open_price, close_price)):
                raise ValueError(f"train open/close prices must be finite and positive: {path}")
            open_row[symbol] = open_price
            close_row[symbol] = close_price
        open_prices.append(open_row)
        close_prices.append(close_row)

    return CalibrationMarketData(
        symbols=symbols,
        sessions=sessions,
        session_open_times=tuple(
            calendar.session_open(label).to_pydatetime() for label in session_labels
        ),
        session_close_times=tuple(
            calendar.session_close(label).to_pydatetime() for label in session_labels
        ),
        open_prices=tuple(open_prices),
        close_prices=tuple(close_prices),
        train_start=train_start,
        train_end=train_end,
        data_version=str(_required(history, "version", name="data.sources.history")),
        source_manifest_sha256=audit.manifest_sha256,
    )


def _calibration_document_from_config(
    *,
    config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    market = _load_verified_train_market(config=config, repo_root=repo_root)
    execution = _mapping(_required(config, "execution", name="config"), name="execution")
    calibration = calibrate_reward_parameters(
        market=market,
        initial_cash=float(_required(execution, "initial_cash", name="execution")),
        execution_policy=ExecutionPolicy(
            commission_bps=float(_required(execution, "commission_bps", name="execution")),
            slippage_bps=float(_required(execution, "slippage_bps", name="execution")),
        ),
    )
    return build_pending_calibration_document(market=market, calibration=calibration)


def _write_new_or_identical(path: Path, document: dict[str, Any]) -> None:
    rendered = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to overwrite a different calibration table: {path}")
        return
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(rendered)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.chmod(temporary_path, 0o644)
        os.link(temporary_path, path)
    except FileExistsError as error:
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(
                f"refusing to overwrite a different calibration table: {path}"
            ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default="configs/reward_calibration.pending.yaml")
    parser.add_argument("--check")
    args = parser.parse_args()
    config_path = _resolved(args.config, repo_root=PROJECT_ROOT)
    output_path = _resolved(args.output, repo_root=PROJECT_ROOT)
    document = _calibration_document_from_config(
        config_path=config_path,
        repo_root=PROJECT_ROOT,
    )
    if args.check is not None:
        check_path = _resolved(args.check, repo_root=PROJECT_ROOT)
        existing = _mapping(
            yaml.safe_load(check_path.read_text(encoding="utf-8")),
            name="calibration document",
        )
        verify_calibration_document(existing)
        existing_hash = _required(
            _mapping(_required(existing, "hashes", name="calibration document"), name="hashes"),
            "freeze_table_sha256",
            name="hashes",
        )
        expected_hash = document["hashes"]["freeze_table_sha256"]
        if existing_hash != expected_hash:
            raise ValueError(
                "calibration table does not match the current verified train inputs and protocol"
            )
        print(json.dumps({"checked": str(check_path), "freeze_table_sha256": existing_hash}))
        return 0
    _write_new_or_identical(output_path, document)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "reward": document["reward"],
                "hashes": document["hashes"],
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Behavioral specification for train-only reward calibration."""

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, time, timezone
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd
import pytest
import yaml

from src.data import audit_raw_daily_manifest
from src.env import ExecutionPolicy
from src.rewards import (
    CalibrationMarketData,
    build_pending_calibration_document,
    calibrate_reward_parameters,
    rolling_volatility_epsilon,
    verify_calibration_document,
)

UTC = timezone.utc
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _market_data(*, session_count: int = 21) -> CalibrationMarketData:
    symbols = tuple(f"S{index:02d}" for index in range(20))
    calendar = xcals.get_calendar("XNYS")
    sessions = tuple(
        timestamp.date()
        for timestamp in calendar.sessions_in_range("2024-01-02", "2024-03-29")[:session_count]
    )
    open_times = tuple(datetime.combine(day, time(14, 30), UTC) for day in sessions)
    close_times = tuple(datetime.combine(day, time(21, 0), UTC) for day in sessions)
    closes: list[dict[str, float]] = []
    opens: list[dict[str, float]] = []
    for day_index in range(session_count):
        close_row: dict[str, float] = {}
        open_row: dict[str, float] = {}
        for symbol_index, symbol in enumerate(symbols):
            close = (
                100.0
                + symbol_index
                + day_index * (symbol_index - 9.5) * 0.15
                + ((day_index % 4) - 1.5) * ((symbol_index % 3) + 1) * 0.4
            )
            close_row[symbol] = close
            open_row[symbol] = close * (1.0 + 0.002 * (((day_index + symbol_index) % 3) - 1))
        closes.append(close_row)
        opens.append(open_row)
    return CalibrationMarketData(
        symbols=symbols,
        sessions=sessions,
        session_open_times=open_times,
        session_close_times=close_times,
        open_prices=tuple(opens),
        close_prices=tuple(closes),
        train_start=sessions[0],
        train_end=sessions[-1],
        data_version="fixture-train-only-v1",
        source_manifest_sha256="a" * 64,
    )


def _write_cli_fixture(tmp_path: Path) -> Path:
    market = _market_data()
    history_root = tmp_path / "history"
    for session_index, session in enumerate(market.sessions):
        for symbol in market.symbols:
            day_root = history_root / "parquet" / symbol / "day"
            day_root.mkdir(parents=True, exist_ok=True)
            open_price = market.open_prices[session_index][symbol]
            close_price = market.close_prices[session_index][symbol]
            pd.DataFrame(
                {
                    "date": [session],
                    "open": [open_price],
                    "high": [max(open_price, close_price)],
                    "low": [min(open_price, close_price)],
                    "close": [close_price],
                    "volume": [1000.0],
                }
            ).to_parquet(day_root / f"{session.isoformat()}.parquet", index=False)
    audit = audit_raw_daily_manifest(
        manifest_root=history_root,
        storage_root=history_root,
        symbols=market.symbols,
        start=market.train_start,
        end=market.train_end,
    )
    config = {
        "data": {
            "version": "fixture-composite-v1",
            "split_source": {"train": "history"},
            "sources": {
                "history": {
                    "version": "fixture-history-v1",
                    "manifest_root": str(history_root),
                    "storage_root": str(history_root),
                    "input_window": {
                        "start": market.train_start.isoformat(),
                        "end": market.train_end.isoformat(),
                    },
                    "input_manifest_sha256": audit.manifest_sha256,
                    "files": audit.file_count,
                    "sessions_per_symbol": len(market.sessions),
                }
            },
            "windows": {
                "train": {
                    "start": market.train_start.isoformat(),
                    "end": market.train_end.isoformat(),
                }
            },
            "universe": {"symbols": list(market.symbols)},
            "availability": {"exchange_calendar": "XNYS"},
        },
        "execution": {
            "initial_cash": 1_000_000.0,
            "commission_bps": 10.0,
            "slippage_bps": 5.0,
        },
    }
    config_path = tmp_path / "base.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_epsilon_is_ten_percent_of_median_20_return_volatility() -> None:
    equity = [100.0]
    for daily_return in (0.01, -0.01) * 10:
        equity.append(equity[-1] * (1.0 + daily_return))

    epsilon = rolling_volatility_epsilon(equity)

    assert epsilon == pytest.approx(0.001)


def test_train_only_calibration_caps_each_average_penalty_at_one_quarter_of_core() -> None:
    result = calibrate_reward_parameters(
        market=_market_data(),
        initial_cash=1_000_000.0,
        execution_policy=ExecutionPolicy(commission_bps=10.0, slippage_bps=5.0),
    )

    assert result.turnover_limit == 1.0
    assert result.position_limit == 0.30
    assert result.statistics.calibration_window_count == 1
    assert result.statistics.trajectory_count == 40
    assert result.statistics.step_count == 800
    assert result.statistics.maximum_daily_bilateral_turnover > 1.0
    assert result.risk_lambda * result.statistics.mean_risk_penalty == pytest.approx(
        0.25 * result.statistics.mean_absolute_finance_core
    )
    assert result.drawdown_coefficient * result.statistics.mean_drawdown == pytest.approx(
        0.25 * result.statistics.mean_absolute_step_core
    )
    assert result.turnover_coefficient * result.statistics.mean_daily_bilateral_turnover == (
        pytest.approx(0.25 * result.statistics.mean_absolute_step_core)
    )


def test_pending_freeze_table_is_deterministic_and_self_hashing() -> None:
    market = _market_data()
    result = calibrate_reward_parameters(
        market=market,
        initial_cash=1_000_000.0,
        execution_policy=ExecutionPolicy(commission_bps=10.0, slippage_bps=5.0),
    )

    first = build_pending_calibration_document(market=market, calibration=result)
    second = build_pending_calibration_document(market=market, calibration=result)

    assert first == second
    assert first["status"] == "pending_human_signature"
    assert first["amendment_date"] == "2026-08-24"
    assert first["reward"]["turnover_limit"] == 1.0
    assert first["protocol"]["turnover_semantics"] == "bilateral_daily"
    unhashed = dict(first)
    unhashed["hashes"] = dict(first["hashes"])
    del unhashed["hashes"]["freeze_table_sha256"]
    del unhashed["signature"]
    del unhashed["status"]
    expected = hashlib.sha256(
        json.dumps(
            unhashed,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    assert first["hashes"]["freeze_table_sha256"] == expected


def test_freeze_table_signature_can_be_completed_without_invalidating_payload_hash() -> None:
    market = _market_data()
    calibration = calibrate_reward_parameters(
        market=market,
        initial_cash=1_000_000.0,
        execution_policy=ExecutionPolicy(commission_bps=10.0, slippage_bps=5.0),
    )
    signed = build_pending_calibration_document(market=market, calibration=calibration)
    signed["status"] = "signed"
    signed["signature"] = {
        "reviewer": "reviewer-id",
        "signed_at": "2026-08-24T20:00:00+08:00",
    }

    verify_calibration_document(signed)

    signed["reward"]["epsilon"] *= 2.0
    with pytest.raises(ValueError, match="freeze_table_sha256 mismatch"):
        verify_calibration_document(signed)


def test_cli_writes_a_pending_freeze_table_from_verified_train_files(tmp_path: Path) -> None:
    config_path = _write_cli_fixture(tmp_path)
    output_path = tmp_path / "reward_calibration.pending.yaml"

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "calibrate_reward.py"),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    document = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert document["status"] == "pending_human_signature"
    assert document["source"]["data_version"] == "fixture-history-v1"
    assert document["statistics"]["calibration_window_count"] == 1
    assert "freeze_table_sha256" in document["hashes"]

    document["status"] = "signed"
    document["signature"] = {
        "reviewer": "reviewer-id",
        "signed_at": "2026-08-24T20:00:00+08:00",
    }
    output_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    checked = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "calibrate_reward.py"),
            "--config",
            str(config_path),
            "--check",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stderr


def test_calibration_fails_closed_on_out_of_train_nonfinite_or_zero_evidence() -> None:
    market = _market_data()
    with pytest.raises(ValueError, match="exactly equal the declared train window"):
        replace(market, train_start=market.sessions[1])

    invalid_opens = [dict(row) for row in market.open_prices]
    invalid_opens[0][market.symbols[0]] = float("inf")
    with pytest.raises(ValueError, match="finite and positive"):
        replace(market, open_prices=tuple(invalid_opens))

    constant_rows = tuple({symbol: 100.0 for symbol in market.symbols} for _ in market.sessions)
    constant_market = replace(
        market,
        open_prices=constant_rows,
        close_prices=constant_rows,
    )
    with pytest.raises(ValueError, match="epsilon calibration produced"):
        calibrate_reward_parameters(
            market=constant_market,
            initial_cash=1_000_000.0,
            execution_policy=ExecutionPolicy(commission_bps=0.0, slippage_bps=0.0),
        )

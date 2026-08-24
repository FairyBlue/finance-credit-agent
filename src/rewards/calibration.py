"""Deterministic, train-only calibration for the frozen reward protocol."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from hashlib import sha256
from types import MappingProxyType
from typing import Any

import numpy as np

from src.env import ExecutionPolicy, PortfolioLedger

_SYMBOL_COUNT = 20
_ROLLING_RETURNS = 20
_CALIBRATION_WINDOW_SESSIONS = 21
_EPSILON_FRACTION = 0.10
_PENALTY_FRACTION = 0.25
_TURNOVER_LIMIT = 1.0
_POSITION_LIMIT = 0.30
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class CalibrationMarketData:
    """Exact train-window prices and source identity used for calibration."""

    symbols: tuple[str, ...]
    sessions: tuple[date, ...]
    session_open_times: tuple[datetime, ...]
    session_close_times: tuple[datetime, ...]
    open_prices: tuple[Mapping[str, float], ...]
    close_prices: tuple[Mapping[str, float], ...]
    train_start: date
    train_end: date
    data_version: str
    source_manifest_sha256: str

    def __post_init__(self) -> None:
        symbols = tuple(symbol.strip().upper() for symbol in self.symbols)
        if len(symbols) != _SYMBOL_COUNT or len(set(symbols)) != _SYMBOL_COUNT:
            raise ValueError("calibration requires the frozen ordered universe of 20 symbols")
        if any(not symbol for symbol in symbols):
            raise ValueError("calibration symbols must not be empty")
        object.__setattr__(self, "symbols", symbols)

        sessions = tuple(self.sessions)
        if len(sessions) < _CALIBRATION_WINDOW_SESSIONS:
            raise ValueError("calibration needs at least 21 train sessions")
        if any(left >= right for left, right in zip(sessions, sessions[1:], strict=False)):
            raise ValueError("calibration sessions must be strictly increasing")
        if self.train_start != sessions[0] or self.train_end != sessions[-1]:
            raise ValueError("calibration data must exactly equal the declared train window")
        if any(not self.train_start <= session <= self.train_end for session in sessions):
            raise ValueError("calibration data contains an out-of-train session")
        object.__setattr__(self, "sessions", sessions)

        aligned = (
            self.session_open_times,
            self.session_close_times,
            self.open_prices,
            self.close_prices,
        )
        if any(len(values) != len(sessions) for values in aligned):
            raise ValueError("calibration timestamps and price rows must align with sessions")
        for index, (session, open_time, close_time) in enumerate(
            zip(sessions, self.session_open_times, self.session_close_times, strict=True)
        ):
            if (
                open_time.tzinfo is None
                or open_time.utcoffset() is None
                or close_time.tzinfo is None
                or close_time.utcoffset() is None
            ):
                raise ValueError("calibration timestamps must be timezone-aware")
            if open_time >= close_time:
                raise ValueError("every calibration session open must precede its close")
            if index and self.session_close_times[index - 1] >= open_time:
                raise ValueError("every execution open must follow the prior decision close")
            if open_time.date() != session or close_time.date() != session:
                raise ValueError("calibration timestamp dates must match their sessions")

        expected_symbols = set(symbols)
        normalized_rows: list[tuple[Mapping[str, float], ...]] = []
        for row_kind, rows in (("open", self.open_prices), ("close", self.close_prices)):
            normalized_kind: list[Mapping[str, float]] = []
            for row in rows:
                if set(row) != expected_symbols:
                    raise ValueError(f"every {row_kind} price row must contain exactly 20 symbols")
                normalized = {symbol: float(row[symbol]) for symbol in symbols}
                if any(not math.isfinite(value) or value <= 0.0 for value in normalized.values()):
                    raise ValueError(f"{row_kind} prices must be finite and positive")
                normalized_kind.append(MappingProxyType(normalized))
            normalized_rows.append(tuple(normalized_kind))
        object.__setattr__(self, "open_prices", normalized_rows[0])
        object.__setattr__(self, "close_prices", normalized_rows[1])

        if not self.data_version.strip():
            raise ValueError("data_version must not be empty")
        if _SHA256.fullmatch(self.source_manifest_sha256) is None:
            raise ValueError("source_manifest_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class CalibrationStatistics:
    train_session_count: int
    epsilon_return_count: int
    epsilon_rolling_window_count: int
    calibration_window_count: int
    trajectory_count: int
    step_count: int
    median_20_return_volatility: float
    mean_absolute_finance_core: float
    mean_risk_penalty: float
    mean_absolute_step_core: float
    mean_drawdown: float
    mean_daily_bilateral_turnover: float
    maximum_daily_bilateral_turnover: float


@dataclass(frozen=True, slots=True)
class RewardCalibration:
    epsilon: float
    risk_lambda: float
    drawdown_coefficient: float
    turnover_coefficient: float
    turnover_limit: float
    position_limit: float
    statistics: CalibrationStatistics


@dataclass(frozen=True, slots=True)
class _Replay:
    equity_curve: tuple[float, ...]
    transaction_costs: tuple[float, ...]
    daily_bilateral_turnovers: tuple[float, ...]
    drawdowns: tuple[float, ...]
    maximum_position_weights: tuple[float, ...]


def _canonical_json_sha256(document: Mapping[str, Any]) -> str:
    payload = json.dumps(
        document,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256(payload).hexdigest()


def _freeze_table_sha256(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload.pop("status", None)
    payload.pop("signature", None)
    hashes = dict(payload.get("hashes", {}))
    hashes.pop("freeze_table_sha256", None)
    payload["hashes"] = hashes
    return _canonical_json_sha256(payload)


def _protocol_document() -> dict[str, Any]:
    return {
        "train_only": True,
        "epsilon": {
            "nav": "shared_ledger_initial_equal_weight_buy_and_hold_continuous_train",
            "rolling_returns": _ROLLING_RETURNS,
            "standard_deviation_ddof": 0,
            "reducer": "median",
            "multiplier": _EPSILON_FRACTION,
        },
        "calibration_windows": {
            "sessions": _CALIBRATION_WINDOW_SESSIONS,
            "overlapping": True,
        },
        "trajectories_per_window": {
            "single_stock_buy_and_hold": _SYMBOL_COUNT,
            "cyclic_midpoint_full_rotation": _SYMBOL_COUNT,
            "rotation_pairing": "symbol_i_to_symbol_i_plus_1_mod_20",
            "midpoint_decision_session_offset_zero_based": 10,
            "execution": "next_open",
        },
        "penalty_fraction_of_core_mean_magnitude": _PENALTY_FRACTION,
        "risk_lambda": {
            "core": "mean_absolute_A_fin_before_risk_penalty",
            "exposure": (
                "mean_per_trajectory_max_of_max_drawdown_"
                "max_daily_bilateral_turnover_excess_1.00_"
                "max_position_excess_0.30"
            ),
        },
        "a2_coefficients": {
            "core": "mean_absolute_unpenalized_step_core",
            "drawdown_exposure": "mean_step_drawdown",
            "turnover_exposure": "mean_daily_bilateral_turnover",
            "selection": "maximum_coefficient_at_equality",
        },
        "threshold_semantics": "soft_penalty",
        "turnover_semantics": "bilateral_daily",
        "turnover_limit_interpretation": (
            "1.00_bilateral_is_approximately_50_percent_one_way_rotation"
        ),
    }


def build_pending_calibration_document(
    *,
    market: CalibrationMarketData,
    calibration: RewardCalibration,
) -> dict[str, Any]:
    """Build the deterministic, unsigned reward freeze table and its content hashes."""

    protocol = _protocol_document()
    session_bytes = "".join(f"{session.isoformat()}\n" for session in market.sessions).encode()
    symbol_bytes = "".join(f"{symbol}\n" for symbol in market.symbols).encode()
    document: dict[str, Any] = {
        "status": "pending_human_signature",
        "amendment_date": "2026-08-24",
        "source": {
            "data_version": market.data_version,
            "history_manifest_sha256": market.source_manifest_sha256,
            "train_window": {
                "start": market.train_start.isoformat(),
                "end": market.train_end.isoformat(),
                "inclusive": True,
            },
            "frozen_symbol_order": list(market.symbols),
        },
        "protocol": protocol,
        "statistics": asdict(calibration.statistics),
        "reward": {
            "epsilon": calibration.epsilon,
            "risk_lambda": calibration.risk_lambda,
            "drawdown_coefficient": calibration.drawdown_coefficient,
            "turnover_coefficient": calibration.turnover_coefficient,
            "turnover_limit": calibration.turnover_limit,
            "position_limit": calibration.position_limit,
        },
        "hashes": {
            "history_manifest_sha256": market.source_manifest_sha256,
            "train_session_list_sha256": sha256(session_bytes).hexdigest(),
            "frozen_symbol_order_sha256": sha256(symbol_bytes).hexdigest(),
            "protocol_sha256": _canonical_json_sha256(protocol),
            "freeze_table_hash_format": (
                "canonical_json_excluding_status_signature_and_freeze_table_sha256_v1"
            ),
        },
        "signature": {
            "reviewer": None,
            "signed_at": None,
        },
    }
    document["hashes"]["freeze_table_sha256"] = _freeze_table_sha256(document)
    return document


def verify_calibration_document(document: Mapping[str, Any]) -> None:
    """Verify the immutable calibration payload while allowing signatures to be added."""

    hashes = document.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("calibration hashes must be a mapping")
    expected_format = "canonical_json_excluding_status_signature_and_freeze_table_sha256_v1"
    if hashes.get("freeze_table_hash_format") != expected_format:
        raise ValueError("unsupported calibration freeze-table hash format")
    recorded = hashes.get("freeze_table_sha256")
    if not isinstance(recorded, str) or _SHA256.fullmatch(recorded) is None:
        raise ValueError("calibration freeze_table_sha256 must be a lowercase SHA-256")
    actual = _freeze_table_sha256(document)
    if actual != recorded:
        raise ValueError(
            f"calibration freeze_table_sha256 mismatch: expected {recorded}, got {actual}"
        )

    status = document.get("status")
    signature = document.get("signature")
    if not isinstance(signature, Mapping):
        raise ValueError("calibration signature must be a mapping")
    reviewer = signature.get("reviewer")
    signed_at = signature.get("signed_at")
    if status == "pending_human_signature":
        if reviewer is not None or signed_at is not None:
            raise ValueError("pending calibration signatures must remain empty")
        return
    if status != "signed":
        raise ValueError("calibration status must be pending_human_signature or signed")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("signed calibration requires a non-empty reviewer")
    if not isinstance(signed_at, str):
        raise ValueError("signed calibration requires a timezone-aware signed_at")
    try:
        timestamp = datetime.fromisoformat(signed_at)
    except ValueError as error:
        raise ValueError("signed calibration signed_at must be ISO-8601") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("signed calibration signed_at must be timezone-aware")


def rolling_volatility_epsilon(
    equity_curve: Sequence[float],
    *,
    rolling_returns: int = 20,
    fraction: float = 0.10,
) -> float:
    """Return ``fraction * median(rolling std of adjacent NAV returns)``."""

    equity = np.asarray(tuple(float(value) for value in equity_curve), dtype=np.float64)
    if rolling_returns < 1:
        raise ValueError("rolling_returns must be positive")
    if equity.size < rolling_returns + 1:
        raise ValueError(
            f"equity_curve needs at least {rolling_returns + 1} values for calibration"
        )
    if not np.all(np.isfinite(equity)) or np.any(equity <= 0.0):
        raise ValueError("equity_curve values must be finite and positive")
    if not math.isfinite(fraction) or fraction <= 0.0:
        raise ValueError("fraction must be finite and positive")

    returns = equity[1:] / equity[:-1] - 1.0
    rolling = np.asarray(
        [
            np.std(returns[index - rolling_returns : index], ddof=0)
            for index in range(rolling_returns, returns.size + 1)
        ],
        dtype=np.float64,
    )
    epsilon = float(np.median(rolling) * fraction)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon calibration produced a non-positive or non-finite value")
    return epsilon


def _mean(values: Sequence[float], *, name: str) -> float:
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{name} calibration evidence must be finite and non-negative")
    result = math.fsum(values) / len(values)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} calibration evidence has zero or non-finite mean")
    return result


def _replay(
    *,
    market: CalibrationMarketData,
    start: int,
    targets: Sequence[Mapping[str, float]],
    initial_cash: float,
    execution_policy: ExecutionPolicy,
) -> _Replay:
    if not targets or start < 0 or start + len(targets) >= len(market.sessions):
        raise ValueError("calibration replay targets must fit wholly inside train sessions")
    ledger = PortfolioLedger(initial_cash=initial_cash, policy=execution_policy)
    equities = [ledger.equity(market.close_prices[start])]
    costs: list[float] = []
    turnovers: list[float] = []
    drawdowns: list[float] = []
    position_weights: list[float] = []
    high_water_mark = equities[0]

    for offset, target_row in enumerate(targets):
        decision_index = start + offset
        execution_index = decision_index + 1
        for symbol, target_weight in target_row.items():
            ledger.queue_target_order(
                symbol,
                target_weight=float(target_weight),
                decision_at=market.session_close_times[decision_index],
            )
        equity_before = equities[-1]
        fills = ledger.execute_next_open(
            market.session_open_times[execution_index],
            market.open_prices[execution_index],
        )
        equity_after = ledger.equity(market.close_prices[execution_index])
        if not math.isfinite(equity_after) or equity_after <= 0.0:
            raise ValueError("calibration replay produced non-positive or non-finite equity")
        transaction_cost = math.fsum(fill.total_cost for fill in fills)
        traded_notional = math.fsum(fill.notional for fill in fills)
        turnover = traded_notional / equity_before
        high_water_mark = max(high_water_mark, equity_after)
        drawdown = max(0.0, (high_water_mark - equity_after) / high_water_mark)
        maximum_weight = max(
            (
                shares * market.close_prices[execution_index][symbol] / equity_after
                for symbol, shares in ledger.positions.items()
            ),
            default=0.0,
        )
        values = (transaction_cost, turnover, drawdown, maximum_weight)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("calibration replay produced invalid trajectory evidence")
        equities.append(equity_after)
        costs.append(transaction_cost)
        turnovers.append(turnover)
        drawdowns.append(drawdown)
        position_weights.append(maximum_weight)

    return _Replay(
        equity_curve=tuple(equities),
        transaction_costs=tuple(costs),
        daily_bilateral_turnovers=tuple(turnovers),
        drawdowns=tuple(drawdowns),
        maximum_position_weights=tuple(position_weights),
    )


def _hold_targets(symbols: Sequence[str]) -> tuple[dict[str, float], ...]:
    return (
        {symbol: 1.0 for symbol in symbols},
        *({} for _ in range(_CALIBRATION_WINDOW_SESSIONS - 2)),
    )


def _equal_weight_targets(
    symbols: Sequence[str],
    *,
    decision_count: int,
) -> tuple[dict[str, float], ...]:
    return (
        {symbol: 1.0 / len(symbols) for symbol in symbols},
        *({} for _ in range(decision_count - 1)),
    )


def _rotation_targets(source: str, destination: str) -> tuple[dict[str, float], ...]:
    targets: list[dict[str, float]] = [{} for _ in range(_CALIBRATION_WINDOW_SESSIONS - 1)]
    targets[0] = {source: 1.0}
    midpoint_decision = (_CALIBRATION_WINDOW_SESSIONS - 1) // 2
    targets[midpoint_decision] = {source: 0.0, destination: 1.0}
    return tuple(targets)


def _calibration_coefficient(*, core_mean: float, exposure_mean: float, name: str) -> float:
    coefficient = _PENALTY_FRACTION * core_mean / exposure_mean
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError(f"{name} calibration produced a non-positive or non-finite coefficient")
    return coefficient


def calibrate_reward_parameters(
    *,
    market: CalibrationMarketData,
    initial_cash: float,
    execution_policy: ExecutionPolicy,
) -> RewardCalibration:
    """Calibrate all reward constants from the exact, complete train window."""

    cash = float(initial_cash)
    if not math.isfinite(cash) or cash <= 0.0:
        raise ValueError("initial_cash must be finite and positive")

    full_benchmark = _replay(
        market=market,
        start=0,
        targets=_equal_weight_targets(
            market.symbols,
            decision_count=len(market.sessions) - 1,
        ),
        initial_cash=cash,
        execution_policy=execution_policy,
    )
    epsilon = rolling_volatility_epsilon(full_benchmark.equity_curve)

    absolute_finance_cores: list[float] = []
    risk_penalties: list[float] = []
    absolute_step_cores: list[float] = []
    drawdown_exposures: list[float] = []
    turnover_exposures: list[float] = []
    maximum_daily_turnover = 0.0
    calibration_window_count = len(market.sessions) - _CALIBRATION_WINDOW_SESSIONS + 1

    for start in range(calibration_window_count):
        benchmark = _replay(
            market=market,
            start=start,
            targets=_equal_weight_targets(market.symbols, decision_count=20),
            initial_cash=cash,
            execution_policy=execution_policy,
        )
        trajectory_targets: list[tuple[dict[str, float], ...]] = [
            _hold_targets((symbol,)) for symbol in market.symbols
        ] + [
            _rotation_targets(symbol, market.symbols[(index + 1) % len(market.symbols)])
            for index, symbol in enumerate(market.symbols)
        ]
        benchmark_return = benchmark.equity_curve[-1] / benchmark.equity_curve[0] - 1.0
        benchmark_step_returns = tuple(
            following / current - 1.0
            for current, following in zip(
                benchmark.equity_curve,
                benchmark.equity_curve[1:],
                strict=False,
            )
        )

        for targets in trajectory_targets:
            trajectory = _replay(
                market=market,
                start=start,
                targets=targets,
                initial_cash=cash,
                execution_policy=execution_policy,
            )
            net_returns = np.asarray(
                [
                    following / current - 1.0
                    for current, following in zip(
                        trajectory.equity_curve,
                        trajectory.equity_curve[1:],
                        strict=False,
                    )
                ],
                dtype=np.float64,
            )
            episode_return = trajectory.equity_curve[-1] / trajectory.equity_curve[0] - 1.0
            finance_core = (episode_return - benchmark_return) / (
                float(np.std(net_returns, ddof=0)) + epsilon
            )
            if not math.isfinite(finance_core):
                raise ValueError("A_fin core calibration produced a non-finite value")
            absolute_finance_cores.append(abs(finance_core))
            risk_penalties.append(
                max(
                    max(trajectory.drawdowns),
                    max(0.0, max(trajectory.daily_bilateral_turnovers) - _TURNOVER_LIMIT),
                    max(0.0, max(trajectory.maximum_position_weights) - _POSITION_LIMIT),
                )
            )

            for step_index, (equity_before, equity_after, transaction_cost) in enumerate(
                zip(
                    trajectory.equity_curve[:-1],
                    trajectory.equity_curve[1:],
                    trajectory.transaction_costs,
                    strict=True,
                )
            ):
                gross_return = (equity_after + transaction_cost) / equity_before - 1.0
                cost_fraction = transaction_cost / equity_before
                step_core = gross_return - benchmark_step_returns[step_index] - cost_fraction
                if not math.isfinite(step_core):
                    raise ValueError("A2 step core calibration produced a non-finite value")
                absolute_step_cores.append(abs(step_core))
            drawdown_exposures.extend(trajectory.drawdowns)
            turnover_exposures.extend(trajectory.daily_bilateral_turnovers)
            maximum_daily_turnover = max(
                maximum_daily_turnover,
                max(trajectory.daily_bilateral_turnovers),
            )

    finance_core_mean = _mean(absolute_finance_cores, name="A_fin core")
    risk_mean = _mean(risk_penalties, name="A3 risk penalty")
    step_core_mean = _mean(absolute_step_cores, name="A2 step core")
    drawdown_mean = _mean(drawdown_exposures, name="A2 drawdown exposure")
    turnover_mean = _mean(turnover_exposures, name="A2 turnover exposure")
    statistics = CalibrationStatistics(
        train_session_count=len(market.sessions),
        epsilon_return_count=len(market.sessions) - 1,
        epsilon_rolling_window_count=len(market.sessions) - _ROLLING_RETURNS,
        calibration_window_count=calibration_window_count,
        trajectory_count=len(absolute_finance_cores),
        step_count=len(absolute_step_cores),
        median_20_return_volatility=epsilon / _EPSILON_FRACTION,
        mean_absolute_finance_core=finance_core_mean,
        mean_risk_penalty=risk_mean,
        mean_absolute_step_core=step_core_mean,
        mean_drawdown=drawdown_mean,
        mean_daily_bilateral_turnover=turnover_mean,
        maximum_daily_bilateral_turnover=maximum_daily_turnover,
    )
    return RewardCalibration(
        epsilon=epsilon,
        risk_lambda=_calibration_coefficient(
            core_mean=finance_core_mean,
            exposure_mean=risk_mean,
            name="risk_lambda",
        ),
        drawdown_coefficient=_calibration_coefficient(
            core_mean=step_core_mean,
            exposure_mean=drawdown_mean,
            name="drawdown_coefficient",
        ),
        turnover_coefficient=_calibration_coefficient(
            core_mean=step_core_mean,
            exposure_mean=turnover_mean,
            name="turnover_coefficient",
        ),
        turnover_limit=_TURNOVER_LIMIT,
        position_limit=_POSITION_LIMIT,
        statistics=statistics,
    )

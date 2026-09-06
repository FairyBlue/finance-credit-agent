"""Versioned benchmark-scaled scorer and screen-train-only CPU calibration.

This is a separate prospective protocol. The signed v1 calibration stays intact.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from src.env import ExecutionPolicy

from .calibration import CalibrationMarketData, _hold_targets, _replay, _rotation_targets
from .finance import FinanceAdvantage, RiskComponents, finance_advantage

SCORER_ID = "benchmark_scale_v2"
CALIBRATION_EPISODES = 32
PROXIES_PER_EPISODE = 44


def score_benchmark_scaled(
    *,
    episode_return: float,
    benchmark_return: float,
    benchmark_volatility: float,
    volatility_floor: float,
    risk_lambda: float,
    risk: RiskComponents,
) -> FinanceAdvantage:
    """Compute ``(R-B)/max(sigma_benchmark, floor) - lambda*Risk`` without clipping."""

    values = tuple(
        float(value)
        for value in (
            episode_return,
            benchmark_return,
            benchmark_volatility,
            volatility_floor,
            risk_lambda,
        )
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("screening scorer inputs must be finite")
    result_return, benchmark, volatility, floor, coefficient = values
    if volatility < 0.0 or coefficient < 0.0:
        raise ValueError("benchmark volatility and risk_lambda must be non-negative")
    if floor <= 0.0:
        raise ValueError("volatility_floor must be strictly positive")
    excess = result_return - benchmark
    core = excess / max(volatility, floor)
    result = core - coefficient * risk.penalty
    if not all(math.isfinite(value) for value in (excess, core, result)):
        raise ValueError("screening scorer produced a non-finite result")
    return FinanceAdvantage(result, excess, core, risk.penalty)


def canonical_sha256(document: Any) -> str:
    return sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def screening_payload_sha256(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload["hashes"] = dict(payload.get("hashes", {}))
    payload["hashes"].pop("payload_sha256", None)
    return canonical_sha256(payload)


def verify_screening_calibration(document: Mapping[str, Any]) -> None:
    """Reject altered payloads and unsupported scorer/calibration contracts."""

    if document.get("schema_version") != 1 or document.get("scorer_id") != SCORER_ID:
        raise ValueError("unsupported screening calibration/scorer version")
    hashes = document.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("screening calibration hashes are required")
    if hashes.get("payload_sha256") != screening_payload_sha256(document):
        raise ValueError("screening calibration payload_sha256 mismatch")
    reward = document.get("reward")
    if not isinstance(reward, Mapping):
        raise ValueError("screening calibration reward is required")
    for name in ("volatility_floor", "risk_lambda"):
        value = float(reward.get(name, math.nan))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"screening calibration {name} must be finite and positive")
    if reward.get("turnover_limit") != 1.0 or reward.get("position_limit") != 0.30:
        raise ValueError("screening risk limits must retain their frozen definitions")
    statistics = document.get("statistics", {})
    if (
        statistics.get("calibration_window_count") != CALIBRATION_EPISODES
        or statistics.get("trajectory_count") != CALIBRATION_EPISODES * PROXIES_PER_EPISODE
    ):
        raise ValueError("screening calibration must contain exactly 32 x 44 proxies")
    ids = document.get("source", {}).get("calibration_episode_ids", [])
    if len(ids) != CALIBRATION_EPISODES or len(set(ids)) != CALIBRATION_EPISODES:
        raise ValueError("screening calibration requires 32 distinct source episode IDs")


def load_screening_calibration(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("screening calibration must be a JSON mapping")
    verify_screening_calibration(document)
    return document


def _summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("screening diagnostics require finite nonempty evidence")
    return {
        "mean": float(np.mean(array)),
        **{
            name: float(np.quantile(array, quantile))
            for name, quantile in (
                ("min", 0.0),
                ("p01", 0.01),
                ("p10", 0.1),
                ("median", 0.5),
                ("p90", 0.9),
                ("p99", 0.99),
                ("max", 1.0),
            )
        },
    }


def screening_proxy_targets(
    symbols: Sequence[str],
) -> tuple[tuple[str, tuple[dict[str, float], ...]], ...]:
    """The 40 historical proxies plus cash and 10/30/100% initial equal weights."""

    if len(symbols) != 20 or len(set(symbols)) != 20:
        raise ValueError("screening proxies require the ordered universe of 20 symbols")
    proxies = [(f"single_stock:{symbol}", _hold_targets((symbol,))) for symbol in symbols]
    proxies.extend(
        (
            f"rotation:{symbol}:{symbols[(index + 1) % 20]}",
            _rotation_targets(symbol, symbols[(index + 1) % 20]),
        )
        for index, symbol in enumerate(symbols)
    )
    for name, exposure in (
        ("cash", 0.0),
        ("equal_weight_10", 0.1),
        ("equal_weight_30", 0.3),
        ("equal_weight_100", 1.0),
    ):
        initial = {} if exposure == 0.0 else {symbol: exposure / 20 for symbol in symbols}
        proxies.append((name, (initial, *({} for _ in range(19)))))
    return tuple(proxies)


def calibrate_screening_reward(
    *,
    markets: Mapping[str, CalibrationMarketData],
    expected_benchmarks: Mapping[str, Sequence[float]],
    initial_cash: float,
    execution_policy: ExecutionPolicy,
    source: Mapping[str, Any],
    legacy_reward: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Fit only on the supplied 32 selected Train windows, each with 44 proxies.

    ``legacy_reward`` is consumed after fitting solely for historical diagnostics.
    No fitted quantity uses those prior full-Train parameters.
    """

    if len(markets) != CALIBRATION_EPISODES or set(expected_benchmarks) != set(markets):
        raise ValueError("screening calibration requires 32 aligned market/benchmark episodes")
    if not math.isfinite(initial_cash) or initial_cash <= 0.0:
        raise ValueError("initial_cash must be finite and positive")
    records: list[dict[str, Any]] = []
    benchmark_volatilities: list[float] = []
    for episode_id, market in markets.items():
        if len(market.sessions) != 21:
            raise ValueError("each screening calibration market must contain exactly 21 sessions")
        targets = screening_proxy_targets(market.symbols)
        benchmark = _replay(
            market=market,
            start=0,
            targets=targets[-1][1],
            initial_cash=initial_cash,
            execution_policy=execution_policy,
        )
        expected = np.asarray(expected_benchmarks[episode_id], dtype=np.float64)
        if (
            expected.shape != (21,)
            or not np.all(np.isfinite(expected))
            or not np.all(expected > 0.0)
            or not np.allclose(benchmark.equity_curve, expected, rtol=1e-9, atol=0.0)
        ):
            raise ValueError(
                f"reviewed benchmark does not match shared-ledger replay: {episode_id}"
            )
        benchmark_returns = np.diff(expected) / expected[:-1]
        benchmark_volatility = float(np.std(benchmark_returns, ddof=0))
        benchmark_volatilities.append(benchmark_volatility)
        benchmark_return = float(expected[-1] / expected[0] - 1.0)
        for proxy_id, target_rows in targets:
            trajectory = (
                benchmark
                if proxy_id == "equal_weight_100"
                else _replay(
                    market=market,
                    start=0,
                    targets=target_rows,
                    initial_cash=initial_cash,
                    execution_policy=execution_policy,
                )
            )
            risk = RiskComponents(
                max_drawdown=max(trajectory.drawdowns),
                turnover_excess=max(0.0, max(trajectory.daily_bilateral_turnovers) - 1.0),
                position_excess=max(0.0, max(trajectory.maximum_position_weights) - 0.30),
            )
            equities = np.asarray(trajectory.equity_curve, dtype=np.float64)
            records.append(
                {
                    "episode_id": episode_id,
                    "proxy_id": proxy_id,
                    "proxy_type": proxy_id.split(":")[0],
                    "episode_return": float(equities[-1] / equities[0] - 1.0),
                    "benchmark_return": benchmark_return,
                    "benchmark_volatility": benchmark_volatility,
                    "agent_volatility": float(np.std(np.diff(equities) / equities[:-1], ddof=0)),
                    "risk_penalty": risk.penalty,
                    "max_drawdown": risk.max_drawdown,
                    "turnover_excess": risk.turnover_excess,
                    "position_excess": risk.position_excess,
                }
            )
    floor = 0.1 * float(np.median(benchmark_volatilities))
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("screening calibration volatility floor must be finite and positive")
    cores = [
        (record["episode_return"] - record["benchmark_return"])
        / max(record["benchmark_volatility"], floor)
        for record in records
    ]
    core_mean = math.fsum(abs(core) for core in cores) / len(cores)
    risk_mean = math.fsum(record["risk_penalty"] for record in records) / len(records)
    if not math.isfinite(risk_mean) or risk_mean <= 0.0:
        raise ValueError("screening calibration meanRisk must be finite and positive")
    coefficient = 0.25 * core_mean / risk_mean
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("screening calibration risk_lambda must be finite and positive")
    for record, core in zip(records, cores, strict=True):
        record["core"] = core
        record["Y"] = core - coefficient * record["risk_penalty"]
        if record["proxy_type"] == "equal_weight_100" and abs(core) > 1e-9:
            raise ValueError("100% equal-weight proxy must reproduce the benchmark core")
        if legacy_reward is not None:
            record["legacy_Y"] = finance_advantage(
                episode_return=record["episode_return"],
                benchmark_return=record["benchmark_return"],
                volatility=record["agent_volatility"],
                epsilon=legacy_reward["epsilon"],
                risk_lambda=legacy_reward["risk_lambda"],
                risk=RiskComponents(
                    record["max_drawdown"], record["turnover_excess"], record["position_excess"]
                ),
            ).advantage
    fields: tuple[str, ...] = ("core", "Y", "risk_penalty", "agent_volatility")
    if legacy_reward is not None:
        fields += ("legacy_Y",)
    diagnostics = {
        proxy_type: {
            field: _summarize([row[field] for row in records if row["proxy_type"] == proxy_type])
            for field in fields
        }
        for proxy_type in sorted({row["proxy_type"] for row in records})
    }
    document: dict[str, Any] = {
        "schema_version": 1,
        "scorer_id": SCORER_ID,
        "status": "computed_for_authorized_screening",
        "source": {**source, "calibration_episode_ids": list(markets)},
        "protocol": {
            "formula": "(R_net-B)/max(sigma_benchmark,volatility_floor)-risk_lambda*RiskPenalty",
            "clip": None,
            "benchmark_standard_deviation_ddof": 0,
            "benchmark_annualized": False,
            "floor_median_multiplier": 0.1,
            "risk_mean_absolute_core_fraction": 0.25,
            "proxy_weighting": "equal_episode_and_proxy",
            "proxies_per_episode": PROXIES_PER_EPISODE,
            "proxy_definitions": [
                name for name, _ in screening_proxy_targets(next(iter(markets.values())).symbols)
            ],
            "execution": {
                "initial_cash": initial_cash,
                "commission_bps": execution_policy.commission_bps,
                "slippage_bps": execution_policy.slippage_bps,
                "fill": "next_open",
                "turnover": "bilateral_daily",
            },
            "legacy_parameters_used_for_fitting": False,
        },
        "reward": {
            "volatility_floor": floor,
            "risk_lambda": coefficient,
            "turnover_limit": 1.0,
            "position_limit": 0.30,
        },
        "statistics": {
            "calibration_window_count": len(markets),
            "trajectory_count": len(records),
            "step_count": len(records) * 20,
            "mean_absolute_finance_core": core_mean,
            "mean_risk_penalty": risk_mean,
            "benchmark_volatility": _summarize(benchmark_volatilities),
            "benchmark_floor_hit_count": sum(value < floor for value in benchmark_volatilities),
        },
        "diagnostics_by_proxy_type": diagnostics,
        "proxy_evidence": records,
        "hashes": {},
    }
    document["hashes"]["payload_sha256"] = screening_payload_sha256(document)
    verify_screening_calibration(document)
    return document

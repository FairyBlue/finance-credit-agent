"""Train-only, four-decision portfolio residual demo on the existing ledger.

This is a new task profile, not a replacement for the single-order TradingEnv.
Only ``visible()`` is an agent observation; MarketEpisode also holds future
prices needed by execution and must never be serialized into an agent prompt.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.env import ExecutionPolicy, PortfolioLedger

ROOT = Path(__file__).resolve().parents[2]
PROFILE = "portfolio_management_residual_v1"
ACTIONS = ("hold", "momentum", "reversal", "defensive")
DECISION_SESSIONS = (1, 6, 11, 16)
MAX_ACTIVE_L1 = 0.2
MAX_TARGET_WEIGHT = 0.1
LOOKBACK_RETURNS = 20
TRAIN_ADAPTER_CACHE = ROOT / "runs/m2-same-prefix-online-demo-20261024-v2/shared_adapter.pkl"


@dataclass(frozen=True)
class MarketEpisode:
    """Execution data, explicitly separate from the PIT agent observation."""

    episode_id: str
    kwargs: dict[str, Any]
    opens: list[dict[str, float]]
    closes: list[dict[str, float]]
    visible_histories: list[dict[str, list[float]]]


def load_train_markets(prepared_path, output_dir) -> dict[str, MarketEpisode]:
    """Reuse the existing Train-only adapter and prepared market windows.

    No Valid/OOS inputs, calibration fit, repeated market checksums, or model
    inference are performed here. Histories use the adapter's availability gate.
    """
    import pyarrow.parquet as pq

    from src.agents.same_prefix_online_collect import (
        _adapter_configuration,
        _install_adapter,
    )

    path, output = Path(prepared_path).resolve(), Path(output_dir).resolve()
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError("nonempty prepared Train episodes required")
    prepared = {}
    for row in rows:
        eid = row["extra_info"]["episode_id"]
        if not eid.startswith("formal-train-") or row["extra_info"]["source_split"] != "train":
            raise ValueError("portfolio management step1 is restricted to Train")
        kwargs = json.loads(row["env_kwargs"])
        dates = [str(day)[:10] for day in kwargs["episode_dates"]]
        if (
            len(dates) != 21
            or len(kwargs["episode_open_times"]) != 21
            or len(kwargs["symbols"]) != 20
            or len(set(kwargs["symbols"])) != 20
            or not all("2023-01-03" <= day <= "2024-10-31" for day in dates)
            or kwargs["data_end"] != "2024-10-31"
            or kwargs["input_sessions_per_symbol"] != 461
        ):
            raise ValueError("existing 20-stock, 21-session Train-only identity required")
        if eid in prepared:
            raise ValueError("duplicate prepared episode identity")
        prepared[eid] = {"kwargs": kwargs, "prompt": row.get("prompt", [])}
    expected = _adapter_configuration(prepared)
    if any(_adapter_configuration({eid: data}) != expected for eid, data in prepared.items()):
        raise ValueError("all episodes must share the same Train-only market identity")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "third_party/agent-r1"))
    # An explicitly Train-only cache: never inherit the previous Valid worker's
    # environment variable or load its broader history object.
    os.environ["FCA_ONLINE_ADAPTER_CACHE"] = str(TRAIN_ADAPTER_CACHE)
    _install_adapter(prepared, output)
    from src.env import trading

    adapter = trading._cached_storage_adapter(**expected)
    symbols = tuple(next(iter(prepared.values()))["kwargs"]["symbols"])
    cache: dict[str, tuple[dict, dict, dict]] = {}
    markets = {}
    for eid, data in prepared.items():
        opens, closes, histories = [], [], []
        for day in data["kwargs"]["episode_dates"]:
            if day not in cache:
                as_of = datetime.fromisoformat(day)
                visible = adapter.get_observations(as_of)
                history: dict[str, list[float]] = {s: [] for s in symbols}
                current = {}
                for observation in visible:
                    if observation.kind != "market" or observation.symbol not in history:
                        continue
                    payload = observation.payload
                    close = float(payload["close"])
                    if not math.isfinite(close) or close <= 0:
                        raise ValueError("positive finite visible closes required")
                    history[observation.symbol].append(close)
                    if payload.get("date") == day[:10]:
                        if observation.symbol in current:
                            raise ValueError("duplicate current-session market bar")
                        current[observation.symbol] = payload
                if set(current) != set(symbols):
                    raise ValueError("complete current-session 20-stock market required")
                opening = {s: float(current[s]["open"]) for s in symbols}
                closing = {s: float(current[s]["close"]) for s in symbols}
                if any(not math.isfinite(v) or v <= 0 for v in opening.values()):
                    raise ValueError("positive finite opens required")
                cache[day] = (
                    opening,
                    closing,
                    {s: history[s][-(LOOKBACK_RETURNS + 1) :] for s in symbols},
                )
            opening, closing, history = cache[day]
            opens.append(opening)
            closes.append(closing)
            histories.append(history)
        markets[eid] = MarketEpisode(eid, data["kwargs"], opens, closes, histories)
    return markets


class PortfolioManagement:
    """Four macro actions, paid next-open execution, and natural share drift."""

    def __init__(self, market: MarketEpisode, *, initialize_equal_weight: bool = True):
        self.market = market
        self.episode_id = market.episode_id
        self.symbols = tuple(market.kwargs["symbols"])
        self._dates = [datetime.fromisoformat(v) for v in market.kwargs["episode_dates"]]
        self._open_times = [datetime.fromisoformat(v) for v in market.kwargs["episode_open_times"]]
        if len(self.symbols) != 20 or len(self._dates) != 21:
            raise ValueError("the new profile requires exactly 20 stocks and 21 sessions")
        if any(not "2023-01-03" <= d.date().isoformat() <= "2024-10-31" for d in self._dates):
            raise ValueError("portfolio management step1 only permits historical Train")
        self.ledger = PortfolioLedger(
            initial_cash=float(market.kwargs["initial_cash"]),
            policy=ExecutionPolicy(
                float(market.kwargs["commission_bps"]),
                float(market.kwargs["slippage_bps"]),
            ),
        )
        self.session_index = 0
        self.decisions: list[dict[str, Any]] = []
        self.daily_states: list[dict[str, Any]] = []
        self._peak = self.ledger.initial_cash
        self._record_day((), previous_equity=self.ledger.initial_cash)
        self.initialize_equal_weight = initialize_equal_weight
        if initialize_equal_weight:
            # Frozen benchmark order is intentional: match the existing reviewed
            # sequential target ledger, including its small fee-induced drift.
            for symbol in self.symbols:
                self.ledger.queue_target_order(
                    symbol, target_weight=1 / len(self.symbols), decision_at=self._dates[0]
                )
        self._advance_one()

    @property
    def done(self) -> bool:
        return self.session_index == 20

    def _weights(self) -> dict[str, float]:
        closes = self.market.closes[self.session_index]
        equity = self.ledger.equity(closes)
        return {s: self.ledger.positions.get(s, 0.0) * closes[s] / equity for s in self.symbols}

    def _history_features(self) -> dict[str, dict[str, Any]]:
        features = {}
        for symbol in self.symbols:
            history = self.market.visible_histories[self.session_index][symbol]
            returns = np.diff(history) / np.asarray(history[:-1])
            features[symbol] = {
                "close_records_used": len(history),
                "lookback_used_returns": len(returns),
                "past_return": float(history[-1] / history[0] - 1) if len(history) >= 2 else None,
                "daily_volatility": float(returns.std(ddof=0)) if len(returns) else None,
            }
        return features

    def _action(self, action_id: str) -> dict[str, Any]:
        if action_id not in ACTIONS:
            raise ValueError(f"action_id must be one of {ACTIONS}")
        current = self._weights()
        exposure = math.fsum(current.values())
        if action_id == "hold":
            return {
                "action_id": action_id,
                "target_weights": None,
                "requested_stock_exposure": exposure,
                "active_weight_l1": 0.0,
                "orders": 0,
                "definition": "No orders; preserve actual shares and cash, allow market drift.",
            }
        base = {s: min(MAX_TARGET_WEIGHT, current[s]) for s in self.symbols}
        mandatory_clip = math.fsum(current[s] - base[s] for s in self.symbols)
        if mandatory_clip > MAX_ACTIVE_L1 + 1e-10:
            raise ValueError("current cap breach cannot be corrected inside the active L1 budget")
        if action_id == "defensive":
            requested_exposure = max(0.0, exposure - MAX_ACTIVE_L1)
            base_exposure = math.fsum(base.values())
            scale = requested_exposure / base_exposure if base_exposure else 0.0
            targets = {s: base[s] * scale for s in self.symbols}
            definition = "Reduce current total stock exposure by 20 percentage points, floor zero."
            desired = None
        else:
            features = self._history_features()
            order = sorted(
                self.symbols,
                key=lambda s: (
                    features[s]["past_return"] if features[s]["past_return"] is not None else 0,
                    self.symbols.index(s),
                ),
            )
            # Stable universe order breaks ties, without filling missing history
            # using any later bar. Reversal uses the exact opposite rank tilt.
            desired = {s: 0.05 for s in self.symbols}
            lower, upper = (0.03, 0.07) if action_id == "momentum" else (0.07, 0.03)
            for symbol in order[:5]:
                desired[symbol] = lower
            for symbol in order[-5:]:
                desired[symbol] = upper
            distance = math.fsum(abs(desired[s] - base[s]) for s in self.symbols)
            budget = max(0.0, MAX_ACTIVE_L1 - mandatory_clip)
            scale = min(1.0, budget / distance) if distance else 1.0
            targets = {s: base[s] + scale * (desired[s] - base[s]) for s in self.symbols}
            definition = (
                "Move toward a full-exposure 3%/5%/7% rank template, inside 20pp active L1; "
                "can gradually restore stock exposure after defensive."
            )
        active_l1 = math.fsum(abs(targets[s] - current[s]) for s in self.symbols)
        if (
            any(
                not math.isfinite(v) or not -1e-12 <= v <= MAX_TARGET_WEIGHT + 1e-12
                for v in targets.values()
            )
            or math.fsum(targets.values()) > 1 + 1e-10
            or active_l1 > MAX_ACTIVE_L1 + 1e-10
        ):
            raise ValueError("invalid constrained portfolio action")
        return {
            "action_id": action_id,
            "target_weights": targets,
            "requested_stock_exposure": math.fsum(targets.values()),
            "active_weight_l1": active_l1,
            "definition": definition,
            "desired_template": desired,
            "mandatory_cap_correction_l1": mandatory_clip,
        }

    def visible(self) -> dict[str, Any]:
        """Only current/past state. No future bar, benchmark path, or outcome Y."""
        weights = self._weights()
        return {
            "profile": PROFILE,
            "episode_id": self.episode_id,
            "source_split": "train",
            "current_date": self._dates[self.session_index].isoformat(),
            "session_index": self.session_index,
            "done": self.done,
            "decision_sessions": list(DECISION_SESSIONS),
            "cash": self.ledger.cash,
            "portfolio_equity": self.daily_states[-1]["equity"],
            "stock_exposure": math.fsum(weights.values()),
            "weights": weights,
            "positions": {s: self.ledger.positions.get(s, 0.0) for s in self.symbols},
            "market_closes": dict(self.market.closes[self.session_index]),
            "past_returns": self._history_features(),
            "history": {
                "requested_return_lookback": LOOKBACK_RETURNS,
                "definition": "Up to 21 already-visible closes, yielding up to 20 daily returns.",
                "as_of": self._dates[self.session_index].isoformat(),
                "insufficient_history": "Use available past closes; missing return ranks as zero.",
                "ranking_ties": "Fixed symbols universe order.",
            },
            "actions": [] if self.done else [self._action(a) for a in ACTIONS],
            "constraints": {
                "long_only": True,
                "leverage": False,
                "max_requested_symbol_weight": MAX_TARGET_WEIGHT,
                "max_requested_stock_exposure": 1.0,
                "max_active_weight_l1": MAX_ACTIVE_L1,
                "l1_definition": "Absolute stock target changes, buys plus sells; excludes cash.",
                "hold_preserves_drift": True,
                "execution": "Next open; sells before buys, fixed symbol order within each side.",
                "commission_bps": self.ledger.policy.commission_bps,
                "slippage_bps": self.ledger.policy.slippage_bps,
                "target_vs_realized": "Gaps, sequential fees and cash clipping affect fills.",
            },
        }

    def step(self, action_id: str) -> dict[str, Any]:
        if self.done or self.session_index not in DECISION_SESSIONS:
            raise ValueError("step requires one of the four unfinished decision sessions")
        action = self._action(action_id)
        start = self.session_index
        trace = {
            **action,
            "decision_session": start,
            "decision_at": self._dates[start].isoformat(),
            "weights_before": self._weights(),
            "fills": [],
        }
        if action["target_weights"] is not None:
            # Decide targets now. The execution engine orders those fixed targets
            # using only the following open when that open actually arrives.
            self._queue_at_next_open(action["target_weights"])
        end = next((j for j in DECISION_SESSIONS if j > start), 20)
        first_fill = len(self.ledger.fills)
        while self.session_index < end:
            self._advance_one()
        trace["fills"] = [
            {
                **asdict(f),
                "decision_at": f.decision_at.isoformat(),
                "executed_at": f.executed_at.isoformat(),
            }
            for f in self.ledger.fills[first_fill:]
        ]
        trace.update(
            next_session=self.session_index,
            equity_after=self.daily_states[-1]["equity"],
            stock_exposure_after=math.fsum(self._weights().values()),
        )
        self.decisions.append(trace)
        return trace

    def _queue_at_next_open(self, targets: dict[str, float]) -> None:
        opens = self.market.opens[self.session_index + 1]
        equity_at_open = self.ledger.equity(opens)
        order = sorted(
            self.symbols,
            key=lambda s: (
                targets[s] >= self.ledger.positions.get(s, 0.0) * opens[s] / equity_at_open,
                self.symbols.index(s),
            ),
        )
        for symbol in order:
            # Generic target orders preserve the original ledger's cash/fee and
            # overnight gap semantics; no artificial financing or cost override.
            self.ledger.queue_target_order(
                symbol,
                target_weight=targets[symbol],
                decision_at=self._dates[self.session_index],
                direction=None,
            )

    def _advance_one(self) -> None:
        previous_equity = self.daily_states[-1]["equity"]
        self.session_index += 1
        fills = self.ledger.execute_next_open(
            self._open_times[self.session_index], self.market.opens[self.session_index]
        )
        self._record_day(fills, previous_equity=previous_equity)

    def _record_day(self, fills, *, previous_equity: float) -> None:
        equity = self.ledger.equity(self.market.closes[self.session_index])
        self._peak = max(self._peak, equity)
        weights = self._weights()
        self.daily_states.append(
            {
                "session_index": self.session_index,
                "date": self._dates[self.session_index].isoformat(),
                "equity": equity,
                "cash": self.ledger.cash,
                "stock_exposure": math.fsum(weights.values()),
                "maximum_position_weight": max(weights.values(), default=0.0),
                "weights": weights,
                "turnover": math.fsum(fill.notional for fill in fills) / previous_equity,
                "drawdown": max(0.0, (self._peak - equity) / self._peak),
                "transaction_cost": math.fsum(fill.total_cost for fill in fills),
                "fill_count": len(fills),
            }
        )

    def result(self) -> dict[str, Any]:
        if not self.done:
            raise ValueError("result requires all four decisions to reach original session20")
        from src.rewards import RiskComponents
        from src.rewards.screening import score_benchmark_scaled

        k = self.market.kwargs
        curve = [row["equity"] for row in self.daily_states]
        benchmark = np.asarray(k["benchmark_closes"], dtype=float)
        net_return = curve[-1] / curve[0] - 1
        benchmark_return = float(benchmark[-1] / benchmark[0] - 1)
        benchmark_volatility = float(np.std(np.diff(benchmark) / benchmark[:-1], ddof=0))
        max_drawdown = max(row["drawdown"] for row in self.daily_states)
        max_turnover = max(row["turnover"] for row in self.daily_states)
        max_position = max(row["maximum_position_weight"] for row in self.daily_states)
        risk = RiskComponents(
            max_drawdown=max_drawdown,
            turnover_excess=max(0.0, max_turnover - k["turnover_limit"]),
            position_excess=max(0.0, max_position - k["position_limit"]),
        )
        score = score_benchmark_scaled(
            episode_return=net_return,
            benchmark_return=benchmark_return,
            benchmark_volatility=benchmark_volatility,
            volatility_floor=k["finance_volatility_floor"],
            risk_lambda=k["risk_lambda"],
            risk=risk,
        )
        return {
            "profile": PROFILE,
            "episode_id": self.episode_id,
            "status": "COMPLETE",
            "source_split": "train",
            "formal_valid_access": False,
            "formal_final_oos_access": False,
            "metrics": {
                "raw_net_return": net_return,
                "benchmark_return": benchmark_return,
                "benchmark_excess": net_return - benchmark_return,
                "terminal_financial_Y": score.advantage,
                "financial_Y_diagnostic_only": True,
                "scorer_id": k["finance_scorer_id"],
                "transaction_cost": math.fsum(f.total_cost for f in self.ledger.fills),
                "commission": math.fsum(f.commission for f in self.ledger.fills),
                "slippage_cost": math.fsum(f.slippage_cost for f in self.ledger.fills),
                "fill_count": len(self.ledger.fills),
                "max_drawdown": max_drawdown,
                "turnover": math.fsum(row["turnover"] for row in self.daily_states),
                "max_daily_turnover": max_turnover,
                "maximum_position_weight": max_position,
                "risk": asdict(risk),
                "terminal_session_index": self.session_index,
                "decision_count": len(self.decisions),
                "initial_equal_weight_establishment": self.initialize_equal_weight,
                "initial_establishment_cost": self.daily_states[1]["transaction_cost"],
            },
            "decisions": self.decisions,
            "daily_states": self.daily_states,
            "equity_curve": curve,
            "action_spec": {
                "action_ids": list(ACTIONS),
                "decision_sessions": list(DECISION_SESSIONS),
                "rank_template": {"bottom5": 0.03, "middle10": 0.05, "top5": 0.07},
                "max_active_weight_l1": MAX_ACTIVE_L1,
                "max_requested_symbol_weight": MAX_TARGET_WEIGHT,
                "momentum_reversal_can_restore_exposure": True,
                "defensive_exposure_reduction_pp": 20,
                "hold_is_natural_share_drift": True,
                "requested_constraints_do_not_bound_overnight_realized_drift": True,
            },
        }

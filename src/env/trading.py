"""Point-in-time trading environment implemented as an Agent-R1 ToolEnv plugin."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date, datetime, time
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from agent_r1.env import AgentEnv, Observation, ToolEnv
from agent_r1.env.base import Action
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.data import (
    AvailabilityPolicy,
    MarketObservation,
    ObservationSnapshot,
    StockBenchAdapter,
    validate_episode_session_boundaries,
    verify_raw_daily_manifest,
)
from src.rewards import RiskComponents, finance_advantage

from . import tools as _registered_tools  # noqa: F401  # registry side effects
from .accounting import ExecutionPolicy, PortfolioLedger, _aware_utc


class _StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TradingDataIntegrityError(RuntimeError):
    """A required bar is absent or ambiguous for the exact replay session."""


class RetrieveArgs(_StrictArgs):
    query: str = Field(min_length=1)
    symbol: str | None = None
    limit: int = Field(default=10, ge=1, le=10)


class ComputeArgs(_StrictArgs):
    metric: Literal["price_change", "mean_return", "volatility"]
    symbol: str = Field(min_length=1)
    lookback: int = Field(ge=2, le=60)


class TargetArgs(_StrictArgs):
    symbol: str = Field(min_length=1)
    target_weight: float = Field(ge=0.0, le=1.0)


class HoldArgs(_StrictArgs):
    pass


class AbstainArgs(_StrictArgs):
    reason: str | None = Field(default=None, max_length=500)


_ARG_MODELS: dict[str, type[_StrictArgs]] = {
    "retrieve": RetrieveArgs,
    "compute": ComputeArgs,
    "buy": TargetArgs,
    "sell": TargetArgs,
    "hold": HoldArgs,
    "abstain": AbstainArgs,
}

_MAX_STATE_SIGNALS = 10
_MAX_PAYLOAD_KEYS = 12
_MAX_SEQUENCE_ITEMS = 20
_MAX_TEXT_CHARS = 256


def _compact_json(value: object) -> object:
    """Return a deterministic, size-bounded JSON value for an LLM observation."""

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, str):
        return value[:_MAX_TEXT_CHARS]
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: str(item[0]))[:_MAX_PAYLOAD_KEYS]
        return {str(key): _compact_json(item) for key, item in items}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_compact_json(item) for item in value[:_MAX_SEQUENCE_ITEMS]]
    return str(value)[:_MAX_TEXT_CHARS]


def _parse_datetime(value: datetime | str) -> datetime:
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(value.replace("Z", "+00:00"))
    )
    return _aware_utc(parsed, name="episode date")


@lru_cache(maxsize=8)
def _cached_storage_adapter(
    *,
    manifest_root: str,
    storage_root: str,
    symbols: tuple[str, ...],
    data_version: str,
    data_start: str,
    data_end: str,
    input_manifest_sha256: str,
    input_file_count: int,
    input_sessions_per_symbol: int,
    exchange_calendar: str,
    market_timezone: str,
    market_close_time: str,
) -> StockBenchAdapter:
    """Verify and share one immutable raw snapshot per rollout worker process."""

    start = date.fromisoformat(data_start)
    end = date.fromisoformat(data_end)
    manifest_root_path = Path(manifest_root).expanduser().resolve()
    audit = verify_raw_daily_manifest(
        manifest_root=manifest_root,
        storage_root=storage_root,
        symbols=symbols,
        start=start,
        end=end,
        expected_sha256=input_manifest_sha256,
        expected_file_count=input_file_count,
        expected_sessions_per_symbol=input_sessions_per_symbol,
        exchange_calendar=exchange_calendar,
    )
    expected_raw_sha256_by_path: dict[str, str] = {}
    for raw_line in audit.manifest_bytes.splitlines():
        relative_path, raw_sha256 = raw_line.decode("utf-8").split("\t", maxsplit=1)
        absolute_path = str((manifest_root_path / relative_path).resolve())
        expected_raw_sha256_by_path[absolute_path] = raw_sha256

    return StockBenchAdapter.from_storage(
        Path(storage_root).expanduser().resolve(),
        symbols=symbols,
        data_version=data_version,
        availability_policy=AvailabilityPolicy(
            market_timezone,
            time.fromisoformat(market_close_time),
            exchange_calendar,
        ),
        expected_raw_sha256_by_path=expected_raw_sha256_by_path,
        start_date=start,
        end_date=end,
    )


@AgentEnv.register("trading")
class TradingEnv(ToolEnv):
    """Long-only daily replay with retrieve/compute/buy/sell/hold/abstain tools.

    Unlike upstream `ToolEnv.step`, this override validates arguments and executes
    exactly one tool serially.  Malformed actions become error observations, so an
    LLM mistake cannot crash a rollout or concurrently mutate the ledger.
    """

    TOOL_NAMES = ("retrieve", "compute", "buy", "sell", "hold", "abstain")

    def __init__(
        self,
        *,
        episode_dates: Sequence[datetime | str],
        episode_open_times: Sequence[datetime | str],
        symbols: Sequence[str],
        initial_cash: float,
        commission_bps: float,
        slippage_bps: float,
        drawdown_coefficient: float,
        turnover_coefficient: float,
        benchmark_closes: Sequence[float],
        reward_mode: str = "generic_step",
        finance_epsilon: float | None = None,
        risk_lambda: float | None = None,
        turnover_limit: float | None = None,
        position_limit: float | None = None,
        adapter: StockBenchAdapter | None = None,
        manifest_root: str | Path | None = None,
        storage_root: str | Path | None = None,
        data_version: str | None = None,
        data_start: date | str | None = None,
        data_end: date | str | None = None,
        input_manifest_sha256: str | None = None,
        input_file_count: int | None = None,
        input_sessions_per_symbol: int | None = None,
        exchange_calendar: str = "XNYS",
        market_timezone: str | None = None,
        market_close_time: str | None = None,
        tool_format: str = "hermes",
        **kwargs: Any,
    ) -> None:
        super().__init__(tools=list(self.TOOL_NAMES), tool_format=tool_format)
        self._dates = tuple(_parse_datetime(value) for value in episode_dates)
        if len(self._dates) < 2:
            raise ValueError("an episode needs at least two strictly ordered market dates")
        if any(
            current >= following
            for current, following in zip(self._dates, self._dates[1:], strict=False)
        ):
            raise ValueError("episode_dates must be strictly increasing")
        self._open_times = tuple(_parse_datetime(value) for value in episode_open_times)
        if len(self._open_times) != len(self._dates):
            raise ValueError("episode_open_times must align one-to-one with episode_dates")
        if any(
            open_time >= close_time
            for open_time, close_time in zip(self._open_times, self._dates, strict=True)
        ):
            raise ValueError("every session open must be earlier than its after-close observation")
        if any(
            previous_close >= next_open
            for previous_close, next_open in zip(
                self._dates[:-1], self._open_times[1:], strict=True
            )
        ):
            raise ValueError("each next open must occur strictly after the previous decision close")
        validate_episode_session_boundaries(
            session_closes=self._dates,
            session_opens=self._open_times,
            exchange_calendar=exchange_calendar,
        )
        self._symbols = tuple(dict.fromkeys(symbol.strip().upper() for symbol in symbols))
        if not self._symbols or any(not symbol for symbol in self._symbols):
            raise ValueError("symbols must not be empty")

        if adapter is None:
            if not all(
                (
                    manifest_root,
                    storage_root,
                    data_version,
                    data_start,
                    data_end,
                    input_manifest_sha256,
                    input_file_count,
                    input_sessions_per_symbol,
                    market_timezone,
                    market_close_time,
                )
            ):
                raise ValueError(
                    "provide adapter, or manifest_root/storage_root/data_version/data_start/"
                    "data_end/input_manifest_sha256/input_file_count/"
                    "input_sessions_per_symbol/market_timezone/market_close_time"
                )
            assert manifest_root is not None
            assert data_start is not None
            assert data_end is not None
            assert input_manifest_sha256 is not None
            assert input_file_count is not None
            assert input_sessions_per_symbol is not None
            adapter = _cached_storage_adapter(
                manifest_root=str(Path(str(manifest_root)).expanduser().resolve()),
                storage_root=str(Path(str(storage_root)).expanduser().resolve()),
                symbols=self._symbols,
                data_version=str(data_version),
                data_start=(
                    data_start
                    if isinstance(data_start, str)
                    else data_start.isoformat()
                ),
                data_end=(data_end if isinstance(data_end, str) else data_end.isoformat()),
                input_manifest_sha256=input_manifest_sha256,
                input_file_count=input_file_count,
                input_sessions_per_symbol=input_sessions_per_symbol,
                exchange_calendar=exchange_calendar,
                market_timezone=str(market_timezone),
                market_close_time=str(market_close_time),
            )
        self._adapter = adapter
        self._initial_cash = float(initial_cash)
        self._execution_policy = ExecutionPolicy(float(commission_bps), float(slippage_bps))
        self._drawdown_coefficient = self._coefficient(
            drawdown_coefficient, name="drawdown_coefficient"
        )
        self._turnover_coefficient = self._coefficient(
            turnover_coefficient, name="turnover_coefficient"
        )
        self._benchmark_closes = tuple(float(value) for value in benchmark_closes)
        if len(self._benchmark_closes) != len(self._dates):
            raise ValueError("benchmark_closes must align one-to-one with episode_dates")
        if any(not math.isfinite(value) or value <= 0.0 for value in self._benchmark_closes):
            raise ValueError("benchmark closes must be finite and positive")
        if reward_mode not in {"outcome_only", "generic_step", "finance_trajectory"}:
            raise ValueError(
                "reward_mode must be outcome_only, generic_step, or finance_trajectory"
            )
        self._reward_mode = reward_mode
        finance_values = (finance_epsilon, risk_lambda, turnover_limit, position_limit)
        if reward_mode == "finance_trajectory" and any(value is None for value in finance_values):
            raise ValueError(
                "finance_trajectory requires finance_epsilon, risk_lambda, turnover_limit, "
                "and position_limit"
            )
        self._finance_epsilon = None if finance_epsilon is None else float(finance_epsilon)
        self._risk_lambda = None if risk_lambda is None else float(risk_lambda)
        self._turnover_limit = None if turnover_limit is None else float(turnover_limit)
        self._position_limit = None if position_limit is None else float(position_limit)
        if self._finance_epsilon is not None and self._finance_epsilon <= 0.0:
            raise ValueError("finance_epsilon must be positive")
        for name, value in (
            ("risk_lambda", self._risk_lambda),
            ("turnover_limit", self._turnover_limit),
            ("position_limit", self._position_limit),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        self._base_raw_prompt = list(kwargs.get("raw_prompt", []))
        self._episode_prompt: list[dict[str, Any]] = []
        self.ledger = PortfolioLedger(
            initial_cash=self._initial_cash,
            policy=self._execution_policy,
        )
        self._index = 0
        self._done = False
        self._high_water_mark = self._initial_cash
        self._invalid_calls = 0
        self._tool_calls = 0
        self._abstentions = 0
        self._equity_curve = [self._initial_cash]
        self._net_returns: list[float] = []
        self._turnovers: list[float] = []
        self._maximum_position_weight = 0.0

    @staticmethod
    def _coefficient(value: float, *, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return result

    @property
    def current_date(self) -> datetime:
        return self._dates[self._index]

    @property
    def symbols(self) -> tuple[str, ...]:
        """Return the immutable frozen universe used by prompt-budget preflight."""

        return self._symbols

    def reset(self, **kwargs: Any) -> Observation:
        self.ledger = PortfolioLedger(
            initial_cash=self._initial_cash,
            policy=self._execution_policy,
        )
        self._index = 0
        self._done = False
        self._high_water_mark = self._initial_cash
        self._invalid_calls = 0
        self._tool_calls = 0
        self._abstentions = 0
        self._equity_curve = [self._initial_cash]
        self._net_returns = []
        self._turnovers = []
        self._maximum_position_weight = 0.0
        raw_prompt = list(kwargs.get("raw_prompt", self._base_raw_prompt))
        if not raw_prompt:
            raw_prompt = [
                {
                    "role": "system",
                    "content": (
                        "Use exactly one available tool per turn. Trading decisions fill at the "
                        "next open; shorting and leverage are forbidden."
                    ),
                }
            ]
        self._episode_prompt = [dict(message) for message in raw_prompt]
        super().reset(raw_prompt=raw_prompt)
        return self._replace_visible_observation(self._state_text())

    async def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._done:
            return self._invalid("episode is already complete", terminal=True)
        if not isinstance(action, Action) or action.text is None:
            return self._invalid("expected Agent-R1 Action with text")

        try:
            _, calls = self.format_wrapper.parse_response(action.text)
            if len(calls) != 1:
                raise ValueError("exactly one tool call is required per environment step")
            call = calls[0]
            if call.name not in _ARG_MODELS:
                raise ValueError(
                    f"unknown tool {call.name!r}; available tools: {', '.join(self.TOOL_NAMES)}"
                )
            validated = _ARG_MODELS[call.name].model_validate(call.arguments)
            normalized_args = validated.model_dump(exclude_none=True)
            # Exercise Agent-R1's BaseTool plugin contract before stateful dispatch.
            await self.tools[call.name].run(normalized_args, tools_kwargs=self.tools_kwargs)
            self._tool_calls += 1

            if call.name == "retrieve":
                result = self._retrieve(validated)
                return self._success(
                    self._state_text(
                        extra={"last_tool": {"name": "retrieve", "result": result}}
                    ),
                    info={"tool": "retrieve"},
                )
            if call.name == "compute":
                result = self._compute(validated)
                return self._success(
                    self._state_text(
                        extra={"last_tool": {"name": "compute", "result": result}}
                    ),
                    info={"tool": "compute"},
                )
            if call.name in {"buy", "sell"}:
                assert isinstance(validated, TargetArgs)
                self._queue_directional_target(call.name, validated)
                return self._advance(call.name)
            if call.name == "abstain":
                self._abstentions += 1
                return self._advance("abstain")
            return self._advance("hold")
        except (KeyError, TypeError, ValidationError, ValueError) as error:
            return self._invalid(str(error))

    def _retrieve(self, args: _StrictArgs) -> dict[str, object]:
        assert isinstance(args, RetrieveArgs)
        query = args.query.casefold()
        symbol = args.symbol.strip().upper() if args.symbol else None
        matches: list[dict[str, object]] = []
        for observation in self._snapshot():
            if symbol is not None and observation.symbol != symbol:
                continue
            record = self._compact_observation(observation)
            # Search exactly what is returned.  A match in a discarded payload
            # field would otherwise produce a result that contains no evidence
            # for the match and silently change retrieve semantics.
            if query not in json.dumps(record, ensure_ascii=False, sort_keys=True).casefold():
                continue
            matches.append(record)
            if len(matches) >= args.limit:
                break
        return {"as_of": self.current_date.isoformat(), "results": matches}

    def _compute(self, args: _StrictArgs) -> dict[str, object]:
        assert isinstance(args, ComputeArgs)
        symbol = args.symbol.strip().upper()
        closes = [
            float(observation.payload["close"])
            for observation in self._snapshot()
            if observation.kind == "market"
            and observation.symbol == symbol
            and "close" in observation.payload
        ][-args.lookback :]
        if len(closes) < 2:
            raise ValueError(f"compute needs at least two visible close prices for {symbol}")
        array = np.asarray(closes, dtype=np.float64)
        returns = array[1:] / array[:-1] - 1.0
        values = {
            "price_change": float(array[-1] / array[0] - 1.0),
            "mean_return": float(returns.mean()),
            "volatility": float(returns.std(ddof=0)),
        }
        return {
            "as_of": self.current_date.isoformat(),
            "lookback_used": len(closes),
            "metric": args.metric,
            "symbol": symbol,
            "value": values[args.metric],
        }

    def _queue_directional_target(self, action_name: str, args: TargetArgs) -> None:
        symbol = args.symbol.strip().upper()
        if symbol not in self._symbols:
            raise ValueError(f"symbol {symbol!r} is outside the frozen episode universe")
        current_weight = self._current_weight(symbol)
        if action_name == "buy" and args.target_weight <= current_weight + 1e-12:
            raise ValueError("buy target_weight must be above the current weight")
        if action_name == "sell" and args.target_weight >= current_weight - 1e-12:
            raise ValueError("sell target_weight must be below the current weight")
        self.ledger.queue_target_order(
            symbol,
            target_weight=args.target_weight,
            decision_at=self.current_date,
        )

    def _advance(self, action_name: str) -> tuple[Observation, float, bool, dict[str, Any]]:
        current_closes = self._latest_market_prices(self._snapshot(), field="close")
        equity_before = self.ledger.equity(current_closes)
        current_benchmark = self._benchmark_closes[self._index]

        next_index = self._index + 1
        next_session = self._dates[next_index]
        next_snapshot = self._adapter.get_observations(next_session)
        next_opens = self._latest_market_prices(
            next_snapshot, field="open", session_at=next_session
        )
        next_closes = self._latest_market_prices(
            next_snapshot, field="close", session_at=next_session
        )
        self._index = next_index
        fills = self.ledger.execute_next_open(self._open_times[self._index], next_opens)
        equity_after = self.ledger.equity(next_closes)

        transaction_cost = sum(fill.total_cost for fill in fills)
        turnover = sum(fill.notional for fill in fills) / equity_before
        gross_return = (equity_after + transaction_cost) / equity_before - 1.0
        cost_fraction = transaction_cost / equity_before
        benchmark_return = self._benchmark_closes[self._index] / current_benchmark - 1.0
        self._high_water_mark = max(self._high_water_mark, equity_after)
        drawdown = max(0.0, (self._high_water_mark - equity_after) / self._high_water_mark)
        reward = (
            gross_return
            - benchmark_return
            - cost_fraction
            - self._drawdown_coefficient * drawdown
            - self._turnover_coefficient * turnover
        )
        self._done = self._index == len(self._dates) - 1
        portfolio_return_net = equity_after / equity_before - 1.0
        self._equity_curve.append(equity_after)
        self._net_returns.append(portfolio_return_net)
        self._turnovers.append(turnover)
        if equity_after > 0.0:
            self._maximum_position_weight = max(
                self._maximum_position_weight,
                max(
                    (
                        shares * next_closes[symbol] / equity_after
                        for symbol, shares in self.ledger.positions.items()
                    ),
                    default=0.0,
                ),
            )
        emitted_reward = self._emitted_reward(generic_step_reward=reward)
        info = {
            "action": action_name,
            "benchmark_return": benchmark_return,
            "cost_fraction": cost_fraction,
            "drawdown": drawdown,
            "equity": equity_after,
            "fills": [asdict(fill) for fill in fills],
            "invalid_action": False,
            "portfolio_return_net": portfolio_return_net,
            "reward_mode": self._reward_mode,
            "turnover": turnover,
        }
        if self._done:
            info["episode_return"] = equity_after / self._initial_cash - 1.0
            info["trajectory_reward"] = emitted_reward
        return self._success(
            self._state_text(extra={"last_action": action_name, "last_reward": emitted_reward}),
            reward=emitted_reward,
            done=self._done,
            info=info,
        )

    def _emitted_reward(self, *, generic_step_reward: float) -> float:
        if self._reward_mode == "generic_step":
            return generic_step_reward
        if not self._done:
            return 0.0
        episode_return = self._equity_curve[-1] / self._initial_cash - 1.0
        if self._reward_mode == "outcome_only":
            return episode_return

        assert self._finance_epsilon is not None
        assert self._risk_lambda is not None
        assert self._turnover_limit is not None
        assert self._position_limit is not None
        benchmark_return = self._benchmark_closes[-1] / self._benchmark_closes[0] - 1.0
        running_peak = self._equity_curve[0]
        maximum_drawdown = 0.0
        for equity in self._equity_curve:
            running_peak = max(running_peak, equity)
            maximum_drawdown = max(maximum_drawdown, (running_peak - equity) / running_peak)
        risk = RiskComponents(
            max_drawdown=maximum_drawdown,
            turnover_excess=max(0.0, max(self._turnovers, default=0.0) - self._turnover_limit),
            position_excess=max(0.0, self._maximum_position_weight - self._position_limit),
        )
        result = finance_advantage(
            episode_return=episode_return,
            benchmark_return=benchmark_return,
            volatility=float(np.std(self._net_returns, ddof=0)),
            epsilon=self._finance_epsilon,
            risk_lambda=self._risk_lambda,
            risk=risk,
        )
        return result.advantage

    def _snapshot(self) -> ObservationSnapshot:
        return self._adapter.get_observations(self.current_date)

    def _latest_market_prices(
        self,
        snapshot: ObservationSnapshot,
        *,
        field: str,
        session_at: datetime | None = None,
    ) -> dict[str, float]:
        expected_date = (session_at or self.current_date).date().isoformat()
        prices: dict[str, float] = {}
        for observation in snapshot:
            if (
                observation.kind == "market"
                and observation.symbol in self._symbols
                and field in observation.payload
                and observation.payload.get("date") == expected_date
            ):
                if observation.symbol in prices:
                    raise TradingDataIntegrityError(
                        f"duplicate current-session {field} price for {observation.symbol} "
                        f"on {expected_date}"
                    )
                prices[observation.symbol] = float(observation.payload[field])
        missing = [symbol for symbol in self._symbols if symbol not in prices]
        if missing:
            raise TradingDataIntegrityError(
                f"missing current-session {field} prices on {expected_date} for: {missing}"
            )
        return prices

    def _current_weight(self, symbol: str) -> float:
        closes = self._latest_market_prices(self._snapshot(), field="close")
        equity = self.ledger.equity(closes)
        return self.ledger.positions.get(symbol, 0.0) * closes[symbol] / equity

    @staticmethod
    def _compact_observation(observation: MarketObservation) -> dict[str, object]:
        if observation.kind == "market":
            record: dict[str, object] = {"symbol": observation.symbol}
            for key in ("date", "open", "high", "low", "close", "volume"):
                if key in observation.payload:
                    record[key] = _compact_json(observation.payload[key])
            return record
        payload = _compact_json(observation.payload)
        assert isinstance(payload, dict)
        return {
            "available_at": observation.available_at.isoformat(),
            "event_at": observation.event_at.isoformat(),
            "kind": observation.kind,
            "payload": payload,
            "symbol": observation.symbol,
        }

    def _state_text(self, *, extra: Mapping[str, object] | None = None) -> str:
        snapshot = self._snapshot()
        closes = self._latest_market_prices(snapshot, field="close")
        expected_date = self.current_date.date().isoformat()
        market_by_symbol: dict[str, dict[str, object]] = {}
        signal_observations: list[MarketObservation] = []
        for observation in snapshot:
            if (
                observation.kind == "market"
                and observation.symbol in self._symbols
                and observation.payload.get("date") == expected_date
            ):
                if observation.symbol in market_by_symbol:
                    raise TradingDataIntegrityError(
                        "duplicate current-session market observation for "
                        f"{observation.symbol} on {expected_date}"
                    )
                market_by_symbol[observation.symbol] = self._compact_observation(observation)
            elif observation.kind != "market":
                signal_observations.append(observation)
        missing_market = [symbol for symbol in self._symbols if symbol not in market_by_symbol]
        if missing_market:
            raise TradingDataIntegrityError(
                "missing current-session market observation on "
                f"{expected_date} for: {missing_market}"
            )
        market = [market_by_symbol[symbol] for symbol in self._symbols]
        signals = [
            self._compact_observation(observation)
            for observation in signal_observations[-_MAX_STATE_SIGNALS:]
        ]
        state: dict[str, object] = {
            "as_of": self.current_date.isoformat(),
            "cash": round(self.ledger.cash, 8),
            "equity": round(self.ledger.equity(closes), 8),
            "market": market,
            "positions": {
                symbol: round(shares, 8)
                for symbol, shares in sorted(self.ledger.positions.items())
            },
        }
        if signals:
            state["signals"] = signals
        if extra:
            compact_extra = _compact_json(extra)
            assert isinstance(compact_extra, dict)
            state.update(compact_extra)
        return json.dumps(
            state,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def _replace_visible_observation(self, text: str) -> Observation:
        """Expose the task plus only the current bounded state, never full history."""

        messages = [dict(message) for message in self._episode_prompt]
        if messages and messages[-1].get("role") == "user":
            current = str(messages[-1].get("content", ""))
            messages[-1] = {
                **messages[-1],
                "content": f"{current}\n\nCurrent observation:\n{text}",
            }
        else:
            messages.append({"role": "user", "content": text})
        self._messages = messages
        return Observation(messages=list(messages))

    def _success(
        self,
        text: str,
        *,
        reward: float = 0.0,
        done: bool = False,
        info: dict[str, Any] | None = None,
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        observation = self._replace_visible_observation(text)
        return observation, float(reward), done, dict(info or {})

    def _invalid(
        self, message: str, *, terminal: bool = False
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        self._invalid_calls += 1
        error = f"Error: {message}"
        observation = self._replace_visible_observation(
            self._state_text(extra={"error": error})
        )
        return (
            observation,
            0.0,
            terminal,
            {"error": message, "invalid_action": True},
        )

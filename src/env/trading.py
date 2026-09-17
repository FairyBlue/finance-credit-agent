"""Point-in-time trading environment implemented as an Agent-R1 ToolEnv plugin."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from functools import lru_cache
from hashlib import sha256
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
from .observation_features import compute_close_metrics, render_decision_features
from .observation_memory import (
    MEMORY_OBSERVATION_PROFILES,
    ToolResultMemory,
    validate_observation_profile,
)


class _StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TradingDataIntegrityError(RuntimeError):
    """A required bar is absent or ambiguous for the exact replay session."""


@dataclass(frozen=True, slots=True)
class TradingStepTrace:
    """Minimal causal step evidence retained after Agent-R1 drops env ``info``."""

    step_index: int
    action: str
    decision_session_index: int
    decision_at: str
    result_session_index: int
    reward: float
    done: bool
    invalid_action: bool


MAX_RETRIEVE_QUERY_CHARS = 200


class RetrieveArgs(_StrictArgs):
    query: str = Field(min_length=1, max_length=MAX_RETRIEVE_QUERY_CHARS)
    symbol: str | None = None
    limit: int = Field(default=3, ge=1, le=3)


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
    MAX_RETRIEVE_QUERY_CHARS = MAX_RETRIEVE_QUERY_CHARS
    MAX_RETRIEVE_RESULTS = 3
    MAX_RETRIEVE_BYTES = 1024
    MAX_RETRIEVE_FIELDS = 8
    MAX_RETRIEVE_TEXT_CHARS = 160
    MAX_RECENT_EVENTS = 3
    MAX_EVENT_TEXT_CHARS = 80
    MAX_ERROR_CHARS = 240
    MAX_DIAGNOSTIC_TEXT_CHARS = 65536
    MARKET_STATE_FIELDS = ("open", "high", "low", "close", "volume")

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
        screening: bool = False,
        finance_scorer_id: str | None = None,
        finance_volatility_floor: float | None = None,
        screening_calibration_sha256: str | None = None,
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
        capture_interactions: bool = False,
        observation_profile: str = "legacy",
        **kwargs: Any,
    ) -> None:
        super().__init__(tools=list(self.TOOL_NAMES), tool_format=tool_format)
        if not isinstance(capture_interactions, bool):
            raise ValueError("capture_interactions must be a boolean")
        self._capture_interactions = capture_interactions
        self._interaction_trace: list[dict[str, Any]] = []
        self._observation_profile = validate_observation_profile(observation_profile)
        self._tool_result_memory = ToolResultMemory()
        self._dates = tuple(_parse_datetime(value) for value in episode_dates)
        if not isinstance(screening, bool):
            raise ValueError("screening must be a boolean")
        self._screening = screening
        self._finance_scorer_id = finance_scorer_id
        self._screening_calibration_sha256 = screening_calibration_sha256
        self._finance_volatility_floor = finance_volatility_floor
        if screening:
            if finance_scorer_id != "benchmark_scale_v2":
                raise ValueError("screening requires finance_scorer_id=benchmark_scale_v2")
            if (
                finance_volatility_floor is None
                or not math.isfinite(finance_volatility_floor)
                or finance_volatility_floor <= 0
            ):
                raise ValueError("screening finance_volatility_floor must be finite and positive")
            if (
                not isinstance(screening_calibration_sha256, str)
                or len(screening_calibration_sha256) != 64
                or any(char not in "0123456789abcdef" for char in screening_calibration_sha256)
            ):
                raise ValueError("screening requires its calibration SHA-256")
            if not self._dates or any(
                not date(2023, 1, 3) <= timestamp.date() <= date(2024, 10, 31)
                for timestamp in self._dates
            ):
                raise ValueError("screening episode dates must remain inside historical Train")
            if adapter is None and (
                str(data_start) != "2023-01-03" or str(data_end) != "2024-10-31"
            ):
                raise ValueError("screening storage ingestion must use the Train-only snapshot")
            if reward_mode not in {"outcome_only", "finance_trajectory"}:
                raise ValueError("screening forbids historical dense rewards")
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
                data_start=(data_start if isinstance(data_start, str) else data_start.isoformat()),
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
        if (
            not screening
            and reward_mode == "finance_trajectory"
            and any(value is None for value in finance_values)
        ):
            raise ValueError(
                "finance_trajectory requires finance_epsilon, risk_lambda, turnover_limit, "
                "and position_limit"
            )
        if screening and any(
            value is None for value in (risk_lambda, turnover_limit, position_limit)
        ):
            raise ValueError("screening requires risk_lambda, turnover_limit and position_limit")
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
        self._prompt_prefix: list[dict[str, Any]] = []
        self._recent_events: list[dict[str, object]] = []
        self._credit_step_trace: list[TradingStepTrace] = []
        self._active_decision_session_index = 0

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
        return self._symbols

    @property
    def credit_step_trace(self) -> tuple[TradingStepTrace, ...]:
        return tuple(self._credit_step_trace)

    @property
    def episode_complete(self) -> bool:
        return self._done

    @property
    def reward_mode(self) -> str:
        return self._reward_mode

    @property
    def screening(self) -> bool:
        return self._screening

    @property
    def episode_session_count(self) -> int:
        return len(self._dates)

    @property
    def current_session_index(self) -> int:
        return self._index

    def screening_terminal_metrics(self) -> dict[str, Any]:
        """Common scorer and ledger outcomes; never a model-visible observation."""
        if not self._screening or not self._done or self._index != len(self._dates) - 1:
            raise RuntimeError("screening metrics require a complete screening trajectory")
        from src.rewards.screening import score_benchmark_scaled

        assert self._risk_lambda is not None
        assert self._turnover_limit is not None
        assert self._position_limit is not None
        assert self._finance_volatility_floor is not None
        episode_return = self._equity_curve[-1] / self._initial_cash - 1.0
        benchmark_return = self._benchmark_closes[-1] / self._benchmark_closes[0] - 1.0
        benchmark_returns = [
            following / current - 1.0
            for current, following in zip(
                self._benchmark_closes[:-1], self._benchmark_closes[1:], strict=True
            )
        ]
        peak = self._equity_curve[0]
        maximum_drawdown = 0.0
        for equity in self._equity_curve:
            peak = max(peak, equity)
            maximum_drawdown = max(maximum_drawdown, (peak - equity) / peak)
        maximum_turnover = max(self._turnovers, default=0.0)
        risk = RiskComponents(
            max_drawdown=maximum_drawdown,
            turnover_excess=max(0.0, maximum_turnover - self._turnover_limit),
            position_excess=max(0.0, self._maximum_position_weight - self._position_limit),
        )
        benchmark_volatility = float(np.std(benchmark_returns, ddof=0))
        result = score_benchmark_scaled(
            episode_return=episode_return,
            benchmark_return=benchmark_return,
            benchmark_volatility=benchmark_volatility,
            volatility_floor=self._finance_volatility_floor,
            risk_lambda=self._risk_lambda,
            risk=risk,
        )
        metrics: dict[str, Any] = {
            "scorer_id": self._finance_scorer_id,
            "calibration_sha256": self._screening_calibration_sha256,
            "terminal_score": result.advantage,
            "raw_net_return": episode_return,
            "benchmark_return": benchmark_return,
            "benchmark_excess": episode_return - benchmark_return,
            "agent_volatility": float(np.std(self._net_returns, ddof=0)),
            "benchmark_volatility": benchmark_volatility,
            "score_denominator": max(benchmark_volatility, self._finance_volatility_floor),
            "max_drawdown": maximum_drawdown,
            "turnover": math.fsum(self._turnovers),
            "max_daily_turnover": maximum_turnover,
            "maximum_position_weight": self._maximum_position_weight,
            "risk_penalty": max(risk.max_drawdown, risk.turnover_excess, risk.position_excess),
            "transaction_cost": math.fsum(fill.total_cost for fill in self.ledger.fills),
            "commission": math.fsum(fill.commission for fill in self.ledger.fills),
            "slippage_cost": math.fsum(fill.slippage_cost for fill in self.ledger.fills),
            "invalid_count": self._invalid_calls,
            "step_count": len(self._credit_step_trace),
            "session_count": len(self._dates),
        }
        if any(
            isinstance(value, (int, float)) and not math.isfinite(value)
            for value in metrics.values()
        ):
            raise RuntimeError("non-finite screening terminal metrics")
        return metrics

    def reset(self, **kwargs: Any) -> Observation:
        self._tool_result_memory = ToolResultMemory()
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
        self._recent_events = []
        self._credit_step_trace = []
        self._interaction_trace = []
        self._active_decision_session_index = 0
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
        super().reset(raw_prompt=raw_prompt)
        self._prompt_prefix = [dict(message) for message in self._messages]
        return self._observation(self._state_text(), tool_response=False)

    async def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        decision_index = self._index
        result = await self._step(action)
        if self._capture_interactions:
            observation, reward, done, info = result
            raw_text = action.text if isinstance(action, Action) else None
            observation_text = (
                json.dumps(observation.messages, ensure_ascii=True, sort_keys=True, allow_nan=False)
                if observation.messages is not None
                else observation.text
            )
            # Keep diagnostics separate from observations and credit traces. In
            # particular, neither rewards nor this audit history enter prompts.
            self._interaction_trace.append(
                {
                    "step_index": len(self._interaction_trace),
                    "decision_session_index": decision_index,
                    "result_session_index": self._index,
                    "response": self._diagnostic_text(raw_text),
                    "observation": self._diagnostic_text(observation_text),
                    "reward": reward,
                    "done": done,
                    "info": json.loads(json.dumps(
                        info, default=self._diagnostic_json_default, allow_nan=False,
                    )),
                }
            )
        return result

    @property
    def interaction_trace(self) -> list[dict[str, Any]]:
        """Detached, bounded diagnostic records; empty unless explicitly enabled."""
        return deepcopy(self._interaction_trace)

    @classmethod
    def _diagnostic_text(cls, text: Any) -> dict[str, Any]:
        if text is not None and not isinstance(text, str):
            # Action is an unvalidated upstream dataclass. A non-text action
            # remains an ordinary invalid step, even with evidence enabled.
            return {
                "text": None, "char_count": None, "truncated": False, "sha256": None,
                "unavailable_reason": "non_text_value", "value_type": type(text).__name__,
            }
        return {
            "text": None if text is None else text[:cls.MAX_DIAGNOSTIC_TEXT_CHARS],
            "char_count": 0 if text is None else len(text),
            "truncated": text is not None and len(text) > cls.MAX_DIAGNOSTIC_TEXT_CHARS,
            "sha256": None if text is None else sha256(text.encode("utf-8")).hexdigest(),
        }

    @staticmethod
    def _diagnostic_json_default(value: Any) -> str:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        raise TypeError(f"unsupported diagnostic value: {type(value).__name__}")

    async def _step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        self._active_decision_session_index = self._index
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
                if self._observation_profile in MEMORY_OBSERVATION_PROFILES:
                    self._tool_result_memory.remember(call.name, normalized_args, result)
                self._remember_event(call.name, normalized_args)
                return self._success(
                    self._state_text(extra={"last_tool_result": result}),
                    info={"result_count": result["returned"], "tool": "retrieve"},
                )
            if call.name == "compute":
                result = self._compute(validated)
                if self._observation_profile in MEMORY_OBSERVATION_PROFILES:
                    self._tool_result_memory.remember(call.name, normalized_args, result)
                self._remember_event(call.name, normalized_args)
                return self._success(
                    self._state_text(extra={"last_tool_result": result}),
                    info={"tool": "compute"},
                )
            if call.name in {"buy", "sell"}:
                assert isinstance(validated, TargetArgs)
                self._queue_directional_target(call.name, validated)
                self._remember_event(call.name, normalized_args)
                return self._advance(call.name)
            if call.name == "abstain":
                self._abstentions += 1
                self._remember_event(call.name, normalized_args)
                return self._advance("abstain")
            self._remember_event(call.name, normalized_args)
            return self._advance("hold")
        except (KeyError, TypeError, ValidationError, ValueError) as error:
            return self._invalid(str(error))

    def _retrieve(self, args: _StrictArgs) -> dict[str, object]:
        assert isinstance(args, RetrieveArgs)
        query = args.query.casefold()
        symbol = args.symbol.strip().upper() if args.symbol else None
        matches: list[dict[str, object]] = []
        truncated = False
        for observation in self._snapshot():
            if symbol is not None and observation.symbol != symbol:
                continue
            compact = self._compact_retrieval_record(observation)
            if query not in json.dumps(compact, ensure_ascii=False, sort_keys=True).casefold():
                continue
            if len(matches) >= args.limit:
                truncated = True
                break
            candidate = {
                "as_of": self.current_date.isoformat(),
                "results": [*matches, compact],
                "returned": len(matches) + 1,
                "truncated": False,
            }
            if self._json_size(candidate) > self.MAX_RETRIEVE_BYTES:
                truncated = True
                break
            matches.append(compact)
        result: dict[str, object] = {
            "as_of": self.current_date.isoformat(),
            "results": matches,
            "returned": len(matches),
            "truncated": truncated,
        }
        if self._json_size(result) > self.MAX_RETRIEVE_BYTES:
            raise RuntimeError("bounded retrieve result exceeds its byte contract")
        return result

    @staticmethod
    def _json_size(value: object) -> int:
        return len(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        )

    def _compact_retrieval_record(self, observation: MarketObservation) -> dict[str, object]:
        keys: tuple[str, ...]
        if observation.kind == "market":
            keys = self.MARKET_STATE_FIELDS
        else:
            keys = tuple(sorted(observation.payload))[: self.MAX_RETRIEVE_FIELDS]
        fields: dict[str, object] = {}
        for key in keys:
            if key not in observation.payload:
                continue
            value = observation.payload[key]
            if isinstance(value, str):
                suffix = "..." if len(value) > self.MAX_RETRIEVE_TEXT_CHARS else ""
                fields[key] = value[: self.MAX_RETRIEVE_TEXT_CHARS] + suffix
            elif value is None or isinstance(value, (bool, int)):
                fields[key] = value
            elif isinstance(value, float) and math.isfinite(value):
                fields[key] = round(value, 8)
        return {
            "available_at": observation.available_at.isoformat(),
            "fields": fields,
            "kind": observation.kind,
            "symbol": observation.symbol,
        }

    def _remember_event(self, tool: str, arguments: Mapping[str, object]) -> None:
        compact_arguments: dict[str, object] = {}
        for key in (
            "error",
            "limit",
            "lookback",
            "metric",
            "query",
            "reason",
            "symbol",
            "target_weight",
        ):
            if key not in arguments:
                continue
            value = arguments.get(key)
            if isinstance(value, str):
                compact_arguments[key] = self._bounded_ascii_text(
                    value, limit=self.MAX_EVENT_TEXT_CHARS
                )
            elif value is None or isinstance(value, (bool, int, float)):
                compact_arguments[key] = value
        self._recent_events.append(
            {
                "arguments": compact_arguments,
                "session_index": self._index,
                "tool": tool,
            }
        )
        self._recent_events = self._recent_events[-self.MAX_RECENT_EVENTS :]

    @staticmethod
    def _bounded_ascii_text(value: str, *, limit: int) -> str:
        """Keep user-controlled summaries bounded even when JSON escapes Unicode."""

        return value.encode("unicode_escape").decode("ascii")[:limit]

    @property
    def observation_profile(self) -> str:
        return self._observation_profile

    def _visible_close_prices(
        self, symbol: str, snapshot: ObservationSnapshot | None = None,
    ) -> list[float]:
        # Preserve the existing PIT-record order and selection semantics. This
        # counts records, not distinct dates or sessions since episode reset.
        return [
            float(observation.payload["close"])
            for observation in (self._snapshot() if snapshot is None else snapshot)
            if observation.kind == "market"
            and observation.symbol == symbol
            and "close" in observation.payload
        ]

    def _compute(self, args: _StrictArgs) -> dict[str, object]:
        assert isinstance(args, ComputeArgs)
        symbol = args.symbol.strip().upper()
        closes = self._visible_close_prices(symbol)[-args.lookback :]
        if len(closes) < 2:
            raise ValueError(f"compute needs at least two visible close prices for {symbol}")
        values = compute_close_metrics(closes)
        return {
            "as_of": self.current_date.isoformat(),
            "lookback_used": len(closes),
            "metric": args.metric,
            "symbol": symbol,
            "value": values[args.metric],
        }

    def _queue_directional_target(self, action_name: str, args: TargetArgs) -> None:
        if action_name not in ("buy", "sell"):
            raise ValueError("directional target action must be buy or sell")
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
            direction="buy" if action_name == "buy" else "sell",
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
        skipped_orders = [
            {
                **asdict(order),
                "decision_at": order.decision_at.isoformat(),
                "executed_at": order.executed_at.isoformat(),
            }
            for order in self.ledger.last_execution_skips
        ]
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
            "skipped_orders": skipped_orders,
            "invalid_action": False,
            "portfolio_return_net": portfolio_return_net,
            "reward_mode": self._reward_mode,
            "turnover": turnover,
        }
        if self._done:
            info["episode_return"] = equity_after / self._initial_cash - 1.0
            info["trajectory_reward"] = emitted_reward
        extra: dict[str, Any] = {"last_action": action_name, "last_reward": emitted_reward}
        if skipped_orders:
            extra["skipped_orders"] = skipped_orders
        return self._success(
            self._state_text(extra=extra),
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
        if self._screening:
            return float(self.screening_terminal_metrics()["terminal_score"])

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

    def _state_text(self, *, extra: Mapping[str, object] | None = None) -> str:
        snapshot = self._snapshot()
        closes = self._latest_market_prices(snapshot, field="close")
        equity = self.ledger.equity(closes)
        expected_date = self.current_date.date().isoformat()
        market: dict[str, dict[str, object]] = {}
        for observation in snapshot:
            if (
                observation.kind != "market"
                or observation.symbol not in self._symbols
                or observation.payload.get("date") != expected_date
            ):
                continue
            if observation.symbol in market:
                raise TradingDataIntegrityError(
                    f"duplicate current-session state for {observation.symbol} on {expected_date}"
                )
            fields: dict[str, object] = {}
            for key in self.MARKET_STATE_FIELDS:
                value = observation.payload.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    number = float(value)
                    if math.isfinite(number):
                        fields[key] = round(number, 8)
            market[observation.symbol] = fields
        positions = {
            symbol: {
                "shares": round(float(shares), 8),
                "weight": round(float(shares) * closes[symbol] / equity, 8),
            }
            for symbol, shares in sorted(self.ledger.positions.items())
            if abs(shares) > 1e-12
        }
        state: dict[str, object] = {
            "cash": round(self.ledger.cash, 8),
            "current_date": self.current_date.isoformat(),
            "market": market,
            "portfolio_equity": round(equity, 8),
            "positions": positions,
            "recent_events": list(self._recent_events),
            "session_index": self._index,
            "sessions_remaining": len(self._dates) - self._index - 1,
        }
        if self._observation_profile in MEMORY_OBSERVATION_PROFILES:
            visible_closes = {
                symbol: self._visible_close_prices(symbol, snapshot) for symbol in self._symbols
            }
            state["compute_history"] = {
                "minimum_closes": 2,
                "max_lookback": 60,
                "available_close_records": {
                    symbol: len(history) for symbol, history in visible_closes.items()
                },
            }
            current_result = extra.get("last_tool_result") if extra else None
            state["tool_result_memory"] = self._tool_result_memory.render(
                current_result if isinstance(current_result, Mapping) else None,
            )
            if self._observation_profile == "history_features_v1":
                state["decision_features"] = render_decision_features(
                    visible_closes, as_of=self.current_date.isoformat(),
                )
        if extra:
            state.update(
                {key: value for key, value in extra.items() if key != "last_reward"}
                if self._screening
                else extra
            )
        return json.dumps(
            state,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def _observation(self, text: str, *, tool_response: bool = True) -> Observation:
        content = self.format_wrapper.format_observation(text) if tool_response else text
        self._messages = [
            *[dict(message) for message in self._prompt_prefix],
            {"role": "user", "content": content},
        ]
        return Observation(messages=list(self._messages))

    def _success(
        self,
        text: str,
        *,
        reward: float = 0.0,
        done: bool = False,
        info: dict[str, Any] | None = None,
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        info_document = dict(info or {})
        action = info_document.get("action") or info_document.get("tool")
        if not isinstance(action, str) or not action:
            raise RuntimeError("successful trading steps must identify their action")
        self._record_credit_step(
            action=action,
            reward=reward,
            done=done,
            invalid_action=False,
        )
        observation = self._observation(text)
        return observation, float(reward), done, info_document

    def _invalid(
        self, message: str, *, terminal: bool = False
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        self._invalid_calls += 1
        safe_message = self._bounded_ascii_text(message, limit=self.MAX_ERROR_CHARS)
        error = f"Error: {safe_message}"
        self._remember_event("invalid", {"error": safe_message})
        self._record_credit_step(
            action="invalid",
            reward=0.0,
            done=terminal,
            invalid_action=True,
        )
        observation = self._observation(self._state_text(extra={"error": error}))
        return (
            observation,
            0.0,
            terminal,
            {"error": safe_message, "invalid_action": True},
        )

    def _record_credit_step(
        self,
        *,
        action: str,
        reward: float,
        done: bool,
        invalid_action: bool,
    ) -> None:
        numeric_reward = float(reward)
        if not math.isfinite(numeric_reward):
            raise RuntimeError("trading step reward must be finite")
        decision_session_index = self._active_decision_session_index
        self._credit_step_trace.append(
            TradingStepTrace(
                step_index=len(self._credit_step_trace),
                action=action,
                decision_session_index=decision_session_index,
                decision_at=self._dates[decision_session_index].isoformat(),
                result_session_index=self._index,
                reward=numeric_reward,
                done=bool(done),
                invalid_action=bool(invalid_action),
            )
        )

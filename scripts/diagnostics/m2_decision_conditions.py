"""CPU-only observation interventions for the fixed Train decision diagnostic.

This module changes displayed information only. The underlying environment,
tools, ledger, reward and bounded result memory are the production implementation.
Feature rows are a diagnostic information intervention: direct, read-only calls to
the real compute implementation, not policy tool calls or sampled experience.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from agent_r1.env.base import Action, Observation

from src.agents.screening_prompt import apply_screening_action_protocol
from src.env.trading import ComputeArgs, TradingEnv

CONDITIONS = ("raw", "features", "history_only", "memory_only", "both")
METRICS = ("price_change", "mean_return", "volatility")
LOOKBACK = 5
FEATURE_COLUMNS = ("symbol", "lookback_used", *METRICS)
FEATURE_KEY = "decision_features"
EPISODES = ("formal-train-0001", "formal-train-0005")
PROMPT_CAP = 8192
RESPONSE_CAP = 4096
CONTEXT_CAP = 12288
HOLD_TEXT = '<tool_call>{"name":"hold","arguments":{}}</tool_call>'


def validate_condition(condition: object, start_session: object) -> str:
    if type(condition) is not str or condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if type(start_session) is not int or start_session not in (0, 4):
        raise ValueError("diagnostic start_session must be integer 0 or 4")
    allowed = {"raw": (0, 4), "features": (4,)}
    if start_session not in allowed.get(condition, (0,)):
        raise ValueError("condition/start_session pair is outside the fixed diagnostic")
    return condition


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


class DecisionConditionEnv(TradingEnv):
    """Render one registered condition over the same production state machine."""

    def __init__(self, *, condition: str, start_session: int, **kwargs: Any) -> None:
        self.decision_condition = validate_condition(condition, start_session)
        self.diagnostic_start_session = start_session
        self._latest_feature_evidence: dict[str, Any] | None = None
        self._latest_pit_counts: dict[str, int] = {}
        self._feature_cache: tuple[str, dict[str, Any], dict[str, Any]] | None = None
        # Keep successful-result collection identical even when hidden. These
        # flags are fixed diagnostic execution contracts, not caller preferences.
        super().__init__(**{
            **kwargs,
            "observation_profile": "history_memory_v1",
            "capture_interactions": True,
            "reward_mode": "outcome_only",
        })
        if len(self.symbols) != 20 or len(self._dates) != 21:
            raise ValueError("diagnostic requires the frozen 20-symbol/21-session universe")

    @property
    def feature_evidence(self) -> dict[str, Any] | None:
        """Detached source/result proof for the most recently rendered observation."""
        return deepcopy(self._latest_feature_evidence)

    @property
    def diagnostic_pit_counts(self) -> dict[str, int]:
        """Counts from the last rendered state, including when hidden from the model."""
        return deepcopy(self._latest_pit_counts)

    def reset(self, **kwargs: Any) -> Observation:
        self._latest_feature_evidence = None
        self._latest_pit_counts = {}
        self._feature_cache = None
        return super().reset(**kwargs)

    def _features(self) -> tuple[dict[str, Any], dict[str, Any]]:
        as_of = self.current_date.isoformat()
        # Each environment owns its cache. The runner freezes the source data;
        # repeated non-advancing calls at one PIT cutoff expose the same exact
        # values. Reset and every date advance force real recomputation.
        if self._feature_cache is not None and self._feature_cache[0] == as_of:
            cached_table, cached_proof = deepcopy(self._feature_cache[1:])
            cached_proof.update(cache_hit=True, actual_compute_calls_this_render=0)
            return cached_table, cached_proof
        snapshot = self._snapshot()
        rows: list[list[Any]] = []
        proofs: list[dict[str, Any]] = []
        for symbol in self.symbols:
            visible = [item for item in snapshot if item.kind == "market"
                       and item.symbol == symbol and "close" in item.payload]
            selected = visible[-LOOKBACK:]
            # Fail closed rather than fabricate history or pad short windows.
            if len(selected) < 2:
                raise ValueError(f"prepared features need two visible close records: {symbol}")
            results = [self._compute(ComputeArgs(
                symbol=symbol, metric=metric, lookback=LOOKBACK,  # type: ignore[arg-type]
            )) for metric in METRICS]
            for metric, result in zip(METRICS, results, strict=True):
                if (result["symbol"] != symbol or result["metric"] != metric
                        or result["as_of"] != as_of
                        or result["lookback_used"] != len(selected)):
                    raise RuntimeError("real compute result/source contract changed")
            rows.append([symbol, len(selected), *[r["value"] for r in results]])
            proofs.append({
                "symbol": symbol,
                "visible_close_record_count": len(visible),
                "selected_close_records": [{
                    "observation_id": item.observation_id,
                    "event_at": item.event_at.isoformat(),
                    "available_at": item.available_at.isoformat(),
                    "close": float(item.payload["close"]),
                } for item in selected],
                "compute_results": results,
            })
        table = {
            "source": "TradingEnv._compute",
            "as_of": as_of,
            "lookback_requested": LOOKBACK,
            "columns": list(FEATURE_COLUMNS),
            "rows": rows,
        }
        proof = {
            "intervention": "read_only_prepared_PIT_compute_results",
            "as_of": as_of,
            "session_index": self._index,
            "symbol_order": list(self.symbols),
            "metrics": list(METRICS),
            "lookback_requested": LOOKBACK,
            "source_compute_calls": 20 * len(METRICS),
            "actual_compute_calls_this_render": 20 * len(METRICS),
            "cache_hit": False,
            "model_requests": 0,
            "environment_steps": 0,
            "selection": "last visible close records in the production adapter order",
            "table_sha256": hashlib.sha256(_canonical(table).encode("ascii")).hexdigest(),
            "sources": proofs,
        }
        # Validate finite JSON before an observation or proof can be emitted.
        _canonical(proof)
        self._feature_cache = (as_of, deepcopy(table), deepcopy(proof))
        return table, proof

    def _state_text(self, *, extra: Mapping[str, object] | None = None) -> str:
        state = json.loads(super()._state_text(extra=extra))
        self._latest_pit_counts = deepcopy(state["compute_history"]["available_close_records"])
        if self.decision_condition not in ("history_only", "both"):
            del state["compute_history"]
        if self.decision_condition not in ("memory_only", "both"):
            del state["tool_result_memory"]
        self._latest_feature_evidence = None
        # Four scripted holds construct a shared cash state before any model
        # request. Prefix observations are retained but never sent to the model.
        if (self.decision_condition == "features"
                and self._index >= self.diagnostic_start_session):
            table, proof = self._features()
            state[FEATURE_KEY] = table
            self._latest_feature_evidence = proof
        return _canonical(state)


def create_env(
    rows: list[dict[str, Any]], index: int, condition: str, start_session: int,
) -> DecisionConditionEnv:
    """Construct from the two already source-verified pilot Train parquet rows.

    The caller must bind the parquet/raw manifest hashes before construction;
    this function additionally rejects a different split or window identity.
    """
    validate_condition(condition, start_session)
    if len(rows) != 2 or type(index) is not int or index not in (0, 1):
        raise ValueError("expected two fixed pilot rows and index 0 or 1")
    row = rows[index]
    extra = row["extra_info"]
    if (extra.get("episode_id") != EPISODES[index]
            or extra.get("source_index") != 4 * index
            or extra.get("index") != index
            or extra.get("source_split") != "train"
            or extra.get("screening_role") != "pilot_train"):
        raise ValueError("wrong fixed Train window identity")
    kwargs = json.loads(row["env_kwargs"])
    if kwargs.get("screening") is not True:
        raise ValueError("diagnostic requires screening scorer inputs")
    prompt = row["prompt"]
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("missing original raw prompt")
    return DecisionConditionEnv(
        condition=condition, start_session=start_session,
        **{**kwargs, "raw_prompt": deepcopy(prompt)},
    )


async def reset_with_prefix(
    env: DecisionConditionEnv,
) -> tuple[Observation, dict[str, Any]]:
    """Reset and retain the prescribed untrained hold prefix, without model calls."""
    observation = env.reset()
    initial_observation = deepcopy(observation.messages)
    for index in range(env.diagnostic_start_session):
        observation, reward, done, info = await env.step(Action(text=HOLD_TEXT))
        trace = env.credit_step_trace[-1]
        if (reward != 0.0 or done or info.get("invalid_action")
                or trace.action != "hold" or trace.decision_session_index != index
                or trace.result_session_index != index + 1):
            raise RuntimeError("scripted hold prefix violated the registered state construction")
    if (env._index != env.diagnostic_start_session or env.ledger.positions
            or env.ledger.fills or env.ledger.cash != env._initial_cash
            or len(env.credit_step_trace) != env.diagnostic_start_session):
        raise RuntimeError("prefix did not construct the common cash/zero-position state")
    return observation, {
        "kind": "scripted_state_construction_not_model_experience",
        "decision_condition": env.decision_condition,
        "start_session": env.diagnostic_start_session,
        "model_requests": 0,
        "scripted_hold_count": env.diagnostic_start_session,
        "prefix_action_count": env.diagnostic_start_session,
        "is_model_request": False,
        "initial_observation": initial_observation,
        "credit_step_trace": [asdict(step) for step in env.credit_step_trace],
        "interaction_trace": env.interaction_trace,
        "final_as_of": env.current_date.isoformat(),
        "cash": env.ledger.cash,
        "positions": dict(env.ledger.positions),
        "feature_evidence": env.feature_evidence,
    }


def render_prompt(tokenizer: Any, observation: Observation, tools: Any) -> dict[str, Any]:
    """Use the real shared protocol/template; never truncate over-budget input."""
    if (observation.messages is None or observation.text is not None
            or observation.token_ids is not None):
        raise ValueError("diagnostic requires a message observation")
    original = deepcopy(observation.messages)
    messages = apply_screening_action_protocol(observation.messages, "single_tool_v1")
    ids = list(tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=True, enable_thinking=False,
    ))
    if observation.messages != original:
        raise RuntimeError("prompt helper mutated environment observation")
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("invalid prompt token IDs")
    return {
        "messages": messages,
        "tools": deepcopy(tools),
        "template_kwargs": {"enable_thinking": False},
        "action_protocol": "single_tool_v1",
        "prompt_ids": ids,
        "prompt_sha256": hashlib.sha256(json.dumps(ids).encode("ascii")).hexdigest(),
        "prompt_token_count": len(ids),
    }

"""Diagnostic-only availability/time explanations over the frozen memory baseline.

The two factors alter only current-state JSON. Every condition retains the same
production result memory, actions, state transitions, data and reward. The old
memory_only rendering is the exact control, not a newly worded baseline.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from agent_r1.env.base import Observation

from src.env.trading import TradingEnv

ROOT = Path(__file__).resolve().parents[2]
LEGACY_PATH = ROOT / "scripts/diagnostics/m2_decision_conditions.py"
LEGACY_SHA256 = "39c5fda47d09bfb63cd41b1b5b5ea6a12721a6429bafb65de66e7aea1ee3c87f"
CONDITIONS = ("memory_only", "availability", "time_semantics", "availability_time")
AVAILABILITY_KEY = "compute_availability"
TIME_KEY = "decision_time_semantics"
AVAILABILITY_TEXT = (
    "Availability uses only point-in-time close history. unavailable means compute lacks "
    "sufficient visible closes for that symbol and will fail in this state. ready means "
    "the history requirement is met; arguments must still be valid."
)
TIME_TEXT = {
    "advances_one_session": (
        "A successful buy, sell, hold, or abstain advances exactly one session."
    ),
    "does_not_advance": "compute, retrieve, and invalid actions do not advance the session.",
    "repeated_compute": (
        "At the same date, repeating compute with the same arguments does not make "
        "new data available."
    ),
    "saved_results": (
        "Saved tool results retain their original as_of. After the date advances, assess whether "
        "those results remain suitable for the new decision."
    ),
}


def _load_legacy() -> Any:
    if hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest() != LEGACY_SHA256:
        raise RuntimeError("Frozen memory baseline implementation changed")
    spec = importlib.util.spec_from_file_location("semantics_legacy_conditions", LEGACY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = _load_legacy()
EPISODES = LEGACY.EPISODES
PROMPT_CAP = LEGACY.PROMPT_CAP
RESPONSE_CAP = LEGACY.RESPONSE_CAP
CONTEXT_CAP = LEGACY.CONTEXT_CAP
# Preserve the exact task/protocol/token rendering and reset-receipt implementation.
render_prompt = LEGACY.render_prompt
reset_with_prefix = LEGACY.reset_with_prefix


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def validate_condition(condition: object, start_session: object = 0) -> str:
    if type(condition) is not str or condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if type(start_session) is not int or start_session != 0:
        raise ValueError("semantics diagnostic starts at integer session 0")
    return condition


class SemanticsConditionEnv(TradingEnv):
    """Display two independent explanations without modifying the environment rules."""

    def __init__(self, *, condition: str, start_session: int = 0, **kwargs: Any) -> None:
        self.decision_condition = validate_condition(condition, start_session)
        self.diagnostic_start_session = start_session
        self._latest_pit_counts: dict[str, int] = {}
        self._latest_semantic_evidence: dict[str, Any] | None = None
        super().__init__(
            **{
                **kwargs,
                "observation_profile": "history_memory_v1",
                "capture_interactions": True,
                "reward_mode": "outcome_only",
            }
        )
        if len(self.symbols) != 20 or self.episode_session_count != 21:
            raise ValueError("diagnostic requires the frozen 20-symbol/21-session universe")

    @property
    def diagnostic_pit_counts(self) -> dict[str, int]:
        """Detached actual counts from the last rendered state; no new snapshot scan."""
        return deepcopy(self._latest_pit_counts)

    @property
    def semantic_evidence(self) -> dict[str, Any] | None:
        return deepcopy(self._latest_semantic_evidence)

    @property
    def feature_evidence(self) -> dict[str, Any] | None:
        """Explicit transport alias for the frozen executor, not a feature intervention."""
        return self.semantic_evidence

    def reset(self, **kwargs: Any) -> Observation:
        self._latest_pit_counts = {}
        self._latest_semantic_evidence = None
        return super().reset(**kwargs)

    def _state_text(self, *, extra: Mapping[str, object] | None = None) -> str:
        # The production implementation already makes one PIT snapshot and
        # supplies it to its per-symbol close counts. Reuse those exact counts.
        state = json.loads(super()._state_text(extra=extra))
        counts = state.pop("compute_history")["available_close_records"]
        self._latest_pit_counts = deepcopy(counts)
        if set(counts) != set(self.symbols) or any(
            type(n) is not int or n < 0 for n in counts.values()
        ):
            raise RuntimeError("invalid production PIT close counts")
        statuses = {
            symbol: "ready" if counts[symbol] >= 2 else "unavailable" for symbol in self.symbols
        }
        availability = self.decision_condition in ("availability", "availability_time")
        time_semantics = self.decision_condition in ("time_semantics", "availability_time")
        displayed: dict[str, Any] = {}
        if availability:
            displayed[AVAILABILITY_KEY] = {
                "as_of": state["current_date"],
                "explanation": AVAILABILITY_TEXT,
                "columns": ["symbol", "status"],
                "rows": [[symbol, statuses[symbol]] for symbol in self.symbols],
            }
        if time_semantics:
            displayed[TIME_KEY] = deepcopy(TIME_TEXT)
        state.update(displayed)
        self._latest_semantic_evidence = {
            "schema_version": 1,
            "kind": "semantic_display",
            "condition": self.decision_condition,
            "as_of": state["current_date"],
            "session_index": state["session_index"],
            "transport_field": "feature_evidence",
            "meaning": "semantics diagnostic evidence, not precomputed features",
            "factors": {"availability": availability, "time_semantics": time_semantics},
            "diagnostic_pit_counts": deepcopy(counts),
            "availability_by_symbol": statuses,
            "availability_rule": "ready iff actual visible close record count >= 2",
            "displayed_state_fields": deepcopy(displayed),
            "displayed_fields_sha256": _sha(displayed),
            "memory_display_sha256": _sha(state["tool_result_memory"]),
            "model_requests": 0,
            "environment_steps": 0,
        }
        return _canonical(state)


def create_env(
    rows: list[dict[str, Any]],
    index: int,
    condition: str,
    start_session: int = 0,
) -> SemanticsConditionEnv:
    """Construct from already SHA-verified, fixed pilot Train rows only."""
    validate_condition(condition, start_session)
    if len(rows) != 2 or type(index) is not int or index not in (0, 1):
        raise ValueError("expected two fixed pilot rows and index 0 or 1")
    row = rows[index]
    extra = row["extra_info"]
    if (
        extra.get("episode_id") != EPISODES[index]
        or extra.get("source_index") != 4 * index
        or extra.get("index") != index
        or extra.get("source_split") != "train"
        or extra.get("screening_role") != "pilot_train"
    ):
        raise ValueError("wrong fixed Train window identity")
    kwargs = json.loads(row["env_kwargs"])
    if kwargs.get("screening") is not True:
        raise ValueError("diagnostic requires screening scorer inputs")
    prompt = row["prompt"]
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("missing original raw prompt")
    return SemanticsConditionEnv(
        condition=condition,
        start_session=start_session,
        **{**kwargs, "raw_prompt": deepcopy(prompt)},
    )

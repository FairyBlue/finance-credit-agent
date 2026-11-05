"""Isolated n=4 same-prefix gate operators; not wired into production training.

The caller supplies complete fixed-hold terminal financial scores. For a common
hold baseline H, center(Y) == center(Y-H); H is never an input to this module.
Only the action-diversity gate changes. There is no pooling or whitening.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

ROLLOUT_N = 4
ETA = 0.25
COMPLETE_LABEL = "COMPLETE_FIXED_HOLD"
EconomicOrderKey = tuple[str] | tuple[str, str, float]
Signals = tuple[float, ...]


class IncompleteSamePrefixGroup(ValueError):
    """The whole group has no n=4 support; no smaller-group result is returned."""


@dataclass(frozen=True, slots=True)
class SamePrefixGateSlot:
    anchor_id: str
    slot_id: int
    prompt_sha256: str
    financial_state_sha256: str
    policy_snapshot_sha256: str
    economic_order_key: str | None
    score: float | None
    label_status: str


@dataclass(frozen=True, slots=True)
class GateSignals:
    zero: Signals
    ungated_plain: Signals
    family_gate_plain: Signals
    order_gate_plain: Signals


@dataclass(frozen=True, slots=True)
class SamePrefixGateResult:
    """Signals follow the supplied row order, including after a row permutation."""

    zero: Signals
    ungated_plain: Signals
    family_gate_plain: Signals
    order_gate_plain: Signals
    scaled_eta025: GateSignals
    normalized_order_keys: tuple[EconomicOrderKey, ...]
    family_gate: bool
    order_gate: bool
    mean_score: float
    eta: float = ETA


def _sha256(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def normalize_economic_order_key(value: str) -> EconomicOrderKey:
    """Normalize economic identity only, without parsing or repairing an API call.

    Input is a JSON array from an already accepted advancing order. Symbols are
    stripped and uppercased, numerical zero has one identity, and reasons do not
    enter hold/abstain keys. Universe membership/actual execution belong to the
    caller's environment evidence, not to this pure operator.
    """
    if not isinstance(value, str):
        raise ValueError("economic_order_key must be a JSON array string")
    try:
        key = json.loads(value)
    except (TypeError, ValueError) as error:
        raise ValueError("economic_order_key must be a JSON array string") from error
    if not isinstance(key, list) or not key or not isinstance(key[0], str):
        raise ValueError("economic_order_key must be a nonempty action array")
    name = key[0]
    if name in {"hold", "abstain"}:
        if len(key) != 1:
            raise ValueError("hold/abstain economic keys have no parameters")
        return (name,)
    if name not in {"buy", "sell"} or len(key) != 3:
        raise ValueError("economic key must identify buy, sell, hold or abstain")
    symbol = key[1]
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("economic order symbol must be nonempty text")
    target = _finite_number(key[2], "target_weight")
    if not 0.0 <= target <= 1.0:
        raise ValueError("target_weight must lie within [0,1]")
    return name, symbol.strip().upper(), 0.0 if target == 0.0 else target


def compute_same_prefix_gate(rows: Sequence[SamePrefixGateSlot]) -> SamePrefixGateResult:
    """Return four centered-score operators for one complete same-prefix group.

    tau_i = score_i - mean_4(score). Family/order gates use only normalized
    economic actions. Raw scores, response lengths and outcomes never determine
    gate eligibility. Missing slots or labels raise IncompleteSamePrefixGroup;
    identity, slot or numerical errors raise ValueError and must not be treated
    as ordinary missing scientific support.
    """
    if len(rows) < ROLLOUT_N:
        raise IncompleteSamePrefixGroup("same-prefix gate requires all four original slots")
    if len(rows) > ROLLOUT_N:
        raise ValueError("same-prefix gate cannot accept extra slots")
    if any(not isinstance(row, SamePrefixGateSlot) for row in rows):
        raise ValueError("rows must be SamePrefixGateSlot instances")
    identities = set()
    slot_ids = []
    for row in rows:
        if not isinstance(row.anchor_id, str) or not row.anchor_id.strip():
            raise ValueError("anchor_id must be nonempty text")
        for name in ("prompt_sha256", "financial_state_sha256", "policy_snapshot_sha256"):
            _sha256(getattr(row, name), name)
        identities.add(
            (
                row.anchor_id,
                row.prompt_sha256,
                row.financial_state_sha256,
                row.policy_snapshot_sha256,
            )
        )
        if type(row.slot_id) is not int or row.slot_id not in range(ROLLOUT_N):
            raise ValueError("slot IDs must be integers 0,1,2,3")
        slot_ids.append(row.slot_id)
        if not isinstance(row.label_status, str) or row.label_status not in {COMPLETE_LABEL, "NA"}:
            raise ValueError("label_status must be COMPLETE_FIXED_HOLD or NA")
    if len(identities) != 1:
        raise ValueError("same-prefix rows must share anchor, prompt, financial state and policy")
    if set(slot_ids) != set(range(ROLLOUT_N)):
        raise ValueError("slot IDs must be unique 0,1,2,3")
    if any(row.label_status != COMPLETE_LABEL for row in rows):
        raise IncompleteSamePrefixGroup("partial labels cannot enter n=4 centering")
    scores = tuple(_finite_number(row.score, "score") for row in rows)
    keys = tuple(normalize_economic_order_key(row.economic_order_key) for row in rows)
    try:
        # This is the same raw score-centering convention as sibling_credit.
        mean = math.fsum(scores) / ROLLOUT_N
        plain = tuple(score - mean for score in scores)
    except OverflowError as error:
        raise ValueError("centered score exceeds the finite numerical domain") from error
    if not math.isfinite(mean) or not all(math.isfinite(value) for value in plain):
        raise ValueError("centered score exceeds the finite numerical domain")
    family_gate = len({key[0] for key in keys}) > 1
    order_gate = len(set(keys)) > 1
    zero = (0.0,) * ROLLOUT_N
    family = plain if family_gate else zero
    order = plain if order_gate else zero
    scaled = GateSignals(
        zero=zero,
        ungated_plain=tuple(ETA * value for value in plain),
        family_gate_plain=tuple(ETA * value for value in family),
        order_gate_plain=tuple(ETA * value for value in order),
    )
    return SamePrefixGateResult(
        zero=zero,
        ungated_plain=plain,
        family_gate_plain=family,
        order_gate_plain=order,
        scaled_eta025=scaled,
        normalized_order_keys=keys,
        family_gate=family_gate,
        order_gate=order_gate,
        mean_score=mean,
    )

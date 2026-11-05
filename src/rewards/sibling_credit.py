"""Strict session-aligned sibling residuals for the approved M2 screening.

This module is independent of the trainer and never rewrites rewards. Both
residuals use the same diversity gate; information and invalid steps get zero.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

ACTION_FAMILIES = frozenset({"buy", "sell", "hold", "abstain"})
OTHER_ACTIONS = frozenset({"retrieve", "compute", "invalid"})
ROLLOUT_N = 4
DECISION_SESSIONS = 20
DIAGNOSTIC_ATOL = 1e-8
DIAGNOSTIC_RTOL = 1e-6


@dataclass(frozen=True, slots=True)
class SiblingStep:
    """One real trainer row, including its completed trajectory's identity."""

    trajectory_uid: str
    group_uid: str
    episode_id: str
    policy_step: int
    rollout_id: int
    rollout_n: int
    step_index: int
    step_count: int
    decision_session_index: int
    result_session_index: int
    action: str
    invalid_action: bool
    done: bool
    episode_complete: bool
    terminal_score: float


@dataclass(frozen=True, slots=True)
class SiblingCredit:
    """Residuals and eligibility in the exact input-row order."""

    plain: tuple[float, ...]
    pooled: tuple[float, ...]
    advancing: tuple[bool, ...]
    diagnostics: dict[str, object]


def _validate_row(row: SiblingStep) -> None:
    for name in ("trajectory_uid", "group_uid", "episode_id"):
        if not isinstance(getattr(row, name), str) or not getattr(row, name):
            raise ValueError(f"{name} must be a nonempty string")
    for name in (
        "policy_step",
        "rollout_id",
        "rollout_n",
        "step_index",
        "step_count",
        "decision_session_index",
        "result_session_index",
    ):
        value = getattr(row, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name in ("invalid_action", "done", "episode_complete"):
        if type(getattr(row, name)) is not bool:
            raise ValueError(f"{name} must be bool")
    if row.rollout_n != ROLLOUT_N or row.rollout_id not in range(ROLLOUT_N):
        raise ValueError("screening requires four siblings with rollout IDs 0,1,2,3")
    if row.step_count < 1 or row.step_index >= row.step_count:
        raise ValueError("step_index is outside the declared complete trajectory")
    if not row.episode_complete:
        raise ValueError("screening cannot credit an incomplete trajectory")
    if not isinstance(row.action, str) or row.action not in ACTION_FAMILIES | OTHER_ACTIONS:
        raise ValueError(f"unknown trading action: {row.action!r}")
    if row.invalid_action != (row.action == "invalid"):
        raise ValueError("invalid-action flag does not match trading action")
    if isinstance(row.terminal_score, bool) or not math.isfinite(row.terminal_score):
        raise ValueError("terminal score must be finite")


def compute_sibling_credit(rows: Sequence[SiblingStep]) -> SiblingCredit:
    """Validate complete n=4 groups and compute plain and family-pooled S.

    At each advancing session j, let mean_Y be the four siblings' mean original
    financial outcome and D_j indicate at least two action families. Then plain
    is D_j * (Y_i - mean_Y); pooled replaces Y_i by its action-family mean.
    Both are zero on non-advancing rows. Neither includes eta or whitening;
    these are actor-only residuals added to the shared critic/GAE signal later.

    Validation is before arithmetic: no partial trajectory, missing sibling,
    cross-policy group, or duplicated step is repaired or silently discarded.
    Rows may have been permuted by the trainer's DP load balancing.
    """

    if not rows:
        raise ValueError("screening requires a nonempty logical rollout batch")
    trajectories: dict[str, list[int]] = defaultdict(list)
    groups: dict[str, dict[int, str]] = defaultdict(dict)
    group_context: dict[str, tuple[str, int]] = {}
    policy_steps = set()
    for index, row in enumerate(rows):
        _validate_row(row)
        trajectories[row.trajectory_uid].append(index)
        policy_steps.add(row.policy_step)
        context = (row.episode_id, row.policy_step)
        if group_context.setdefault(row.group_uid, context) != context:
            raise ValueError("sibling UID crosses episode occurrence or policy snapshot")
        old_uid = groups[row.group_uid].setdefault(row.rollout_id, row.trajectory_uid)
        if old_uid != row.trajectory_uid:
            raise ValueError("duplicate rollout ID inside sibling group")
    if len(policy_steps) != 1:
        raise ValueError("logical rollout batch mixes pre-update policy snapshots")
    for group in groups.values():
        if set(group) != set(range(ROLLOUT_N)):
            raise ValueError("missing sibling: each group must contain rollout IDs 0,1,2,3")

    session_groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    advancing = [False] * len(rows)
    for indices in trajectories.values():
        ordered = sorted(indices, key=lambda index: rows[index].step_index)
        first = rows[ordered[0]]
        identity = (
            first.group_uid,
            first.episode_id,
            first.policy_step,
            first.rollout_id,
            first.step_count,
            first.terminal_score,
        )
        if [rows[index].step_index for index in ordered] != list(range(first.step_count)):
            raise ValueError("trajectory step indices are missing, duplicated, or non-contiguous")
        session = 0
        for offset, index in enumerate(ordered):
            row = rows[index]
            if (
                row.group_uid,
                row.episode_id,
                row.policy_step,
                row.rollout_id,
                row.step_count,
                row.terminal_score,
            ) != identity:
                raise ValueError("trajectory identity or terminal outcome changes across rows")
            if row.done != (offset == len(ordered) - 1):
                raise ValueError("only the final retained step may be terminal")
            if row.decision_session_index != session or session >= DECISION_SESSIONS:
                raise ValueError("trajectory decision sessions are not contiguous")
            advancing[index] = row.action in ACTION_FAMILIES
            if advancing[index]:
                session_groups[(row.group_uid, session)].append(index)
                session += 1
            if row.result_session_index != session:
                raise ValueError("trading action does not match its session transition")
        if session != DECISION_SESSIONS or not advancing[ordered[-1]]:
            raise ValueError("complete 21-session trajectory requires 20 advancing decisions")

    plain = [0.0] * len(rows)
    pooled = [0.0] * len(rows)
    diverse_count = 0
    pooling_opportunities = 0
    histogram: Counter[str] = Counter()
    for indices in session_groups.values():
        if len(indices) != ROLLOUT_N or {rows[i].rollout_id for i in indices} != set(
            range(ROLLOUT_N)
        ):
            raise ValueError("session group must contain exactly one row per sibling")
        cells: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            cells[rows[index].action].append(index)
            histogram[rows[index].action] += 1
        if len(cells) == 1:
            continue
        diverse_count += 1
        pooling_opportunities += int(max(map(len, cells.values())) >= 2)
        group_mean = math.fsum(rows[i].terminal_score for i in indices) / ROLLOUT_N
        for cell in cells.values():
            cell_mean = math.fsum(rows[i].terminal_score for i in cell) / len(cell)
            for index in cell:
                plain[index] = rows[index].terminal_score - group_mean
                pooled[index] = cell_mean - group_mean
                if not math.isfinite(plain[index]) or not math.isfinite(pooled[index]):
                    raise ValueError("sibling residual overflowed its finite input domain")

    session_count = len(session_groups)
    advancing_count = sum(advancing)
    changed_count = sum(
        abs(pooled[i] - plain[i]) > DIAGNOSTIC_ATOL + DIAGNOSTIC_RTOL * abs(plain[i])
        for i in range(len(rows))
        if advancing[i]
    )
    return SiblingCredit(
        plain=tuple(plain),
        pooled=tuple(pooled),
        advancing=tuple(advancing),
        diagnostics={
            "policy_step": next(iter(policy_steps)),
            "episode_group_count": len(groups),
            "trajectory_count": len(trajectories),
            "real_step_count": len(rows),
            "advancing_row_count": advancing_count,
            "session_group_count": session_count,
            "action_diversity_rate": diverse_count / session_count,
            "pooling_opportunity_rate": pooling_opportunities / session_count,
            "pooling_changed_row_count": changed_count,
            "pooling_changed_row_rate": changed_count / advancing_count,
            "action_family_histogram": {
                action: histogram[action] for action in sorted(ACTION_FAMILIES)
            },
            "invalid_step_count": sum(row.invalid_action for row in rows),
            "diagnostic_atol": DIAGNOSTIC_ATOL,
            "diagnostic_rtol": DIAGNOSTIC_RTOL,
        },
    )

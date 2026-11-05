#!/usr/bin/env python3
"""Read-only-input CPU readout of fixed-state, fixed-endpoint order branches.

This script does not construct environments, run a model, or change a run's
evidence. It exclusively creates descriptive readout artifacts from a complete
state_results.jsonl. Deterministic identities are checks, not denoising evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

PROFILE = "m2_real_train_state_order_readout_v1"
SEEDS = (42, 43)
DAYS = (0, 5, 10, 15)
N = 4
TOL = 1e-9
METHODS = ("plain", "family_pool", "exact_same_gate", "exact_order_gate")
OUTPUTS = (
    "readout.json",
    "readout_metrics.csv",
    "readout_coverage.csv",
    "readout_algebra.csv",
    "readout_matched_rows.csv",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def close(left: float, right: float, name: str) -> None:
    if not math.isclose(left, right, abs_tol=TOL, rel_tol=1e-10):
        raise ValueError(f"{name} differs: {left!r} != {right!r}")


def center(values: np.ndarray) -> np.ndarray:
    return values - np.mean(values)


def pooled(values: np.ndarray, labels: list[object]) -> np.ndarray:
    """Keep the original global mean and the original four-slot weights."""
    means = {
        label: float(np.mean([v for v, key in zip(values, labels, strict=True) if key == label]))
        for label in set(labels)
    }
    return np.asarray([means[key] for key in labels]) - np.mean(values)


def predictions(
    values: np.ndarray,
    families: list[str],
    orders: list[str],
    states: list[str] | None = None,
) -> dict[str, np.ndarray]:
    family_gate = len(set(families)) > 1
    order_gate = len(set(orders)) > 1
    zero = np.zeros(N)
    result = {
        "zero": zero.copy(),
        "ungated_plain": center(values),
        "plain": center(values) if family_gate else zero.copy(),
        "family_pool": pooled(values, families) if family_gate else zero.copy(),
        "exact_same_gate": pooled(values, orders) if family_gate else zero.copy(),
        "exact_order_gate": pooled(values, orders) if order_gate else zero.copy(),
    }
    if states is not None:
        labels = list(zip(states, orders, strict=True))
        result["state_order"] = pooled(values, labels) if order_gate else zero.copy()
    return result


def rank_counts(predicted: np.ndarray, target: np.ndarray) -> dict[str, int]:
    result = {"rank_pairs": 0, "rank_correct": 0, "rank_wrong": 0, "rank_ties": 0}
    result["oracle_tied_pairs"] = 0
    for i, j in combinations(range(len(target)), 2):
        truth = float(target[i] - target[j])
        prediction = float(predicted[i] - predicted[j])
        if abs(truth) <= TOL:
            result["oracle_tied_pairs"] += 1
            continue
        result["rank_pairs"] += 1
        if abs(prediction) <= TOL:
            result["rank_ties"] += 1
        elif truth * prediction > 0:
            result["rank_correct"] += 1
        else:
            result["rank_wrong"] += 1
    return result


def metrics(predicted: np.ndarray, target: np.ndarray, *, rank: bool = True) -> dict[str, object]:
    if len(target) == 0:
        raise ValueError("cannot evaluate empty target support")
    error = predicted - target
    counts = (
        rank_counts(predicted, target)
        if rank
        else {
            "rank_pairs": 0,
            "rank_correct": 0,
            "rank_wrong": 0,
            "rank_ties": 0,
            "oracle_tied_pairs": 0,
        }
    )
    nonzero = np.abs(target) > TOL
    neutral = np.abs(predicted) <= TOL
    correct = nonzero & ~neutral & (predicted * target > 0)
    wrong = nonzero & ~neutral & (predicted * target < 0)
    denom = int(counts["rank_pairs"])
    return {
        "member_count": len(target),
        "squared_error_sum": float(np.dot(error, error)),
        "mse": float(np.mean(error**2)),
        "prediction_energy": float(np.mean(predicted**2)),
        "target_energy": float(np.mean(target**2)),
        "nonzero_target_count": int(np.sum(nonzero)),
        "sign_correct": int(np.sum(correct)),
        "sign_wrong": int(np.sum(wrong)),
        "sign_neutral": int(np.sum(nonzero & neutral)),
        "zero_target_count": int(np.sum(~nonzero)),
        "zero_target_nonzero_prediction": int(np.sum(~nonzero & ~neutral)),
        "predicted_all_zero": bool(np.all(neutral)),
        "rank_evaluated": rank,
        **counts,
        "rank_score": (
            (counts["rank_correct"] + 0.5 * counts["rank_ties"]) / denom if denom else None
        ),
    }


def unique_order_vectors(
    predicted: np.ndarray, target: np.ndarray, orders: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """An alternate descriptive rank view; unique orders are not independent n."""
    keep = [i for i, order in enumerate(orders) if order not in orders[:i]]
    for i, order in enumerate(orders):
        first = orders.index(order)
        close(float(target[i]), float(target[first]), "duplicate order oracle")
        close(float(predicted[i]), float(predicted[first]), "duplicate order prediction")
    return predicted[keep], target[keep]


def validate(rows: list[dict], *, require_design: bool = True) -> list[list[dict]]:
    grouped: dict[tuple[int, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        for name in ("group_id", "episode_id", "full_state_sha256"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(f"missing {name}")
        state = row["full_state_sha256"]
        if len(state) != 64 or any(c not in "0123456789abcdef" for c in state):
            raise ValueError("full_state_sha256 must be a lowercase SHA256")
        seed = integer(row.get("seed"), "seed")
        day = integer(row.get("decision_session_index"), "decision_session_index")
        sibling = integer(row.get("sibling_id"), "sibling_id")
        integer(row.get("policy_step"), "policy_step")
        if seed not in SEEDS or day not in DAYS or sibling not in range(N):
            raise ValueError("row is outside the fixed seed/day/sibling design")
        finite(row.get("source_Y"), "source_Y")
        hold = finite(row.get("hold_Y"), "hold_Y")
        cells = row.get("cells")
        if not isinstance(cells, list) or len(cells) != N:
            raise ValueError("each anchor must retain all four original slots")
        if {c.get("slot") for c in cells} != set(range(N)):
            raise ValueError("duplicate or missing candidate slot")
        cells.sort(key=lambda cell: cell["slot"])
        for cell in cells:
            if type(cell.get("slot")) is not int:
                raise ValueError("slot must be integer")
            key = cell.get("action_key")
            if not isinstance(key, str):
                raise ValueError("action_key must be canonical JSON text")
            parsed = json.loads(key)
            canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if key != canonical:
                raise ValueError("noncanonical action_key")
            if cell.get("family") not in {"buy", "sell", "hold", "abstain"}:
                raise ValueError("unknown advancing action family")
            if not isinstance(cell.get("branch_id"), str) or not cell["branch_id"]:
                raise ValueError("missing branch_id")
            status = cell.get("status")
            if status == "LEGAL_ADVANCING":
                y = finite(cell.get("Y"), "branch Y")
                q = finite(cell.get("q"), "branch q")
                close(q, y - hold, "q = Y - hold_Y")
                if type(cell.get("zero_fill")) is not bool:
                    raise ValueError("legal branch zero_fill must be bool")
                integer(cell.get("skipped_count"), "skipped_count")
                if finite(cell.get("first_cost"), "first_cost") < 0:
                    raise ValueError("negative first cost")
            elif status == "INADMISSIBLE_AT_DECISION":
                if cell.get("Y") is not None or cell.get("q") is not None:
                    raise ValueError("inadmissible branches cannot have financial labels")
            else:
                raise ValueError(f"unknown branch status {status!r}")
        if cells[sibling]["status"] != "LEGAL_ADVANCING":
            raise ValueError("the historical own-state diagonal must be legal")
        grouped[seed, row["group_id"], day].append(row)

    groups = []
    episodes: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    episode_policy: dict[tuple[int, str], int] = {}
    source_scores: dict[tuple[int, str, int, int], float] = {}
    for key, group in sorted(grouped.items()):
        if len(group) != N or {r["sibling_id"] for r in group} != set(range(N)):
            raise ValueError(f"missing or duplicate source sibling in {key}")
        group.sort(key=lambda row: row["sibling_id"])
        contexts = {(r["episode_id"], r["policy_step"]) for r in group}
        if len(contexts) != 1:
            raise ValueError("source group mixes episode or policy snapshots")
        orders = [(c["action_key"], c["family"]) for c in group[0]["cells"]]
        seen: dict[tuple[str, str], float] = {}
        status_seen: dict[tuple[str, str], str] = {}
        hold_by_state: dict[str, float] = {}
        for row in group:
            if [(c["action_key"], c["family"]) for c in row["cells"]] != orders:
                raise ValueError("candidate multiset changes across anchor states")
            state = row["full_state_sha256"]
            close(hold_by_state.setdefault(state, row["hold_Y"]), row["hold_Y"], "state hold")
            score_key = key[0], row["episode_id"], row["policy_step"], row["sibling_id"]
            close(
                source_scores.setdefault(score_key, row["source_Y"]),
                row["source_Y"],
                "historical terminal score across decision days",
            )
            for cell in row["cells"]:
                state_order = state, cell["action_key"]
                if status_seen.setdefault(state_order, cell["status"]) != cell["status"]:
                    raise ValueError("same-state same-order admissibility changes")
                if cell["status"] == "LEGAL_ADVANCING":
                    close(seen.setdefault(state_order, cell["Y"]), cell["Y"], "determinism")
        by_order: dict[str, str] = {}
        for order, family in orders:
            if by_order.setdefault(order, family) != family:
                raise ValueError("one concrete order has multiple families")
        if key[2] == 0 and len({r["full_state_sha256"] for r in group}) != 1:
            raise ValueError("session zero must be the common initial-state positive control")
        episode_key = key[0], group[0]["episode_id"]
        policy = group[0]["policy_step"]
        if episode_policy.setdefault(episode_key, policy) != policy:
            raise ValueError("one selected window appears under multiple policy snapshots")
        if key[2] in episodes[key[0]][group[0]["episode_id"]]:
            raise ValueError("duplicate source group for the same seed/window/day")
        episodes[key[0]][group[0]["episode_id"]].add(key[2])
        groups.append(group)
    if require_design:
        if len(rows) != 256 or len(groups) != 64 or set(episodes) != set(SEEDS):
            raise ValueError("fixed primary sample must contain 256 states / 64 source groups")
        for seed in SEEDS:
            if len(episodes[seed]) != 8 or any(v != set(DAYS) for v in episodes[seed].values()):
                raise ValueError("each seed must retain eight original windows and all four days")
    return groups


def context(group: list[dict]) -> dict:
    first = group[0]
    return {
        name: first[name]
        for name in ("seed", "episode_id", "policy_step", "group_id", "decision_session_index")
    }


def peer_counts(labels: list[object]) -> int:
    return sum(labels.count(label) > 1 for label in labels)


def strata(row: dict) -> tuple[tuple[str, str], ...]:
    return (
        ("overall", "all"),
        ("seed", str(row["seed"])),
        ("episode", f"{row['seed']}:{row['episode_id']}"),
        ("day", str(row["decision_session_index"])),
        ("seed_day", f"{row['seed']}:{row['decision_session_index']}"),
    )


def summarize_coverage(records: list[dict]) -> list[dict]:
    bins: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        for scope in strata(row):
            bins[scope].append(row)
    result = []
    for (scope, value), items in sorted(bins.items()):
        anchors = len(items)
        complete = sum(row["complete_anchor"] for row in items)
        legal = sum(row["legal_candidate_slots"] for row in items)
        source_groups = len(
            {(r["seed"], r["group_id"], r["decision_session_index"]) for r in items}
        )
        result.append(
            {
                "scope": scope,
                "scope_value": value,
                "source_states": anchors,
                "source_groups": source_groups,
                "candidate_slots": anchors * N,
                "legal_candidate_slots": legal,
                "legal_candidate_rate": legal / (anchors * N),
                "complete_anchors": complete,
                "complete_anchor_rate": complete / anchors,
                "all_16_legal_source_groups": sum(r["group_all_16_legal"] for r in items) // N,
                "source_state_peer_rows": sum(r["own_source_state_has_peer"] for r in items),
                "source_state_order_peer_rows": sum(
                    r["own_source_state_order_has_peer"] for r in items
                ),
                "family_gate_closed_source_groups": (
                    sum(not r["family_gate_open"] for r in items) // N
                ),
                "family_closed_order_open_source_groups": (
                    sum(not r["family_gate_open"] and r["order_gate_open"] for r in items) // N
                ),
            }
        )
    return result


def summarize_algebra(records: list[dict]) -> list[dict]:
    bins: dict[tuple[str, str], list[dict]] = defaultdict(list)
    fields = (
        "hold_energy",
        "q_energy",
        "continuation_energy",
        "twice_cov_hold_q",
        "twice_cov_hold_continuation",
        "twice_cov_q_continuation",
        "source_energy",
    )
    for row in records:
        for scope in strata(row):
            bins[scope].append(row)
    return [
        {
            "scope": scope,
            "scope_value": value,
            "source_groups": len(items),
            **{f"mean_{name}": float(np.mean([row[name] for row in items])) for name in fields},
        }
        for (scope, value), items in sorted(bins.items())
    ]


def summarize_metrics(records: list[dict]) -> list[dict]:
    bins: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        for scope, value in strata(row):
            bins[(row["analysis"], row["lane"], row["method"], scope, value)].append(row)
    result = []
    for (analysis, lane, method, scope, value), items in sorted(bins.items()):
        members = sum(row["member_count"] for row in items)
        pairs = sum(row["rank_pairs"] for row in items)
        nonzero = sum(row["nonzero_target_count"] for row in items)
        scored = [row["rank_score"] for row in items if row["rank_score"] is not None]
        ranked_units = sum(row["rank_evaluated"] for row in items)

        def total(key: str, rows=items):
            return sum(row[key] for row in rows)

        result.append(
            {
                "analysis": analysis,
                "lane": lane,
                "method": method,
                "scope": scope,
                "scope_value": value,
                "evaluation_units": len(items),
                "member_count": members,
                "mse_member_weighted": total("squared_error_sum") / members,
                "mse_unit_mean": float(np.mean([row["mse"] for row in items])),
                "mean_prediction_energy": float(np.mean([r["prediction_energy"] for r in items])),
                "mean_target_energy": float(np.mean([r["target_energy"] for r in items])),
                "rank_pairs": pairs,
                "rank_evaluated_units": ranked_units,
                "rank_units_with_nontied_oracle": len(scored),
                "rank_units_all_oracle_ties": ranked_units - len(scored),
                "rank_pair_weighted": (
                    (total("rank_correct") + 0.5 * total("rank_ties")) / pairs if pairs else None
                ),
                "rank_unit_mean": float(np.mean(scored)) if scored else None,
                "rank_correct": total("rank_correct"),
                "rank_wrong": total("rank_wrong"),
                "rank_prediction_ties": total("rank_ties"),
                "oracle_tied_pairs": total("oracle_tied_pairs"),
                "sign_nonzero_target_count": nonzero,
                "sign_correct": total("sign_correct"),
                "sign_wrong": total("sign_wrong"),
                "sign_neutral": total("sign_neutral"),
                "sign_correct_rate": total("sign_correct") / nonzero if nonzero else None,
                "zero_target_count": total("zero_target_count"),
                "zero_target_nonzero_prediction": total("zero_target_nonzero_prediction"),
                "all_zero_prediction_units": total("predicted_all_zero"),
            }
        )
    return result


def analyze(rows: list[dict], *, require_design: bool = True) -> dict:
    groups = validate(rows, require_design=require_design)
    records: list[dict] = []
    coverage: list[dict] = []
    algebra: list[dict] = []
    matched_rows: list[dict] = []
    decomposition: list[dict] = []
    identity_errors: dict[str, list[float]] = defaultdict(list)

    def add(ctx: dict, analysis: str, lane: str, method: str, pred, target, **extra) -> None:
        records.append(
            {
                **ctx,
                "analysis": analysis,
                "lane": lane,
                "method": method,
                **metrics(np.asarray(pred), np.asarray(target), **extra),
            }
        )

    for group in groups:
        ctx = context(group)
        orders = [cell["action_key"] for cell in group[0]["cells"]]
        families = [cell["family"] for cell in group[0]["cells"]]
        states = [row["full_state_sha256"] for row in group]
        complete = np.asarray(
            [all(c["status"] == "LEGAL_ADVANCING" for c in row["cells"]) for row in group]
        )
        exact_peers = peer_counts(orders)
        state_order_peers = peer_counts(list(zip(states, orders, strict=True)))
        state_peers = peer_counts(states)
        for row in group:
            count = sum(c["status"] == "LEGAL_ADVANCING" for c in row["cells"])
            legal = [c for c in row["cells"] if c["status"] == "LEGAL_ADVANCING"]
            coverage.append(
                {
                    **ctx,
                    "sibling_id": row["sibling_id"],
                    "candidate_slots": N,
                    "legal_candidate_slots": count,
                    "inadmissible_candidate_slots": N - count,
                    "complete_anchor": count == N,
                    "group_all_16_legal": bool(np.all(complete)),
                    "family_gate_open": len(set(families)) > 1,
                    "order_gate_open": len(set(orders)) > 1,
                    "unique_orders": len(set(orders)),
                    "family_peer_slots": peer_counts(families),
                    "exact_order_peer_slots": exact_peers,
                    "source_state_peer_slots": state_peers,
                    "source_state_order_peer_slots": state_order_peers,
                    "own_source_state_has_peer": states.count(row["full_state_sha256"]) > 1,
                    "own_source_state_order_has_peer": (
                        list(zip(states, orders, strict=True)).count(
                            (row["full_state_sha256"], orders[row["sibling_id"]])
                        )
                        > 1
                    ),
                    "legal_zero_fill_slots": sum(c["zero_fill"] for c in legal),
                    "legal_skipped_slots": sum(c["skipped_count"] > 0 for c in legal),
                    "legal_positive_first_cost_slots": sum(c["first_cost"] > 0 for c in legal),
                }
            )
            if count != N:
                continue
            y = np.asarray([c["Y"] for c in row["cells"]], dtype=float)
            q = np.asarray([c["q"] for c in row["cells"]], dtype=float)
            target = center(q)
            preds = predictions(y, families, orders)
            identity_errors["same_state_ungated_plain_equals_oracle"].append(
                float(np.max(np.abs(preds["ungated_plain"] - target)))
            )
            identity_errors["same_state_exact_same_gate_equals_gated_plain"].append(
                float(np.max(np.abs(preds["exact_same_gate"] - preds["plain"])))
            )
            identity_errors["same_state_exact_order_gate_equals_oracle"].append(
                float(np.max(np.abs(preds["exact_order_gate"] - target)))
            )
            for method, pred in preds.items():
                anchor_ctx = {**ctx, "sibling_id": row["sibling_id"]}
                add(anchor_ctx, "same_anchor_oracle", "four_original_slots", method, pred, target)
                dedup_pred, dedup_target = unique_order_vectors(pred, target, orders)
                add(
                    anchor_ctx,
                    "same_anchor_oracle",
                    "unique_order_alternate",
                    method,
                    dedup_pred,
                    dedup_target,
                )

        source = np.asarray([row["source_Y"] for row in group], dtype=float)
        hold = np.asarray([row["hold_Y"] for row in group], dtype=float)
        a = np.asarray([row["cells"][i]["Y"] for i, row in enumerate(group)], dtype=float)
        qdiag = np.asarray([row["cells"][i]["q"] for i, row in enumerate(group)], dtype=float)
        future = source - a
        source_c, hold_c, q_c, future_c = map(center, (source, hold, qdiag, future))
        terms = {
            "hold_energy": float(np.mean(hold_c**2)),
            "q_energy": float(np.mean(q_c**2)),
            "continuation_energy": float(np.mean(future_c**2)),
            "twice_cov_hold_q": float(2 * np.mean(hold_c * q_c)),
            "twice_cov_hold_continuation": float(2 * np.mean(hold_c * future_c)),
            "twice_cov_q_continuation": float(2 * np.mean(q_c * future_c)),
        }
        source_energy = float(np.mean(source_c**2))
        algebra.append(
            {
                **ctx,
                **terms,
                "source_energy": source_energy,
                "component_energy_sum": sum(terms.values()),
                "energy_identity_error": source_energy - sum(terms.values()),
                "source_equals_hold_q_continuation_max_error": float(
                    np.max(np.abs(source - hold - qdiag - future))
                ),
                "A_equals_hold_q_max_error": float(np.max(np.abs(a - hold - qdiag))),
                "complete_anchor_count": int(np.sum(complete)),
                "all_16_legal": bool(np.all(complete)),
            }
        )
        identity_errors["source_component_energy"].append(abs(source_energy - sum(terms.values())))
        lanes = {
            "historical_source_Y": source,
            "fixed_hold_A": a,
            "fixedhold_minus_oracle_hold": qdiag,
            "original_minus_oracle_hold": source - hold,
        }
        all_predictions = {
            lane: predictions(values, families, orders, states) for lane, values in lanes.items()
        }
        tau = np.full(N, np.nan)
        for i, row in enumerate(group):
            if complete[i]:
                tau[i] = qdiag[i] - np.mean([c["q"] for c in row["cells"]])
            matched_rows.append(
                {
                    **ctx,
                    "sibling_id": i,
                    "four_candidate_support": bool(complete[i]),
                    "tau": float(tau[i]) if complete[i] else None,
                    "centered_diagonal_q": float(q_c[i]),
                    "hold_Y": float(hold[i]),
                    "fixed_hold_A": float(a[i]),
                    "diagonal_q": float(qdiag[i]),
                    "historical_source_Y": float(source[i]),
                    "continuation_difference": float(future[i]),
                }
            )
        for lane, preds in all_predictions.items():
            for method in ("zero", *METHODS, "state_order"):
                pred = preds[method]
                add(ctx, "diagonal_component_concordance", lane, method, pred, q_c)
                if np.any(complete):
                    add(
                        ctx,
                        "matched_tau_supported_rows",
                        lane,
                        method,
                        pred[complete],
                        tau[complete],
                        rank=False,
                    )
                if np.all(complete):
                    tau_c = center(tau)
                    add(ctx, "matched_centered_tau_full_groups", lane, method, pred, tau_c)
                    mean_term = float(np.mean(tau) ** 2)
                    mse_tau = float(np.mean((pred - tau) ** 2))
                    mse_centered = float(np.mean((pred - tau_c) ** 2))
                    error = mse_tau - mse_centered - mean_term
                    identity_errors["matched_tau_centering_decomposition"].append(abs(error))
                    decomposition.append(
                        {
                            **ctx,
                            "lane": lane,
                            "method": method,
                            "mse_raw_tau": mse_tau,
                            "mse_centered_tau": mse_centered,
                            "mean_tau_squared": mean_term,
                            "identity_error": error,
                        }
                    )

    state_count = len(rows)
    complete_count = sum(row["complete_anchor"] for row in coverage)
    full_groups = sum(row["all_16_legal"] for row in algebra)
    identity_summary = {
        name: {"checks": len(values), "max_abs_error": max(values) if values else None}
        for name, values in sorted(identity_errors.items())
    }
    for name, item in identity_summary.items():
        if item["max_abs_error"] is not None and item["max_abs_error"] > 1e-7:
            raise ValueError(f"identity failed: {name}: {item}")
    return {
        "profile": PROFILE,
        "status": "DESCRIPTIVE_READOUT_COMPLETE",
        "sample": {
            "source_groups": len(groups),
            "source_states": state_count,
            "candidate_slots": state_count * N,
            "legal_candidate_slots": sum(r["legal_candidate_slots"] for r in coverage),
            "inadmissible_candidate_slots": sum(
                r["inadmissible_candidate_slots"] for r in coverage
            ),
            "complete_four_candidate_anchors": complete_count,
            "complete_anchor_rate_all_states": complete_count / state_count,
            "all_16_legal_source_groups": full_groups,
            "all_16_legal_rate_all_groups": full_groups / len(groups),
            "matched_tau_supported_rows": complete_count,
            "family_gate_open_source_groups": sum(
                len({c["family"] for c in group[0]["cells"]}) > 1 for group in groups
            ),
            "order_gate_open_source_groups": sum(
                len({c["action_key"] for c in group[0]["cells"]}) > 1 for group in groups
            ),
            "source_state_peer_rows": sum(r["own_source_state_has_peer"] for r in coverage),
            "source_state_order_peer_rows": sum(
                r["own_source_state_order_has_peer"] for r in coverage
            ),
            "source_state_peer_rate_all_states": (
                sum(r["own_source_state_has_peer"] for r in coverage) / state_count
            ),
            "source_state_order_peer_rate_all_states": (
                sum(r["own_source_state_order_has_peer"] for r in coverage) / state_count
            ),
        },
        "definitions": {
            "H": "hold_Y, using the complete score-relevant prefix and original window endpoint",
            "A": "own-state diagonal Y after original order, then explicit hold suffix",
            "q": "A-H; q(s,a) is deterministic fixed-hold counterfactual increment",
            "F": "historical source_Y-A; continuation-target difference, not identified noise",
            "same_anchor_target": "q(s,a_slot)-mean_four_original_slots q(s,a_slot)",
            "component_target": "H4 diag(q), cross-state component concordance only",
            "matched_tau": "q(s_i,a_i)-mean_four_original_slots q(s_i,a_slot)",
            "matched_centered_target": "H4 tau, only when all 16 candidates are legal",
            "state_order": "pool by (full_state_sha256,action_key), global mean, exact-order gate",
            "gate": (
                "plain/family_pool/exact_same_gate: family diversity; "
                "exact_order_gate: order diversity"
            ),
            "units": "original terminal score units; MSE uses squared score units",
            "rank": "oracle ties excluded, predicted ties score one half, all oracle ties = NA",
            "unique_order_alternate": (
                "first distinct order only within an anchor; no independent-n claim"
            ),
            "support": "no target for an incomplete four-candidate row; no masked recentering",
            "fixedhold_minus_oracle_hold": (
                "q=A-H: privileged future hold baseline, fixed-hold continuation, not deployable"
            ),
            "original_minus_oracle_hold": (
                "source_Y-H=q+F: privileged hold baseline, original continuation retained, "
                "not deployable and no guaranteed improvement"
            ),
            "matched_tau_supported_rows": (
                "supplemental supported-row descriptive errors, no ranking or recentering; "
                "the primary four-dimensional tau/Htau decomposition uses full-16 groups only"
            ),
        },
        "limits": [
            "This audits deterministic structure, bias, support and target sensitivity; "
            "not denoising gains.",
            "Ungated plain and exact-order pooling equal the same-state oracle by algebra.",
            "Historical source_Y uses another continuation than the fixed-hold target.",
            "A q input lane uses unavailable future hold information; "
            "no pooling credit is assigned to it.",
            "No oracle reference is claimed to guarantee better MSE for every target.",
            "States/days/slots within a training episode are dependent; "
            "no p-values or independent-n claims.",
            "Partial-support tau errors use the same supported rows for every method, "
            "without recentering.",
            "Legal next-open skips remain legal; inadmissible orders are not replaced by hold.",
        ],
        "identity_checks": identity_summary,
        "metrics": summarize_metrics(records),
        "coverage_summary": summarize_coverage(coverage),
        "component_algebra_summary": summarize_algebra(algebra),
        "coverage_by_anchor": coverage,
        "component_algebra_by_group": algebra,
        "matched_tau_by_source_row": matched_rows,
        "matched_tau_decomposition_full_groups": decomposition,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty CSV {path.name}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def readout(run: Path) -> dict:
    run = run.resolve(strict=True)
    source = run / "state_results.jsonl"
    source_hash = sha(source)
    for name in OUTPUTS:
        if (run / name).exists():
            raise FileExistsError(f"readout output already exists: {run / name}")
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank input line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"input line {line_number} must be an object")
            rows.append(row)
    result = analyze(rows)
    if sha(source) != source_hash:
        raise RuntimeError("state_results.jsonl changed while reading")
    result["input"] = {
        "path": str(source),
        "sha256": source_hash,
        "records": len(rows),
        "readout_source": str(Path(__file__).resolve()),
        "readout_source_sha256": sha(Path(__file__).resolve()),
    }
    tables = {
        "readout_metrics.csv": result["metrics"],
        "readout_coverage.csv": result["coverage_by_anchor"],
        "readout_algebra.csv": result["component_algebra_by_group"],
        "readout_matched_rows.csv": result["matched_tau_by_source_row"],
    }
    for name, table in tables.items():
        write_csv(run / name, table)
    result["output_csv_sha256"] = {name: sha(run / name) for name in tables}
    with (run / "readout.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    result = readout(args.run)
    print(json.dumps({"status": result["status"], "sample": result["sample"]}, allow_nan=False))


if __name__ == "__main__":
    main()

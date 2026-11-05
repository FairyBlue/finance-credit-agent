"""Pure same-prefix local corrections on original complete-parent RLOO rows.

EST continuations provide terminal-Y request returns; they are never actor rows.
Missing support zeros only the local correction and never fills an EST outcome.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

from src.agents.same_prefix_online_train import economic_key, finite, records
from src.agents.same_prefix_training_demo import require


def slot_credit(values: list[float], orders: list[tuple]) -> dict:
    """Four native slots retain their multiplicity in both arm baselines."""
    require(
        len(values) == len(orders) == 4 and all(finite(v) for v in values),
        "four finite original slot returns required",
    )
    average = math.fsum(values) / 4
    peers = [q for q, order in zip(values, orders, strict=True) if order == orders[0]]
    diverse = len(set(orders)) > 1
    return dict(
        slot_mean_Q=average,
        slot0_order_mean_Q=math.fsum(peers) / len(peers),
        plain_S=values[0] - average if diverse else 0.0,
        exact_S=math.fsum(peers) / len(peers) - average if diverse else 0.0,
        order_gate=diverse,
        distinct_orders=len(set(orders)),
        duplicate_slots=4 - len(set(orders)),
        slot0_order_slots=len(peers),
    )


def apply_credit(
    batch: list[dict],
    episodes: list[str],
    snapshots: list[dict],
    heads: list[dict],
    estimates: list[dict],
    arm: str,
    alpha: float = 0.5,
    eta: float = 0.25,
    penalty: float = 0.05,
) -> dict:
    require(arm in {"plain", "exact"}, "local credit arm")
    require(alpha == 0.5 and eta == 0.25 and penalty == 0.05, "fixed demo recipe")
    parents = {(r["trajectory_key"], r["request_index"]): r for r in batch}
    for row in batch:
        row.update(
            raw_RLOO=row["unwhitened_advantage"],
            local_credit_S=0.0,
            local_credit_status="NOT_ANCHOR",
            local_credit_arm=arm,
            local_credit_alpha=alpha,
            local_credit_eta=eta,
            local_credit_coefficient=alpha * eta,
            local_credit_adjustment=0.0,
        )
    anchor_map, anchor_ids = {}, set()
    for snapshot in snapshots:
        key = (snapshot["episode_id"], snapshot["session_index"])
        require(
            key not in anchor_map and snapshot["anchor_id"] not in anchor_ids,
            "duplicate local anchor identity",
        )
        anchor_map[key] = snapshot
        anchor_ids.add(snapshot["anchor_id"])
    head_map, est_map = {}, {}
    for head in heads:
        key = (head["anchor_id"], head["slot_id"])
        require(
            key not in head_map and type(key[1]) is int and 0 <= key[1] < 4,
            "duplicate/unexpected local head slot",
        )
        head_map[key] = head
    for estimate in estimates:
        if estimate.get("role") not in {"estimate", "est"}:
            continue
        key = (estimate["anchor_id"], estimate["slot_id"], estimate["estimate_replicate"])
        require(
            key not in est_map
            and type(key[1]) is int
            and 0 <= key[1] < 4
            and type(key[2]) is int
            and key[2] in (0, 1),
            "duplicate/unexpected local EST",
        )
        est_map[key] = estimate
    planned = {(episode, session) for episode in episodes for session in (5, 15)}
    require(
        set(anchor_map) <= planned and {key[0] for key in est_map} <= anchor_ids,
        "unexpected local anchor",
    )
    diagnostics, affected = [], set()
    for episode in episodes:
        for session in (5, 15):
            snap = anchor_map.get((episode, session))
            if snap is not None and snap.get("parent_request_index") is None:
                snap = None
            anchor = snap["anchor_id"] if snap is not None else f"{episode}/j{session}/MISSING"
            diagnostic = dict(
                anchor_id=anchor,
                episode_id=episode,
                decision_session_index=session,
                arm=arm,
                alpha=alpha,
                eta=eta,
                planned_slots=4,
                planned_EST=8,
                supported=False,
                status="NA",
                S=None,
                applied_S=0.0,
                reasons=[],
            )
            diagnostics.append(diagnostic)
            reasons = diagnostic["reasons"]
            target = None
            if snap is None:
                reasons.append("missing-anchor")
            else:
                parent_key, index = snap["parent_trajectory_key"], snap["parent_request_index"]
                require(type(index) is int and index >= 0, "parent request index")
                diagnostic.update(parent_trajectory_key=parent_key, parent_request_index=index)
                target = parents.get((parent_key, index))
                if target is None:
                    reasons.append("parent-window-not-admitted")
                else:
                    require(
                        target["episode_id"] == episode
                        and target["replicate_index"] == 0
                        and target["decision_session_index"] == session
                        and target["prompt_token_ids"] == snap["prompt_token_ids"],
                        "local anchor must be original replica0 request at exact prompt",
                    )
                    identity = (parent_key, index)
                    require(identity not in affected, "multiple anchors on same parent request")
                    affected.add(identity)
            members, values, orders, surfaces = [], [], [], []
            supported_slots = 0
            for slot in range(4):
                head = head_map.get((anchor, slot))
                if head is None:
                    reasons.append(f"slot{slot}-missing-head")
                    continue
                members.append(head)
                if snap is not None:
                    require(
                        head.get("parent_trajectory_key") == snap["parent_trajectory_key"]
                        and head.get("parent_request_index") == snap["parent_request_index"],
                        "local head parent identity changed",
                    )
                    if head.get("prompt_token_ids"):
                        require(
                            head["prompt_token_ids"] == snap["prompt_token_ids"],
                            "all native heads share original visible prompt",
                        )
                if slot == 0 and target is not None and head.get("response_token_ids"):
                    require(
                        head["response_token_ids"] == target["response_token_ids"]
                        and head["prompt_token_ids"] == target["prompt_token_ids"],
                        "slot0 must be actual original parent request",
                    )
                if head.get("status") != "ADVANCING" or head.get("legal_head") is not True:
                    reasons.append(f"slot{slot}-illegal-head")
                    continue
                draws = [est_map.get((anchor, slot, rep)) for rep in (0, 1)]
                if any(draw is None or draw.get("status") != "COMPLETE" for draw in draws):
                    reasons.append(f"slot{slot}-incomplete-EST")
                    continue
                returns = []
                for draw in draws:
                    score, invalids = (
                        draw.get("terminal_financial_Y", draw.get("Y")),
                        draw.get("remaining_invalid_count"),
                    )
                    require(
                        finite(score) and type(invalids) is int and invalids >= 0,
                        "complete EST requires finite terminal Y and invalid count",
                    )
                    if "timesteps" in draw:
                        require(
                            sum(bool(s["invalid"]) for s in draw["timesteps"]) == invalids,
                            "EST remaining invalid count differs",
                        )
                        require(
                            bool(draw["timesteps"])
                            and draw["timesteps"][0]["response_token_ids"]
                            == head["response_token_ids"],
                            "EST replays declared head",
                        )
                    returns.append(float(score) - penalty * invalids)
                supported_slots += 1
                values.append(math.fsum(returns) / 2)
                orders.append(economic_key(head["economic_order_key"]))
                surfaces.append(tuple(head["response_token_ids"]))
                diagnostic.setdefault("slots", []).append(
                    dict(
                        slot_id=slot,
                        economic_order_key=head["economic_order_key"],
                        EST_returns=returns,
                        mean_Q=values[-1],
                    )
                )
            diagnostic["supported_slots"] = supported_slots
            if reasons or supported_slots != 4:
                if target is not None:
                    target.update(local_credit_status="NA", local_credit_anchor_id=anchor)
                continue
            stats = slot_credit(values, orders)
            fragments = defaultdict(set)
            for order, surface in zip(orders, surfaces, strict=True):
                fragments[order].add(surface)
            correction = stats[f"{arm}_S"]
            diagnostic.update(
                **stats,
                supported=True,
                status="COMPLETE",
                S=correction,
                applied_S=correction,
                distinct_native_fragments=len(set(surfaces)),
                orders_with_multiple_fragments=sum(len(v) > 1 for v in fragments.values()),
                extra_order_fragments=sum(max(0, len(v) - 1) for v in fragments.values()),
            )
            target.update(
                local_credit_S=correction,
                local_credit_status="COMPLETE",
                local_credit_anchor_id=anchor,
                local_credit_adjustment=alpha * eta * correction,
                unwhitened_advantage=target["raw_RLOO"] + alpha * eta * correction,
            )
    return dict(
        arm=arm,
        alpha=alpha,
        eta=eta,
        coefficient=alpha * eta,
        Q_definition="mean_EST(terminal_financial_Y - 0.05 * remaining_invalid_count)",
        credit_definition="slot0_Q - native_four_slot_mean_Q"
        if arm == "plain"
        else "slot0_economic_order_pooled_mean_Q - native_four_slot_mean_Q",
        planned_anchors=len(planned),
        supported_anchors=sum(d["supported"] for d in diagnostics),
        NA_anchors=sum(not d["supported"] for d in diagnostics),
        diverse_supported_anchors=sum(d.get("order_gate", False) for d in diagnostics),
        duplicate_order_anchors=sum(d.get("duplicate_slots", 0) > 0 for d in diagnostics),
        nonzero_parent_rows=sum(row["local_credit_S"] != 0 for row in batch),
        mean_abs_applied_credit=(
            math.fsum(abs(row["local_credit_adjustment"]) for row in batch) / len(planned)
        ),
        row_status_counts=dict(Counter(row["local_credit_status"] for row in batch)),
        parent_only_actor_rows=True,
        EST_actor_rows=False,
        anchors=diagnostics,
    )


def apply_from_directory(
    batch: list[dict], episodes: list[str], directory: Path, arm: str, alpha: float, eta: float
) -> dict:
    def optional(name):
        path = directory / name
        return records(path) if path.exists() else []

    return apply_credit(
        batch,
        episodes,
        optional("anchors.jsonl"),
        optional("heads.jsonl"),
        optional("trajectory_results.jsonl"),
        arm,
        alpha,
        eta,
    )

"""Replace selected RLOO requests with fixed-action double-return advantages.

All actor examples remain the genuine parent samples. Auxiliary continuations
only supply Q estimates; the original full-request whitening and loss weights
are applied before the calculation/parse-failure actor mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

from src.agents import mandate_repair_train as base
from src.agents.same_prefix_online_collect import canonical

ACTOR_UPDATE_SCOPE = "calculate_and_parse_failures"
ESTIMATOR = "parent_plus_independent_post_action_return_mean_v1"
_BASE_BUILD_BATCH = base.build_batch
_BASE_RECIPE = base.recipe


def _history_identity(row):
    return hashlib.sha256(canonical(row["messages"]).encode()).hexdigest()


def _validated_records(path, admitted, collection):
    records = {}
    for record in base.read_rows(path):
        key = record.get("trajectory_key")
        base.require(key in admitted, "action Q record belongs to an admitted actual parent")
        base.require(key not in records, "unique action Q record per parent")
        parent = admitted[key]["parent"]
        base.require(record.get("task_id") == parent["task_id"], "action Q task identity")
        records[key] = record
        if record.get("status") != "COMPLETE":
            continue
        request_map = {r["request_key"]: r for r in admitted[key]["requests"]}
        anchor = record.get("parent_request_key")
        base.require(anchor in request_map, "action Q identifies a genuine parent request")
        base.require(
            record.get("adapter_sha256") == collection.get("adapter_sha256"),
            "action Q continuation uses the sampled parent adapter",
        )
        base.require(record.get("estimator") == ESTIMATOR, "fixed-action Q estimator identity")
        base.require(record.get("m") == 2, "exactly parent plus one auxiliary return")
        base.require(
            record.get("parent_action_executed_once") is True
            and record.get("auxiliary_only") is True
            and record.get("continuation_policy") == "current_policy",
            "fixed actual action and auxiliary-only current-policy continuation required",
        )
        base.require(
            all(
                base.finite(record.get(k)) for k in ("parent_return", "auxiliary_return", "q_mean")
            ),
            "finite parent and auxiliary returns and Q mean",
        )
        base.require(
            math.isclose(record["parent_return"], parent["terminal_reward"], abs_tol=1e-10),
            "action Q parent return matches the actual parent outcome",
        )
        expected = (record["parent_return"] + record["auxiliary_return"]) / 2
        base.require(
            math.isclose(record["q_mean"], expected, rel_tol=1e-10, abs_tol=1e-10),
            "action Q mean is the parent plus auxiliary arithmetic mean",
        )
        if record.get("prefix_history_sha256") is not None:
            base.require(
                record["prefix_history_sha256"] == _history_identity(request_map[anchor]),
                "action Q prefix history matches the actual parent messages",
            )
    return records


def build_batch(
    directory,
    arm="rloo",
    local_credit=None,
    local_weight=0.25,
    local_credit_target=None,
    credit_shuffle_seed=42,
    mc_baselines=None,
    *,
    action_q_values,
):
    base.require(
        arm == "rloo" and local_credit is None and mc_baselines is None,
        "action Q only supports RLOO without other credit or state-baseline replacements",
    )
    directory, action_q_values = Path(directory), Path(action_q_values)
    admitted, metadata = _BASE_BUILD_BATCH(
        directory,
        arm,
        local_credit,
        local_weight,
        local_credit_target,
        credit_shuffle_seed=credit_shuffle_seed,
        mc_baselines=mc_baselines,
    )
    collection = json.loads((directory / "collection_started.json").read_text())
    base.require(
        isinstance(collection.get("adapter_sha256"), str)
        and len(collection["adapter_sha256"]) == 64,
        "parent collection declares its actual adapter identity",
    )
    records = _validated_records(action_q_values, admitted, collection)
    groups = defaultdict(list)
    for key, group in admitted.items():
        groups[group["parent"]["task_id"]].append(key)
        for row in group["requests"]:
            row["raw_rloo_advantage"] = row["raw_combined_advantage"]
            row["action_q_used"] = False
    replaced, changed, zero_turned_on, sign_changes = 0, 0, 0, 0
    delta_mass = []
    complete_tasks, unsupported = [], []
    for task_id, keys in groups.items():
        supported = len(keys) == 4 and all(
            key in records and records[key].get("status") == "COMPLETE" for key in keys
        )
        if not supported:
            unsupported.append(task_id)
            continue
        anchors = {
            key: next(
                r
                for r in admitted[key]["requests"]
                if r["request_key"] == records[key]["parent_request_key"]
            )
            for key in keys
        }
        if len({_history_identity(row) for row in anchors.values()}) != 1:
            unsupported.append(task_id)
            continue
        total = math.fsum(records[key]["q_mean"] for key in keys)
        for key, row in anchors.items():
            q = records[key]["q_mean"]
            new_raw = q - (total - q) / 3
            old_raw = row["raw_combined_advantage"]
            row.update(
                raw_combined_advantage=new_raw,
                action_q_used=True,
                action_q_mean=q,
                action_q_parent_return=records[key]["parent_return"],
                action_q_auxiliary_return=records[key]["auxiliary_return"],
            )
            replaced += 1
            changed += abs(new_raw - old_raw) > 1e-12
            zero_turned_on += abs(old_raw) <= 1e-12 < abs(new_raw)
            sign_changes += old_raw * new_raw < 0
            delta_mass.append(abs(new_raw - old_raw))
        complete_tasks.append(task_id)
    all_rows = [r for group in admitted.values() for r in group["requests"]]
    mean = math.fsum(r["raw_combined_advantage"] for r in all_rows) / len(all_rows)
    std = math.sqrt(
        math.fsum((r["raw_combined_advantage"] - mean) ** 2 for r in all_rows) / len(all_rows)
    )
    counts = dict(calculation=0, parse_failure=0, suppressed=0, positive=0, negative=0, zero=0)
    for group in admitted.values():
        for row in group["requests"]:
            unmasked = (row["raw_combined_advantage"] - mean) / std if std > 1e-12 else 0.0
            action = row.get("action")
            calculation = isinstance(action, dict) and action.get("name") == "calculate"
            invalid = action is None
            eligible = calculation or invalid
            row.update(
                advantage=unmasked if eligible else 0.0,
                unmasked_actor_advantage=unmasked,
                actor_update_scope=ACTOR_UPDATE_SCOPE,
                actor_update_eligible=eligible,
                full_parent_request_count=len(group["requests"]),
            )
            counts["calculation"] += calculation
            counts["parse_failure"] += invalid
            counts["suppressed"] += not eligible
            if invalid:
                counts["positive" if unmasked > 0 else "negative" if unmasked < 0 else "zero"] += 1
    original_identity = metadata["data_sha256"]
    metadata.update(
        data_sha256=hashlib.sha256(
            (original_identity + base.digest(action_q_values)).encode()
        ).hexdigest(),
        original_parent_data_sha256=original_identity,
        action_q_values_path=str(action_q_values.resolve()),
        action_q_values_sha256=base.digest(action_q_values),
        action_advantage_estimator=ESTIMATOR,
        actor_update_scope=ACTOR_UPDATE_SCOPE,
        original_rloo_advantages_mean=metadata["advantages_mean"],
        original_rloo_advantages_std=metadata["advantages_std"],
        advantages_mean=mean,
        advantages_std=std,
        global_advantage_renormalized=True,
        other_raw_advantages_unchanged=True,
        complete_q_task_groups=len(complete_tasks),
        complete_q_task_ids=complete_tasks,
        unsupported_q_task_ids=unsupported,
        replaced_target_requests=replaced,
        changed_target_raw_advantages=changed,
        zero_raw_to_nonzero_targets=zero_turned_on,
        target_raw_sign_changes=sign_changes,
        target_raw_advantage_l1_change=math.fsum(delta_mass),
        action_q_record_count=len(records),
        auxiliary_actor_training_examples=0,
        effective_advantage_requests=sum(r["advantage"] != 0 for r in all_rows),
        eligible_calculation_requests=counts["calculation"],
        eligible_parse_failure_requests=counts["parse_failure"],
        suppressed_other_requests=counts["suppressed"],
        parse_failure_positive_actor_requests=counts["positive"],
        parse_failure_negative_actor_requests=counts["negative"],
        parse_failure_zero_actor_requests=counts["zero"],
        actor_loss_weight_rule="1 / (selected_parent_count * full_parent_request_count)",
    )
    return admitted, metadata


def recipe(args):
    base.require(args.command == "rl" and args.arm == "rloo", "action Q requires the RLOO command")
    return dict(
        _BASE_RECIPE(args),
        actor_update_scope=ACTOR_UPDATE_SCOPE,
        action_advantage_estimator=ESTIMATOR,
    )


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "rl":
        raise SystemExit("Use financial_action_q_train rl with --action-q-values PATH")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--action-q-values", type=Path, required=True)
    args, remaining = parser.parse_known_args(sys.argv[1:])
    previous_batch, previous_recipe, previous_argv = base.build_batch, base.recipe, sys.argv
    base.build_batch = partial(build_batch, action_q_values=args.action_q_values.resolve())
    base.recipe = recipe
    sys.argv = [sys.argv[0], *remaining]
    try:
        base.main()
    finally:
        base.build_batch, base.recipe, sys.argv = previous_batch, previous_recipe, previous_argv


if __name__ == "__main__":
    main()

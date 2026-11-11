"""Restrict the existing RLOO actor loss to genuine calculation responses.

The original K4 advantages and request normalization are computed before the
mask. Full parent histories and request counts remain available to the existing
trainer; this is a change of actor update scope, not a new advantage estimator.
"""

from __future__ import annotations

import math
import sys

from src.agents import mandate_repair_train as base

ACTOR_UPDATE_SCOPE = "calculate_only"
_BASE_BUILD_BATCH = base.build_batch
_BASE_RECIPE = base.recipe


def build_batch(
    directory,
    arm="rloo",
    local_credit=None,
    local_weight=0.25,
    local_credit_target=None,
    credit_shuffle_seed=42,
    mc_baselines=None,
):
    base.require(
        arm == "rloo" and local_credit is None and mc_baselines is None,
        "calculation focus only supports ordinary RLOO without extra credit records",
    )
    admitted, metadata = _BASE_BUILD_BATCH(
        directory,
        arm,
        local_credit,
        local_weight,
        local_credit_target,
        credit_shuffle_seed=credit_shuffle_seed,
        mc_baselines=mc_baselines,
    )
    counts = dict(eligible=0, suppressed=0, suppressed_nonzero=0, eligible_raw_nonzero=0)
    before_mass, after_mass = [], []
    full_counts = {}
    for key, group in admitted.items():
        requests = group["requests"]
        full_counts[key] = len(requests)
        for row in requests:
            unmasked = row["advantage"]
            action = row.get("action")
            eligible = isinstance(action, dict) and action.get("name") == "calculate"
            row.update(
                unmasked_actor_advantage=unmasked,
                actor_update_scope=ACTOR_UPDATE_SCOPE,
                actor_update_eligible=eligible,
                full_parent_request_count=len(requests),
            )
            # Keep the complete request list: the original trainer still uses
            # 1 / (selected_parent_count * full_parent_request_count).
            before_mass.append(abs(unmasked) / len(requests))
            if eligible:
                counts["eligible"] += 1
                counts["eligible_raw_nonzero"] += row["raw_combined_advantage"] != 0
                after_mass.append(abs(unmasked) / len(requests))
            else:
                counts["suppressed"] += 1
                counts["suppressed_nonzero"] += unmasked != 0
                row["advantage"] = 0.0
    metadata.update(
        actor_update_scope=ACTOR_UPDATE_SCOPE,
        mask_applied_after_original_advantage_normalization=True,
        unmasked_effective_advantage_requests=metadata["effective_advantage_requests"],
        effective_advantage_requests=sum(
            row["advantage"] != 0 for group in admitted.values() for row in group["requests"]
        ),
        eligible_calculation_requests=counts["eligible"],
        raw_nonzero_calculation_requests=counts["eligible_raw_nonzero"],
        suppressed_noncalculation_requests=counts["suppressed"],
        suppressed_nonzero_actor_advantage_requests=counts["suppressed_nonzero"],
        full_parent_request_counts=full_counts,
        actor_loss_weight_rule="1 / (selected_parent_count * full_parent_request_count)",
        unmasked_absolute_advantage_request_mass=math.fsum(before_mass) / len(admitted),
        retained_absolute_advantage_request_mass=math.fsum(after_mass) / len(admitted),
        request_mass_is_not_gradient_norm=True,
    )
    return admitted, metadata


def recipe(args):
    base.require(
        args.command == "rl" and args.arm == "rloo",
        "calculation focus only supports the ordinary RLOO command",
    )
    return dict(_BASE_RECIPE(args), actor_update_scope=ACTOR_UPDATE_SCOPE)


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "rl":
        raise SystemExit("Use financial_step_focus_train rl with the existing RL trainer arguments")
    # Patch only this worker process, and restore on return for CPU callers.
    # Saved originals avoid recursively calling the patched functions.
    previous_batch, previous_recipe = base.build_batch, base.recipe
    base.build_batch, base.recipe = build_batch, recipe
    try:
        base.main()
    finally:
        base.build_batch, base.recipe = previous_batch, previous_recipe


if __name__ == "__main__":
    main()

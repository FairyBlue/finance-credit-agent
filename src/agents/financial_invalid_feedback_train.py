"""Keep calculation and unparsed tool responses in the existing RLOO loss.

The mask follows full-batch RLOO and advantage normalization. Invalid responses
keep their original advantage, including positive or zero values; no additional
format reward or sign correction is introduced.
"""

from __future__ import annotations

import math
import sys

from src.agents import mandate_repair_train as base

ACTOR_UPDATE_SCOPE = "calculate_and_parse_failures"
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
        "invalid-output feedback only supports ordinary RLOO without extra credit records",
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
    counts = dict(calculation=0, parse_failure=0, suppressed=0, positive=0, negative=0, zero=0)
    before_mass, retained_mass, invalid_mass = [], [], []
    full_counts = {}
    for key, group in admitted.items():
        requests = group["requests"]
        full_counts[key] = len(requests)
        for row in requests:
            original = row["advantage"]
            action = row.get("action")
            calculation = isinstance(action, dict) and action.get("name") == "calculate"
            # The base builder already requires actual prompt/response tokens.
            # Keep the sampled response, not an invented repaired JSON action.
            parse_failure = action is None
            eligible = calculation or parse_failure
            row.update(
                unmasked_actor_advantage=original,
                actor_update_scope=ACTOR_UPDATE_SCOPE,
                actor_update_eligible=eligible,
                full_parent_request_count=len(requests),
            )
            mass = abs(original) / len(requests)
            before_mass.append(mass)
            if eligible:
                retained_mass.append(mass)
            else:
                row["advantage"] = 0.0
                counts["suppressed"] += 1
            if calculation:
                counts["calculation"] += 1
            if parse_failure:
                counts["parse_failure"] += 1
                counts["positive" if original > 0 else "negative" if original < 0 else "zero"] += 1
                invalid_mass.append(mass)
    metadata.update(
        actor_update_scope=ACTOR_UPDATE_SCOPE,
        mask_applied_after_original_advantage_normalization=True,
        unmasked_effective_advantage_requests=metadata["effective_advantage_requests"],
        effective_advantage_requests=sum(
            row["advantage"] != 0 for group in admitted.values() for row in group["requests"]
        ),
        eligible_calculation_requests=counts["calculation"],
        eligible_parse_failure_requests=counts["parse_failure"],
        suppressed_other_requests=counts["suppressed"],
        parse_failure_positive_actor_requests=counts["positive"],
        parse_failure_negative_actor_requests=counts["negative"],
        parse_failure_zero_actor_requests=counts["zero"],
        full_parent_request_counts=full_counts,
        actor_loss_weight_rule="1 / (selected_parent_count * full_parent_request_count)",
        unmasked_absolute_advantage_request_mass=math.fsum(before_mass) / len(admitted),
        retained_absolute_advantage_request_mass=math.fsum(retained_mass) / len(admitted),
        parse_failure_absolute_advantage_request_mass=math.fsum(invalid_mass) / len(admitted),
        request_mass_is_not_gradient_norm=True,
        parse_failure_advantage_signs_unchanged=True,
    )
    return admitted, metadata


def recipe(args):
    base.require(
        args.command == "rl" and args.arm == "rloo",
        "invalid-output feedback only supports the ordinary RLOO command",
    )
    return dict(_BASE_RECIPE(args), actor_update_scope=ACTOR_UPDATE_SCOPE)


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "rl":
        raise SystemExit(
            "Use financial_invalid_feedback_train rl with the existing trainer arguments"
        )
    previous_batch, previous_recipe = base.build_batch, base.recipe
    base.build_batch, base.recipe = build_batch, recipe
    try:
        base.main()
    finally:
        base.build_batch, base.recipe = previous_batch, previous_recipe


if __name__ == "__main__":
    main()

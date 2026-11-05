"""Offline business and clause diagnostics, never inserted in model observations."""

from __future__ import annotations

import math
from collections import Counter


def business_score(result):
    """Binary executed feasibility/proof quality, independent of citations and cost.

    A completed task without submission is a genuine zero. Collector/system
    failures do not get imputed as counterfactual business outcomes.
    """
    if result.get("status") != "COMPLETE" or result.get("collector_failure"):
        return None
    verified = result.get("verification") or {}
    if "business_correct" in verified:
        return float(verified["business_correct"] is True)
    return float(bool(verified.get("business_feasible") or verified.get("correct_infeasible")))


def _same_number(left, right):
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=1e-8, rel_tol=1e-8)


def constraint_extraction(task, submitted_constraints):
    """Compare explicit submitted clauses, not the teacher's chosen trade vector.

    Default-valued cash/exposure clauses may be omitted without changing their
    meaning. Classification matters when an aggregate industry cap is present.
    This is a diagnostic, separate from acceptance of any genuinely valid plan.
    """
    if not isinstance(submitted_constraints, dict):
        return dict(available=False, complete=None, mismatched_fields=[], missing_clauses=[])
    gold = task["gold"]["constraints"]
    mismatch, missing = [], []
    for key, default in (("cash_min", 0.0), ("exposure_max", 1.0)):
        if not _same_number(submitted_constraints.get(key, default), gold.get(key, default)):
            mismatch.append(key)
            if key not in submitted_constraints:
                missing.append(key)
    for key in ("symbol_max", "sector_max"):
        wanted, supplied = gold.get(key, {}), submitted_constraints.get(key, {})
        if not isinstance(supplied, dict):
            mismatch.append(key)
            missing.extend(f"{key}:{name}" for name in wanted)
            continue
        for name in sorted(set(wanted) | set(supplied)):
            # Weight-one caps and their absence describe the same problem.
            if not _same_number(wanted.get(name, 1.0), supplied.get(name, 1.0)):
                mismatch.append(f"{key}:{name}")
                if name in wanted and name not in supplied:
                    missing.append(f"{key}:{name}")
    for key in ("cannot_sell", "cannot_buy"):
        supplied = submitted_constraints.get(key, [])
        if not isinstance(supplied, list) or any(not isinstance(v, str) for v in supplied):
            mismatch.append(key)
            continue
        wanted = set(gold.get(key, []))
        if set(supplied) != wanted:
            mismatch.append(key)
            missing.extend(f"{key}:{name}" for name in sorted(wanted - set(supplied)))
    if gold.get("sector_max"):
        supplied = submitted_constraints.get("sectors", {})
        if not isinstance(supplied, dict):
            mismatch.append("sectors")
        else:
            for name in sorted(task["quotes"]):
                if supplied.get(name) != gold.get("sectors", {}).get(name):
                    mismatch.append(f"sectors:{name}")
                    if name not in supplied:
                        missing.append(f"sectors:{name}")
    return dict(
        available=True,
        complete=not mismatch,
        mismatched_fields=sorted(set(mismatch)),
        missing_clauses=sorted(set(missing)),
    )


def trajectory_diagnostics(env, result):
    """Read only terminal environment state; expose nothing to a policy prompt."""
    verified = result.get("verification") or {}
    score = business_score(result)
    business_correct = score == 1.0
    submitted = result.get("verification") is not None
    submitted_constraints = None
    for item in reversed(result.get("trace", [])):
        action = item.get("action") or {}
        if action.get("name") == "submit" and item.get("response", {}).get("submitted"):
            submitted_constraints = env.plan_inputs.get(action.get("arguments", {}).get("plan_id"))
            break
    extraction = constraint_extraction(env.task, submitted_constraints)
    evidence = verified.get("evidence_valid") is True
    if score is None:
        category = "collector_or_system_failure"
    elif not submitted:
        category = "no_submission"
    elif business_correct and not evidence:
        category = "evidence_only_failure"
    elif not business_correct:
        category = "business_or_proof_failure"
    else:
        category = "success"
    result = dict(
        business_correct=business_correct,
        business_score=score,
        evidence_valid=evidence,
        submitted=submitted,
        failure_category=category,
        constraint_extraction=extraction,
        incomplete_extraction_but_business_correct=(
            extraction["available"] and not extraction["complete"] and business_correct
        ),
    )
    for key in (
        "metric_correct", "period_correct", "unit_correct", "condition_correct",
        "financial_fact_correct", "financial_business_correct", "execution_business_correct",
        "fact_correct", "portfolio_correct",
    ):
        if key in verified:
            result[key] = verified[key]
    return result


def summarize_outcomes(results):
    categories = Counter(row.get("failure_category", "unavailable") for row in results)
    available = [r for r in results if (r.get("constraint_extraction") or {}).get("available")]
    summary = dict(
        trajectories=len(results),
        business_correct=sum(row.get("business_correct") is True for row in results),
        evidence_valid=sum(row.get("evidence_valid") is True for row in results),
        submitted=sum(row.get("submitted") is True for row in results),
        failure_categories=dict(categories),
        extraction_available=len(available),
        extraction_complete=sum(row["constraint_extraction"]["complete"] for row in available),
        incomplete_extraction_but_business_correct=sum(
            row.get("incomplete_extraction_but_business_correct") is True for row in results
        ),
    )
    for key in (
        "metric_correct", "period_correct", "unit_correct", "condition_correct",
        "financial_fact_correct", "financial_business_correct", "execution_business_correct",
        "fact_correct", "portfolio_correct",
    ):
        if any(key in row for row in results):
            summary[key] = sum(row.get(key) is True for row in results)
    return summary


def local_difference(parent, counterfactuals, target="reward"):
    """Return the selected estimand plus separate main-reward provenance."""
    if target not in {"reward", "business"}:
        raise ValueError("local credit target must be reward or business")
    parent_reward = parent.get("terminal_reward")
    rewards = [row.get("terminal_reward") for row in counterfactuals]
    if not rewards or parent_reward is None or any(value is None for value in rewards):
        raise ValueError("complete real terminal outcomes required")
    parent_value = parent_reward if target == "reward" else business_score(parent)
    values = rewards if target == "reward" else [business_score(row) for row in counterfactuals]
    if parent_value is None or any(value is None for value in values):
        raise ValueError("complete business outcomes required")
    return dict(
        local_credit_target=target,
        parent_reward=parent_reward,
        counterfactual_rewards=rewards,
        parent_target_value=parent_value,
        counterfactual_target_values=values,
        unscaled_local_difference=parent_value - math.fsum(values) / len(values),
        main_reward_difference=parent_reward - math.fsum(rewards) / len(rewards),
        counterfactual_failure_categories=[r.get("failure_category") for r in counterfactuals],
    )

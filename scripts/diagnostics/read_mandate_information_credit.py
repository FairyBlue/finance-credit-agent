"""Read-only mechanism diagnosis of recorded Train information interventions.

Only training/*_seed*/round_*/collect results and anchors are opened. A live
JSONL is sampled at its current byte length and only newline-complete records
are used. Arm/seed/round/current-policy groups stay separate; no IID inference.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

METRICS = (
    "success",
    "business_feasible",
    "evidence_valid",
    "correct_infeasible",
    "decision_correct",
)
ARMS = tuple(f"{arm}_seed{seed}" for arm in ("rloo", "sham", "m3") for seed in (42, 43))


def read_snapshot(path):
    if not path.is_file():
        return [], dict(path=str(path), rows=0, absent=True)
    with path.open("rb") as stream:
        size = path.stat().st_size
        content = stream.read(size)
    rows, malformed, partial = [], 0, 0
    for line in content.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            partial += 1
            continue
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            malformed += 1
    return rows, dict(
        path=str(path),
        snapshot_bytes=len(content),
        rows=len(rows),
        incomplete_trailing_lines_ignored=partial,
        malformed_complete_lines=malformed,
    )


def action_object(raw):
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    raw = raw.strip()
    if raw.startswith("<tool_call>") and raw.endswith("</tool_call>"):
        raw = raw[len("<tool_call>") : -len("</tool_call>")]
    try:
        result = json.loads(raw)
        return result if isinstance(result, dict) else {}
    except ValueError:
        return {}


def outcomes(row):
    checked = row.get("verification")
    if not isinstance(checked, dict):
        return dict(success=bool(row.get("success")), **{m: None for m in METRICS[1:]})
    return dict(
        success=bool(row.get("success")),
        business_feasible=bool(checked.get("business_feasible")),
        evidence_valid=bool(checked.get("evidence_valid")),
        correct_infeasible=bool(checked.get("correct_infeasible")),
        decision_correct=bool(
            checked.get("business_feasible") or checked.get("correct_infeasible")
        ),
    )


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def aggregate(pairs):
    result = dict(
        observed_branch_pairs=len(pairs),
        unique_parent_trajectories=len({p["parent_trajectory_key"] for p in pairs}),
        unique_tasks=len({p["task_id"] for p in pairs}),
        unique_base_scenarios=len({p["task_id"].rsplit("-v", 1)[0] for p in pairs}),
        credit_eligible_pairs=sum(p["credit_eligible"] for p in pairs),
        cf_categories=dict(Counter(p["cf_category"] for p in pairs)),
        cf_violation_types=dict(Counter(v for p in pairs for v in p["cf_violations"])),
        cf_evidence_error_types=dict(Counter(v for p in pairs for v in p["cf_evidence_errors"])),
        continuation_tool_errors=dict(Counter(v for p in pairs for v in p["tail_tool_errors"])),
        collector_failures=dict(
            Counter(p["collector_failure_type"] for p in pairs if p["collector_failure_type"])
        ),
    )
    losses = [
        p for p in pairs if p["parent_outcomes"]["success"] and not p["cf_outcomes"]["success"]
    ]
    gate = [p for p in losses if p["cf_category"] == "masked_read_citation_permission_only"]
    evidence_only = [
        p
        for p in losses
        if p["cf_category"]
        in (
            "masked_read_citation_permission_only",
            "masked_read_required_citation_missing_only",
            "evidence_only_other",
        )
    ]
    masked_evidence_only = [p for p in evidence_only if p["cf_category"] != "evidence_only_other"]
    result.update(
        observed_parent_success_losses=len(losses),
        lost_success_categories=dict(Counter(p["cf_category"] for p in losses)),
        strict_masked_read_permission_losses=len(gate),
        masked_read_evidence_only_losses=len(masked_evidence_only),
        all_evidence_only_losses=len(evidence_only),
        strict_permission_share_of_lost_success=len(gate) / len(losses) if losses else None,
        evidence_only_share_of_lost_success=len(evidence_only) / len(losses) if losses else None,
    )
    result["outcomes"] = {}
    for metric in METRICS:
        judged = [
            p
            for p in pairs
            if p["parent_outcomes"][metric] is not None and p["cf_outcomes"][metric] is not None
        ]
        by_parent = defaultdict(list)
        for pair in judged:
            by_parent[pair["parent_trajectory_key"]].append(
                float(pair["parent_outcomes"][metric]) - float(pair["cf_outcomes"][metric])
            )
        result["outcomes"][metric] = dict(
            parent_true=sum(p["parent_outcomes"][metric] is True for p in pairs),
            cf_true=sum(p["cf_outcomes"][metric] is True for p in pairs),
            cf_false=sum(p["cf_outcomes"][metric] is False for p in pairs),
            cf_unjudged=sum(p["cf_outcomes"][metric] is None for p in pairs),
            judged_parent_cf_pairs=len(judged),
            parent_true_in_judged_pairs=sum(p["parent_outcomes"][metric] is True for p in judged),
            parent_minus_cf_mean=_mean(
                [
                    float(p["parent_outcomes"][metric]) - float(p["cf_outcomes"][metric])
                    for p in judged
                ]
            ),
            parent_balanced_mean_difference=_mean([_mean(v) for v in by_parent.values()]),
            unique_judged_parents=len(by_parent),
        )
    rewards = [p for p in pairs if p["parent_minus_cf_reward"] is not None]
    by_parent = defaultdict(list)
    for pair in rewards:
        by_parent[pair["parent_trajectory_key"]].append(pair["parent_minus_cf_reward"])
    positive = math.fsum(max(0, p["parent_minus_cf_reward"]) for p in rewards)
    gate_positive = math.fsum(
        max(0, p["parent_minus_cf_reward"])
        for p in rewards
        if p["cf_category"] == "masked_read_citation_permission_only"
    )
    evidence_categories = {
        "masked_read_citation_permission_only",
        "masked_read_required_citation_missing_only",
        "evidence_only_other",
    }
    evidence_positive = math.fsum(
        max(0, p["parent_minus_cf_reward"])
        for p in rewards
        if p["cf_category"] in evidence_categories
    )
    parent_counts = Counter(p["parent_trajectory_key"] for p in rewards)
    balanced_positive = math.fsum(
        max(0, p["parent_minus_cf_reward"]) / parent_counts[p["parent_trajectory_key"]]
        for p in rewards
    )
    balanced_evidence_positive = math.fsum(
        max(0, p["parent_minus_cf_reward"]) / parent_counts[p["parent_trajectory_key"]]
        for p in rewards
        if p["cf_category"] in evidence_categories
    )
    result["reward_contrast"] = dict(
        pairs=len(rewards),
        parent_minus_cf_mean=_mean([p["parent_minus_cf_reward"] for p in rewards]),
        parent_balanced_mean_difference=_mean([_mean(v) for v in by_parent.values()]),
        positive_parent_minus_cf_sum=positive,
        strict_citation_permission_share_of_positive_gap=gate_positive / positive
        if positive
        else None,
        all_evidence_only_share_of_positive_gap=evidence_positive / positive if positive else None,
        parent_balanced_evidence_only_share_of_positive_gap=(
            balanced_evidence_positive / balanced_positive if balanced_positive else None
        ),
    )
    return result


def diagnose_collection(collect, arm, seed, round_name):
    # Branch records precede anchor/parent reads, so concurrently appended
    # outcomes cannot outrun the already-emitted link records in our snapshot.
    branches, branch_source = read_snapshot(collect / "branches/trajectory_results.jsonl")
    anchors, anchor_source = read_snapshot(collect / "branches/anchors.jsonl")
    parents, parent_source = read_snapshot(collect / "trajectory_results.jsonl")
    anchor_map = {a["trajectory_key"]: a for a in anchors}
    parent_map = {p["trajectory_key"]: p for p in parents}
    pairs, unlinked = [], []
    for cf in branches:
        key = cf["trajectory_key"]
        anchor = anchor_map.get(key)
        parent = parent_map.get(cf["parent_trajectory_key"])
        if anchor is None or parent is None:
            unlinked.append(key)
            continue
        info = action_object(anchor.get("original_information_action"))
        tool = info.get("name", "unknown")
        masked_doc = info.get("arguments", {}).get("doc_id") if tool == "read_document" else None
        p_flags, cf_flags = outcomes(parent), outcomes(cf)
        checked = cf.get("verification") or {}
        errors = checked.get("evidence_errors", [])
        violations = checked.get("violations", [])
        submits = [
            t
            for t in cf.get("trace", [])
            if (t.get("action") or {}).get("name") == "submit"
            and t.get("response", {}).get("submitted")
        ]
        cited = set(submits[-1]["action"]["arguments"].get("citations", [])) if submits else set()
        unseen_citations = cited - set(cf.get("seen_document_ids", []))
        parent_submits = [
            t
            for t in parent.get("trace", [])
            if (t.get("action") or {}).get("name") == "submit"
            and t.get("response", {}).get("submitted")
        ]
        parent_cited = (
            set(parent_submits[-1]["action"]["arguments"].get("citations", []))
            if parent_submits
            else set()
        )
        if cf_flags["success"]:
            category = "success"
        elif cf.get("status") != "COMPLETE" or cf.get("collector_failure"):
            category = "continuation_or_system_failure"
        elif not cf.get("verification"):
            category = "no_valid_submission"
        elif cf_flags["decision_correct"] and not cf_flags["evidence_valid"]:
            only_permission = (
                masked_doc is not None
                and set(errors) == {"citation_not_observed"}
                and unseen_citations == {masked_doc}
            )
            only_omitted_masked_citation = (
                masked_doc is not None
                and set(errors) == {"missing_required_evidence"}
                and masked_doc not in cf.get("seen_document_ids", [])
                and parent_cited - cited == {masked_doc}
            )
            category = (
                "masked_read_citation_permission_only"
                if only_permission
                else "masked_read_required_citation_missing_only"
                if only_omitted_masked_citation
                else "evidence_only_other"
            )
        elif not cf_flags["decision_correct"]:
            category = "business_or_infeasibility_proof_failure"
        else:
            category = "other_terminal_failure"
        match = re.search(r"/call(\d+)$", cf["parent_request_key"])
        intervention_call = int(match.group(1)) + 1 if match else None
        tail_errors = [
            t["response"].get("error", "unspecified")
            for t in cf.get("trace", [])
            if intervention_call is not None
            and t["call_index"] > intervention_call
            and not t.get("response", {}).get("ok", False)
        ]
        p_reward, c_reward = parent.get("terminal_reward"), cf.get("terminal_reward")
        collector = cf.get("collector_failure")
        pairs.append(
            dict(
                parent_trajectory_key=parent["trajectory_key"],
                cf_trajectory_key=key,
                parent_request_key=cf["parent_request_key"],
                task_id=cf["task_id"],
                original_information_tool=tool,
                masked_document_id=masked_doc,
                continuation_replicate=cf.get("continuation_replicate"),
                parent_outcomes=p_flags,
                cf_outcomes=cf_flags,
                cf_category=category,
                cf_violations=violations,
                cf_evidence_errors=errors,
                cf_unseen_citation_ids=sorted(unseen_citations),
                citations_omitted_relative_to_parent=sorted(parent_cited - cited),
                tail_tool_errors=tail_errors,
                collector_failure_type=collector.get("status", "unknown") if collector else None,
                credit_eligible=(
                    cf.get("status") == "COMPLETE" and p_reward is not None and c_reward is not None
                ),
                parent_minus_cf_reward=float(p_reward) - float(c_reward)
                if p_reward is not None and c_reward is not None
                else None,
                cf_minus_parent_tool_calls=cf.get("calls_used", 0) - parent.get("calls_used", 0),
            )
        )
    summary_file = collect / "summary.json"
    complete = summary_file.is_file()
    grouped = {
        tool: aggregate([p for p in pairs if p["original_information_tool"] == tool])
        for tool in sorted({p["original_information_tool"] for p in pairs})
    }
    read_pairs = [p for p in pairs if p["original_information_tool"] == "read_document"]
    by_document = {
        kind: aggregate(
            [p for p in read_pairs if p["masked_document_id"].rsplit("/", 1)[-1] == kind]
        )
        for kind in sorted(
            {
                p["masked_document_id"].rsplit("/", 1)[-1]
                for p in read_pairs
                if p["masked_document_id"]
            }
        )
    }
    policies = {(p.get("base_model_path"), p.get("adapter_path"), p.get("seed")) for p in parents}
    return dict(
        arm=arm,
        seed=seed,
        round=round_name,
        collection_path=str(collect),
        collection_status="COMPLETE" if complete else "SNAPSHOT",
        interpretation_status="OBSERVATIONAL_DIAGNOSTIC_NOT_FINAL"
        if not complete
        else "ROUND_DIAGNOSTIC",
        current_policy_identifiers=[
            dict(base_model_path=b, adapter_path=a, seed=s) for b, a, s in sorted(policies, key=str)
        ],
        sources=dict(parents=parent_source, branches=branch_source, anchors=anchor_source),
        recorded_parent_rows=len(parents),
        recorded_parent_successes=sum(bool(p.get("success")) for p in parents),
        recorded_anchor_rows=len(anchors),
        recorded_cf_rows=len(branches),
        cf_status_counts=dict(Counter(c.get("status", "unknown") for c in branches)),
        unlinked_cf_keys=unlinked,
        overall=aggregate(pairs),
        by_information_tool=grouped,
        by_read_document_type=by_document,
        paired_details=pairs,
    )


def run(root):
    root = Path(root).resolve()
    groups = []
    inventory = {}
    for label in ARMS:
        folder = root / "training" / label
        inventory[label] = dict(present=folder.is_dir(), collections=0)
        for round_dir in sorted(folder.glob("round_*")):
            collect = round_dir / "collect"
            if not collect.is_dir():
                continue
            arm, seed = label.split("_seed")
            groups.append(diagnose_collection(collect, arm, int(seed), round_dir.name))
            inventory[label]["collections"] += 1
    freeze_path = root / "frozen_models.json"
    frozen = (
        json.loads(freeze_path.read_text()).get("status") == "COMPLETE"
        if freeze_path.is_file()
        else False
    )
    collections_complete = bool(groups) and all(
        g["collection_status"] == "COMPLETE" for g in groups
    )
    report = dict(
        schema="mandate_train_information_credit_diagnostics_v1",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        status="FINAL_TRAIN_COLLECTION_DIAGNOSTIC"
        if frozen and collections_complete
        else "SNAPSHOT",
        all_models_frozen=frozen,
        recorded_collections_complete=collections_complete,
        source_scope="Train collections only; no eval, test or market holdout access",
        arm_inventory=inventory,
        groups=groups,
        no_pooled_cross_policy_estimate=True,
        difference_direction="parent minus information-unavailable current-policy continuation",
        interpretation=[
            "Citation-permission-only: business/proof is correct; the masked unread doc is cited. "
            "The sole evidence error is citation_not_observed.",
            "Masked-citation-missing-only: business/proof is correct; the masked unread doc is "
            "the only citation omitted versus the parent. This is not a literal seen-only error.",
            "Other evidence and business/proof failures remain separate. Unsubmitted branches "
            "have unjudged fields, not imputed business violations.",
            "Parent-balanced differences average within parent first. Tasks, paraphrases and "
            "branches are dependent; no IID confidence intervals are provided.",
            "Contrasts include stochastic continuation and remaining-budget effects. They do not "
            "establish universal causal information or financial decision value.",
        ],
        learning_reward_scoring_or_budget_changed=False,
        reserved_test_opened=False,
        formal_valid_access=False,
        formal_final_oos_access=False,
    )
    target = root / "summary/information_credit_diagnostics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    print(json.dumps(dict(path=str(target), status=report["status"], collections=len(groups))))
    for group in groups:
        if group["recorded_cf_rows"]:
            print(
                json.dumps(
                    {k: group[k] for k in ("arm", "seed", "round", "collection_status")}
                    | {
                        "pairs": group["overall"]["observed_branch_pairs"],
                        "loss_categories": group["overall"]["lost_success_categories"],
                    }
                )
            )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()

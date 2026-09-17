#!/usr/bin/env python3
"""Descriptive paired cold-start results and fixed practical readiness gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/agent-r1"))

from src.agents.coldstart_eval import grid, save, validate_contexts  # noqa: E402

RUN = ROOT / "runs/m2-coldstart-v2-20260913"
PROTOCOL = "single_tool_portfolio_v1"
RECOVERY_SCENARIOS = {"cash", "single_high", "triple_above", "triple_mixed"}


def mean(values):
    return sum(values) / len(values) if values else None


def prefix_errors(row):
    errors = row["context"].get("prefix_expected_errors", [])
    return tuple(error for error in errors if error is not None)


def recovery(row):
    return any(row["context"].get("prefix_expected_invalid", []))


def trading_dates(row):
    return {fill["executed_at"][:10] for fill in row["model_fills"]}


def siblings(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["sampling"] == "sample":
            groups[(row["horizon"], row["context_id"])].append(row)
    details = []
    for (horizon, context_id), selected in sorted(groups.items()):
        complete = [row for row in selected if row["status"] == "COMPLETE"]
        all_four = len(selected) == len(complete) == 4 and {
            row["identity"]["rollout_n"] for row in selected
        } == {0, 1, 2, 3}
        outcomes = [row["takeover_net_return"] for row in complete]
        spread = max(outcomes) - min(outcomes) if all_four else None
        details.append({
            "horizon": horizon, "context_id": context_id,
            "planned_siblings": len(selected), "complete_siblings": len(complete),
            "all_four_complete": all_four,
            "takeover_return_range": spread,
            "distinct_outcomes": spread is not None and spread > 1e-8,
        })
    count = sum(row["distinct_outcomes"] for row in details)
    return {
        "sample_contexts": len(details),
        "contexts_with_four_complete_siblings": sum(row["all_four_complete"] for row in details),
        "contexts_with_distinct_outcomes": count,
        "distinct_fraction_of_planned_contexts": count / len(details) if details else None,
        "details": details,
    }


def describe(rows):
    complete = [row for row in rows if row["status"] == "COMPLETE"]
    fills = [fill for row in rows for fill in row["model_fills"]]
    actions: Counter[str] = Counter()
    for row in rows:
        actions.update(row["actions"])
    requests = sum(row["requests"] for row in rows)
    invalid = sum(row["invalid"] for row in rows)
    executed = sum(actions.values())
    returns = [row["takeover_net_return"] for row in complete]
    return {
        "planned": len(rows), "complete": len(complete),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "model_requests": requests, "executed_model_calls": executed,
        "unexecuted_generation_requests": requests - executed,
        "legal_model_calls": executed - invalid, "invalid_model_calls": invalid,
        "invalid_per_model_request": invalid / requests if requests else None,
        "actions": dict(actions), "model_fill_count": len(fills),
        "trajectories_with_model_fills": sum(bool(row["model_fills"]) for row in rows),
        "complete_trajectories_trading_at_least_two_dates": sum(
            len(trading_dates(row)) >= 2 for row in complete
        ),
        "fill_sides": dict(Counter(fill["side"] for fill in fills)),
        "fill_symbols": dict(Counter(fill["symbol"] for fill in fills)),
        "model_transaction_cost": sum(fill["commission"] + fill["slippage_cost"] for fill in fills),
        "mean_takeover_return_complete_only": mean(returns),
        "min_takeover_return_complete_only": min(returns, default=None),
        "max_takeover_return_complete_only": max(returns, default=None),
        "mean_terminal_score_complete_only": mean(
            [row["terminal_metrics"]["terminal_score"] for row in complete]
        ),
        "mean_final_cash_fraction_complete_only": mean([
            row["final_state"]["cash"] / row["final_state"]["portfolio_equity"] for row in complete
        ]),
        "mean_final_position_count_complete_only": mean([
            len(row["final_state"]["positions"]) for row in complete
        ]),
        "max_position_weight_since_takeover": max(
            (row.get("max_model_position_weight", 0) for row in rows), default=None
        ),
        "max_drawdown_including_prefix_complete_only": max(
            (row["terminal_metrics"]["max_drawdown"] for row in complete), default=None
        ),
        "siblings": siblings(rows),
    }


def breakdown(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["horizon"], row["sampling"], row["state_kind"], prefix_errors(row))].append(row)
    result = []
    for (horizon, sampling, scenario, errors), selected in sorted(groups.items()):
        result.append({
            "horizon": horizon, "sampling": sampling, "scenario": scenario,
            "expected_prefix_errors": list(errors), "metrics": describe(selected),
        })
    horizon_sampling = {}
    for horizon in ("short", "full"):
        for sampling in ("greedy", "sample"):
            horizon_sampling[horizon + "_" + sampling] = describe([
                row for row in rows if row["horizon"] == horizon and row["sampling"] == sampling
            ])
    return {"overall": describe(rows), "by_horizon_sampling": horizon_sampling, "strata": result}


def pair_results(before, after):
    if before.get("status") != "OBSERVED" or after.get("status") != "OBSERVED":
        raise ValueError("both evaluation suites must finish before pairing")
    if before.get("action_protocol") != PROTOCOL or after.get("action_protocol") != PROTOCOL:
        raise ValueError("paired suites must use the frozen shared protocol")
    earlier = {row["id"]: row for row in before["grid"]}
    later = {row["id"]: row for row in after["grid"]}
    if len(earlier) != len(before["grid"]) or len(later) != len(after["grid"]):
        raise ValueError("duplicate trajectory identity")
    if set(earlier) != set(later):
        raise ValueError("before/after trajectory populations differ")
    pairs = []
    for identity, row in later.items():
        old = earlier[identity]
        for field in ("context", "identity", "horizon", "sampling", "context_id", "request_limit"):
            if old[field] != row[field]:
                raise ValueError(f"paired trajectory {identity} differs in {field}")
        complete = old["status"] == row["status"] == "COMPLETE"
        pairs.append({
            "id": identity, "horizon": row["horizon"], "sampling": row["sampling"],
            "scenario": row["state_kind"], "expected_prefix_errors": list(prefix_errors(row)),
            "before_status": old["status"], "after_status": row["status"],
            "both_complete": complete,
            "takeover_return_change": (
                row["takeover_net_return"] - old["takeover_net_return"] if complete else None
            ),
            "invalid_model_calls_change": row["invalid"] - old["invalid"],
            "model_fill_count_change": len(row["model_fills"]) - len(old["model_fills"]),
        })
    by_group = defaultdict(list)
    for pair in pairs:
        by_group[pair["horizon"] + "_" + pair["sampling"]].append(pair)
    return {
        "before": breakdown(before["grid"]), "after": breakdown(after["grid"]),
        "pairs": pairs,
        "paired_return_change": {
            key: {
                "planned_pairs": len(selected),
                "both_complete": sum(pair["both_complete"] for pair in selected),
                "mean_change_both_complete_only": mean([
                    pair["takeover_return_change"] for pair in selected if pair["both_complete"]
                ]),
            }
            for key, selected in by_group.items()
        },
    }


def candidate_gates(suites):
    gates = {}
    for name, rows in suites.items():
        metrics = describe(rows)
        rate = metrics["invalid_per_model_request"]
        diversity = metrics["siblings"]
        gates[name + "_complete"] = {
            "pass": bool(rows) and metrics["complete"] == metrics["planned"],
            "complete": metrics["complete"], "planned": metrics["planned"],
        }
        gates[name + "_invalid"] = {"pass": rate is not None and rate <= 0.05,
                                    "invalid_per_request": rate, "maximum": 0.05}
        fraction = diversity["distinct_fraction_of_planned_contexts"]
        gates[name + "_siblings"] = {
            "pass": fraction is not None and fraction >= 0.5,
            "distinct_complete_four_sibling_contexts": diversity["contexts_with_distinct_outcomes"],
            "planned_sample_contexts": diversity["sample_contexts"],
            "fraction": fraction, "minimum": 0.5, "strict_return_range_threshold": 1e-8,
        }
    selected = [row for row in suites["fresh"] if row["horizon"] == "short" and recovery(row)]
    metrics = describe(selected)
    scenarios = dict(Counter(row["state_kind"] for row in selected))
    populations_match = scenarios == {name: 10 for name in RECOVERY_SCENARIOS}
    rate = metrics["invalid_per_model_request"]
    gates["fresh_error_recovery"] = {
        "pass": populations_match and metrics["planned"] == metrics["complete"] == 40
        and rate is not None and rate <= 0.05,
        "expected_trajectories": 40, "planned": metrics["planned"], "complete": metrics["complete"],
        "scenario_counts": scenarios, "invalid_per_request": rate, "maximum": 0.05,
        "prefix_errors_excluded_from_model_counts": True,
    }
    full_sample = [
        row for rows in suites.values() for row in rows
        if row["horizon"] == "full" and row["sampling"] == "sample"
        and row["status"] == "COMPLETE"
    ]
    active = sum(len(trading_dates(row)) >= 2 for row in full_sample)
    fraction = active / len(full_sample) if full_sample else None
    gates["full_sample_sustained_trading"] = {
        "pass": fraction is not None and fraction >= 0.5,
        "complete_full_sample_trajectories": len(full_sample),
        "trading_on_at_least_two_dates": active, "fraction": fraction, "minimum": 0.5,
    }
    return {
        "status": "PASS" if all(item["pass"] for item in gates.values()) else "FAIL",
        "criteria": gates,
        "interpretation": "Practical common-initialization readiness only; not M2 superiority.",
        "automatic_rl_launch": False,
    }


def validate_population(result, contexts):
    planned = {row["id"]: row for row in grid(contexts)}
    observed = {row["id"]: row for row in result["grid"]}
    if len(observed) != len(result["grid"]) or set(planned) != set(observed):
        raise ValueError("evaluation result differs from the frozen dataset population")
    for identity, row in observed.items():
        for field in ("context", "identity", "horizon", "sampling", "context_id", "request_limit"):
            if row[field] != planned[identity][field]:
                raise ValueError(f"evaluation result differs from frozen {field}")


def model_error_counts(path):
    errors: Counter[str] = Counter()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            record = json.loads(line)
            info = record.get("info", {})
            if info.get("invalid_action") is True:
                message = info.get("error", "<missing error message>")
                errors[message] += 1
    return dict(errors), digest.hexdigest()


def build_readout(run):
    result = {}
    inputs = {}
    after_rows = {}
    for suite, context_name, full in (
        ("regression", "regression_eval_contexts.json", 4), ("fresh", "eval_contexts.json", 1)
    ):
        context_path = run / "dataset" / context_name
        contexts = json.loads(context_path.read_text())
        validate_contexts(contexts, 16, full)
        inputs[str(context_path.relative_to(run))] = hashlib.sha256(
            context_path.read_bytes()
        ).hexdigest()
        evaluations = []
        error_counts = {}
        for phase in ("before", "after"):
            path = run / f"{phase}_{suite}/result.json"
            value = json.loads(path.read_text())
            validate_population(value, contexts)
            evaluations.append(value)
            inputs[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
            request_path = path.parent / "requests.jsonl"
            counts, request_hash = model_error_counts(request_path)
            if sum(counts.values()) != sum(row["invalid"] for row in value["grid"]):
                raise ValueError("request and trajectory model invalid counts differ")
            error_counts[phase] = counts
            inputs[str(request_path.relative_to(run))] = request_hash
        result[suite] = pair_results(*evaluations)
        for phase, counts in error_counts.items():
            result[suite][phase]["model_invalid_error_counts"] = counts
        after_rows[suite] = evaluations[1]["grid"]
    return {
        "status": "COMPLETE",
        "finished_at_SGT": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "action_protocol": PROTOCOL, "inputs_sha256": inputs,
        "suites": result, "gates": candidate_gates(after_rows),
        "limits": [
            "Train diagnostics only; no independent OOS or M2 reward-gain claim.",
            "Shared portfolio protocol is applied to both before and after models.",
            "Fixed prefix fills and errors are excluded from model behavior counts.",
            "Drawdown and terminal scorer include prefixes; short net return starts at takeover.",
            "Same context/seed identities do not imply identical generated trajectories.",
            "Return changes require both sides complete; failures are never zero returns.",
            "Readiness gates are fixed practical checks; returns and risk do not choose prompts.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=RUN)
    args = parser.parse_args()
    output = args.directory / "readout.json"
    if output.exists():
        raise RuntimeError("readout already exists; preserve the previous result")
    result = build_readout(args.directory)
    save(output, result)
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, indent=2))


if __name__ == "__main__":
    main()

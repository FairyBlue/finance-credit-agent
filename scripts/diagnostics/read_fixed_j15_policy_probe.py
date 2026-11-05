#!/usr/bin/env python3
"""Small paired fixed-prefix policy readout; no models, replay or significance."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

COMMON = tuple(f"formal-train-{n:04d}" for n in (1, 9, 65, 73))
ADDED = tuple(f"formal-train-{n:04d}" for n in (17, 25, 149, 157))
EPISODES = tuple(sorted((*COMMON, *ADDED)))
MODELS = ("sft", "trained")
METRICS = ("Y", "raw_net_return", "suffix_net_return", "transaction_cost", "suffix_cost")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)


def mean(values):
    return math.fsum(values) / len(values)


def parse_row(row):
    require(row["role"] == "fixed_continuation", "fixed continuation records only")
    require(row["anchor_id"] == row["episode_id"] + "/j15", "original j15 anchor")
    require(
        type(row["continuation_replicate"]) is int and row["continuation_replicate"] in range(4),
        "four planned draws per anchor",
    )
    steps = row["timesteps"]
    invalid = sum(step["invalid"] is True for step in steps)
    require(invalid == row["remaining_invalid_count"], "suffix invalid accounting")
    first = steps[0]["invalid"] if steps else None
    require(row["first_action_invalid"] is first, "first applied action invalid flag")
    cost = 0.0
    for step in steps:
        for fill in step.get("info", {}).get("fills", []):
            commission, slippage = fill["commission"], fill["slippage_cost"]
            require(
                finite(commission) and finite(slippage) and commission >= 0 and slippage >= 0,
                "finite actual fill fees",
            )
            cost += commission + slippage
    result = dict(
        status=row["status"],
        first_action_invalid=first,
        invalid_count=invalid,
        applied_requests=len(steps),
        observed_suffix_cost=cost,
        anchor_equity=row["anchor_equity"],
        prefix_requests=row["prefix_requests"],
        anchor_prompt_sha256=row["anchor_prompt_sha256"],
        anchor_financial_state_sha256=row["anchor_financial_state_sha256"],
        Y=None,
        raw_net_return=None,
        suffix_net_return=None,
        transaction_cost=None,
        suffix_cost=None,
        final_equity=None,
    )
    require(
        finite(result["anchor_equity"]) and result["anchor_equity"] > 0,
        "positive last-booked prefix equity",
    )
    if row["status"] == "COMPLETE":
        metrics = row["terminal_metrics"]
        values = (
            row["Y"],
            row["terminal_financial_Y"],
            metrics["terminal_score"],
            metrics["raw_net_return"],
            metrics["transaction_cost"],
            row["suffix_net_return"],
            row["final_equity"],
        )
        require(all(finite(value) for value in values), "finite completed financial labels")
        require(
            row["Y"] == row["terminal_financial_Y"] == metrics["terminal_score"],
            "terminal financial Y identity",
        )
        require(
            row["last_session_index"] == 20 and row["final_equity"] > 0,
            "original terminal endpoint and equity",
        )
        require(
            close(row["suffix_net_return"], row["final_equity"] / row["anchor_equity"] - 1),
            "suffix return uses the booked anchor equity",
        )
        result.update(
            Y=row["Y"],
            raw_net_return=metrics["raw_net_return"],
            suffix_net_return=row["suffix_net_return"],
            transaction_cost=metrics["transaction_cost"],
            suffix_cost=cost,
            final_equity=row["final_equity"],
        )
    else:
        require(
            row["Y"] is None
            and row["terminal_financial_Y"] is None
            and row["terminal_metrics"] is None
            and row["suffix_net_return"] is None
            and row["final_equity"] is None,
            "incomplete financial labels stay NA",
        )
    return result


def aggregate(rows, episodes):
    selected = [row for row in rows if row["episode_id"] in episodes]
    complete = [row for row in selected if row["paired_complete"]]
    expected = 4 * len(episodes)
    require(len(selected) == expected, "planned paired denominator retained")
    values = None
    if complete:
        values = {
            f"{model}_{metric}": mean([row[f"{model}_{metric}"] for row in complete])
            for model in MODELS
            for metric in METRICS
        }
        values.update(
            {
                f"delta_{metric}": mean([row[f"delta_{metric}"] for row in complete])
                for metric in METRICS
            }
        )
        for metric in ("raw_net_return", "suffix_net_return"):
            values[f"delta_{metric}_pp"] = 100 * values[f"delta_{metric}"]
    return dict(
        episode_ids=list(episodes),
        planned_pairs=expected,
        complete_pairs=len(complete),
        missing_or_failed_pairs=expected - len(complete),
        all_planned_pairs_complete=len(complete) == expected,
        full_planned_mean=values if len(complete) == expected else None,
        complete_pair_descriptive_mean=values,
    )


def run(directory):
    directory = directory.resolve()
    require(
        not (directory / "paired_results.csv").exists()
        and not (directory / "probe_readout.json").exists(),
        "new outputs required; no overwrite",
    )
    summaries, observed = {}, {}
    expected = {(episode + "/j15", rep) for episode in EPISODES for rep in range(4)}
    for model in MODELS:
        source = directory / model
        summary_path = source / "fixed_prefix_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        summaries[model] = summary
        if summary:
            require(
                set(summary["episode_ids"]) == set(EPISODES)
                and summary["continuations_per_anchor"] == 4
                and summary["planned_trajectories"] == 32,
                "same eight planned Train windows",
            )
            require(
                summary["seed"] == 42 and summary["round_index"] == 0,
                "paired fixed random-number namespace",
            )
        data = {}
        path = source / "trajectory_results.jsonl"
        if path.exists():
            with path.open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    key = (record["anchor_id"], record["continuation_replicate"])
                    require(
                        key in expected and key not in data, "planned unique anchor/draw identity"
                    )
                    data[key] = parse_row(record)
        observed[model] = data
        if summary.get("status") == "COMPLETE":
            require(len(data) == 32, "collector completion retains all planned records")
            require(
                sum(row["status"] == "COMPLETE" for row in data.values())
                == summary["complete_trajectories"],
                "collector completion denominator",
            )
            require(
                sum(row["invalid_count"] for row in data.values()) == summary["invalid_requests"],
                "reported suffix invalid count",
            )
    if all(summaries.values()):
        require(
            summaries["sft"]["fixed_prefix_source"] == summaries["trained"]["fixed_prefix_source"]
            and summaries["sft"]["source_sha256"] == summaries["trained"]["source_sha256"],
            "both policies must use the same frozen prefix source",
        )
    pairs = []
    for anchor, rep in sorted(expected):
        row = dict(
            episode_id=anchor.removesuffix("/j15"), anchor_id=anchor, continuation_replicate=rep
        )
        left, right = (observed[model].get((anchor, rep)) for model in MODELS)
        if left is not None and right is not None:
            require(
                close(left["anchor_equity"], right["anchor_equity"])
                and all(
                    left[k] == right[k]
                    for k in (
                        "prefix_requests",
                        "anchor_prompt_sha256",
                        "anchor_financial_state_sha256",
                    )
                ),
                "paired full prompt and financial prefix identity",
            )
        paired = (
            left is not None
            and right is not None
            and all(r["status"] == "COMPLETE" for r in (left, right))
        )
        row["paired_complete"] = paired
        for model, item in zip(MODELS, (left, right), strict=True):
            row[f"{model}_status"] = item["status"] if item else "MISSING_RECORD"
            row[f"{model}_first_action_invalid"] = item["first_action_invalid"] if item else None
            row[f"{model}_invalid_count"] = item["invalid_count"] if item else None
            for metric in METRICS:
                row[f"{model}_{metric}"] = item[metric] if item else None
        for metric in METRICS:
            row[f"delta_{metric}"] = right[metric] - left[metric] if paired else None
        for metric in ("raw_net_return", "suffix_net_return"):
            row[f"delta_{metric}_pp"] = 100 * row[f"delta_{metric}"] if paired else None
        if paired:
            require(
                1 + left["raw_net_return"] > 0 and 1 + right["raw_net_return"] > 0,
                "positive original initial equity",
            )
            initial_left = left["final_equity"] / (1 + left["raw_net_return"])
            initial_right = right["final_equity"] / (1 + right["raw_net_return"])
            require(close(initial_left, initial_right), "same original starting equity")
            require(
                close(
                    row["delta_raw_net_return"],
                    left["anchor_equity"] / initial_left * row["delta_suffix_net_return"],
                ),
                "full-window and suffix return deltas have different denominators",
            )
            require(
                close(row["delta_transaction_cost"], row["delta_suffix_cost"]),
                "shared prefix fees cancel in paired transaction-cost difference",
            )
        pairs.append(row)
    operational = {}
    for model in MODELS:
        data, summary = observed[model], summaries[model]
        first = [
            r["first_action_invalid"]
            for r in data.values()
            if r["first_action_invalid"] is not None
        ]
        operational[model] = dict(
            planned_paths=32,
            observed_paths=len(data),
            status_counts=dict(Counter(r["status"] for r in data.values())),
            missing_records=32 - len(data),
            first_action_known_paths=len(first),
            first_action_invalid_count=sum(v is True for v in first),
            first_action_invalid_rate=sum(v is True for v in first) / len(first) if first else None,
            total_suffix_invalid=sum(r["invalid_count"] for r in data.values()),
            observed_suffix_transaction_cost=sum(r["observed_suffix_cost"] for r in data.values()),
            reported_model_requests=summary.get("model_requests"),
            reported_prompt_tokens=summary.get("prompt_tokens"),
            reported_response_tokens=summary.get("response_tokens"),
        )
    cohorts = dict(
        all8=aggregate(pairs, EPISODES),
        common4=aggregate(pairs, COMMON),
        added4=aggregate(pairs, ADDED),
    )
    report = dict(
        profile="paired_fixed_j15_SFT_vs_S0_R8_probe_v1",
        status="COMPLETE" if cohorts["all8"]["all_planned_pairs_complete"] else "PARTIAL",
        planned_pairs=32,
        cohorts=cohorts,
        windows={episode: aggregate(pairs, (episode,)) for episode in EPISODES},
        operational=operational,
        definitions=dict(
            delta="trained minus SFT, paired by anchor and continuation draw",
            raw_net_return="original window terminal equity / original initial equity - 1",
            suffix_net_return="terminal equity / same fixed j15 last-booked equity - 1",
            return_delta_pp="fractional return delta multiplied by 100, not by 10000",
            Y="original benchmark-scaled risk-adjusted financial score; never percentage points",
            transaction_cost="whole-window commission plus slippage, currency units",
            suffix_cost="actual suffix fills only: commission plus slippage_cost",
            missing="NA; complete-pair subset means never replace full planned means",
        ),
        limits=[
            "Small same-Train-window functional probe; no power or significance claim.",
            "CRN pairs request namespaces; equal seeds do not guarantee lower variance.",
            "Common prefixes include inherited past performance and fees; "
            "only suffix policy differs.",
            "Full-window and suffix return deltas use different equity denominators.",
            "Common4 and added4 were both previously used Train windows, not held-out dates.",
        ],
    )
    with (directory / "paired_results.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    with (directory / "probe_readout.json").open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            dict(status=report["status"], all8=cohorts["all8"], operational=operational),
            sort_keys=True,
            allow_nan=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    run(parser.parse_args().run)


if __name__ == "__main__":
    main()

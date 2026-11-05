#!/usr/bin/env python3
"""Read the bounded counterfactual-order supervision demo from saved records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from read_rloo_eight_window import EPISODES, collection_metrics, compare
from read_rloo_terminal_demo import finite, mean

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "runs/m2-counterfactual-order-demo-20261027-v1"
POLICIES = ("SFT", "RLOO", "distilled")
NAMESPACES = (5, 6)
CONTRASTS = (("RLOO", "SFT"), ("distilled", "SFT"), ("distilled", "RLOO"))
COST_KEYS = ("model_requests", "prompt_tokens", "response_tokens")
TERMINAL = {"COMPLETE", "COMPLETE_WITH_MODEL_FAILURES", "COMPLETE_NO_USABLE_LABELS", "FAILED"}


def optional(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def jsonl_count(path):
    path = Path(path)
    return (
        sum(bool(line.strip()) for line in path.read_text().splitlines()) if path.is_file() else 0
    )


def pairs(namespace, name, current, reference):
    rows = []
    for index, (now, old) in enumerate(
        zip(current["trajectories"], reference["trajectories"], strict=True)
    ):
        complete = all(
            row is not None
            and row["status"] == "COMPLETE"
            and row["terminal_session_index"] == 20
            and finite(row["net_return"])
            and finite(row["Y"])
            for row in (now, old)
        )
        prompt_equal = (
            (
                now["initial_prompt_sha256"] is not None
                and now["initial_prompt_sha256"] == old["initial_prompt_sha256"]
            )
            if complete
            else None
        )
        seed_equal = (
            (
                type(now["initial_request_seed"]) is int
                and now["initial_request_seed"] == old["initial_request_seed"]
            )
            if complete
            else None
        )
        if complete and not (prompt_equal and seed_equal):
            raise ValueError(f"namespace{namespace}/{name}: initial prompt/seed mismatch")
        rows.append(
            dict(
                namespace=namespace,
                comparison=name,
                episode_id=EPISODES[index // 4],
                replicate_index=index % 4,
                status="OBSERVED" if complete else "PENDING_OR_INCOMPLETE",
                reference_net_return=old["net_return"] if complete else None,
                current_net_return=now["net_return"] if complete else None,
                net_return_delta_pp=100 * (now["net_return"] - old["net_return"])
                if complete
                else None,
                Y_delta=now["Y"] - old["Y"] if complete else None,
                reference_invalid_requests=old["invalid_requests"] if complete else None,
                current_invalid_requests=now["invalid_requests"] if complete else None,
                transaction_cost_delta=now["transaction_cost"] - old["transaction_cost"]
                if complete and finite(now["transaction_cost"]) and finite(old["transaction_cost"])
                else None,
                initial_prompt_equal=prompt_equal,
                initial_seed_equal=seed_equal,
            )
        )
    observed = [row for row in rows if row["status"] == "OBSERVED"]
    return rows, dict(
        planned_pairs=32,
        complete_pairs=len(observed),
        initial_prompt_equal_pairs=sum(row["initial_prompt_equal"] is True for row in observed),
        initial_seed_equal_pairs=sum(row["initial_seed_equal"] is True for row in observed),
        net_return_delta_pp=mean([row["net_return_delta_pp"] for row in observed], 32),
        Y_delta=mean([row["Y_delta"] for row in observed], 32),
    )


def delta_value(current, reference, key, factor=1):
    a, b = current[key], reference[key]
    return factor * (a - b) if finite(a) and finite(b) else None


def training_cost(summary):
    summary = summary or {}
    steps = summary.get("optimizer_updates", 0)
    winners = summary.get("winner_training_rows")
    replay = summary.get("replay_training_rows")
    if winners is None and finite(summary.get("per_update_winner_rows")):
        winners = steps * summary["per_update_winner_rows"]
    if replay is None and finite(summary.get("per_update_replay_rows")):
        replay = steps * summary["per_update_replay_rows"]
    return dict(
        optimizer_updates=steps,
        trained_request_rows=summary.get("training_rows", summary.get("training_sequences", 0)),
        winner_training_rows=winners if summary else 0,
        replay_training_rows=replay if summary else 0,
        training_prompt_tokens=summary.get("training_prompt_tokens", 0),
        training_response_tokens=summary.get("training_response_tokens", 0),
        rule="Supervised request uses, including repeated rows; not new model generation.",
    )


def concentration(comparison):
    windows = comparison["windows"]
    complete = len(windows) == 8 and all(finite(row["net_return_delta_pp"]) for row in windows)
    if not complete:
        return dict(
            largest_positive_contributor=None,
            largest_window_delta_pp=None,
            net_delta_without_largest_window_pp=None,
            all8_positive=None,
            without_largest_positive=None,
            practical_progression_condition=None,
        )
    largest = max(windows, key=lambda row: row["net_return_delta_pp"])
    rest = [row["net_return_delta_pp"] for row in windows if row is not largest]
    dropped = mean(rest, 7)
    overall = comparison["strata"]["all8"]["net_return_delta_pp"]
    return dict(
        largest_positive_contributor=largest["episode_id"]
        if largest["net_return_delta_pp"] > 0
        else None,
        largest_window_delta_pp=largest["net_return_delta_pp"],
        net_delta_without_largest_window_pp=dropped,
        all8_positive=overall > 0,
        without_largest_positive=dropped > 0,
        practical_progression_condition=overall > 0 and dropped > 0,
    )


def summarize(directory=DEFAULT):
    root = Path(directory).resolve()
    execution = optional(root / "completion.json") or optional(root / "results.json") or {}
    stage1 = optional(root / "stage1/collection_summary.json")
    stage2 = optional(root / "stage2/training_summary.json")
    label_count = jsonl_count(root / "stage1/training_samples.jsonl")
    no_labels = bool(stage1 and stage1.get("status") in TERMINAL - {"FAILED"} and label_count == 0)
    collections, comparisons, summary_rows, window_rows, paired_rows = {}, {}, [], [], []
    costs = [stage1] if stage1 else []
    if not no_labels:
        for namespace in NAMESPACES:
            scope, contrast_scope = {}, {}
            for policy in POLICIES:
                path = root / f"eval/{policy}/namespace{namespace}"
                current = collection_metrics(path, 43)
                source = current["summary"]
                if source and (
                    source.get("round_index") != namespace
                    or source.get("optimizer_updates", 0) != 0
                ):
                    raise ValueError("expected declared fresh namespace, collect-only evaluation")
                scope[policy] = current
                overall = current["strata"]["all8"]
                summary_rows.append(dict(namespace=namespace, policy=policy, **overall))
                window_rows.extend(
                    dict(namespace=namespace, policy=policy, **row) for row in current["windows"]
                )
                if source:
                    costs.append(source)
            for now, old in CONTRASTS:
                name = f"{now}_minus_{old}"
                delta = compare(scope[now], scope[old])
                rows, pairing = pairs(namespace, name, scope[now], scope[old])
                a, b = scope[now]["strata"]["all8"], scope[old]["strata"]["all8"]
                delta["strata"]["all8"].update(
                    rejection_rate_delta_pp=delta_value(a, b, "rejection_rate", 100),
                    mean_transaction_cost_delta=delta_value(a, b, "mean_transaction_cost"),
                )
                delta.update(paired_sampling=pairing, concentration=concentration(delta))
                contrast_scope[name] = delta
                paired_rows.extend(rows)
            collections[str(namespace)] = scope
            comparisons[str(namespace)] = contrast_scope
        for row in summary_rows:
            if row["policy"] != "SFT":
                row["versus_SFT_net_delta_pp"] = comparisons[str(row["namespace"])][
                    f"{row['policy']}_minus_SFT"
                ]["strata"]["all8"]["net_return_delta_pp"]
            if row["policy"] == "distilled":
                comparison = comparisons[str(row["namespace"])]["distilled_minus_RLOO"]
                row["versus_RLOO_net_delta_pp"] = comparison["strata"]["all8"][
                    "net_return_delta_pp"
                ]
                row["versus_RLOO_without_largest_window_pp"] = comparison["concentration"][
                    "net_delta_without_largest_window_pp"
                ]
    complete = bool(collections) and all(
        c["strata"]["all8"]["status"] == "OBSERVED"
        for scope in collections.values()
        for c in scope.values()
    )
    progression = [
        scope["distilled_minus_RLOO"]["concentration"]["practical_progression_condition"]
        for scope in comparisons.values()
    ]
    passed = complete and len(progression) == 2 and all(value is True for value in progression)
    training_complete = bool(
        stage2 and stage2.get("status") == "COMPLETE" and stage2.get("optimizer_updates") == 16
    )
    cells = [collection for scope in collections.values() for collection in scope.values()]
    evaluations_finished = len(cells) == 6 and all(
        collection["summary"] is not None
        and collection["summary"].get("status") in {"COMPLETE", "COMPLETE_WITH_MODEL_FAILURES"}
        and len(collection["trajectories"]) == 32
        and all(row is not None for row in collection["trajectories"])
        for collection in cells
    )
    rollout_failure_exit = training_complete and evaluations_finished and not complete
    terminal = (
        execution.get("status") in TERMINAL
        or no_labels
        or (evaluations_finished and training_complete)
    )
    decision = (
        "NO_USABLE_LABELS_STOP"
        if no_labels
        else "WORTH_EXPANDING_NEW_DESIGN"
        if passed and training_complete
        else "STOP_MODEL_ROLLOUT_FAILURES"
        if rollout_failure_exit
        else "STOP_CURRENT_VARIANT"
        if complete and training_complete
        else "INCOMPLETE_NO_CONCLUSION"
    )
    report = dict(
        profile="counterfactual_order_supervision_same_Train_windows_demo_v1",
        generated_at_UTC=datetime.now(timezone.utc).isoformat(),
        status="DESCRIBED"
        if no_labels or (complete and training_complete)
        else "DESCRIBED_WITH_FAILURES"
        if terminal
        else "PARTIAL",
        final_artifacts_allowed=terminal,
        execution_status=execution.get("status", "PENDING"),
        seed=43,
        namespaces=list(NAMESPACES),
        episode_ids=EPISODES,
        parent_repetitions=4,
        stage1=stage1,
        selected_supervised_labels=label_count,
        replay_rows=jsonl_count(root / "stage1/replay_samples.jsonl"),
        stage2=stage2,
        training_cost=training_cost(stage2),
        supervised_label_fit=dict(
            before_NLL=stage2.get("before_NLL") if stage2 else None,
            after_NLL=stage2.get("after_NLL") if stage2 else None,
            scope="Training-label fit only; full-path evaluation determines trading behavior.",
        ),
        stage2_and_evaluation_skipped=no_labels,
        evaluations_finished=evaluations_finished,
        rollout_failure_exit=rollout_failure_exit,
        planned_parent_trajectories=0 if no_labels else 192,
        complete_parent_trajectories=sum(
            c["strata"]["all8"]["complete_trajectories"]
            for scope in collections.values()
            for c in scope.values()
        ),
        collections=collections,
        comparisons=comparisons,
        decision=decision,
        rollout_failure_note=(
            "All six planned evaluation collections ended, including failed budget-bounded "
            "paths. Training completed16 updates. Fixed-denominator candidate returns remain "
            "NA; rollout failure does not establish a lower mean return."
        )
        if rollout_failure_exit
        else None,
        progression_condition=(
            "Distilled minus frozen RLOO net return positive in each namespace, and positive "
            "after removing that namespace's largest positive contributing window. "
            "Behavioral costs are reported alongside this descriptive screen."
        ),
        sampling_cost=dict(
            known_totals={
                key: sum(source.get(key, 0) for source in costs if finite(source.get(key)))
                for key in COST_KEYS
            },
            summaries_observed=len(costs),
            incomplete_costs_are_lower_bounds=not (no_labels or evaluations_finished),
            rule=(
                "Stage1 new branch generation plus each evaluation collection once; "
                "no source replay generation cost."
            ),
        ),
        limitations=[
            "These are previously used Train windows, not new dates or formal Valid/Final OOS.",
            "Both namespaces are held out from this batch, not independent training seeds.",
            "Four replicas within a window are not four independent market windows.",
            "Missing/failed trajectories keep fixed32 collection denominators and NA metrics.",
            "The candidate adds branch-label supervision after RLOO; "
            "this is not equal-budget or exact-pooling evidence.",
            "No intermediate model selection, statistical significance, "
            "or unique failure cause is inferred.",
        ],
    )
    return report, summary_rows, window_rows, paired_rows


def write_table(path, rows, default_columns):
    columns = list(dict.fromkeys(key for row in rows for key in row)) or default_columns
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 4.5))
    for offset, namespace, color in ((-0.18, 5, "#90a4ae"), (0.18, 6, "#1976d2")):
        values = [
            next(
                row["mean_net_return"]
                for row in rows
                if row["policy"] == policy and row["namespace"] == namespace
            )
            for policy in POLICIES
        ]
        axis.bar(
            [index + offset for index in range(3)],
            [100 * value if finite(value) else math.nan for value in values],
            width=0.34,
            color=color,
            label=f"Namespace {namespace}",
        )
        for index, value in enumerate(values):
            if not finite(value):
                row = next(
                    row
                    for row in rows
                    if row["policy"] == POLICIES[index] and row["namespace"] == namespace
                )
                axis.text(
                    index + offset,
                    0.08,
                    f"NA\n{row['complete_trajectories']}/32\ncomplete",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=color,
                    fontweight="bold",
                )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        xticks=range(3),
        xticklabels=POLICIES,
        ylabel="Mean net return (%)",
        xlim=(-0.6, 2.6),
        title="Eight existing Train windows; 32 planned paths per model/namespace",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_readout(root, result):
    report, summaries, windows, paired = result
    if not report["final_artifacts_allowed"]:
        raise ValueError("demo still running; use --preview")
    output = Path(root) / "summary"
    names = (
        "readout.json",
        "summary.csv",
        "window_results.csv",
        "paired_results.csv",
        "comparison.png",
    )
    if any((output / name).exists() for name in names):
        raise FileExistsError("final readout exists; refusing overwrite")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "readout.json").open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    write_table(output / "summary.csv", summaries, ["namespace", "policy", "status"])
    write_table(
        output / "window_results.csv", windows, ["namespace", "policy", "episode_id", "status"]
    )
    write_table(
        output / "paired_results.csv", paired, ["namespace", "comparison", "episode_id", "status"]
    )
    if summaries:
        plot(summaries, output / "comparison.png")
    return output / "readout.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", "--output", dest="directory", type=Path, default=DEFAULT)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    result = summarize(args.directory)
    if args.preview:
        print(
            json.dumps(
                dict(
                    status=result[0]["status"],
                    decision=result[0]["decision"],
                    labels=result[0]["selected_supervised_labels"],
                    policies=result[1],
                    sampling_cost=result[0]["sampling_cost"],
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(write_readout(args.directory, result))


if __name__ == "__main__":
    main()

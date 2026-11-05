#!/usr/bin/env python3
"""Compare saved policies on fresh rollout seeds at the same four Train windows."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from read_rloo_local_credit_demo import (
    COST_KEYS,
    EPISODES,
    RLOO,
    ROOT,
    TERMINAL,
    collection_metrics,
    difference,
    optional,
)
from read_rloo_terminal_demo import finite, mean, write_csv

DEFAULT = ROOT / "runs/m2-rloo-local-credit-fresh-eval-20261026-v1"
LOCAL = ROOT / "runs/m2-rloo-local-credit-demo-20261025-v1"
POLICIES = ("SFT", "RLOO", "plain", "exact")
CONTRASTS = [(policy, "SFT") for policy in POLICIES[1:]] + [
    ("plain", "RLOO"),
    ("exact", "RLOO"),
    ("exact", "plain"),
]


def paired(name, current, reference):
    rows = []
    for index, (a, b) in enumerate(
        zip(current["trajectories"], reference["trajectories"], strict=True)
    ):
        complete = (
            a is not None
            and b is not None
            and a["status"] == b["status"] == "COMPLETE"
            and a["terminal_session_index"] == b["terminal_session_index"] == 20
            and all(finite(row[key]) for row in (a, b) for key in ("Y", "net_return"))
        )
        prompt_equal = (
            (
                a["initial_prompt_sha256"] is not None
                and a["initial_prompt_sha256"] == b["initial_prompt_sha256"]
            )
            if complete
            else None
        )
        seed_equal = (
            (
                type(a["initial_request_seed"]) is int
                and a["initial_request_seed"] == b["initial_request_seed"]
            )
            if complete
            else None
        )
        if complete and (not prompt_equal or not seed_equal):
            raise ValueError(f"{name}: complete fresh pair has mismatched prompt/first seed")
        rows.append(
            dict(
                comparison=name,
                episode_id=EPISODES[index // 4],
                replicate_index=index % 4,
                status="OBSERVED" if complete else "PENDING_OR_INCOMPLETE",
                reference_net_return=b["net_return"] if complete else None,
                current_net_return=a["net_return"] if complete else None,
                net_return_delta_pp=100 * (a["net_return"] - b["net_return"]) if complete else None,
                Y_delta=a["Y"] - b["Y"] if complete else None,
                reference_invalid_requests=b["invalid_requests"] if complete else None,
                current_invalid_requests=a["invalid_requests"] if complete else None,
                initial_prompt_equal=prompt_equal,
                initial_seed_equal=seed_equal,
            )
        )
    observed = [row for row in rows if row["status"] == "OBSERVED"]
    namespace = (
        current["summary"] is not None
        and reference["summary"] is not None
        and current["summary"].get("round_index") == reference["summary"].get("round_index") == 4
    )
    if observed and not namespace:
        raise ValueError(f"{name}: fresh pairs require namespace4")
    return rows, dict(
        planned_pairs=16,
        complete_pairs=len(observed),
        namespace4_pairing=namespace,
        initial_prompt_equal_pairs=sum(r["initial_prompt_equal"] is True for r in observed),
        initial_seed_equal_pairs=sum(r["initial_seed_equal"] is True for r in observed),
        net_return_delta_pp=mean([r["net_return_delta_pp"] for r in observed], 16),
        Y_delta=mean([r["Y_delta"] for r in observed], 16),
    )


def original_collections():
    source = optional(RLOO / "rloo_results.json") or {}
    result = {"SFT": collection_metrics(source.get("baseline_path", RLOO / "common_initial"))}
    for policy, root in [("RLOO", RLOO), ("plain", LOCAL / "plain"), ("exact", LOCAL / "exact")]:
        source = optional(root / "rloo_results.json") or {}
        final = next((entry for entry in source.get("rounds", []) if entry["round"] == 4), {})
        result[policy] = collection_metrics(final.get("collection_path", root / "round4/eval"))
    return result


def direction(value):
    return (
        None
        if not finite(value)
        else "positive"
        if value > 1e-12
        else "negative"
        if value < -1e-12
        else "flat"
    )


def summarize(directory=DEFAULT):
    root = Path(directory).resolve()
    source = optional(root / "results.json") or {}
    if source and (
        source.get("seed") != 42
        or source.get("episode_ids") != EPISODES
        or source.get("round_index") != 4
        or source.get("parent_repetitions") != 4
    ):
        raise ValueError("expected seed42 namespace4 four-window/four-replica fresh evaluation")
    entries = source.get("policies", {})
    if set(entries) - set(POLICIES):
        raise ValueError("unexpected policy entry")
    collections, policies, summaries, windows = {}, {}, [], []
    for policy in POLICIES:
        entry = entries.get(policy, {})
        directory = Path(entry.get("collection_path", root / policy))
        collection = collection_metrics(directory)
        summary = collection["summary"]
        if summary and (summary.get("parents_only") is not True or summary.get("round_index") != 4):
            raise ValueError("fresh evaluation requires namespace4 parent-only collection")
        collections[policy] = collection
        overall = collection["strata"]["all4"]
        policies[policy] = dict(
            execution_status=entry.get("status", "PENDING"),
            model_path=entry.get("model_path"),
            collection=collection,
            new_optimizer_updates=0,
            new_branch_requests=0,
            new_sampling_cost={key: summary.get(key) if summary else None for key in COST_KEYS},
        )
        summaries.append(
            dict(policy=policy, execution_status=entry.get("status", "PENDING"), **overall)
        )
        windows.extend(dict(policy=policy, **window) for window in collection["windows"])
    original = original_collections()
    comparisons, pairs, comparison_windows = {}, [], []
    for current, reference in CONTRASTS:
        name = f"{current}_minus_{reference}"
        rows, pairing = paired(name, collections[current], collections[reference])
        fresh = difference(collections[current], collections[reference])
        prior = difference(original[current], original[reference])
        stability = {}
        for stratum in ("all4", "without0073"):
            a, b = fresh["strata"][stratum], prior["strata"][stratum]
            stability[stratum] = dict(
                fresh_net_return_delta_pp=a["net_return_delta_pp"],
                original_namespace0_net_return_delta_pp=b["net_return_delta_pp"],
                fresh_direction=direction(a["net_return_delta_pp"]),
                original_direction=direction(b["net_return_delta_pp"]),
                net_direction_maintained=(
                    direction(a["net_return_delta_pp"]) == direction(b["net_return_delta_pp"])
                )
                if finite(a["net_return_delta_pp"]) and finite(b["net_return_delta_pp"])
                else None,
                fresh_Y_delta=a["Y_delta"],
                original_namespace0_Y_delta=b["Y_delta"],
                Y_direction_maintained=(direction(a["Y_delta"]) == direction(b["Y_delta"]))
                if finite(a["Y_delta"]) and finite(b["Y_delta"])
                else None,
            )
        comparisons[name] = dict(
            fresh=fresh,
            original_namespace0=prior,
            direction_stability=stability,
            paired_sampling=pairing,
        )
        pairs.extend(rows)
        for fresh_window, prior_window in zip(fresh["windows"], prior["windows"], strict=True):
            comparison_windows.append(
                dict(
                    comparison=name,
                    episode_id=fresh_window["episode_id"],
                    **{
                        f"fresh_{key}": value
                        for key, value in fresh_window.items()
                        if key != "episode_id"
                    },
                    original_namespace0_net_return_delta_pp=prior_window["net_return_delta_pp"],
                    original_namespace0_Y_delta=prior_window["Y_delta"],
                    net_direction_maintained=(
                        direction(fresh_window["net_return_delta_pp"])
                        == direction(prior_window["net_return_delta_pp"])
                    )
                    if finite(fresh_window["net_return_delta_pp"])
                    and finite(prior_window["net_return_delta_pp"])
                    else None,
                )
            )
    for row in summaries:
        policy = row["policy"]
        for reference in ("SFT", "RLOO"):
            if policy == reference:
                continue
            name = f"{policy}_minus_{reference}"
            if name in comparisons:
                delta = comparisons[name]
                for key, value in delta["fresh"]["strata"]["all4"].items():
                    row[f"versus_{reference}_{key}"] = value
                row[f"versus_{reference}_original_net_delta_pp"] = delta["original_namespace0"][
                    "strata"
                ]["all4"]["net_return_delta_pp"]
                row[f"versus_{reference}_net_direction_maintained"] = delta["direction_stability"][
                    "all4"
                ]["net_direction_maintained"]
        if policy != "SFT":
            row["versus_SFT_without0073_net_delta_pp"] = comparisons[f"{policy}_minus_SFT"][
                "fresh"
            ]["strata"]["without0073"]["net_return_delta_pp"]
    terminal = source.get("status") in TERMINAL or all(
        entry.get("status") in TERMINAL
        for entry in (entries.get(policy, {}) for policy in POLICIES)
    )
    complete = all(
        collection["strata"]["all4"]["status"] == "OBSERVED" for collection in collections.values()
    )
    costs = [policy["new_sampling_cost"] for policy in policies.values()]
    columns = list(dict.fromkeys(key for row in summaries for key in row))
    summaries = [{key: row.get(key) for key in columns} for row in summaries]
    report = dict(
        profile="fresh_rollout_namespace4_same_Train_windows_eval_v1",
        generated_at_UTC=datetime.now(timezone.utc).isoformat(),
        status="DESCRIBED"
        if terminal and complete
        else "DESCRIBED_WITH_FAILURES"
        if terminal
        else "PARTIAL",
        execution_status=source.get("status", "PENDING"),
        final_artifacts_allowed=terminal,
        seed=42,
        rollout_seed_namespace=4,
        episode_ids=EPISODES,
        parent_repetitions=4,
        planned_new_parent_trajectories=64,
        complete_new_parent_trajectories=sum(
            collection["strata"]["all4"]["complete_trajectories"]
            for collection in collections.values()
        ),
        new_optimizer_updates=0,
        new_branch_requests=0,
        policies=policies,
        comparisons=comparisons,
        original_namespace0_policy_directories={
            key: value["directory"] for key, value in original.items()
        },
        sampling_cost=dict(
            summaries_observed=sum(
                collection["summary"] is not None for collection in collections.values()
            ),
            planned_collections=4,
            known_totals={
                key: sum(cost[key] for cost in costs if finite(cost[key])) for key in COST_KEYS
            },
            incomplete_costs_are_lower_bounds=any(
                collection["summary"] is None for collection in collections.values()
            ),
            rule="four new parent-only policy collections once each; no reused training costs",
        ),
        limitations=[
            "Namespace4 was not used by these training runs; it changes rollout sampling seeds.",
            "These Train windows may have appeared in SFT; this is not new-date generalization.",
            "Each contrast has four windows with four replicas; paths are not independent windows.",
            "Initial prompts/first seeds match across policies; later action paths can diverge.",
            "Failed or missing replicas retain fixed16 pair denominators and whole-window NA.",
            "Changes versus fresh SFT and namespace0 are descriptive, not significance tests.",
        ],
    )
    return report, summaries, comparison_windows, pairs


def plot(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 4))
    labels = ("RLOO", "plain", "exact")
    for offset, scope, color in [
        (-0.18, "original_namespace0", "#90a4ae"),
        (0.18, "fresh", "#1976d2"),
    ]:
        values = [
            report["comparisons"][f"{p}_minus_SFT"][scope]["strata"]["all4"]["net_return_delta_pp"]
            for p in labels
        ]
        axis.bar(
            [i + offset for i in range(3)],
            [v if finite(v) else math.nan for v in values],
            width=0.34,
            color=color,
            label="Original namespace0" if scope != "fresh" else "Fresh namespace4",
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        xticks=range(3),
        xticklabels=labels,
        ylabel="Net return difference vs own SFT (pp)",
        title="Saved policies on the same four Train windows; no new training",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.text(
        0.5,
        0.01,
        "16 paired paths per comparison; missing means are omitted, not set to zero.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_readout(directory, result):
    report, summaries, windows, pairs = result
    if not report["final_artifacts_allowed"]:
        raise ValueError("fresh evaluation still running; use --preview")
    directory = Path(directory)
    names = (
        "fresh_readout.json",
        "summary.csv",
        "window_results.csv",
        "paired_results.csv",
        "paired_delta_comparison.png",
    )
    if any((directory / name).exists() for name in names):
        raise FileExistsError("fresh final artifacts exist; refusing overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "fresh_readout.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, rows in [
        ("summary.csv", summaries),
        ("window_results.csv", windows),
        ("paired_results.csv", pairs),
    ]:
        write_csv(directory / name, rows)
    plot(report, directory / "paired_delta_comparison.png")
    return directory / "fresh_readout.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    result = summarize(args.directory)
    if args.preview:
        print(
            json.dumps(
                dict(
                    status=result[0]["status"],
                    policies=result[1],
                    sampling_cost=result[0]["sampling_cost"],
                ),
                indent=2,
            )
        )
    else:
        print(write_readout(args.directory, result))


if __name__ == "__main__":
    main()

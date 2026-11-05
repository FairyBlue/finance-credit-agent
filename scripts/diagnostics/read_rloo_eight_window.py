#!/usr/bin/env python3
"""Describe the eight-window, two-seed RLOO replication from saved records."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from read_rloo_terminal_demo import (
    finite,
    mean,
    observed_trajectory,
    read,
    training_metrics,
    training_storage,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "runs/m2-rloo-eight-window-20261025-v1"
EPISODES = [f"formal-train-{i:04d}" for i in (1, 9, 17, 25, 65, 73, 149, 157)]
SEEDS = (42, 43)
TERMINAL = {"COMPLETE", "COMPLETE_WITH_MODEL_FAILURES", "FAILED"}
REPLICATES = range(4)
STRATA = {
    "all8": EPISODES,
    "common4": [f"formal-train-{i:04d}" for i in (1, 9, 65, 73)],
    "added4": [f"formal-train-{i:04d}" for i in (17, 25, 149, 157)],
    "without0073": [eid for eid in EPISODES if eid != "formal-train-0073"],
}


def window_metrics(episode, observed):
    members = [observed.get((episode, replicate)) for replicate in REPLICATES]
    actual = [row for row in members if row is not None]
    complete = all(
        row is not None
        and row["status"] == "COMPLETE"
        and row["terminal_session_index"] == 20
        and finite(row["net_return"])
        and finite(row["Y"])
        for row in members
    )
    requests = sum(row["applied_requests"] for row in actual)
    invalid = sum(row["invalid_requests"] for row in actual)
    return dict(
        episode_id=episode,
        status="OBSERVED" if complete else "PENDING_OR_INCOMPLETE",
        planned_trajectories=4,
        observed_trajectories=len(actual),
        complete_trajectories=sum(row["status"] == "COMPLETE" for row in actual),
        mean_net_return=mean([row["net_return"] for row in actual], 4) if complete else None,
        mean_Y=mean([row["Y"] for row in actual], 4) if complete else None,
        applied_requests=requests,
        invalid_requests=invalid,
        rejection_rate=invalid / requests if requests else None,
        mean_transaction_cost=mean([row["transaction_cost"] for row in actual], 4)
        if complete
        else None,
        fill_count=sum(row["fill_count"] for row in actual),
    )


def stratum_metrics(windows, episodes):
    members = [row for row in windows if row["episode_id"] in episodes]
    complete = len(members) == len(episodes) and all(row["status"] == "OBSERVED" for row in members)
    requests = sum(row["applied_requests"] for row in members)
    invalid = sum(row["invalid_requests"] for row in members)
    return dict(
        status="OBSERVED" if complete else "PENDING_OR_INCOMPLETE",
        planned_windows=len(episodes),
        planned_trajectories=4 * len(episodes),
        observed_trajectories=sum(row["observed_trajectories"] for row in members),
        complete_trajectories=sum(row["complete_trajectories"] for row in members),
        mean_net_return=mean([row["mean_net_return"] for row in members], len(episodes)),
        mean_Y=mean([row["mean_Y"] for row in members], len(episodes)),
        applied_requests=requests,
        invalid_requests=invalid,
        rejection_rate=invalid / requests if requests else None,
        mean_transaction_cost=mean(
            [row["mean_transaction_cost"] for row in members], len(episodes)
        ),
        fill_count=sum(row["fill_count"] for row in members),
    )


def collection_metrics(directory, seed):
    directory = Path(directory)
    summary_path = directory / "collection_summary.json"
    summary = read(summary_path) if summary_path.exists() else None
    if summary is not None and (
        summary.get("seed") != seed
        or summary.get("episode_ids") != EPISODES
        or summary.get("parent_repetitions") != 4
        or summary.get("parents_only") is not True
    ):
        raise ValueError("collection is not this seed's eight-window grouped parent batch")
    observed = {}
    path = directory / "trajectory_results.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("role") != "parent":
                raise ValueError("parent-only replication contains a head/EST record")
            key = row["episode_id"], row["replicate_index"]
            if key[0] not in EPISODES or key[1] not in REPLICATES or key in observed:
                raise ValueError("unexpected/duplicate parent slot")
            observed[key] = observed_trajectory(row)
    windows = [window_metrics(episode, observed) for episode in EPISODES]
    return dict(
        directory=str(directory),
        summary=summary,
        windows=windows,
        strata={key: stratum_metrics(windows, episodes) for key, episodes in STRATA.items()},
        trajectories=[observed.get((episode, rep)) for episode in EPISODES for rep in REPLICATES],
    )


def compare(current, baseline):
    windows = []
    for now, initial in zip(current["windows"], baseline["windows"], strict=True):
        complete = now["status"] == initial["status"] == "OBSERVED"
        windows.append(
            dict(
                episode_id=now["episode_id"],
                status="OBSERVED" if complete else "PENDING_OR_INCOMPLETE",
                net_return_delta_pp=100 * (now["mean_net_return"] - initial["mean_net_return"])
                if complete
                else None,
                Y_delta=now["mean_Y"] - initial["mean_Y"] if complete else None,
            )
        )
    strata = {}
    for name, episodes in STRATA.items():
        members = [row for row in windows if row["episode_id"] in episodes]
        deltas = [row["net_return_delta_pp"] for row in members]
        ys = [row["Y_delta"] for row in members]
        complete = len(members) == len(episodes) and all(map(finite, deltas + ys))
        strata[name] = dict(
            status="OBSERVED" if complete else "PENDING_OR_INCOMPLETE",
            planned_windows=len(episodes),
            net_return_delta_pp=mean(deltas, len(episodes)),
            Y_delta=mean(ys, len(episodes)),
            positive_windows=sum(value > 1e-12 for value in deltas) if complete else None,
            negative_windows=sum(value < -1e-12 for value in deltas) if complete else None,
            tied_windows=sum(abs(value) <= 1e-12 for value in deltas) if complete else None,
        )
    specific = next(row for row in windows if row["episode_id"] == "formal-train-0073")
    contribution = (
        specific["net_return_delta_pp"] / 8 if finite(specific["net_return_delta_pp"]) else None
    )
    overall = strata["all8"]["net_return_delta_pp"]
    return dict(
        strata=strata,
        windows=windows,
        window0073_contribution_to_eight_window_mean_pp=contribution,
        window0073_fraction_of_net_mean_change=contribution / overall
        if finite(contribution) and finite(overall) and abs(overall) > 1e-12
        else None,
        concentration_note="Contribution and leave-one-window-out means are descriptive.",
    )


def paired_final(seed, current, baseline):
    rows = []
    for index, (now, initial) in enumerate(
        zip(current["trajectories"], baseline["trajectories"], strict=True)
    ):
        complete = (
            now is not None
            and initial is not None
            and now["status"] == initial["status"] == "COMPLETE"
            and now["terminal_session_index"] == initial["terminal_session_index"] == 20
            and finite(now["net_return"])
            and finite(initial["net_return"])
            and finite(now["Y"])
            and finite(initial["Y"])
        )
        rows.append(
            dict(
                seed=seed,
                episode_id=EPISODES[index // 4],
                replicate_index=index % 4,
                status="OBSERVED" if complete else "PENDING_OR_INCOMPLETE",
                baseline_net_return=initial["net_return"] if complete else None,
                final_net_return=now["net_return"] if complete else None,
                net_return_delta_pp=100 * (now["net_return"] - initial["net_return"])
                if complete
                else None,
                Y_delta=now["Y"] - initial["Y"] if complete else None,
                initial_prompt_equal=(
                    now["initial_prompt_sha256"] is not None
                    and now["initial_prompt_sha256"] == initial["initial_prompt_sha256"]
                )
                if complete
                else None,
                initial_seed_equal=(
                    type(now["initial_request_seed"]) is int
                    and now["initial_request_seed"] == initial["initial_request_seed"]
                )
                if complete
                else None,
                baseline_invalid_requests=initial["invalid_requests"] if complete else None,
                final_invalid_requests=now["invalid_requests"] if complete else None,
            )
        )
    actual = [row for row in rows if row["status"] == "OBSERVED"]
    summary = dict(
        planned_pairs=32,
        complete_pairs=len(actual),
        net_return_delta_pp=mean([row["net_return_delta_pp"] for row in actual], 32),
        Y_delta=mean([row["Y_delta"] for row in actual], 32),
        initial_prompt_equal_pairs=sum(row["initial_prompt_equal"] is True for row in actual),
        initial_seed_equal_pairs=sum(row["initial_seed_equal"] is True for row in actual),
        namespace0_pairing=(
            current["summary"] is not None
            and baseline["summary"] is not None
            and current["summary"]["round_index"] == baseline["summary"]["round_index"] == 0
        ),
    )
    return rows, summary


def seed_readout(directory, seed):
    path = directory / "rloo_results.json"
    if not path.exists():
        path = directory / "initial_results.json"
    source = read(path) if path.exists() else {}
    if source and (
        source.get("seed") != seed
        or source.get("episode_ids") != EPISODES
        or source.get("planned_updates") != 16
        or source.get("parent_repetitions") != 4
    ):
        raise ValueError("unexpected seed/window/update replication identity")
    entries = source.get("rounds", [])
    indexed = {row["round"]: row for row in entries}
    if len(indexed) != len(entries) or any(number not in range(1, 5) for number in indexed):
        raise ValueError("unexpected or duplicate replication round")
    baseline = collection_metrics(source.get("baseline_path", directory / "common_initial"), seed)
    curve, windows_csv, curve_csv = [], [], []
    for number in range(5):
        entry = indexed.get(number, {})
        current = (
            baseline
            if number == 0
            else collection_metrics(
                entry.get("collection_path", directory / f"round{number}" / "eval"), seed
            )
        )
        delta = compare(current, baseline)
        train = training_metrics(entry.get("training")) if number else None
        point = dict(
            seed=seed,
            round=number,
            optimizer_updates=number * 4,
            execution_status="BASELINE" if number == 0 else entry.get("status", "PENDING"),
            evaluation_seed_namespace=0 if number in (0, 4) else number,
            evaluation=current,
            versus_own_common_SFT=delta,
            training=train,
        )
        curve.append(point)
        overall = current["strata"]["all8"]
        curve_csv.append(
            dict(
                seed=seed,
                optimizer_updates=number * 4,
                execution_status=point["execution_status"],
                **overall,
                net_return_delta_pp=delta["strata"]["all8"]["net_return_delta_pp"],
                Y_delta=delta["strata"]["all8"]["Y_delta"],
                evaluation_seed_namespace=point["evaluation_seed_namespace"],
            )
        )
        for window, difference in zip(current["windows"], delta["windows"], strict=True):
            windows_csv.append(
                dict(
                    seed=seed,
                    optimizer_updates=number * 4,
                    **window,
                    net_return_delta_pp=difference["net_return_delta_pp"],
                    Y_delta=difference["Y_delta"],
                )
            )
    pairs, pairing = paired_final(seed, curve[-1]["evaluation"], baseline)
    driver_complete = (
        source.get("status") in {"COMPLETE", "COMPLETE_WITH_MODEL_FAILURES"}
        and source.get("completed_updates") == 16
        and len(entries) == 4
        and all(point["evaluation"]["summary"] is not None for point in curve)
        and all(point["training"]["status"] == "COMPLETE" for point in curve[1:])
        and sum(point["training"]["optimizer_updates"] for point in curve[1:]) == 16
    )
    metric_complete = (
        driver_complete
        and all(point["evaluation"]["strata"]["all8"]["status"] == "OBSERVED" for point in curve)
        and pairing["complete_pairs"]
        == pairing["initial_prompt_equal_pairs"]
        == pairing["initial_seed_equal_pairs"]
        == 32
        and pairing["namespace0_pairing"]
    )
    if pairing["complete_pairs"] and (
        pairing["initial_prompt_equal_pairs"] != pairing["complete_pairs"]
        or pairing["initial_seed_equal_pairs"] != pairing["complete_pairs"]
        or not pairing["namespace0_pairing"]
    ):
        raise ValueError("observed final pairs violate declared prompt/seed identity")
    terminal = source.get("status") in TERMINAL
    status = (
        "DESCRIBED"
        if metric_complete
        else "DESCRIBED_WITH_MODEL_FAILURES"
        if driver_complete
        else "DESCRIBED_WITH_DRIVER_FAILURES"
        if terminal
        else "PARTIAL"
        if source
        else "PENDING"
    )
    available = [
        point["evaluation"]["summary"]
        for point in curve
        if point["evaluation"]["summary"] is not None
    ]
    storage = training_storage(source)
    report = dict(
        seed=seed,
        status=status,
        execution_status=source.get("status", "PENDING"),
        driver_terminal=terminal,
        all_planned_updates_completed=driver_complete,
        all_planned_metrics_observed=metric_complete,
        source_results=str(path),
        curve=curve,
        final_versus_own_SFT=curve[-1]["versus_own_common_SFT"],
        final_paired_sampling=pairing,
        sampling_cost={
            key: sum(item.get(key, 0) for item in available)
            for key in ("model_requests", "prompt_tokens", "response_tokens")
        },
        baseline_sampling_cost={
            key: baseline["summary"].get(key, 0) if baseline["summary"] is not None else None
            for key in ("model_requests", "prompt_tokens", "response_tokens")
        },
        complete_collection_summaries=len(available),
        training_request_exposures=sum(
            entry["training"].get("training_rows", 0) for entry in entries if entry.get("training")
        ),
        training_invalid_exposures=sum(
            entry["training"].get("invalid_rows", 0) for entry in entries if entry.get("training")
        ),
        training_state=source.get("training_state"),
        training_state_storage=storage,
        cost_note="Each of this seed's five newly generated collections is counted once, "
        "even when a collection supplies both evaluation and subsequent training.",
    )
    first, last = (
        curve[0]["evaluation"]["strata"]["all8"],
        curve[-1]["evaluation"]["strata"]["all8"],
    )
    summary_csv = dict(
        seed=seed,
        status=report["status"],
        baseline_net_return=first["mean_net_return"],
        final_net_return=last["mean_net_return"],
        baseline_Y=first["mean_Y"],
        final_Y=last["mean_Y"],
        **{
            f"{key}_net_return_delta_pp": value["net_return_delta_pp"]
            for key, value in report["final_versus_own_SFT"]["strata"].items()
        },
        baseline_rejection_rate=first["rejection_rate"],
        final_rejection_rate=last["rejection_rate"],
        baseline_invalid_requests=first["invalid_requests"],
        final_invalid_requests=last["invalid_requests"],
        baseline_mean_transaction_cost=first["mean_transaction_cost"],
        final_mean_transaction_cost=last["mean_transaction_cost"],
        complete_pairs=pairing["complete_pairs"],
        planned_pairs=32,
        optimizer_updates=sum(
            point["training"].get("optimizer_updates") or 0 for point in curve[1:]
        ),
        training_request_exposures=report["training_request_exposures"],
        training_invalid_exposures=report["training_invalid_exposures"],
        model_requests=report["sampling_cost"]["model_requests"],
        prompt_tokens=report["sampling_cost"]["prompt_tokens"],
        response_tokens=report["sampling_cost"]["response_tokens"],
    )
    return report, summary_csv, curve_csv, windows_csv, pairs


def summarize(directory=DEFAULT):
    directory = Path(directory).resolve()
    seeds, summaries, curves, windows, pairs = [], [], [], [], []
    for seed in SEEDS:
        report, summary, curve, window, paired = seed_readout(directory / f"seed{seed}", seed)
        seeds.append(report)
        summaries.append(summary)
        curves.extend(curve)
        windows.extend(window)
        pairs.extend(paired)
    complete = all(seed["status"] == "DESCRIBED" for seed in seeds)
    terminal = all(seed["driver_terminal"] for seed in seeds)
    all_updates = all(seed["all_planned_updates_completed"] for seed in seeds)
    deltas = [
        seed["final_versus_own_SFT"]["strata"]["all8"]["net_return_delta_pp"] for seed in seeds
    ]
    report = dict(
        profile="eight_window_two_seed_terminal_RLOO_replication_readout_v1",
        status=(
            "DESCRIBED"
            if complete
            else "DESCRIBED_WITH_MODEL_FAILURES"
            if terminal and all_updates
            else "DESCRIBED_WITH_DRIVER_FAILURES"
            if terminal
            else "PARTIAL"
        ),
        all_seed_drivers_terminal=terminal,
        all_planned_updates_completed=all_updates,
        all_planned_metrics_observed=complete,
        generated_at_UTC=datetime.now(timezone.utc).isoformat(),
        episode_ids=EPISODES,
        seeds=seeds,
        planned_updates_per_seed=16,
        planned_trajectories_per_collection=32,
        strata=STRATA,
        both_seed_net_changes_positive=all(value > 0 for value in deltas)
        if all(map(finite, deltas))
        else None,
        descriptive_mean_of_two_seed_net_changes_pp=mean(deltas, 2),
        total_sampling_cost={
            key: sum(seed["sampling_cost"][key] for seed in seeds)
            for key in ("model_requests", "prompt_tokens", "response_tokens")
        },
        limits=[
            "Eight previously used Train windows, seeds42/43, 16 updates per seed; no Valid/OOS.",
            "Per-window four-replica and per-seed 32-pair denominators stay fixed; missing is NA.",
            "Only initial/final share sampling seeds; intermediate curves use fresh namespaces.",
            "Initial data train each seed; paired finals remain in-sample development evidence.",
            "Common4/added4 and leave0073-out are descriptive strata, not independent datasets.",
            "Seeds are reported separately; windows/replicas across seeds are not pooled as iid.",
            "Baseline and post-update generation costs count each newly collected batch once.",
            "FP32/Adam state is saved on project disk; metadata records its actual location.",
        ],
    )
    return report, summaries, curves, windows, pairs


def plot(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    for ax, result in zip(axes, report["seeds"], strict=True):
        values = [
            100 * value if finite(value) else float("nan")
            for p in result["curve"]
            for value in [p["evaluation"]["strata"]["all8"]["mean_net_return"]]
        ]
        ax.plot([0, 4, 8, 12, 16], values, "o-", color="#355b89", label="RLOO")
        if math.isfinite(values[0]):
            ax.axhline(values[0], color="#6b7280", linestyle="--", label="Own SFT baseline")
        if math.isfinite(values[-1]):
            ax.scatter(16, values[-1], marker="D", color="#8b4a21", zorder=4, label="Paired final")
        ax.set(
            title=f"Seed {result['seed']}",
            xlabel="Effective optimizer updates",
            ylabel="Eight-window mean net return (%)",
            xticks=[0, 4, 8, 12, 16],
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Eight-window RLOO replication")
    fig.text(
        0.5,
        0.015,
        "Initial/final share sampling seeds; intermediate points use fresh seeds. "
        "All windows are previously used Train data.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"learning_curves.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_readout(directory=DEFAULT):
    directory = Path(directory).resolve()
    report, summaries, curves, windows, pairs = summarize(directory)
    if not report["all_seed_drivers_terminal"]:
        raise RuntimeError("both seed drivers must finish; use --preview while they run")
    names = (
        "comparison_readout.json",
        "seed_summary.csv",
        "curve_results.csv",
        "window_results.csv",
        "paired_results.csv",
        "learning_curves.png",
        "learning_curves.svg",
    )
    if any((directory / name).exists() for name in names):
        raise FileExistsError("final readout already exists; refusing overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(names[1:5], (summaries, curves, windows, pairs), strict=True):
        write_csv(directory / name, rows)
    plot(report, directory)
    (directory / names[0]).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    report = summarize(args.directory)[0] if args.preview else write_readout(args.directory)
    print(
        json.dumps(
            dict(
                status=report["status"],
                seeds=[
                    dict(
                        seed=result["seed"],
                        status=result["status"],
                        final=result["final_versus_own_SFT"]["strata"],
                    )
                    for result in report["seeds"]
                ],
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

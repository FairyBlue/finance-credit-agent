#!/usr/bin/env python3
"""Describe the fixed RLOO demo from saved results; no replay or generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "runs/m2-rloo-terminal-demo-20261025-v3"
EPISODES = [f"formal-train-{number:04d}" for number in (1, 9, 65, 73)]
REPLICATES = range(4)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def mean(values, planned):
    complete = len(values) == planned and all(map(finite, values))
    return math.fsum(values) / planned if complete else None


def read(path):
    return json.loads(Path(path).read_text())


def training_storage(source):
    path = source.get("training_state")
    location = Path(path).resolve() if path else None
    if location is not None and location.is_relative_to(ROOT / "runs"):
        return dict(
            kind="project_disk",
            host_reboot_volatile=False,
            description="FP32/Adam shards are on project disk and persist across host reboot.",
        )
    if location is not None and any(
        location.is_relative_to(root) for root in (Path("/dev/shm"), Path("/run"))
    ):
        return dict(
            kind="RAM_filesystem",
            host_reboot_volatile=True,
            description="FP32/Adam shards are on RAM filesystem and do not survive host reboot.",
        )
    return dict(
        kind="unknown",
        host_reboot_volatile=None,
        description="Training-state persistence is unknown until its storage path is recorded.",
    )


def observed_trajectory(row):
    steps = row.get("timesteps", [])
    fills = [fill for step in steps for fill in (step.get("info", {}).get("fills") or [])]
    costs = [fill.get("commission") for fill in fills] + [
        fill.get("slippage_cost") for fill in fills
    ]
    metrics = row.get("terminal_metrics") or {}
    complete = row.get("status") == "COMPLETE"
    requests = len(steps)
    invalid = sum(step.get("invalid") is True for step in steps)
    return dict(
        episode_id=row["episode_id"],
        replicate_index=row["replicate_index"],
        trajectory_key=row["trajectory_key"],
        status=row.get("status"),
        Y=row.get("terminal_financial_Y", row.get("Y")) if complete else None,
        net_return=metrics.get("raw_net_return") if complete else None,
        applied_requests=requests,
        invalid_requests=invalid,
        rejection_rate=invalid / requests if requests else None,
        advancing_requests=sum(step.get("advancing") is True for step in steps),
        transaction_cost=math.fsum(costs) if all(map(finite, costs)) else None,
        fill_count=len(fills),
        action_family_counts=dict(Counter(step.get("action_family") for step in steps)),
        terminal_session_index=row.get("last_session_index"),
        initial_prompt_sha256=hashlib.sha256(
            json.dumps(steps[0]["prompt_token_ids"], separators=(",", ":")).encode()
        ).hexdigest()
        if steps
        else None,
        initial_request_seed=steps[0].get("request_seed") if steps else None,
    )


def collection_metrics(directory):
    directory = Path(directory) if directory is not None else None
    summary_path = directory / "collection_summary.json" if directory else None
    summary = read(summary_path) if summary_path and summary_path.exists() else None
    observed = {}
    path = directory / "trajectory_results.jsonl" if directory else None
    if path is not None and path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("role") != "parent":
                raise ValueError("RLOO parent-only data must not include head/EST records")
            key = row["episode_id"], row["replicate_index"]
            if key[0] not in EPISODES or key[1] not in REPLICATES or key in observed:
                raise ValueError("unexpected or duplicate grouped parent")
            observed[key] = observed_trajectory(row)
    slots = [observed.get((episode, replicate)) for episode in EPISODES for replicate in REPLICATES]
    windows = []
    for episode in EPISODES:
        members = [observed.get((episode, replicate)) for replicate in REPLICATES]
        complete = all(member is not None and member["status"] == "COMPLETE" for member in members)
        actual = [member for member in members if member is not None]
        requests = sum(member["applied_requests"] for member in actual)
        invalid = sum(member["invalid_requests"] for member in actual)
        windows.append(
            dict(
                episode_id=episode,
                status="OBSERVED" if complete else "PENDING_OR_INCOMPLETE",
                planned_trajectories=4,
                observed_trajectories=len(actual),
                complete_trajectories=sum(member["status"] == "COMPLETE" for member in actual),
                mean_net_return=mean([member["net_return"] for member in actual], 4)
                if complete
                else None,
                mean_Y=mean([member["Y"] for member in actual], 4) if complete else None,
                applied_requests=requests,
                invalid_requests=invalid,
                rejection_rate=invalid / requests if requests else None,
                mean_transaction_cost=mean([member["transaction_cost"] for member in actual], 4)
                if complete
                else None,
            )
        )
    actual = [slot for slot in slots if slot is not None]
    complete = all(slot is not None and slot["status"] == "COMPLETE" for slot in slots)
    requests = sum(slot["applied_requests"] for slot in actual)
    invalid = sum(slot["invalid_requests"] for slot in actual)
    overall = dict(
        status="OBSERVED" if complete else "PENDING_OR_INCOMPLETE",
        planned_windows=4,
        planned_trajectories=16,
        observed_trajectories=len(actual),
        complete_trajectories=sum(slot["status"] == "COMPLETE" for slot in actual),
        mean_net_return=mean([window["mean_net_return"] for window in windows], 4),
        mean_Y=mean([window["mean_Y"] for window in windows], 4),
        observed_applied_requests=requests,
        observed_invalid_requests=invalid,
        rejection_rate=invalid / requests if requests else None,
        mean_transaction_cost=mean([window["mean_transaction_cost"] for window in windows], 4),
        fill_count=sum(slot["fill_count"] for slot in actual),
    )
    return dict(
        source_directory=str(directory) if directory else None,
        summary=summary,
        overall=overall,
        windows=windows,
        trajectories=slots,
    )


def compare(current, baseline):
    result = []
    for now, initial in zip(current["windows"], baseline["windows"], strict=True):
        complete = now["status"] == initial["status"] == "OBSERVED"
        result.append(
            dict(
                episode_id=now["episode_id"],
                status="OBSERVED" if complete else "PENDING_OR_INCOMPLETE",
                net_return_delta_pp=100 * (now["mean_net_return"] - initial["mean_net_return"])
                if complete
                else None,
                Y_delta=now["mean_Y"] - initial["mean_Y"] if complete else None,
            )
        )
    deltas = [window["net_return_delta_pp"] for window in result]
    ys = [window["Y_delta"] for window in result]
    complete = all(map(finite, deltas + ys))
    return dict(
        mean_net_return_delta_pp=mean(deltas, 4),
        mean_Y_delta=mean(ys, 4),
        positive_windows=sum(value > 1e-12 for value in deltas) if complete else None,
        negative_windows=sum(value < -1e-12 for value in deltas) if complete else None,
        tied_windows=sum(abs(value) <= 1e-12 for value in deltas) if complete else None,
        windows=result,
    )


def paired_final(final, baseline):
    rows = []
    pairs = zip(final["trajectories"], baseline["trajectories"], strict=True)
    for index, (now, initial) in enumerate(pairs):
        complete = (
            now is not None
            and initial is not None
            and now["status"] == initial["status"] == "COMPLETE"
        )
        rows.append(
            dict(
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
                    now["initial_prompt_sha256"] == initial["initial_prompt_sha256"]
                )
                if complete
                else None,
                initial_seed_equal=now["initial_request_seed"] == initial["initial_request_seed"]
                if complete
                else None,
                initial_invalid_requests=initial["invalid_requests"] if complete else None,
                final_invalid_requests=now["invalid_requests"] if complete else None,
            )
        )
    return rows


def training_metrics(summary):
    if summary is None:
        return dict(status="PENDING", optimizer_updates=None, completed_updates=None)
    diagnostics = summary.get("logprob_diagnostics", [])
    usable = [
        row
        for row in diagnostics
        if finite(row.get("advantage")) and finite(row.get("delta")) and row["advantage"] != 0
    ]
    positive = [row["delta"] for row in usable if row["advantage"] > 0]
    negative = [row["delta"] for row in usable if row["advantage"] < 0]
    keys = (
        "status",
        "training_rows",
        "invalid_rows",
        "admitted_groups",
        "planned_groups",
        "actor_updates",
        "optimizer_updates",
        "completed_updates",
        "optimizer_steps_before",
        "optimizer_steps_after",
        "master_parameter_delta",
        "BF16_parameter_delta",
        "coverage",
        "persistent_state",
    )
    return {
        **{key: summary.get(key) for key in keys},
        "sampled_logprob_diagnostics": diagnostics,
        "logprob_direction": dict(
            sampled_rows=len(usable),
            positive_advantage_rows=len(positive),
            negative_advantage_rows=len(negative),
            positive_advantage_mean_logprob_delta=mean(positive, len(positive))
            if positive
            else None,
            negative_advantage_mean_logprob_delta=mean(negative, len(negative))
            if negative
            else None,
            advantage_weighted_mean_delta=math.fsum(
                row["advantage"] * row["delta"] for row in usable
            )
            / len(usable)
            if usable
            else None,
            aligned_rows=sum(row["advantage"] * row["delta"] > 0 for row in usable),
            scope="saved diagnostic sample only, not all requests or all model parameters",
        ),
    }


def summarize(directory=DEFAULT):
    directory = Path(directory).resolve()
    source_path = directory / "rloo_results.json"
    if not source_path.exists():
        source_path = directory / "initial_results.json"
    source = read(source_path) if source_path.exists() else {}
    if source and (
        source.get("planned_updates") != 16
        or source.get("seed") != 42
        or source.get("episode_ids") != EPISODES
        or source.get("parent_repetitions") != 4
    ):
        raise ValueError("reader requires the fixed seed42 four-window grouped RLOO demo")
    entries = source.get("rounds", [])
    indexed = {entry["round"]: entry for entry in entries}
    if len(indexed) != len(entries) or any(number not in range(1, 5) for number in indexed):
        raise ValueError("unexpected or duplicate RLOO round")
    baseline = collection_metrics(source.get("baseline_path", directory / "common_initial"))
    curve, csv_rows = [], []
    for number in range(5):
        entry = indexed.get(number, {})
        collection = (
            baseline
            if number == 0
            else collection_metrics(
                entry.get("collection_path", directory / f"round{number}" / "eval")
            )
        )
        difference = compare(collection, baseline)
        training = training_metrics(entry.get("training")) if number else None
        point = dict(
            round=number,
            optimizer_updates=number * 4,
            execution_status="BASELINE" if number == 0 else entry.get("status", "PENDING"),
            evaluation_seed_namespace=0 if number in (0, 4) else number,
            paired_with_initial_sampling_seeds=number == 4,
            evaluation=collection,
            versus_common_SFT=difference,
            training=training,
            error=entry.get("error"),
        )
        curve.append(point)
        for window, delta in zip(collection["windows"], difference["windows"], strict=True):
            csv_rows.append(
                dict(
                    round=number,
                    optimizer_updates=number * 4,
                    episode_id=window["episode_id"],
                    evaluation_status=window["status"],
                    planned_trajectories=4,
                    complete_trajectories=window["complete_trajectories"],
                    mean_net_return=window["mean_net_return"],
                    mean_net_return_pct=100 * window["mean_net_return"]
                    if finite(window["mean_net_return"])
                    else None,
                    mean_Y=window["mean_Y"],
                    net_return_delta_pp=delta["net_return_delta_pp"],
                    Y_delta=delta["Y_delta"],
                    applied_requests=window["applied_requests"],
                    invalid_requests=window["invalid_requests"],
                    rejection_rate=window["rejection_rate"],
                    mean_transaction_cost=window["mean_transaction_cost"],
                )
            )
    pairs = paired_final(curve[-1]["evaluation"], baseline)
    observed = [pair for pair in pairs if pair["status"] == "OBSERVED"]
    paired = dict(
        planned_pairs=16,
        complete_pairs=len(observed),
        mean_net_return_delta_pp=mean([pair["net_return_delta_pp"] for pair in observed], 16),
        mean_Y_delta=mean([pair["Y_delta"] for pair in observed], 16),
        initial_prompt_equal_pairs=sum(pair["initial_prompt_equal"] is True for pair in observed),
        initial_seed_equal_pairs=sum(pair["initial_seed_equal"] is True for pair in observed),
        declared_namespace_match=(
            baseline["summary"] is not None
            and curve[-1]["evaluation"]["summary"] is not None
            and baseline["summary"]["round_index"]
            == curve[-1]["evaluation"]["summary"]["round_index"]
            == 0
        ),
    )
    complete = (
        source.get("status") == "COMPLETE"
        and source.get("completed_updates") == 16
        and len(indexed) == 4
        and all(point["evaluation"]["overall"]["status"] == "OBSERVED" for point in curve)
        and all(point["training"]["status"] == "COMPLETE" for point in curve[1:])
        and paired["complete_pairs"]
        == paired["initial_prompt_equal_pairs"]
        == paired["initial_seed_equal_pairs"]
        == 16
        and paired["declared_namespace_match"]
    )
    collections = [point["evaluation"]["summary"] for point in curve]
    available = [item for item in collections if item is not None]
    storage = training_storage(source)
    report = dict(
        profile="persistent_FP32_terminal_Y_RLOO_demo_readout_v3",
        status="DESCRIBED" if complete else "PARTIAL" if source else "PENDING",
        execution_status=source.get("status", "PENDING"),
        generated_at_UTC=datetime.now(timezone.utc).isoformat(),
        source_results=str(source_path),
        episode_ids=EPISODES,
        planned_updates=16,
        parent_repetitions=4,
        curve=curve,
        final_versus_common_SFT=curve[-1]["versus_common_SFT"],
        final_paired_sampling=paired,
        known_completed_collection_cost={
            key: sum(item.get(key, 0) for item in available)
            for key in ("model_requests", "prompt_tokens", "response_tokens")
        },
        new_post_update_collection_cost={
            key: sum(item.get(key, 0) for item in collections[1:] if item is not None)
            for key in ("model_requests", "prompt_tokens", "response_tokens")
        },
        reused_baseline_collection_cost={
            key: collections[0].get(key, 0) if collections[0] is not None else None
            for key in ("model_requests", "prompt_tokens", "response_tokens")
        },
        failed_run_updates_replayed_from_SFT=source.get("failed_run_updates_replayed_from_SFT"),
        complete_collection_summaries=len(available),
        total_training_request_exposures=sum(
            entry["training"].get("training_rows", 0) for entry in entries if entry.get("training")
        ),
        total_training_invalid_exposures=sum(
            entry["training"].get("invalid_rows", 0) for entry in entries if entry.get("training")
        ),
        training_state=source.get("training_state"),
        training_state_storage=storage["description"],
        training_state_storage_kind=storage["kind"],
        training_state_is_host_reboot_volatile=storage["host_reboot_volatile"],
        limits=[
            "Four reused Train windows, seed42, 16 optimizer updates; no Valid/OOS.",
            "Each window mean requires all four replicas; missing values remain NA.",
            "Interim evaluations use fresh seed namespaces; their differences are descriptive.",
            "Final and initial use namespace0, matching first prompts and request seeds.",
            "Initial trajectories also train the model; paired final data are in-sample evidence.",
            "Master and BF16 parameter probes and logprob diagnostics are sampled checks only.",
            storage["description"],
            "Precision, rewards, baseline and updates change together; no single-factor claim.",
        ],
    )
    return report, csv_rows, pairs


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curve(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = report["curve"]
    updates = [point["optimizer_updates"] for point in points]
    returns = [100 * point["evaluation"]["overall"]["mean_net_return"] for point in points]
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    ax.plot(updates, returns, "o-", color="#355b89", linewidth=1.8, label="RLOO curve")
    ax.axhline(returns[0], color="#6b7280", linestyle="--", label="Common SFT baseline")
    ax.scatter(updates[-1], returns[-1], marker="D", s=66, color="#8b4a21", zorder=4,
               label="Final: paired initial sampling seeds")
    for x, y in zip(updates, returns, strict=True):
        ax.annotate(f"{y:.3f}%", (x, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set(title="RLOO terminal-reward demo", xlabel="Effective optimizer updates",
           ylabel="Mean net return (%)", xticks=updates, xlim=(-0.6, 16.6))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8.5, loc="best")
    fig.text(0.5, 0.055,
             "Four Train windows, seed42. Initial/final share sampling seeds; "
             "updates 4-12 use fresh seeds.\n"
             "Interim differences are descriptive; the final paired comparison is in-sample.",
             ha="center", fontsize=8.5, color="#4b5563")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"learning_curve.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_readout(directory=DEFAULT):
    directory = Path(directory).resolve()
    report, windows, pairs = summarize(directory)
    if report["status"] != "DESCRIBED":
        raise RuntimeError(
            "use --preview for a pending or partial run; final artifacts not written"
        )
    paths = [
        directory / name
        for name in (
            "rloo_readout.json",
            "window_results.csv",
            "paired_results.csv",
            "curve_results.csv",
            "learning_curve.png",
            "learning_curve.svg",
        )
    ]
    if any(path.exists() for path in paths):
        raise FileExistsError("readout exists; refusing overwrite")
    write_csv(paths[1], windows)
    write_csv(paths[2], pairs)
    curve = []
    for point in report["curve"]:
        overall = point["evaluation"]["overall"]
        curve.append(
            {
                "optimizer_updates": point["optimizer_updates"],
                **overall,
                "net_return_delta_pp": point["versus_common_SFT"]["mean_net_return_delta_pp"],
                "Y_delta": point["versus_common_SFT"]["mean_Y_delta"],
                "evaluation_seed_namespace": point["evaluation_seed_namespace"],
                "paired_with_initial_sampling_seeds": point["paired_with_initial_sampling_seeds"],
            }
        )
    write_csv(paths[3], curve)
    plot_learning_curve(report, directory)
    report["learning_curve_files"] = [str(path) for path in paths[4:]]
    paths[0].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
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
                execution_status=report["execution_status"],
                final=report["final_versus_common_SFT"],
                paired=report["final_paired_sampling"],
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read the bounded RLOO/plain/exact demo without generation or checkpoint loading."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from read_rloo_eight_window import stratum_metrics, window_metrics
from read_rloo_terminal_demo import (
    EPISODES,
    finite,
    mean,
    observed_trajectory,
    read,
    training_metrics,
    training_storage,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "runs/m2-rloo-local-credit-demo-20261025-v1"
RLOO = ROOT / "runs/m2-rloo-terminal-demo-20261025-v3"
ARMS = ("RLOO", "plain", "exact")
TERMINAL = {"COMPLETE", "COMPLETE_WITH_MODEL_FAILURES", "FAILED"}
STRATA = {"all4": EPISODES, "without0073": EPISODES[:-1]}
COST_KEYS = ("model_requests", "prompt_tokens", "response_tokens")


def optional(path):
    return read(path) if Path(path).exists() else None


def collection_metrics(directory):
    """Only the sixteen original parent slots contribute evaluation metrics."""
    directory = Path(directory)
    summary = optional(directory / "collection_summary.json")
    if summary and (
        summary.get("episode_ids") != EPISODES
        or summary.get("seed") != 42
        or summary.get("parent_repetitions") != 4
    ):
        raise ValueError("expected fixed four Train windows, seed42 and four parent replicas")
    observed = {}
    path = directory / "trajectory_results.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("role") != "parent":
                continue
            key = row["episode_id"], row["replicate_index"]
            if key[0] not in EPISODES or type(key[1]) is not int or key[1] not in range(4):
                raise ValueError("unexpected parent slot")
            if key in observed:
                raise ValueError("duplicate parent slot")
            observed[key] = observed_trajectory(row)
    windows = [window_metrics(episode, observed) for episode in EPISODES]
    return dict(
        directory=str(directory),
        summary=summary,
        windows=windows,
        strata={name: stratum_metrics(windows, ids) for name, ids in STRATA.items()},
        trajectories=[observed.get((episode, rep)) for episode in EPISODES for rep in range(4)],
    )


def difference(now, initial):
    def subtract(a, b, scale=1):
        return scale * (a - b) if finite(a) and finite(b) else None

    def metrics(a, b):
        return dict(
            net_return_delta_pp=subtract(a["mean_net_return"], b["mean_net_return"], 100),
            Y_delta=subtract(a["mean_Y"], b["mean_Y"]),
            rejection_rate_delta_pp=subtract(a["rejection_rate"], b["rejection_rate"], 100),
            mean_transaction_cost_delta=subtract(
                a["mean_transaction_cost"], b["mean_transaction_cost"]
            ),
        )

    windows = [
        dict(episode_id=a["episode_id"], **metrics(a, b))
        for a, b in zip(now["windows"], initial["windows"], strict=True)
    ]
    strata = {}
    for name, ids in STRATA.items():
        deltas = [w["net_return_delta_pp"] for w in windows if w["episode_id"] in ids]
        complete = all(map(finite, deltas))
        strata[name] = dict(
            **metrics(now["strata"][name], initial["strata"][name]),
            planned_windows=len(ids),
            positive_windows=sum(v > 1e-12 for v in deltas) if complete else None,
            negative_windows=sum(v < -1e-12 for v in deltas) if complete else None,
        )
    contribution = windows[-1]["net_return_delta_pp"]
    contribution = contribution / 4 if finite(contribution) else None
    total = strata["all4"]["net_return_delta_pp"]
    return dict(
        strata=strata,
        windows=windows,
        window0073_contribution_to_four_window_mean_pp=contribution,
        window0073_fraction_of_net_mean_change=contribution / total
        if finite(contribution) and finite(total) and abs(total) > 1e-12
        else None,
    )


def pairing(name, now, initial):
    rows = []
    for index, (a, b) in enumerate(zip(now["trajectories"], initial["trajectories"], strict=True)):
        complete = (
            a is not None
            and b is not None
            and a["status"] == b["status"] == "COMPLETE"
            and a["terminal_session_index"] == b["terminal_session_index"] == 20
            and all(finite(r[k]) for r in (a, b) for k in ("Y", "net_return"))
        )
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
                transaction_cost_delta=a["transaction_cost"] - b["transaction_cost"]
                if complete and finite(a["transaction_cost"]) and finite(b["transaction_cost"])
                else None,
                initial_prompt_equal=(
                    a["initial_prompt_sha256"] is not None
                    and a["initial_prompt_sha256"] == b["initial_prompt_sha256"]
                )
                if complete
                else None,
                initial_seed_equal=(
                    type(a["initial_request_seed"]) is int
                    and a["initial_request_seed"] == b["initial_request_seed"]
                )
                if complete
                else None,
            )
        )
    observed = [row for row in rows if row["status"] == "OBSERVED"]
    namespace = (
        now["summary"] is not None
        and initial["summary"] is not None
        and now["summary"].get("round_index") == initial["summary"].get("round_index") == 0
    )
    if any(
        row["initial_prompt_equal"] is not True or row["initial_seed_equal"] is not True
        for row in observed
    ):
        raise ValueError(f"{name}: complete final pair has mismatched initial prompt/seed")
    if observed and not namespace:
        raise ValueError(f"{name}: final pairing requires declared namespace0")
    return rows, dict(
        planned_pairs=16,
        complete_pairs=len(observed),
        namespace0_pairing=namespace,
        initial_prompt_equal_pairs=sum(r["initial_prompt_equal"] is True for r in observed),
        initial_seed_equal_pairs=sum(r["initial_seed_equal"] is True for r in observed),
        net_return_delta_pp=mean([r["net_return_delta_pp"] for r in observed], 16),
        Y_delta=mean([r["Y_delta"] for r in observed], 16),
    )


def local_support(training):
    local = training.get("local_credit") if training else None
    if local is None:
        return dict(status="PENDING", planned_anchors=8, supported_anchors=None)
    anchors = local.get("anchors", [])
    result = {key: value for key, value in local.items() if key != "anchors"}
    result.update(
        status="OBSERVED",
        planned_anchors=8,
        distinct_native_fragments=sum(a.get("distinct_native_fragments", 0) for a in anchors),
        orders_with_multiple_fragments=sum(
            a.get("orders_with_multiple_fragments", 0) for a in anchors
        ),
        extra_order_fragments=sum(a.get("extra_order_fragments", 0) for a in anchors),
        fragment_scope="supported anchors only; unsupported local items remain NA",
        anchors=anchors,
    )
    return result


def state_description(directory, source):
    manifest = optional(directory / "training_state/manifest.json")
    retirement = optional(directory / "training_state/optimizer_moment_retirement.json")
    retired = bool(
        source.get("optimizer_moments_retired") or (manifest or {}).get("optimizer_moments_retired")
    )
    return dict(
        **training_storage(source),
        manifest_status=(manifest or {}).get("status"),
        optimizer_moments_retired=retired,
        lossless_training_resume_available=False
        if retired
        else True
        if manifest and manifest.get("status") == "COMPLETE"
        else None,
        optimizer_moment_retirement=retirement or source.get("optimizer_moment_retirement"),
        scope="Recorded manifest/retirement status; no model or tensor audit performed.",
    )


def arm_readout(root, arm, baseline):
    directory = RLOO if arm == "RLOO" else root / arm
    source = optional(directory / "rloo_results.json") or {}
    if source and (
        source.get("episode_ids") != EPISODES
        or source.get("seed") != 42
        or source.get("parent_repetitions") != 4
        or source.get("planned_updates") != 16
    ):
        raise ValueError("unexpected arm/window/update identity")
    if (
        source
        and arm != "RLOO"
        and (source.get("arm") != arm or source.get("alpha") != 0.5 or source.get("eta") != 0.25)
    ):
        raise ValueError("unexpected local-credit recipe")
    entries = source.get("rounds", [])
    indexed = {entry["round"]: entry for entry in entries}
    if len(entries) != len(indexed) or any(n not in range(1, 5) for n in indexed):
        raise ValueError("unexpected/duplicate arm round")
    curve, curve_rows, window_rows, support = [], [], [], []
    actual_updates, training_rows, invalid_rows = 0, 0, 0
    for number in range(5):
        entry = indexed.get(number, {})
        current = (
            baseline
            if number == 0
            else collection_metrics(entry.get("collection_path", directory / f"round{number}/eval"))
        )
        training = (
            optional(directory / f"round{number}/train/training_summary.json") if number else None
        )
        training = training or entry.get("training")
        if training:
            actual_updates += training.get("optimizer_updates", 0) or 0
            training_rows += training.get("training_rows", 0) or 0
            invalid_rows += training.get("invalid_rows", 0) or 0
        local = local_support(training) if number and arm != "RLOO" else None
        if local is not None:
            support.append(dict(round=number, **local))
        delta = difference(current, baseline)
        point = dict(
            arm=arm,
            round=number,
            planned_optimizer_updates=number * 4,
            actual_optimizer_updates=actual_updates,
            reused=arm == "RLOO",
            execution_status="BASELINE" if not number else entry.get("status", "PENDING"),
            evaluation_seed_namespace=0 if number in (0, 4) else number,
            evaluation=current,
            versus_common_SFT=delta,
            training=training_metrics(training) if number else None,
            local_credit=local,
        )
        curve.append(point)
        overall = current["strata"]["all4"]
        curve_rows.append(
            dict(
                arm=arm,
                round=number,
                planned_updates=number * 4,
                actual_updates=actual_updates,
                reused=arm == "RLOO",
                status=overall["status"],
                mean_net_return=overall["mean_net_return"],
                mean_Y=overall["mean_Y"],
                rejection_rate=overall["rejection_rate"],
                invalid_requests=overall["invalid_requests"],
                applied_requests=overall["applied_requests"],
                mean_transaction_cost=overall["mean_transaction_cost"],
                **delta["strata"]["all4"],
            )
        )
        for window, change in zip(current["windows"], delta["windows"], strict=True):
            window_rows.append(
                dict(
                    arm=arm,
                    round=number,
                    planned_updates=number * 4,
                    actual_updates=actual_updates,
                    **{**window, **change},
                )
            )
    terminal = source.get("status") in TERMINAL
    final_observed = curve[-1]["evaluation"]["strata"]["all4"]["status"] == "OBSERVED"
    return (
        dict(
            arm=arm,
            status="DESCRIBED"
            if terminal and actual_updates == 16 and final_observed
            else "DESCRIBED_WITH_FAILURES"
            if terminal
            else "PARTIAL"
            if source
            else "PENDING",
            execution_status=source.get("status", "PENDING"),
            driver_terminal=terminal,
            source_results=str(directory / "rloo_results.json"),
            reused=arm == "RLOO",
            planned_optimizer_updates=16,
            actual_optimizer_updates=actual_updates,
            training_request_exposures=training_rows,
            training_invalid_exposures=invalid_rows,
            curve=curve,
            final_versus_common_SFT=curve[-1]["versus_common_SFT"],
            local_credit_rounds=support,
            planned_local_anchors=32 if arm != "RLOO" else 0,
            supported_local_anchors=sum(s.get("supported_anchors", 0) or 0 for s in support),
            local_round_summaries_observed=sum(s["status"] == "OBSERVED" for s in support),
            training_state=state_description(directory, source),
        ),
        curve_rows,
        window_rows,
    )


def summarize(directory=DEFAULT):
    root = Path(directory).resolve()
    reference_source = optional(RLOO / "rloo_results.json") or {}
    baseline_path = Path(reference_source.get("baseline_path", RLOO / "common_initial"))
    baseline = collection_metrics(baseline_path)
    arms, curve_rows, window_rows = [], [], []
    for arm in ARMS:
        report, points, windows = arm_readout(root, arm, baseline)
        arms.append(report)
        curve_rows.extend(points)
        window_rows.extend(windows)
    finals = {arm["arm"]: arm["curve"][-1]["evaluation"] for arm in arms}
    comparisons, pairs = {}, []
    contrasts = [(arm, "SFT") for arm in ARMS] + [
        ("plain", "RLOO"),
        ("exact", "RLOO"),
        ("exact", "plain"),
    ]
    for current, reference in contrasts:
        name = f"{current}_minus_{reference}"
        initial = baseline if reference == "SFT" else finals[reference]
        rows, paired = pairing(name, finals[current], initial)
        comparisons[name] = dict(**difference(finals[current], initial), paired_sampling=paired)
        pairs.extend(rows)
    costs, cost_paths = [], set()
    for path in [root / "common_initial"] + [
        Path(point["evaluation"]["directory"])
        for arm in arms
        if arm["arm"] != "RLOO"
        for point in arm["curve"][1:]
    ]:
        path = path.resolve()
        if path in cost_paths:
            raise ValueError("a new collection was scheduled for cost accounting twice")
        cost_paths.add(path)
        summary = optional(path / "collection_summary.json")
        costs.append(
            dict(
                directory=str(path),
                summary_available=summary is not None,
                **{key: summary.get(key) if summary else None for key in COST_KEYS},
            )
        )
    suite = optional(root / "suite_results.json") or {}
    terminal = suite.get("status") in TERMINAL or all(arm["driver_terminal"] for arm in arms)
    complete = all(arm["status"] == "DESCRIBED" for arm in arms)
    summary_rows = []
    for arm in arms:
        initial, final = baseline["strata"]["all4"], finals[arm["arm"]]["strata"]["all4"]
        summary_rows.append(
            dict(
                arm=arm["arm"],
                status=arm["status"],
                reused=arm["reused"],
                actual_optimizer_updates=arm["actual_optimizer_updates"],
                baseline_net_return=initial["mean_net_return"],
                final_net_return=final["mean_net_return"],
                baseline_Y=initial["mean_Y"],
                final_Y=final["mean_Y"],
                **{
                    f"SFT_{key}": value
                    for key, value in arm["final_versus_common_SFT"]["strata"]["all4"].items()
                },
                without0073_net_return_delta_pp=arm["final_versus_common_SFT"]["strata"][
                    "without0073"
                ]["net_return_delta_pp"],
                versus_RLOO_net_return_delta_pp=comparisons.get(f"{arm['arm']}_minus_RLOO", {})
                .get("strata", {})
                .get("all4", {})
                .get("net_return_delta_pp")
                if arm["arm"] != "RLOO"
                else 0.0,
                final_rejection_rate=final["rejection_rate"],
                final_invalid_requests=final["invalid_requests"],
                final_requests=final["applied_requests"],
                final_mean_transaction_cost=final["mean_transaction_cost"],
                supported_local_anchors=arm["supported_local_anchors"],
                planned_local_anchors=arm["planned_local_anchors"],
                training_request_exposures=arm["training_request_exposures"],
                optimizer_moments_retired=arm["training_state"]["optimizer_moments_retired"],
                lossless_training_resume_available=arm["training_state"][
                    "lossless_training_resume_available"
                ],
            )
        )
    report = dict(
        profile="terminal_RLOO_plus_fixed_alpha_local_credit_demo_readout_v1",
        generated_at_UTC=datetime.now(timezone.utc).isoformat(),
        status="DESCRIBED"
        if terminal and complete
        else "DESCRIBED_WITH_FAILURES"
        if terminal
        else "PARTIAL",
        suite_execution_status=suite.get("status", "PENDING"),
        final_artifacts_allowed=terminal,
        episode_ids=EPISODES,
        seed=42,
        parent_repetitions=4,
        common_SFT_baseline_path=str(baseline_path),
        arms=arms,
        final_comparisons=comparisons,
        new_optimizer_updates=sum(a["actual_optimizer_updates"] for a in arms if not a["reused"]),
        reused_RLOO_optimizer_updates=arms[0]["actual_optimizer_updates"],
        new_sampling_cost=dict(
            collections=costs,
            planned_collections=9,
            summaries_observed=sum(c["summary_available"] for c in costs),
            known_totals={k: sum(c[k] for c in costs if finite(c[k])) for k in COST_KEYS},
            reused_initial_parent_new_requests=0,
            rule="common_initial branches once + plain/exact four eval batches once; RLOO excluded",
            incomplete_costs_are_lower_bounds=any(not c["summary_available"] for c in costs),
        ),
        limitations=[
            "Four training windows and one training seed; no date generalization evidence.",
            "CRN matches initial prompts/seeds; later policy requests can diverge.",
            "Sampling and compute differ from reused RLOO; this is not an equal-cost comparison.",
            "Missing replicas preserve fixed16 denominators and whole-window NA; no filling.",
            "Intermediate points change rollout seeds; final namespace0 is paired with SFT.",
            "Methods share four windows; these are not independent statistical samples.",
        ],
    )
    return report, summary_rows, curve_rows, window_rows, pairs


def plots(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for arm in report["arms"]:
        x = [p["planned_optimizer_updates"] for p in arm["curve"]]
        label = arm["arm"] + (" (reused)" if arm["reused"] else "")
        for axis, metric, factor in zip(axes, ("mean_net_return", "mean_Y"), (100, 1), strict=True):
            values = [p["evaluation"]["strata"]["all4"][metric] for p in arm["curve"]]
            axis.plot(x, [v * factor if finite(v) else math.nan for v in values], "o-", label=label)
    baseline = report["arms"][0]["curve"][0]["evaluation"]["strata"]["all4"]
    for axis, metric, factor, label in zip(
        axes,
        ("mean_net_return", "mean_Y"),
        (100, 1),
        ("Net return (%)", "Financial Y"),
        strict=True,
    ):
        if finite(baseline[metric]):
            axis.axhline(baseline[metric] * factor, color="grey", linestyle="--", linewidth=1)
        axis.set(xlabel="Planned optimizer updates", ylabel=label, xticks=[0, 4, 8, 12, 16])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Four training windows, seed42; final16 paired with SFT")
    fig.text(
        0.5,
        0.01,
        "Intermediate points use different rollout seeds; missing means remain gaps.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(directory / "learning_curves.png", dpi=180)
    fig.savefig(directory / "learning_curves.svg")
    plt.close(fig)


def write_readout(directory, result):
    report, summaries, curves, windows, pairs = result
    if not report["final_artifacts_allowed"]:
        raise ValueError("suite still running; use --preview")
    output = Path(directory) / "summary"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("final summary exists; refusing overwrite")
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison_readout.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, rows in [
        ("arm_summary", summaries),
        ("curve_results", curves),
        ("window_results", windows),
        ("paired_results", pairs),
    ]:
        write_csv(output / f"{name}.csv", rows)
    plots(report, output)
    return output


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
                    arms=result[1],
                    new_sampling_cost=result[0]["new_sampling_cost"],
                ),
                indent=2,
            )
        )
    else:
        print(write_readout(args.directory, result))


if __name__ == "__main__":
    main()

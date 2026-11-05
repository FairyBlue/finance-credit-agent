#!/usr/bin/env python3
"""Read one frozen broad-RLOO/order-scoring experiment, keeping fixed denominators."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from read_rloo_terminal_demo import finite, mean, observed_trajectory

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "runs/m2-broad-rloo-order-main-20261028-v1"
ARMS = ("SFT", "old_RLOO", "broad_RLOO", "menu_control", "learned_ranker")
TERMINAL = {"COMPLETE", "COMPLETE_WITH_MODEL_FAILURES", "FAILED"}
COST_KEYS = ("model_requests", "prompt_tokens", "response_tokens")


def optional(path):
    return json.loads(path.read_text()) if path.is_file() else None


def records(path):
    if path.is_file():
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def aggregate(windows, planned):
    complete = len(windows) == planned and all(w["status"] == "OBSERVED" for w in windows)
    requests = sum(w["applied_requests"] for w in windows)
    invalid = sum(w["invalid_requests"] for w in windows)
    return dict(
        status="OBSERVED" if complete else "INCOMPLETE_FIXED_DENOMINATOR",
        planned_windows=planned,
        planned_parents=sum(w["planned_parents"] for w in windows),
        observed_parents=sum(w["observed_parents"] for w in windows),
        complete_parents=sum(w["complete_parents"] for w in windows),
        mean_net_return=mean([w["mean_net_return"] for w in windows], planned),
        mean_Y=mean([w["mean_Y"] for w in windows], planned),
        mean_transaction_cost=mean([w["mean_transaction_cost"] for w in windows], planned),
        mean_benchmark_excess=mean([w["mean_benchmark_excess"] for w in windows], planned),
        applied_requests=requests,
        invalid_requests=invalid,
        rejection_rate=invalid / requests if requests else None,
    )


def collection(path, ids, repetitions, split, namespace):
    summary = optional(path / "collection_summary.json")
    if summary and (
        summary.get("episode_ids") != ids
        or summary.get("seed") != 43
        or summary.get("parent_repetitions") != repetitions
        or summary.get("round_index") != namespace
        or summary.get("parents_only") is not True
        or summary.get("source_split") != split
        or summary.get("formal_final_oos_access", False) is not False
    ):
        raise ValueError(f"unexpected collection identity: {path}")
    indexed = {}
    for source in records(path / "trajectory_results.jsonl"):
        key = source["episode_id"], source["replicate_index"]
        if (
            source.get("role") != "parent"
            or key[0] not in ids
            or key[1] not in range(repetitions)
            or key in indexed
        ):
            raise ValueError(f"unexpected or duplicate parent slot: {path}")
        row = observed_trajectory(source)
        row["benchmark_excess"] = (source.get("terminal_metrics") or {}).get("benchmark_excess")
        indexed[key] = row
    windows, slots = [], []
    for eid in ids:
        members = [indexed.get((eid, rep)) for rep in range(repetitions)]
        slots.extend(members)
        actual = [row for row in members if row is not None]
        complete = len(actual) == repetitions and all(
            row["status"] == "COMPLETE"
            and row["terminal_session_index"] == 20
            and finite(row["net_return"])
            and finite(row["Y"])
            for row in actual
        )
        requests = sum(row["applied_requests"] for row in actual)
        invalid = sum(row["invalid_requests"] for row in actual)
        windows.append(
            dict(
                episode_id=eid,
                status="OBSERVED" if complete else "INCOMPLETE_FIXED_DENOMINATOR",
                planned_parents=repetitions,
                observed_parents=len(actual),
                complete_parents=sum(row["status"] == "COMPLETE" for row in actual),
                mean_net_return=mean([row["net_return"] for row in actual], repetitions)
                if complete
                else None,
                mean_Y=mean([row["Y"] for row in actual], repetitions) if complete else None,
                mean_transaction_cost=mean([row["transaction_cost"] for row in actual], repetitions)
                if complete
                else None,
                mean_benchmark_excess=mean([row["benchmark_excess"] for row in actual], repetitions)
                if complete
                else None,
                applied_requests=requests,
                invalid_requests=invalid,
                rejection_rate=invalid / requests if requests else None,
            )
        )
    return dict(
        path=str(path),
        summary=summary,
        windows=windows,
        trajectories=slots,
        overall=aggregate(windows, len(ids)),
    )


def compare(name, now, old, ids, repetitions=2):
    rows, windows = [], []
    for index, (a, b) in enumerate(zip(now["trajectories"], old["trajectories"], strict=True)):
        has_identity = (
            a is not None
            and b is not None
            and all(
                row.get("initial_prompt_sha256") is not None
                and row.get("initial_request_seed") is not None
                for row in (a, b)
            )
        )
        prompt_equal = (
            a["initial_prompt_sha256"] == b["initial_prompt_sha256"] if has_identity else None
        )
        seed_equal = (
            a["initial_request_seed"] == b["initial_request_seed"] if has_identity else None
        )
        if has_identity and not (prompt_equal and seed_equal):
            raise ValueError(f"first-request prompt/seed differs in {name}")
        complete = (
            a is not None
            and b is not None
            and all(
                row["status"] == "COMPLETE" and finite(row["net_return"]) and finite(row["Y"])
                for row in (a, b)
            )
        )
        rows.append(
            dict(
                comparison=name,
                episode_id=ids[index // repetitions],
                replicate_index=index % repetitions,
                status="OBSERVED" if complete else "NA",
                initial_prompt_equal=prompt_equal,
                initial_seed_equal=seed_equal,
                net_return_delta_pp=100 * (a["net_return"] - b["net_return"]) if complete else None,
                Y_delta=a["Y"] - b["Y"] if complete else None,
            )
        )
    for a, b in zip(now["windows"], old["windows"], strict=True):
        good = a["status"] == b["status"] == "OBSERVED"
        windows.append(
            dict(
                episode_id=a["episode_id"],
                status="OBSERVED" if good else "NA",
                net_return_delta_pp=100 * (a["mean_net_return"] - b["mean_net_return"])
                if good
                else None,
                Y_delta=a["mean_Y"] - b["mean_Y"] if good else None,
            )
        )
    deltas = [row["net_return_delta_pp"] for row in windows]
    good = len(deltas) == len(ids) and all(map(finite, deltas))
    maximum = max(range(len(ids)), key=lambda index: deltas[index]) if good else None
    return dict(
        planned_pairs=len(ids) * repetitions,
        observed_pairs=sum(row["status"] == "OBSERVED" for row in rows),
        net_return_delta_pp=mean(deltas, len(ids)),
        Y_delta=mean([row["Y_delta"] for row in windows], len(ids)),
        positive_windows=sum(value > 1e-12 for value in deltas) if good else None,
        negative_windows=sum(value < -1e-12 for value in deltas) if good else None,
        maximum_positive_contributor=ids[maximum]
        if maximum is not None and deltas[maximum] > 0
        else None,
        without_largest_window_net_delta_pp=mean(
            [value for index, value in enumerate(deltas) if index != maximum], len(ids) - 1
        )
        if good
        else None,
        windows=windows,
    ), rows


def actual_sampling_cost(root):
    paths = [
        *sorted((root / "stage1").glob("*")),
        root / "stage2/branches",
        *sorted((root / "stage3/eval").glob("*")),
    ]
    sources, seen = [], set()
    for path in paths:
        if not path.is_dir() or path.resolve() in seen:
            continue
        seen.add(path.resolve())
        summary = optional(path / "collection_summary.json")
        request_file = path / "requests.jsonl"
        if summary:
            values = {key: summary.get(key, 0) for key in COST_KEYS}
            kind = "completed_collection_summary"
        elif request_file.is_file():
            values = dict.fromkeys(COST_KEYS, 0)
            for row in records(request_file):
                values["model_requests"] += 1
                values["prompt_tokens"] += len(row.get("prompt_token_ids", []))
                values["response_tokens"] += len(row.get("response_token_ids", []))
            kind = "partial_actual_request_log"
        else:
            continue
        sources.append(dict(path=str(path), source=kind, **values))
    return dict(
        totals={key: sum(row[key] for row in sources) for key in COST_KEYS},
        sources=sources,
        counting_rule="Each actual generation directory once; source replay is not new generation.",
    )


def summarize(root=DEFAULT):
    root = Path(root).resolve()
    config = json.loads((root / "config.json").read_text())
    result = optional(root / "results.json") or {}
    completion = optional(root / "completion.json") or {}
    terminal = completion.get("status") in TERMINAL and result.get("status") in TERMINAL
    ids = config["valid_episode_ids"]
    arms = {name: collection(root / f"stage3/eval/{name}", ids, 2, "valid", 20) for name in ARMS}
    for name, entry in arms.items():
        execution = result.get("evaluations", {}).get(name, {})
        entry["execution_status"] = execution.get("status", "NOT_FINISHED")
        entry["unavailable_reason"] = execution.get("reason")
    summaries = [dict(arm=name, **entry["overall"]) for name, entry in arms.items()]
    windows = [dict(arm=name, **row) for name, entry in arms.items() for row in entry["windows"]]
    comparisons, pairs = {}, []
    selected_native = "broad_RLOO" if result.get("stage1_status") == "COMPLETE" else "old_RLOO"
    definitions = [
        ("old_RLOO", "SFT"),
        ("broad_RLOO", "old_RLOO"),
        ("broad_RLOO", "SFT"),
        ("menu_control", selected_native),
        ("learned_ranker", "menu_control"),
        ("learned_ranker", selected_native),
        ("learned_ranker", "SFT"),
    ]
    for current, baseline in dict.fromkeys(definitions):
        name = f"{current}_minus_{baseline}"
        comparisons[name], rows = compare(name, arms[current], arms[baseline], ids)
        pairs.extend(rows)
    baseline_result = optional(root / "stage3/cpu_baselines/results.json") or {}
    baseline_rows = baseline_result.get("results", [])
    cpu = {}
    for name in ("cash", "buy_and_hold"):
        subset = [row for row in baseline_rows if row["baseline"] == name]
        good = len(subset) == 60 and {row["episode_id"] for row in subset} == set(ids)
        cpu[name] = dict(
            status="OBSERVED" if good else "NA",
            planned_deterministic_trajectories=60,
            observed_trajectories=len(subset),
            mean_net_return=mean([row["raw_net_return"] for row in subset], 60),
            mean_Y=mean([row["terminal_financial_Y"] for row in subset], 60),
            mean_transaction_cost=mean([row["transaction_cost"] for row in subset], 60),
            model_requests=0,
        )
        summaries.append(dict(arm=name, **cpu[name]))
    baseline_deltas = {}
    for name, entry in arms.items():
        baseline_deltas[name] = {}
        for reference in ("cash", "buy_and_hold"):
            a, b = entry["overall"]["mean_net_return"], cpu[reference]["mean_net_return"]
            baseline_deltas[name][reference] = 100 * (a - b) if finite(a) and finite(b) else None
    train = []
    for label, updates, namespace in (
        ("initial_train", 0, 10),
        ("after32_train", 32, 11),
        ("final_train", 64, 10),
    ):
        current = collection(
            root / f"stage1/{label}", config["train_episode_ids"], 4, "train", namespace
        )
        train.append(
            dict(
                label=label,
                added_updates=updates,
                cumulative_updates=16 + updates,
                sampling_namespace=namespace,
                paired_with_initial_sampling=namespace == 10,
                **current["overall"],
            )
        )
    updates = [
        row
        for stage in (1, 2)
        for row in records(root / f"round{stage}/train/optimizer_updates.jsonl")
    ]
    full = all(entry["overall"]["complete_parents"] == 120 for entry in arms.values())
    ranker_delta = comparisons["learned_ranker_minus_menu_control"]["net_return_delta_pp"]
    native_delta = comparisons[f"learned_ranker_minus_{selected_native}"]["net_return_delta_pp"]
    versus_sft = comparisons["learned_ranker_minus_SFT"]["net_return_delta_pp"]
    ranker_gain = (
        finite(ranker_delta)
        and finite(native_delta)
        and finite(versus_sft)
        and ranker_delta > 0
        and native_delta > 0
        and versus_sft > 0
    )

    def beats_simple(name):
        values = list(baseline_deltas[name].values())
        return len(values) == 2 and all(finite(value) and value > 0 for value in values)

    eligible_native = [
        name
        for name in ("old_RLOO", "broad_RLOO")
        if arms[name]["overall"]["complete_parents"] == 120
        and beats_simple(name)
        and finite(comparisons[f"{name}_minus_SFT"]["net_return_delta_pp"])
        and comparisons[f"{name}_minus_SFT"]["net_return_delta_pp"] > 0
    ]
    decision = "INCOMPLETE_NO_CONCLUSION"
    if terminal:
        if (
            ranker_gain
            and arms["learned_ranker"]["overall"]["complete_parents"] == 120
            and beats_simple("learned_ranker")
        ):
            decision = "RANKER_WORTH_FOLLOWUP_WITH_BEHAVIOR_REVIEW"
        elif eligible_native:
            decision = "KEEP_NATIVE_RLOO_STOP_CURRENT_RANKER"
        elif arms["SFT"]["overall"]["complete_parents"] == 120 and beats_simple("SFT"):
            decision = "KEEP_SFT_DEMO_NO_RL_INCREMENT"
        elif all(cpu[name]["status"] == "OBSERVED" for name in cpu):
            decision = "NO_COMPLETE_LEARNING_ADVANTAGE_OVER_SIMPLE_BASELINES"
    states = [
        optional(root / "state.json") or {},
        optional(root / "bootstrap/initial_state.json") or {},
    ]
    report = dict(
        profile="broad_rloo_order_main_readout_v1",
        generated_at_UTC=datetime.now(timezone.utc).isoformat(),
        status="DESCRIBED"
        if terminal and full
        else "DESCRIBED_WITH_FAILURES"
        if terminal
        else "PARTIAL",
        execution_status=completion.get("status", result.get("status", "PENDING")),
        final_artifacts_allowed=terminal,
        decision=decision,
        config=config,
        planned_valid_parent_slots=600,
        observed_valid_parent_slots=sum(
            row["overall"]["observed_parents"] for row in arms.values()
        ),
        complete_valid_parent_slots=sum(
            row["overall"]["complete_parents"] for row in arms.values()
        ),
        selected_native_for_menu=selected_native,
        selected_model=result.get("selected_model"),
        stage1_status=result.get("stage1_status"),
        stage2_status=result.get("stage2_status"),
        stage3_status=result.get("stage3_status"),
        train_curve=train,
        valid_arms=arms,
        comparisons=comparisons,
        cpu_baselines=cpu,
        learning_minus_simple_baseline_net_delta_pp=baseline_deltas,
        sampling_cost=actual_sampling_cost(root),
        training_cost=dict(
            added_optimizer_updates=len(updates),
            cumulative_optimizer_steps=result.get("cumulative_updates"),
            full_trajectory_uses=sum(row.get("trajectories", 0) for row in updates),
            request_uses=sum(row.get("requests", 0) for row in updates),
            original16_updates_reused=True,
        ),
        stage_wall_seconds=sum(
            stage.get("seconds", 0) for state in states for stage in state.get("stages", [])
        ),
        formal_valid_access=result.get("formal_valid_access", False),
        formal_final_oos_access=False,
        limitations=[
            "Valid windows overlap; no IID confidence interval or concatenated portfolio return.",
            "Two model replicas are sampling repetitions, not independent market dates.",
            "Any failed parent makes that arm's fixed120 financial mean NA; no survivor mean.",
            "Candidate interfaces differ; compare learned scoring to the same menu control.",
            "Models/rules freeze before Valid; this is development comparison, not Final OOS.",
            "Decision labels are descriptive; behavioral costs must be read alongside net returns.",
        ],
    )
    return report, summaries, windows, pairs


def write_table(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def figures(root, report, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(ARMS) + ["cash", "buy_and_hold"]
    by_name = {row["arm"]: row for row in summaries}
    values = [by_name[name]["mean_net_return"] for name in names]
    ax.bar(
        range(len(names)),
        [100 * value if finite(value) else math.nan for value in values],
        color=["#90a4ae", "#607d8b", "#1565c0", "#66a4a4", "#2e7d32", "#a1887f", "#ef9a30"],
    )
    for index, value in enumerate(values):
        if not finite(value):
            row = by_name[names[index]]
            count = row.get("complete_parents", row.get("observed_trajectories", 0))
            planned = 120 if names[index] in ARMS else 60
            ax.text(
                index,
                0,
                f"NA\n{count}/{planned} complete",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        xticks=range(len(names)),
        xticklabels=[*ARMS, "cash\n(CPU)", "buy_and_hold\n(CPU)"],
        xlim=(-0.6, len(names) - 0.4),
        ylabel="Mean rolling-window net return (%)",
        title="All60 overlapping Valid windows\n120 paths per LLM arm;60 windows per CPU baseline",
    )
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    fig.text(
        0.5,
        0.01,
        "All policies and CPU baselines use the same next-open ledger, commission and slippage.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(root / "valid_comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    points = report["train_curve"]
    ax.plot(
        [row["added_updates"] for row in points],
        [
            100 * row["mean_net_return"] if finite(row["mean_net_return"]) else math.nan
            for row in points
        ],
        marker="o",
    )
    ax.set(
        xlabel="Added optimizer updates (source checkpoint already has16)",
        ylabel="Mean Train net return (%)",
        title="Fixed32 Train windows;128 paths per point\n"
        "Endpoints paired; middle point uses another sampling namespace",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "train_curve.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    report, summaries, windows, pairs = summarize(args.output)
    if args.preview:
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in {"valid_arms", "comparisons"}},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not report["final_artifacts_allowed"]:
        raise RuntimeError("experiment is still running; use --preview")
    output = args.output.resolve() / "summary"
    if (output / "readout.json").exists():
        raise FileExistsError("final readout exists; refusing overwrite")
    output.mkdir(exist_ok=True)
    with (output / "readout.json").open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    write_table(output / "summary.csv", summaries)
    write_table(output / "window_results.csv", windows)
    write_table(output / "paired_results.csv", pairs)
    write_table(output / "train_curve.csv", report["train_curve"])
    figures(output, report, summaries)
    from finalize_broad_main import finalize

    finalize(args.output, report)
    print(
        json.dumps(
            dict(
                status=report["status"],
                decision=report["decision"],
                complete_valid_parent_slots=report["complete_valid_parent_slots"],
                sampling_cost=report["sampling_cost"]["totals"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

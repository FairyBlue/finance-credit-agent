#!/usr/bin/env python3
"""Read the fixed exact-auxiliary pilot against saved same-namespace references."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from read_rloo_fresh_eval import paired as fresh_pairing
from read_rloo_local_credit_demo import (
    COST_KEYS,
    EPISODES,
    ROOT,
    TERMINAL,
    collection_metrics,
    difference,
    local_support,
    optional,
    pairing,
)
from read_rloo_terminal_demo import finite, write_csv

DEFAULT = ROOT / "runs/m2-rloo-local-aux-demo-20261026-v1"
LOCAL = ROOT / "runs/m2-rloo-local-credit-demo-20261025-v1"
RLOO = ROOT / "runs/m2-rloo-terminal-demo-20261025-v3"
FRESH = ROOT / "runs/m2-rloo-local-credit-fresh-eval-20261026-v1"
ARM = "exact_aux"
REFERENCES = ("SFT", "RLOO", "plain", "exact")


def records(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def final_path(source, directory):
    final = next((entry for entry in source.get("rounds", []) if entry["round"] == 4), {})
    return Path(final.get("collection_path", directory / "round4/eval"))


def metric_row(policy, namespace, collection, **extra):
    return dict(
        policy=policy,
        evaluation_seed_namespace=namespace,
        **extra,
        **collection["strata"]["all4"],
    )


def training_readout(directory, entry, number):
    path = directory / f"round{number}/train"
    summary = optional(path / "training_summary.json") or entry.get("training")
    events = records(path / "optimizer_updates.jsonl")
    if not events and summary:
        events = summary.get("updates", [])
    local = local_support(summary)
    return dict(
        round=number,
        status=(summary or {}).get("status", "PENDING"),
        committed_optimizer_updates=(summary or {}).get("optimizer_updates", 0) or 0,
        observed_optimizer_events=len(events),
        training_rows=(summary or {}).get("training_rows"),
        invalid_rows=(summary or {}).get("invalid_rows"),
        base_RLOO_whitening_mean=(summary or {}).get("base_RLOO_whitening_mean"),
        base_RLOO_whitening_std_sample=(summary or {}).get("base_RLOO_whitening_std_sample"),
        local_credit=local,
        actual_optimizer_losses=events,
        base_loss_sum=math.fsum(e["base_loss"] for e in events)
        if events and all(finite(e.get("base_loss")) for e in events)
        else None,
        auxiliary_loss_sum=math.fsum(e["auxiliary_loss"] for e in events)
        if events and all(finite(e.get("auxiliary_loss")) for e in events)
        else None,
        local_aux_loss_weights=sorted(
            {e["local_aux_loss_weight"] for e in events if finite(e.get("local_aux_loss_weight"))}
        ),
    )


def summarize(directory=DEFAULT):
    root = Path(directory).resolve()
    actor_directory = root / ARM
    source = optional(actor_directory / "rloo_results.json") or {}
    if source and (
        source.get("arm") != ARM
        or source.get("seed") != 42
        or source.get("episode_ids") != EPISODES
        or source.get("parent_repetitions") != 4
        or source.get("planned_updates") != 16
        or source.get("alpha") != 0.5
        or source.get("eta") != 0.25
    ):
        raise ValueError("expected fixed exact_aux, four-window seed42, alpha .5 eta .25, 16 steps")
    baseline = collection_metrics(source.get("baseline_path", LOCAL / "common_initial"))
    reference_paths = source.get("reference_results", {})
    old_sources, old_directories = {}, {}
    for policy, default in (("RLOO", RLOO), ("plain", LOCAL / "plain"), ("exact", LOCAL / "exact")):
        path = Path(reference_paths.get(policy, default / "rloo_results.json"))
        old_directories[policy] = path.parent
        old_sources[policy] = optional(path) or {}
    indexed = {entry["round"]: entry for entry in source.get("rounds", [])}
    if len(indexed) != len(source.get("rounds", [])) or set(indexed) - set(range(1, 5)):
        raise ValueError("unexpected/duplicate auxiliary round")

    curve, curve_rows, window_rows, training = [], [], [], []
    finals0 = {"SFT": baseline}
    for policy in (*REFERENCES[1:], ARM):
        is_aux = policy == ARM
        policy_source = source if is_aux else old_sources[policy]
        policy_directory = actor_directory if is_aux else old_directories[policy]
        entries = {entry["round"]: entry for entry in policy_source.get("rounds", [])}
        committed = 0
        for number in range(5):
            entry = entries.get(number, {})
            namespace = 0 if number in (0, 4) else number
            collection = (
                baseline
                if not number
                else collection_metrics(
                    entry.get("collection_path", policy_directory / f"round{number}/eval")
                )
            )
            current_summary = collection["summary"]
            if current_summary and current_summary.get("round_index") != namespace:
                raise ValueError("curve collection has unexpected sampling namespace")
            train = None
            if number:
                if is_aux:
                    train = training_readout(policy_directory, entry, number)
                    training.append(train)
                    committed += train["committed_optimizer_updates"]
                else:
                    old = optional(policy_directory / f"round{number}/train/training_summary.json")
                    old = old or entry.get("training") or {}
                    committed += old.get("optimizer_updates", 0) or 0
            point = dict(
                policy=policy,
                round=number,
                planned_updates=number * 4,
                committed_updates=committed,
                evaluation_seed_namespace=namespace,
                reused=not is_aux or number == 0,
                execution_status=entry.get("status", "PENDING") if number else "BASELINE_REUSED",
                evaluation=collection,
                training=train,
            )
            curve.append(point)
            curve_rows.append(
                metric_row(
                    policy,
                    namespace,
                    collection,
                    round=number,
                    planned_updates=number * 4,
                    committed_updates=committed,
                    reused=point["reused"],
                    execution_status=point["execution_status"],
                )
            )
            window_rows.extend(
                dict(
                    policy=policy,
                    round=number,
                    planned_updates=number * 4,
                    evaluation_seed_namespace=namespace,
                    reused=point["reused"],
                    **window,
                )
                for window in collection["windows"]
            )
            if number == 4:
                finals0[policy] = collection

    fresh_source_path = Path(source.get("fresh_reference_results", FRESH / "results.json"))
    fresh_source = optional(fresh_source_path) or {}
    if fresh_source and (
        fresh_source.get("round_index") != 4
        or fresh_source.get("seed") != 42
        or fresh_source.get("episode_ids") != EPISODES
    ):
        raise ValueError("expected existing namespace4 fresh reference")
    fresh_entries = fresh_source.get("policies", {})
    finals4 = {}
    for policy in (*REFERENCES, ARM):
        path = (
            source.get("fresh_collection_path", actor_directory / "fresh_eval")
            if policy == ARM
            else (
                fresh_entries.get(policy, {}).get(
                    "collection_path", fresh_source_path.parent / policy
                )
            )
        )
        collection = collection_metrics(path)
        if collection["summary"] and collection["summary"].get("round_index") != 4:
            raise ValueError("fresh collection must use namespace4")
        finals4[policy] = collection
        curve_rows.append(
            metric_row(
                policy,
                4,
                collection,
                round="fresh",
                planned_updates=0 if policy == "SFT" else 16,
                committed_updates=0
                if policy == "SFT"
                else 16
                if policy != ARM
                else sum(t["committed_optimizer_updates"] for t in training),
                reused=policy != ARM,
                execution_status=source.get("status", "PENDING")
                if policy == ARM
                else fresh_entries.get(policy, {}).get("status", "PENDING"),
            )
        )
        window_rows.extend(
            dict(
                policy=policy,
                round="fresh",
                planned_updates=0 if policy == "SFT" else 16,
                evaluation_seed_namespace=4,
                reused=policy != ARM,
                **window,
            )
            for window in collection["windows"]
        )

    comparisons, comparison_rows, pairs = {}, [], []
    for namespace, finals, pair_function in ((0, finals0, pairing), (4, finals4, fresh_pairing)):
        namespace_results = {}
        for reference in REFERENCES:
            name = f"{ARM}_minus_{reference}"
            delta = difference(finals[ARM], finals[reference])
            paired_rows, paired_summary = pair_function(name, finals[ARM], finals[reference])
            observed = [p for p in paired_rows if p["status"] == "OBSERVED"]
            paired_summary.update(
                positive_pairs=sum(p["net_return_delta_pp"] > 1e-12 for p in observed),
                negative_pairs=sum(p["net_return_delta_pp"] < -1e-12 for p in observed),
                tied_pairs=sum(abs(p["net_return_delta_pp"]) <= 1e-12 for p in observed),
            )
            namespace_results[name] = dict(**delta, paired_sampling=paired_summary)
            pairs.extend(dict(evaluation_seed_namespace=namespace, **p) for p in paired_rows)
            current = finals[ARM]["strata"]["all4"]
            initial = finals[reference]["strata"]["all4"]
            comparison_rows.append(
                dict(
                    comparison=name,
                    evaluation_seed_namespace=namespace,
                    current_policy=ARM,
                    reference_policy=reference,
                    current_status=current["status"],
                    reference_status=initial["status"],
                    current_mean_net_return=current["mean_net_return"],
                    reference_mean_net_return=initial["mean_net_return"],
                    current_mean_Y=current["mean_Y"],
                    reference_mean_Y=initial["mean_Y"],
                    current_rejection_rate=current["rejection_rate"],
                    reference_rejection_rate=initial["rejection_rate"],
                    current_invalid_requests=current["invalid_requests"],
                    current_requests=current["applied_requests"],
                    current_mean_transaction_cost=current["mean_transaction_cost"],
                    reference_mean_transaction_cost=initial["mean_transaction_cost"],
                    **delta["strata"]["all4"],
                    without0073_net_return_delta_pp=delta["strata"]["without0073"][
                        "net_return_delta_pp"
                    ],
                    planned_pairs=16,
                    complete_pairs=paired_summary["complete_pairs"],
                    paired_net_return_delta_pp=paired_summary["net_return_delta_pp"],
                    positive_pairs=paired_summary["positive_pairs"],
                    negative_pairs=paired_summary["negative_pairs"],
                )
            )
        comparisons[f"namespace{namespace}"] = namespace_results

    costs = []
    for number in range(1, 5):
        entry = indexed.get(number, {})
        path = Path(entry.get("collection_path", actor_directory / f"round{number}/eval"))
        costs.append((f"round{number}_eval", path))
    costs.append(
        ("fresh_eval", Path(source.get("fresh_collection_path", actor_directory / "fresh_eval")))
    )
    if len({path.resolve() for _, path in costs}) != 5:
        raise ValueError("new sampling batches must be counted once each")
    cost_rows = []
    for name, path in costs:
        summary = optional(path / "collection_summary.json")
        cost_rows.append(
            dict(
                batch=name,
                directory=str(path),
                summary_available=summary is not None,
                **{key: summary.get(key) if summary else None for key in COST_KEYS},
                parent_requests=summary.get("parent_requests") if summary else None,
                local_requests=summary.get("local_requests", 0) if summary else None,
            )
        )
    terminal = source.get("status") in TERMINAL
    complete = all(
        finals[ARM]["strata"]["all4"]["status"] == "OBSERVED" for finals in (finals0, finals4)
    )
    updates = sum(t["committed_optimizer_updates"] for t in training)
    exceeds = {}
    for reference in ("RLOO", "exact"):
        changes = [
            comparisons[f"namespace{ns}"][f"{ARM}_minus_{reference}"]["strata"]["all4"][
                "net_return_delta_pp"
            ]
            for ns in (0, 4)
        ]
        exceeds[reference] = all(v > 1e-12 for v in changes) if all(map(finite, changes)) else None
    manifest = optional(actor_directory / "training_state/manifest.json") or {}
    report = dict(
        profile="fixed16_exact_auxiliary_vs_saved_same_namespace_references_v1",
        generated_at_UTC=datetime.now(timezone.utc).isoformat(),
        status="DESCRIBED"
        if terminal and complete and updates == 16
        else "DESCRIBED_WITH_FAILURES"
        if terminal
        else "PARTIAL",
        execution_status=source.get("status", "PENDING"),
        final_artifacts_allowed=terminal,
        arm=ARM,
        seed=42,
        episode_ids=EPISODES,
        parent_repetitions=4,
        planned_pairs_per_contrast=16,
        source_results=str(actor_directory / "rloo_results.json"),
        baseline_directory=baseline["directory"],
        curve=curve,
        final_collections={"namespace0": finals0, "namespace4": finals4},
        comparisons=comparisons,
        outcome=dict(
            exceeds_RLOO_net_return_in_both_namespaces=exceeds["RLOO"],
            exceeds_prior_exact_net_return_in_both_namespaces=exceeds["exact"],
            interpretation="PENDING_OR_INCOMPLETE"
            if not complete
            else "positive_in_both_namespaces_against_both"
            if all(exceeds.values())
            else "no_consistent_increment_over_both_references",
            scope="Descriptive fixed final16; no best-round selection or significance claim",
        ),
        training=dict(
            planned_new_optimizer_updates=16,
            committed_new_optimizer_updates=updates,
            observed_optimizer_events=sum(t["observed_optimizer_events"] for t in training),
            training_request_exposures=sum(t["training_rows"] or 0 for t in training),
            training_invalid_exposures=sum(t["invalid_rows"] or 0 for t in training),
            rounds=training,
            planned_local_anchors=32,
            supported_local_anchors=sum(
                t["local_credit"].get("supported_anchors", 0) or 0 for t in training
            ),
            nonzero_local_requests=sum(
                t["local_credit"].get("nonzero_parent_rows", 0) or 0 for t in training
            ),
            objective=dict(
                base="clipped PPO with raw RLOO centered and whitened alone",
                auxiliary="separately clipped PPO with exact S / base_RLOO_std",
                alpha=0.5,
                eta=0.25,
                fixed_planned_collection_anchors=8,
                local_aux_loss_weight=0.015625,
                base_request_loss_weight=0.0125,
                scalar_local_weight_relative_to_mixed=10,
                weighting_note="Independent episode replica permutations distribute local rows "
                "across optimizer minibatches; fixed denominator8 is per entire collection. "
                "10x describes the coefficient of a local request, not measured gradient norm.",
            ),
            state=dict(
                directory=str(actor_directory / "training_state"),
                manifest_status=manifest.get("status"),
                completed_optimizer_steps=manifest.get("completed_optimizer_steps"),
                optimizer_moments_retired=manifest.get("optimizer_moments_retired", False),
                evidence="manifest only; no checkpoint tensors loaded",
            ),
        ),
        new_sampling_cost=dict(
            planned_collections=5,
            collections=cost_rows,
            summaries_observed=sum(c["summary_available"] for c in cost_rows),
            known_totals={k: sum(c[k] for c in cost_rows if finite(c[k])) for k in COST_KEYS},
            reused_initial_parent_new_requests=0,
            reused_initial_branch_new_requests=0,
            historical_reference_new_sampling_cost=0,
            rule="aux round1..4 eval and fresh_eval each once; initial and old runs reused",
            incomplete_costs_are_lower_bounds=any(not c["summary_available"] for c in cost_rows),
        ),
        limitations=[
            "Four existing Train windows, one training seed; no date generalization claim.",
            "Namespaces are rollout random seeds, not independent datasets or training seeds.",
            "Middle rounds retain namespaces1..3; final namespace0 and fresh namespace4 are "
            "compared only with references in the same namespace.",
            "Every contrast keeps sixteen planned pairs and whole-window NA; no missing fill.",
            "CRN checks initial prompt and request seed; subsequent policy paths may differ.",
            "Pilot and reference compute costs differ; no equal-cost claim.",
        ],
    )
    return report, curve_rows, comparison_rows, window_rows, pairs


def plots(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for policy in (*REFERENCES[1:], ARM):
        points = [p for p in report["curve"] if p["policy"] == policy]
        for axis, metric, scale in zip(axes, ("mean_net_return", "mean_Y"), (100, 1), strict=True):
            (line,) = axis.plot(
                [p["planned_updates"] for p in points],
                [
                    p["evaluation"]["strata"]["all4"][metric] * scale
                    if finite(p["evaluation"]["strata"]["all4"][metric])
                    else math.nan
                    for p in points
                ],
                "o-" if policy == ARM else "o--",
                label=policy,
            )
            fresh = report["final_collections"]["namespace4"][policy]["strata"]["all4"][metric]
            axis.scatter(
                [20],
                [fresh * scale if finite(fresh) else math.nan],
                marker="D",
                color=line.get_color(),
            )
    initial = report["final_collections"]["namespace0"]["SFT"]["strata"]["all4"]
    fresh = report["final_collections"]["namespace4"]["SFT"]["strata"]["all4"]
    for axis, metric, scale, label in zip(
        axes,
        ("mean_net_return", "mean_Y"),
        (100, 1),
        ("Net return (%)", "Financial Y"),
        strict=True,
    ):
        if finite(initial[metric]):
            axis.axhline(initial[metric] * scale, color="grey", linestyle=":", label="SFT ns0")
        axis.scatter(
            [20],
            [fresh[metric] * scale if finite(fresh[metric]) else math.nan],
            color="grey",
            marker="D",
            label="SFT ns4",
        )
        axis.set(
            xlabel="Optimizer updates / final fresh sampling",
            ylabel=label,
            xticks=[0, 4, 8, 12, 16, 20],
            xticklabels=["0", "4", "8", "12", "16", "16 ns4"],
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Exact auxiliary pilot; saved references reused; four Train windows")
    fig.text(
        0.5,
        0.01,
        "Middle points use namespaces1..3; final16 namespace0; diamonds namespace4.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"learning_curve.{suffix}", dpi=180)
    plt.close(fig)


def write_readout(directory, result):
    report, curves, comparisons, windows, pairs = result
    if not report["final_artifacts_allowed"]:
        raise ValueError("pilot still running; use --preview")
    output = Path(directory) / "summary"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("auxiliary final artifacts exist; refusing overwrite")
    output.mkdir(parents=True, exist_ok=True)
    (output / "aux_readout.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, rows in (
        ("curve_results", curves),
        ("comparison_summary", comparisons),
        ("window_results", windows),
        ("paired_results", pairs),
    ):
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
        report = result[0]
        print(
            json.dumps(
                dict(
                    status=report["status"],
                    execution_status=report["execution_status"],
                    committed_new_optimizer_updates=report["training"][
                        "committed_new_optimizer_updates"
                    ],
                    comparisons=result[2],
                    outcome=report["outcome"],
                    new_sampling_cost=report["new_sampling_cost"],
                ),
                indent=2,
                allow_nan=False,
            )
        )
    else:
        print(write_readout(args.directory, result))


if __name__ == "__main__":
    main()

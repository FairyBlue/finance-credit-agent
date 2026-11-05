"""Complete-trajectory evaluation of portfolio SFT initializations on Train only.

The date-separated learning/development subsets are declared by the teacher-data
manifest. Both models use the same visible prompt, temperature zero and strict
four-action decoding. This module performs no optimization or teacher lookahead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from src.agents.portfolio_management_llm_demo import (
    ACTION_IDS,
    DECISION_SESSIONS,
    ENV_PROFILE,
    MAX_MODEL_LEN,
    MAX_RESPONSE,
    aggregate_metrics,
    choice_sampling_factory,
    run_waves,
)
from src.agents.same_prefix_online_collect import _helper, canonical, require, write

PROFILE = "portfolio_teacher_trajectory_eval_v1"
METRICS = (
    "raw_net_return",
    "benchmark_excess",
    "terminal_financial_Y",
    "transaction_cost",
    "max_drawdown",
    "turnover",
)


def paired_readout(reference: list[dict], candidate: list[dict], episode_ids: list[str]) -> dict:
    """Preserve failed slots and expose full-set aggregates only for complete pairs."""
    reference_map = {row["episode_id"]: row for row in reference}
    candidate_map = {row["episode_id"]: row for row in candidate}
    require(
        len(reference_map) == len(reference) == len(episode_ids)
        and len(candidate_map) == len(candidate) == len(episode_ids)
        and set(reference_map) == set(candidate_map) == set(episode_ids),
        "one retained trajectory per declared evaluation episode in each policy",
    )
    rows = []
    for episode_id in episode_ids:
        before, after = reference_map[episode_id], candidate_map[episode_id]
        complete = before["status"] == after["status"] == "COMPLETE"
        deltas = {}
        for metric in METRICS:
            a, b = before.get("metrics", {}).get(metric), after.get("metrics", {}).get(metric)
            finite = (
                type(a) in (int, float) and type(b) in (int, float)
                and math.isfinite(a) and math.isfinite(b)
            )
            deltas[metric] = b - a if complete and finite else None
        rows.append(
            dict(
                episode_id=episode_id,
                reference_status=before["status"],
                candidate_status=after["status"],
                complete_pair=complete,
                reference_action_ids=before.get("action_ids"),
                candidate_action_ids=after.get("action_ids"),
                delta_candidate_minus_reference=deltas,
            )
        )
    all_complete = all(row["complete_pair"] for row in rows)
    means = {
        metric: (
            math.fsum(row["delta_candidate_minus_reference"][metric] for row in rows) / len(rows)
            if all_complete
            and all(row["delta_candidate_minus_reference"][metric] is not None for row in rows)
            else None
        )
        for metric in METRICS
    }
    net_deltas = [row["delta_candidate_minus_reference"]["raw_net_return"] for row in rows]
    net_complete = all(delta is not None for delta in net_deltas)
    return dict(
        planned_pairs=len(rows),
        complete_pairs=sum(row["complete_pair"] for row in rows),
        all_planned_pairs_complete=all_complete,
        mean_delta_all_planned=means,
        mean_net_return_delta_pp=(means["raw_net_return"] * 100 if net_complete else None),
        net_return_window_wins=sum(delta > 0 for delta in net_deltas) if net_complete else None,
        net_return_window_losses=sum(delta < 0 for delta in net_deltas) if net_complete else None,
        net_return_window_ties=sum(delta == 0 for delta in net_deltas) if net_complete else None,
        failed_slots_imputed=False,
        survivor_aggregate_used=False,
        rows=rows,
    )


def run(args) -> dict:
    from src.agents.portfolio_management import PortfolioManagement, load_train_markets

    out, model = args.output.resolve(), args.model.resolve()
    manifest_path = args.split_manifest.resolve()
    require(args.seed == 43, "fixed initialization evaluation seed43")
    require(
        args.prepared is None or args.subset != "both",
        "both subsets use the manifest's two prepared paths; omit --prepared",
    )
    require(model.is_dir(), "existing evaluation model")
    manifest = json.loads(manifest_path.read_text())
    subset_names = ("learning", "development") if args.subset == "both" else (args.subset,)
    groups = {}
    prepared_paths = {}
    for subset in subset_names:
        episode_ids = manifest[f"evaluation_{subset}_episode_ids"]
        require(
            len(episode_ids) == len(set(episode_ids)) == 32
            and all(eid.startswith("formal-train-") for eid in episode_ids),
            "manifest-declared thirty-two Train-only evaluation episodes per subset",
        )
        groups[subset] = episode_ids
        path = (
            args.prepared.resolve() if args.prepared is not None and args.subset != "both"
            else Path(manifest["prepared_paths"][f"{subset}_eval"]).resolve()
        )
        require(path.is_file(), "existing manifest-declared prepared Train subset")
        prepared_paths[subset] = path
    episode_ids = [eid for group in groups.values() for eid in group]
    require(len(episode_ids) == len(set(episode_ids)), "disjoint Train evaluation subsets")
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / "collection_started.json").exists(), "fresh evaluation output required")
    started = time.monotonic()
    zero_shot = args.policy_role == "common_sft"
    provenance = dict(
        profile=PROFILE,
        environment_profile=ENV_PROFILE,
        policy_role=args.policy_role,
        strategy=args.policy_role,
        model_path=str(model),
        prepared_paths={key: str(path) for key, path in prepared_paths.items()},
        split_manifest_path=str(manifest_path),
        evaluation_subset=args.subset,
        episode_ids=episode_ids,
        seed=args.seed,
        zero_shot=zero_shot,
        constrained_decoding=True,
        evaluation_optimizer_updates=0,
        model_learning_claim=False,
        training_split="train",
        formal_valid_access=False,
        formal_final_oos_access=False,
        future_information_in_model_prompt=False,
        teacher_labels_in_model_prompt=False,
        automatic_hold_or_retries=False,
        date_isolation_scope="new teacher supervision only; base model has prior Train exposure",
    )
    write(
        out / "collection_started.json",
        dict(
            **provenance,
            planned_episodes=len(episode_ids),
            planned_model_requests=len(episode_ids) * len(DECISION_SESSIONS),
        ),
    )
    if args.subset == "both":
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Load the Train adapter once, as well as sharing one GPU engine. This
        # temporary parquet only concatenates the two declared evaluation lists.
        inputs = out / "inputs"
        inputs.mkdir(exist_ok=False)
        combined = inputs / "evaluation.parquet"
        pq.write_table(
            pa.concat_tables([pq.read_table(p) for p in prepared_paths.values()]), combined
        )
        markets = load_train_markets(combined, inputs / "adapter")
        require(set(markets) == set(episode_ids), "prepared episodes equal both declared subsets")
    else:
        subset = subset_names[0]
        markets = load_train_markets(prepared_paths[subset], out / "inputs" / subset)
        require(set(markets) == set(groups[subset]), "prepared episodes equal declared subset")
    environments = {
        eid: PortfolioManagement(markets[eid], initialize_equal_weight=True) for eid in episode_ids
    }
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        USE_TORCH="1",
        USE_TF="0",
        USE_FLAX="0",
        VLLM_USE_V1="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    from vllm import LLM, sampling_params

    sampling_factory, choice_settings = choice_sampling_factory(sampling_params, args.seed)
    write(out / "constrained_decoding.json", choice_settings)
    helper = _helper("portfolio_teacher_eval_native_worker", "m2_continuation_gpu_worker.py")
    compiler, version = helper.resolve_compiler()
    os.environ["CC"] = compiler
    settings = helper.engine_settings(str(model))
    settings.update(max_model_len=MAX_MODEL_LEN, seed=args.seed, gpu_memory_utilization=0.35)
    settings["override_generation_config"].update(temperature=0.0, max_new_tokens=MAX_RESPONSE)
    write(out / "engine_settings.json", settings)
    write(out / "compiler.json", dict(CC=compiler, version=version))
    print(f"Loading {args.policy_role} for Train {args.subset} evaluation", str(model), flush=True)
    llm = LLM(**settings)
    tokenizer = llm.get_tokenizer()
    summary = run_waves(
        environments, llm, tokenizer, sampling_factory, out, args.seed,
        strategy=args.policy_role, zero_shot=zero_shot,
    )
    all_complete = summary["complete_trajectories"] == len(episode_ids)
    summary.update(
        **provenance,
        planned_episodes=len(episode_ids),
        planned_model_requests=len(episode_ids) * len(DECISION_SESSIONS),
        action_ids=list(ACTION_IDS),
        decision_session_indices=list(DECISION_SESSIONS),
        all_planned_trajectories_complete=all_complete,
        metric_means_all_planned=(summary["metric_means_over_complete"] if all_complete else None),
        constrained_decoding_api=choice_settings["api"],
        trajectory_results_path=str(out / "trajectory_results.jsonl"),
        elapsed_seconds=time.monotonic() - started,
        engine_shared_across_evaluation_subsets=args.subset == "both",
    )
    if args.reference is not None:
        reference_dir = args.reference.resolve()
        reference_summary = json.loads((reference_dir / "summary.json").read_text())
        require(
            reference_summary["profile"] == PROFILE
            and reference_summary["evaluation_subset"] == args.subset
            and reference_summary["episode_ids"] == episode_ids
            and reference_summary["seed"] == args.seed,
            "same subset and seed in declared reference evaluation",
        )
        reference = [
            json.loads(line)
            for line in (reference_dir / "trajectory_results.jsonl").read_text().splitlines()
        ]
        readout = paired_readout(reference, summary["results"], episode_ids)
        write(out / "comparison_to_reference.json", readout)
        summary["comparison_to_reference"] = dict(
            reference_path=str(reference_dir), **{k: v for k, v in readout.items() if k != "rows"}
        )
    if args.subset == "both":
        requests = [
            json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()
        ]
        summary["subset_summaries"] = {}
        for subset, group_ids in groups.items():
            partition_dir = out / subset
            partition_dir.mkdir(exist_ok=False)
            group_set = set(group_ids)
            rows = [row for row in summary["results"] if row["episode_id"] in group_set]
            group_requests = [row for row in requests if row["episode_id"] in group_set]
            submitted = [row for row in group_requests if row["model_request_submitted"]]
            aggregates = aggregate_metrics(rows)
            complete = aggregates["complete_trajectories"] == len(group_ids)
            partition = dict(
                **provenance,
                status="COMPLETE" if complete else "COMPLETE_WITH_FAILURES",
                planned_episodes=len(group_ids),
                planned_model_requests=len(group_ids) * len(DECISION_SESSIONS),
                model_requests=len(submitted),
                prompt_tokens=sum(row["prompt_tokens"] for row in submitted),
                response_tokens=sum(len(row["response_token_ids"]) for row in submitted),
                complete_requests=sum(row["status"] == "COMPLETE" for row in group_requests),
                failed_requests=sum(row["status"] != "COMPLETE" for row in group_requests),
                all_planned_trajectories_complete=complete,
                metric_means_all_planned=(
                    aggregates["metric_means_over_complete"] if complete else None
                ),
                results=rows,
                shared_requests_path=str(out / "requests.jsonl"),
                constrained_decoding_api=choice_settings["api"],
                engine_shared_across_evaluation_subsets=True,
                **aggregates,
            )
            partition.update(
                evaluation_subset=subset,
                episode_ids=group_ids,
                prepared_path=str(prepared_paths[subset]),
                trajectory_results_path=str(partition_dir / "trajectory_results.jsonl"),
            )
            with (partition_dir / "trajectory_results.jsonl").open("x") as stream:
                for row in rows:
                    stream.write(canonical(row) + "\n")
            if args.reference is not None:
                reference_rows = [row for row in reference if row["episode_id"] in group_set]
                readout = paired_readout(reference_rows, rows, group_ids)
                write(partition_dir / "comparison_to_reference.json", readout)
                partition["comparison_to_reference"] = {
                    key: value for key, value in readout.items() if key != "rows"
                }
            write(partition_dir / "summary.json", partition)
            summary["subset_summaries"][subset] = str(partition_dir / "summary.json")
    write(out / "summary.json", summary)
    temporary = out / "collection_state.tmp"
    public_summary = {key: value for key, value in summary.items() if key != "results"}
    temporary.write_text(canonical(public_summary) + "\n")
    temporary.replace(out / "collection_state.json")
    print(canonical(public_summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--subset", choices=("learning", "development", "both"), required=True)
    parser.add_argument("--policy-role", choices=("common_sft", "teacher_sft"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--reference", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

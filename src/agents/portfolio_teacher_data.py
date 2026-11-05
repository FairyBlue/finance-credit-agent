"""Deterministic Train-only hindsight teachers for the residual portfolio demo.

Labels compare one candidate macro action followed by hold to the original j20.
They are future-outcome supervised labels, not current-policy Q estimates. Agent
messages contain only the replayed state's public point-in-time observation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

from src.agents.broad_prepare import ROOT, TEMPLATE
from src.agents.portfolio_management import ACTIONS, PortfolioManagement, load_train_markets
from src.agents.portfolio_management_llm_demo import prompt_messages

PROFILE = "portfolio_teacher_sft_train_v1"
LEARNING_END = "2024-06-28"
DEVELOPMENT_START = "2024-08-01"
PREFIX_SCHEDULES = {
    "hold": ("hold", "hold", "hold", "hold"),
    "momentum": ("momentum", "momentum", "momentum", "momentum"),
    "mixed_defensive_reversal": ("defensive", "reversal", "momentum", "defensive"),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(canonical(value) + "\n")


def uniform_ids(ids, count=32):
    if len(ids) < count:
        raise ValueError("insufficient unique windows for the fixed evaluation subset")
    return [ids[round(i * (len(ids) - 1) / (count - 1))] for i in range(count)]


def prepare_splits(output) -> dict:
    """Copy old Train settings and select date-separated learning/development."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    destination = Path(output).resolve() / "data"
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "split_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("profile") != PROFILE:
            raise ValueError("existing data belongs to another task profile")
        return manifest
    source = ROOT / "data/processed/train.parquet"
    rows = pq.read_table(source).to_pylist()
    template_row = pq.read_table(TEMPLATE).slice(0, 1).to_pylist()[0]
    frozen = json.loads(template_row["env_kwargs"])
    if len(rows) != 441:
        raise ValueError("the existing full 441-window Train source is required")
    partitions = {"all_train": [], "learning": [], "development": [], "excluded": []}
    dates_by_id, all_sessions = {}, set()
    for index, original in enumerate(rows):
        eid = original["extra_info"]["episode_id"]
        actual = json.loads(original["env_kwargs"])
        dates = [v[:10] for v in actual["episode_dates"]]
        if (
            not eid.startswith("formal-train-")
            or len(dates) != 21
            or not all("2023-01-03" <= day <= "2024-10-31" for day in dates)
        ):
            raise ValueError("teacher input must remain inside the existing historical Train")
        all_sessions.update(dates)
        dates_by_id[eid] = {"start": dates[0], "end": dates[-1]}
        split = (
            "learning"
            if dates[-1] <= LEARNING_END
            else "development"
            if dates[0] >= DEVELOPMENT_START
            else "excluded"
        )
        kwargs = dict(frozen)
        for key in ("episode_dates", "episode_open_times", "benchmark_closes"):
            kwargs[key] = actual[key]
        row = dict(original)
        row["env_kwargs"] = canonical(kwargs)
        row["extra_info"] = dict(
            original["extra_info"],
            index=index,
            source_split="train",
            teacher_partition=split,
            portfolio_teacher_profile=PROFILE,
        )
        partitions["all_train"].append(row)
        partitions[split].append(row)
    gap_sessions = sorted(day for day in all_sessions if LEARNING_END < day < DEVELOPMENT_START)
    if len(gap_sessions) < 20:
        raise ValueError("learning/development boundaries need at least 20 unused Train sessions")
    ids = {key: [r["extra_info"]["episode_id"] for r in value] for key, value in partitions.items()}
    ids["learning_eval"] = uniform_ids(ids["learning"])
    ids["development_eval"] = uniform_ids(ids["development"])
    for partition in ("learning_eval", "development_eval"):
        lookup = {r["extra_info"]["episode_id"]: r for r in partitions["all_train"]}
        partitions[partition] = [lookup[eid] for eid in ids[partition]]
    paths = {}
    for name, selected in partitions.items():
        if name == "excluded":
            continue
        path = destination / f"{name}.parquet"
        if path.exists():
            raise FileExistsError(f"prepared subset already exists without a manifest: {path}")
        pq.write_table(pa.Table.from_pylist(selected), path)
        paths[name] = str(path)
    manifest = dict(
        profile=PROFILE,
        source=str(source),
        frozen_template=str(TEMPLATE),
        source_split="train",
        learning_end=LEARNING_END,
        development_start=DEVELOPMENT_START,
        unused_gap_sessions=gap_sessions,
        unused_gap_session_count=len(gap_sessions),
        prepared_paths=paths,
        learning_episode_ids=ids["learning"],
        development_episode_ids=ids["development"],
        excluded_episode_ids=ids["excluded"],
        evaluation_learning_episode_ids=ids["learning_eval"],
        evaluation_development_episode_ids=ids["development_eval"],
        episode_dates=dates_by_id,
        counts={name: len(value) for name, value in ids.items()},
        prefix_schedules=PREFIX_SCHEDULES,
        planned_learning_states=len(ids["learning"]) * 10,
        future_teacher_outcomes_restricted_to_learning=True,
        formal_valid_access=False,
        formal_final_oos_access=False,
    )
    write_new(manifest_path, manifest)
    return manifest


def fork_environment(env):
    """Fork mutable ledger/state, share read-only execution market data."""
    branch = copy.copy(env)
    branch.ledger = copy.deepcopy(env.ledger)
    branch.decisions = list(env.decisions)
    branch.daily_states = list(env.daily_states)
    return branch


def teacher_state(env, *, prefix_policy, prefix_actions, hold_threshold):
    visible = env.visible()
    q_values, returns, costs = {}, {}, {}
    for action in ACTIONS:
        branch = fork_environment(env)
        branch.step(action)
        while not branch.done:
            branch.step("hold")
        result = branch.result()["metrics"]
        q_values[action] = result["benchmark_excess"]
        returns[action] = result["raw_net_return"]
        costs[action] = result["transaction_cost"]
    raw_best = max(ACTIONS, key=lambda a: q_values[a])
    # Fixed numerical tie handling, not an economic improvement threshold.
    # ACTIONS starts with hold, then preserves the shared menu's fixed order.
    tied_best = next(a for a in ACTIONS if q_values[raw_best] - q_values[a] <= 1e-12)
    gains = {action: q_values[action] - q_values["hold"] for action in ACTIONS}
    label = (
        tied_best
        if hold_threshold is not None and gains[tied_best] > hold_threshold
        else "hold"
        if hold_threshold is not None
        else None
    )
    identifier = hashlib.sha256(canonical(visible).encode()).hexdigest()[:24]
    row = dict(
        profile=PROFILE,
        state_id=identifier,
        episode_id=env.episode_id,
        partition="learning",
        session_index=env.session_index,
        decision_at=visible["current_date"],
        prefix_policy=prefix_policy,
        prefix_actions=list(prefix_actions),
        visible=visible,
        prompt_messages=prompt_messages(visible),
        current_weights=visible["weights"],
        candidate_weights={a["action_id"]: a["target_weights"] for a in visible["actions"]},
        q_by_action=q_values,
        delta_to_hold=gains,
        raw_net_return_by_action=returns,
        total_transaction_cost_by_action=costs,
        teacher_action_raw=raw_best,
        teacher_action=label,
        hold_threshold=hold_threshold,
        sample_weight=1.0,
        q_definition="net-return benchmark excess: one action then hold to original j20",
        continuation="deterministic_hold_to_original_j20",
        current_policy_q_estimate=False,
        future_outcomes_used_only_for_supervised_label=True,
        formal_valid_access=False,
        formal_final_oos_access=False,
    )
    return row


def supervised_row(row):
    if row["teacher_action"] not in ACTIONS:
        raise ValueError("a root-selected hold threshold is required before making SFT labels")
    return dict(
        state_id=row["state_id"],
        episode_id=row["episode_id"],
        session_index=row["session_index"],
        messages=[*row["prompt_messages"], {"role": "assistant", "content": row["teacher_action"]}],
        teacher_action=row["teacher_action"],
        sample_weight=row["sample_weight"],
    )


def build(output, *, hold_threshold=None, max_states=5000):
    output = Path(output).resolve()
    if hold_threshold is not None and (not math.isfinite(hold_threshold) or hold_threshold < 0):
        raise ValueError("hold threshold must be a nonnegative finite net-return fraction")
    if max_states < 2000 or max_states > 5000:
        raise ValueError("the authorized teacher demo is bounded to 2000–5000 states")
    manifest = prepare_splits(output)
    data = output / "data"
    started = time.monotonic()
    write_new(
        data / "dataset_started.json",
        dict(
            profile=PROFILE,
            hold_threshold=hold_threshold,
            maximum_states=max_states,
            planned_states=manifest["planned_learning_states"],
            prefix_schedules=PREFIX_SCHEDULES,
            formal_valid_access=False,
            formal_final_oos_access=False,
        ),
    )
    markets = load_train_markets(manifest["prepared_paths"]["learning"], data / "market_adapter")
    counts, labels, raw_labels = Counter(), Counter(), Counter()
    seen = set()
    files = {"teacher": (data / "teacher_states.jsonl").open("x")}
    if hold_threshold is not None:
        files["train"] = (data / "train.jsonl").open("x")
        files["first"] = (data / "first_batch.jsonl").open("x")
    try:
        for eid, market in markets.items():
            for policy, schedule in PREFIX_SCHEDULES.items():
                env = PortfolioManagement(market)
                prefix = []
                for action in schedule:
                    # j1 is identical for all three prefix trajectories. Later
                    # states are deduplicated as well if exact observations agree.
                    fingerprint = hashlib.sha256(canonical(env.visible()).encode()).hexdigest()[:24]
                    if fingerprint not in seen and counts["states"] < max_states:
                        row = teacher_state(
                            env,
                            prefix_policy=policy,
                            prefix_actions=prefix,
                            hold_threshold=hold_threshold,
                        )
                        seen.add(fingerprint)
                        files["teacher"].write(canonical(row) + "\n")
                        counts["states"] += 1
                        counts[f"j{env.session_index}"] += 1
                        counts["candidate_rollouts"] += 4
                        raw_labels[row["teacher_action_raw"]] += 1
                        if hold_threshold is not None:
                            training_row = canonical(supervised_row(row)) + "\n"
                            files["train"].write(training_row)
                            labels[row["teacher_action"]] += 1
                            if counts["states"] <= 256:
                                files["first"].write(training_row)
                            if counts["states"] == 256:
                                files["first"].flush()
                                files["train"].flush()
                                write_new(
                                    data / "first_batch.ready.json",
                                    dict(
                                        profile=PROFILE,
                                        rows=256,
                                        path=str(data / "first_batch.jsonl"),
                                        hold_threshold=hold_threshold,
                                    ),
                                )
                                print(
                                    "FIRST_BATCH_READY", str(data / "first_batch.jsonl"), flush=True
                                )
                    env.step(action)
                    prefix.append(action)
            counts["windows"] += 1
            if counts["windows"] % 20 == 0:
                for stream in files.values():
                    stream.flush()
                print(
                    canonical(
                        dict(
                            status="BUILDING",
                            counts=dict(counts),
                            last_episode=eid,
                            elapsed_seconds=time.monotonic() - started,
                        )
                    ),
                    flush=True,
                )
            if counts["states"] >= max_states:
                break
    finally:
        for stream in files.values():
            stream.close()
    summary = dict(
        status="COMPLETE",
        profile=PROFILE,
        counts=dict(counts),
        teacher_action_counts=dict(labels),
        raw_argmax_counts=dict(raw_labels),
        hold_threshold=hold_threshold,
        hold_threshold_units="net-return fraction",
        sample_weight=1.0,
        teacher_states=str(data / "teacher_states.jsonl"),
        train_jsonl=str(data / "train.jsonl") if hold_threshold is not None else None,
        split_manifest=str(data / "split_manifest.json"),
        labels_use_future_outcomes=True,
        observations_point_in_time=True,
        current_policy_q_estimate=False,
        development_teacher_labels_created=False,
        formal_valid_access=False,
        formal_final_oos_access=False,
        elapsed_seconds=time.monotonic() - started,
    )
    write_new(data / "dataset_ready.json", summary)
    print(canonical(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hold-threshold", type=float)
    parser.add_argument("--max-states", type=int, default=5000)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        print(canonical(prepare_splits(args.output)), flush=True)
    else:
        build(args.output, hold_threshold=args.hold_threshold, max_states=args.max_states)


if __name__ == "__main__":
    main()

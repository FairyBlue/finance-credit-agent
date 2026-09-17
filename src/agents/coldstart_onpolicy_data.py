"""Aggregate real Train rollout states for offline common SFT supervision.

The original policy requests remain immutable evidence, including rejections.
Teacher branches run in independent environments and never repair those runs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from agent_r1.env.base import Action

from src.agents.coldstart_data import (
    ROOT,
    TRAIN_IDS,
    candidate_calls,
    parse_state,
    render_prompt,
    select_label,
    sha,
    tool_call,
)
from src.env.trading import TradingEnv

PROTOCOL = "single_tool_portfolio_v1"
V2_ROOT = ROOT / "runs/m2-coldstart-v2-20260913"
V3_ROOT = ROOT / "runs/m2-coldstart-v3-20260914"
SOURCE_IDS = [TRAIN_IDS[index] for index in (0, 3, 5, 8, 11, 14, 16, 19)]
MAX_NEW_ROWS = 640
LAST_TRAIN_DATE = "2023-05-22"


def json_default(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def normalized(value):
    return json.loads(json.dumps(value, default=json_default, allow_nan=False))


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, default=json_default, allow_nan=False)
        stream.write("\n")


def complete_request(record):
    tokens = record.get("response_token_ids")
    return (
        record.get("finish_reason") == "stop"
        and isinstance(tokens, list)
        and bool(tokens)
        and len(tokens) < record.get("sampling_params", {}).get("max_tokens", 4096)
        and isinstance(record.get("trace"), dict)
        and isinstance(record.get("info"), dict)
    )


def group_requests(result, records):
    if result.get("status") != "OBSERVED":
        raise ValueError("collection must have a final OBSERVED result")
    rows = result["grid"]
    by_id = {row["id"]: row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError("duplicate collection trajectory ID")
    grouped = {row["id"]: [] for row in rows}
    for record in records:
        if record["trajectory_id"] not in grouped:
            raise ValueError("unknown request trajectory")
        grouped[record["trajectory_id"]].append(record)
    for row in rows:
        group = grouped[row["id"]]
        if [record["request_index"] for record in group] != list(range(len(group))):
            raise ValueError("missing or unordered collection requests")
        if len(group) != row["requests"]:
            raise ValueError("collection request count mismatch")
        invalid = sum(bool(record.get("trace", {}).get("invalid_action")) for record in group)
        if invalid != row["invalid"]:
            raise ValueError("collection invalid count mismatch")
    return rows, grouped


def round_robin(groups):
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index < len(group):
                yield group[index]


def source_reference(source):
    record, row = source["record"], source["row"]
    return {
        "trajectory_id": row["id"],
        "request_index": record["request_index"],
        "source_episode": row["context"]["source_episode"],
        "strata": source["strata"],
        "collection_trajectory_status": row["status"],
        "collection_trajectory_failure": row.get("failure"),
        "source_line_number": record.get("_source_line_number"),
        "source_line_sha256": record.get("_source_line_sha256"),
    }


def select_states(result, records, *, cap=MAX_NEW_ROWS):
    """Select by executed policy errors and fixed indexes before teacher scoring."""
    rows, grouped = group_requests(result, records)
    aliases = defaultdict(list)
    priority_queues, clean_queues = [], []
    exclusions = Counter()
    excluded_errors = Counter()
    excluded_details = []
    prompt_texts = {}
    for row in rows:
        priority, clean = [], []
        records_for_row = grouped[row["id"]]
        prefix_complete = True
        dates = row["context"]["env_kwargs"]["episode_dates"]
        for index, record in enumerate(records_for_row):
            is_invalid = bool(record.get("trace", {}).get("invalid_action"))
            post_error = index > 0 and bool(
                records_for_row[index - 1].get("trace", {}).get("invalid_action")
            )
            reason = None
            if not prefix_complete:
                reason = "incomplete_prior_request"
            elif not complete_request(record):
                reason = "incomplete_or_unexecuted_request"
            else:
                session = record["trace"]["decision_session_index"]
                if not isinstance(session, int) or not 0 <= session < len(dates):
                    raise ValueError("source trace has impossible decision session")
                if record["trace"].get("decision_at") != dates[session]:
                    raise ValueError("source trace date and original full context disagree")
                if dates[session][:10] > LAST_TRAIN_DATE:
                    reason = "outside_original_train_dates"
                elif session + 3 >= len(dates):
                    reason = "fewer_than_three_future_sessions"
            if reason:
                exclusions[reason] += 1
                if is_invalid or post_error:
                    excluded_errors[reason] += 1
                excluded_details.append(
                    {
                        "trajectory_id": row["id"],
                        "request_index": index,
                        "reason": reason,
                        "invalid": is_invalid,
                        "post_error": post_error,
                    }
                )
            else:
                prompt = record["prompt"]
                digest = text_sha(prompt)
                if digest in prompt_texts and prompt_texts[digest] != prompt:
                    raise ValueError("prompt hash collision")
                prompt_texts[digest] = prompt
                strata = []
                if is_invalid:
                    strata.append("invalid")
                if post_error:
                    strata.append("post_error")
                if not strata:
                    strata.append("clean_even" if index % 2 == 0 else "clean_odd")
                source = {
                    "row": row,
                    "record": record,
                    "records": records_for_row,
                    "prompt_sha256": digest,
                    "strata": strata,
                }
                aliases[digest].append(source)
                if is_invalid or post_error:
                    priority.append(source)
                elif index % 2 == 0:
                    clean.append(source)
            prefix_complete = prefix_complete and complete_request(record)
        priority_queues.append(priority)
        clean_queues.append(clean)
    selected, seen = [], set()
    for source in round_robin(priority_queues):
        if source["prompt_sha256"] not in seen:
            selected.append(source)
            seen.add(source["prompt_sha256"])
    priority_count = len(selected)
    blocked = priority_count > cap
    if not blocked:
        for source in round_robin(clean_queues):
            if len(selected) >= cap:
                break
            if source["prompt_sha256"] not in seen:
                selected.append(source)
                seen.add(source["prompt_sha256"])
    for source in selected:
        source["aliases"] = [source_reference(alias) for alias in aliases[source["prompt_sha256"]]]
    selection = {
        "status": "BLOCKED_PRIORITY_EXCEEDS_CAP" if blocked else "SELECTED",
        "cap": cap,
        "priority_unique_prompts": priority_count,
        "eligible_unique_prompts": len(aliases),
        "selected_unique_prompts": len(selected),
        "selected_clean_fillers": 0 if blocked else len(selected) - priority_count,
        "excluded_request_counts": dict(exclusions),
        "excluded_invalid_or_post_error_counts": dict(excluded_errors),
        "excluded_requests": excluded_details,
        "selection_uses_teacher_scores_or_returns": False,
        "all_eligible_priority_prompts_retained": not blocked,
        "same_prompt_sources_merged_before_priority_selection": True,
        "selected_sources": [
            source_reference(source)
            | {
                "prompt_sha256": source["prompt_sha256"],
                "all_sources": source["aliases"],
            }
            for source in selected
        ],
    }
    return selected, selection


def verify_prompt(tokenizer, env, obs, record):
    prompt = render_prompt(tokenizer, env, obs, action_protocol=PROTOCOL)
    if prompt != record["prompt"]:
        raise ValueError(
            f"replayed prompt mismatch: {record['trajectory_id']}:{record['request_index']}"
        )
    if tokenizer.encode(prompt, add_special_tokens=False) != record["prompt_token_ids"]:
        raise ValueError("replayed prompt token IDs differ from actual policy request")
    return prompt


async def apply_original(env, record):
    if not complete_request(record):
        raise ValueError("cannot replay an incomplete model request")
    obs, reward, done, info = await env.step(
        Action(
            text=record["response"],
            token_ids=record["response_token_ids"],
        )
    )
    trace = normalized(asdict(env.credit_step_trace[-1]))
    if (
        trace != record["trace"]
        or normalized(info) != record["info"]
        or reward != record.get("reward")
    ):
        raise ValueError("original request replay differs in trace, info, or reward")
    return obs, done


async def replay_to_request(source, tokenizer, *, check_prompts=True):
    context = source["row"]["context"]
    env = TradingEnv(**context["env_kwargs"])
    obs = env.reset(raw_prompt=context["raw_prompt"])
    if context.get("prefix"):
        raise ValueError("on-policy collection must begin from cash with no scripted prefix")
    target_index = source["record"]["request_index"]
    for record in source["records"][:target_index]:
        if check_prompts:
            verify_prompt(tokenizer, env, obs, record)
        obs, done = await apply_original(env, record)
        if done:
            raise ValueError("collection prefix terminated before selected request")
    verify_prompt(tokenizer, env, obs, source["record"])
    if env.current_session_index != source["record"]["trace"]["decision_session_index"]:
        raise ValueError("selected state session differs after replay")
    return env, obs


async def score_candidate(source, tokenizer, completion):
    # Prefix requests are replayed and checked for every branch; prompt encoding
    # of all earlier steps is checked once per selected state, then at its endpoint.
    env, _ = await replay_to_request(source, tokenizer, check_prompts=False)
    start_index = env.current_session_index
    horizon = start_index + 3
    dates = source["row"]["context"]["env_kwargs"]["episode_dates"]
    if horizon >= len(dates) or dates[horizon][:10] > LAST_TRAIN_DATE:
        raise ValueError("teacher horizon escapes original Train dates")
    _, _, _, first_info = await env.step(Action(text=completion))
    if first_info.get("invalid_action"):
        raise ValueError(f"teacher candidate illegal: {completion}")
    info = first_info
    while env.current_session_index < horizon:
        _, _, _, info = await env.step(Action(text=tool_call("hold")))
        if info.get("invalid_action"):
            raise ValueError("teacher horizon hold failed")
    if env.current_session_index != horizon:
        raise ValueError("teacher valuation is not exactly three sessions later")
    return {
        "completion": completion,
        # Kept for the unchanged select_label interface. This is the terminal
        # value of the three-session branch, not necessarily the full episode.
        "terminal_equity": info["equity"],
        "valuation_session_index": horizon,
        "valuation_date": dates[horizon],
        "full_episode_terminal_at_valuation": env.episode_complete,
        "candidate_fills": normalized(first_info["fills"]),
        "candidate_skipped_orders": normalized(first_info.get("skipped_orders", [])),
        "candidate_cost": sum(
            fill["commission"] + fill["slippage_cost"] for fill in first_info["fills"]
        ),
        "zero_fill": not first_info["fills"],
    }


async def make_example(source, tokenizer):
    env, obs = await replay_to_request(source, tokenizer)
    prompt = source["record"]["prompt"]
    state = parse_state(obs)
    # Confirm the selected original action itself, including a genuine refusal.
    # Its outcome is evidence, never the teacher action's outcome.
    await apply_original(env, source["record"])
    scores = [await score_candidate(source, tokenizer, call) for call in candidate_calls(state)]
    winner = select_label(scores)
    session = state["session_index"]
    context, record = source["row"]["context"], source["record"]
    return {
        "id": "onpolicy-state-" + source["prompt_sha256"],
        "kind": "decision",
        "prompt": prompt,
        "completion": winner["completion"],
        "metadata": {
            "example_family": "onpolicy_state_teacher_label",
            "action_protocol": PROTOCOL,
            "source_episode": context["source_episode"],
            "decision_date": state["current_date"],
            "decision_session_index": session,
            "sessions_remaining": state["sessions_remaining"],
            "horizon_dates": context["env_kwargs"]["episode_dates"][session + 1 : session + 4],
            "source_request": source_reference(source),
            "all_source_aliases": source["aliases"],
            "original_response": record["response"],
            "original_response_token_ids": record["response_token_ids"],
            "original_trace": record["trace"],
            "original_info": record["info"],
            "executed_prefix_requests": [
                {
                    "request_index": prior["request_index"],
                    "response": prior["response"],
                    "source_line_sha256": prior.get("_source_line_sha256"),
                    "invalid_action": prior["trace"]["invalid_action"],
                }
                for prior in source["records"][: record["request_index"]]
            ],
            "visible_weights": {
                symbol: position["weight"] for symbol, position in state["positions"].items()
            },
            "prompt_sha256": source["prompt_sha256"],
            "prompt_token_ids_sha256": text_sha(
                json.dumps(record["prompt_token_ids"], separators=(",", ":"))
            ),
            "prompt_and_tokens_exactly_replayed": True,
            "original_request_replayed_unchanged": True,
            "teacher_target": "actual net equity three sessions later; full context retained",
            "teacher_label": json.loads(winner["completion"][11:-12]),
            "candidate_scores": scores,
            "future_information_in_prompt": False,
            "is_on_policy_rl_sample": False,
        },
    }


def verify_collection_sources(result, source_rows):
    by_id = {row["extra_info"]["episode_id"]: row for row in source_rows}
    used = set()
    for row in result["grid"]:
        context = row["context"]
        episode = context.get("source_episode")
        if episode not in SOURCE_IDS:
            raise ValueError("unexpected Train collection window")
        original = by_id[episode]
        if original["extra_info"]["source_split"] != "train":
            raise ValueError("non-Train collection source")
        expected_kwargs = json.loads(original["env_kwargs"])
        expected_kwargs.update(
            observation_profile="history_features_v1", reward_mode="finance_trajectory"
        )
        if context["env_kwargs"] != expected_kwargs or context["raw_prompt"] != list(
            original["prompt"]
        ):
            raise ValueError("collection context differs from original full Train episode")
        if len(expected_kwargs["episode_dates"]) != 21 or context.get("prefix"):
            raise ValueError("collection context was shortened or manually prefixed")
        used.add(episode)
    if used != set(SOURCE_IDS) or len(result["grid"]) != 40:
        raise ValueError("expected exactly eight Train windows and five lanes each")
    if result.get("action_protocol") != PROTOCOL:
        raise ValueError("collection prompt protocol differs")


async def build_dataset(collection, prepared, output, model):
    import pandas as pd
    from transformers import AutoTokenizer

    from src.agents.coldstart_sft import read_examples

    output.mkdir(parents=True, exist_ok=False)
    old_dataset = V2_ROOT / "dataset"
    old_manifest = json.loads((old_dataset / "manifest.json").read_text())
    for filename, key in (
        ("train.jsonl", "train_jsonl_sha256"),
        ("eval_contexts.json", "eval_contexts_sha256"),
        ("regression_eval_contexts.json", "regression_eval_contexts_sha256"),
    ):
        if sha(old_dataset / filename) != old_manifest[key]:
            raise ValueError("immutable v2 dataset checksum mismatch")
    old_bytes = (old_dataset / "train.jsonl").read_bytes()
    if len(old_bytes.splitlines()) != 740 or not old_bytes.endswith(b"\n"):
        raise ValueError("v2 replay must retain exactly 740 original JSON lines")
    result_path, requests_path = collection / "result.json", collection / "requests.jsonl"
    result = json.loads(result_path.read_text())
    prepared_manifest = json.loads((prepared / "manifest.json").read_text())
    old_readout_inputs = json.loads((V2_ROOT / "readout.json").read_text())["inputs_sha256"]
    for filename in ("eval_contexts.json", "regression_eval_contexts.json"):
        digest = sha(old_dataset / filename)
        if digest != prepared_manifest["reserved_contexts_sha256"].get(str(old_dataset / filename)):
            raise ValueError("prepared reserved evaluation identity differs from v2")
        if digest != old_readout_inputs.get("dataset/" + filename):
            raise ValueError("reserved context identity differs from completed v2 readout")
    context_path = prepared / "collection_contexts.json"
    if prepared_manifest.get("status") != "COMPLETE" or (
        sha(context_path) != prepared_manifest["collection_contexts_sha256"]
    ):
        raise ValueError("prepared collection context identity mismatch")
    prepared_contexts = json.loads(context_path.read_text())
    prepared_by_id = {context["id"]: context for context in prepared_contexts["full"]}
    if any(row["context"] != prepared_by_id.get(row["context_id"]) for row in result["grid"]):
        raise ValueError("actual collection contexts differ from prepared contexts")
    if model.resolve() != Path(prepared_manifest["tokenizer_model_path"]).resolve():
        raise ValueError("tokenizer model differs from collection model")
    records = []
    with requests_path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            record.update(_source_line_number=line_number, _source_line_sha256=text_sha(line))
            records.append(record)
    artifact = ROOT / "artifacts/m2_screening"
    artifact_manifest = json.loads((artifact / "manifest.json").read_text())
    if sha(artifact / "train.parquet") != artifact_manifest["datasets"]["train"]["sha256"]:
        raise ValueError("original Train parquet checksum mismatch")
    verify_collection_sources(
        result, pd.read_parquet(artifact / "train.parquet").to_dict("records")
    )
    selected, selection = select_states(result, records)
    write_json(output / "selection.json", selection)
    if selection["status"] != "SELECTED":
        write_json(
            output / "manifest.json",
            {
                "status": selection["status"],
                "priority_unique_prompts": selection["priority_unique_prompts"],
                "max_new_rows": MAX_NEW_ROWS,
                "collection_result_sha256": sha(result_path),
                "collection_requests_sha256": sha(requests_path),
                "original_collection_preserved": True,
            },
        )
        raise RuntimeError("eligible error-priority prompts exceed 640; refusing to truncate them")
    if not selected:
        raise ValueError("no labelable on-policy collection states")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    new_rows = []
    for index, source in enumerate(selected):
        new_rows.append(await make_example(source, tokenizer))
        if index % 10 == 0:
            print(
                json.dumps(
                    {
                        "teacher_states_completed": index + 1,
                        "teacher_states_selected": len(selected),
                    }
                ),
                flush=True,
            )
    with (output / "train.jsonl").open("xb") as stream:
        stream.write(old_bytes)
        for row in new_rows:
            stream.write(
                (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            )
    if (output / "train.jsonl").read_bytes()[: len(old_bytes)] != old_bytes:
        raise ValueError("v2 replay bytes were altered")
    for filename in ("eval_contexts.json", "regression_eval_contexts.json"):
        with (output / filename).open("xb") as stream:
            stream.write((old_dataset / filename).read_bytes())
    examples = read_examples(output / "train.jsonl", tokenizer, 8192)
    dates = {date for row in new_rows for date in row["metadata"]["horizon_dates"]}
    dates.update(row["metadata"]["decision_date"] for row in new_rows)
    if any(day[:10] > LAST_TRAIN_DATE for day in dates):
        raise ValueError("new labels escape original Train dates")
    label_counts = Counter(json.loads(row["completion"][11:-12])["name"] for row in new_rows)
    target_counts = Counter(
        json.dumps(json.loads(row["completion"][11:-12]), sort_keys=True) for row in new_rows
    )
    manifest = {
        "status": "COMPLETE",
        "created_at_SGT": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "row_count": len(examples),
        "new_rows": len(new_rows),
        "original_rows_retained": 740,
        "train_jsonl_sha256": sha(output / "train.jsonl"),
        "eval_contexts_sha256": sha(output / "eval_contexts.json"),
        "regression_eval_contexts_sha256": sha(output / "regression_eval_contexts.json"),
        "generator_sha256": sha(Path(__file__)),
        "action_protocol": PROTOCOL,
        "original_v2_train_sha256": sha(old_dataset / "train.jsonl"),
        "original_v2_manifest_sha256": sha(old_dataset / "manifest.json"),
        "original_v2_readout_sha256": sha(V2_ROOT / "readout.json"),
        "original_740_lines_byte_identical": True,
        "eval_contexts_byte_identical_to_v2": True,
        "collection_result_sha256": sha(result_path),
        "collection_requests_sha256": sha(requests_path),
        "prepared_manifest_sha256": sha(prepared / "manifest.json"),
        "prepared_collection_contexts_sha256": sha(context_path),
        "source_train_sha256": sha(artifact / "train.parquet"),
        "source_train_ids": SOURCE_IDS,
        "selection_sha256": sha(output / "selection.json"),
        "max_new_rows": MAX_NEW_ROWS,
        "priority_unique_prompts": selection["priority_unique_prompts"],
        "selected_clean_fillers": selection["selected_clean_fillers"],
        "excluded_request_counts": selection["excluded_request_counts"],
        "excluded_invalid_or_post_error_counts": selection["excluded_invalid_or_post_error_counts"],
        "new_label_distribution": dict(label_counts),
        "new_target_distribution": dict(target_counts),
        "new_session_index_distribution": dict(
            Counter(row["metadata"]["decision_session_index"] for row in new_rows)
        ),
        "new_sessions_remaining_distribution": dict(
            Counter(row["metadata"]["sessions_remaining"] for row in new_rows)
        ),
        "new_source_status_distribution": dict(
            Counter(
                row["metadata"]["source_request"]["collection_trajectory_status"]
                for row in new_rows
            )
        ),
        "new_decision_and_horizon_dates_min": min(dates),
        "new_decision_and_horizon_dates_max": max(dates),
        "max_prompt_tokens": max(row["prompt_tokens"] for row in examples),
        "max_total_tokens_including_eos": max(len(row["input_ids"]) for row in examples),
        "target_tokens_per_epoch": sum(row["target_tokens"] for row in examples),
        "all_selected_prompts_and_token_ids_exactly_replayed": True,
        "all_original_selected_requests_replayed_unchanged": True,
        "teacher_preserves_original_21_session_context": True,
        "teacher_horizon_sessions": 3,
        "teacher_priors": "unchanged GS/MSFT/HD grid buy0.1/0.2,sell0/0.1,hold; "
        "net-equity argmax and hold-first economic ties; no hold-label balancing",
        "prior_warning": (
            "20% buy/full-liquidation preference may remain; state aggregation does not prove alpha"
        ),
        "tokenizer_snapshot": str(model),
        "non_thinking": True,
        "future_information_in_prompt": False,
        "is_on_policy_rl_data": False,
        "original_collection_requests_modified": False,
        "source_incomplete_trajectories_are_not_complete_evaluations": True,
    }
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "validation_receipt.json",
        {
            "status": "PASS",
            "all_selected_original_prompt_tokens_matched": True,
            "all_selected_original_requests_replayed": True,
            "all_teacher_branches_start_at_same_real_state": True,
            "original_v2_740_lines_byte_identical": True,
            "eval_contexts_byte_identical": True,
            "sft_read_examples_passed": len(examples),
            "max_total_tokens": manifest["max_total_tokens_including_eos"],
            "no_truncation": True,
            "selection_before_teacher_scoring": True,
        },
    )
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=V3_ROOT / "collection/rollout")
    parser.add_argument("--prepared", type=Path, default=V3_ROOT / "collection/prepared")
    parser.add_argument("--output", type=Path, default=V3_ROOT / "dataset")
    parser.add_argument("--model", type=Path, default=V2_ROOT / "merged_model")
    args = parser.parse_args()
    asyncio.run(build_dataset(args.collection, args.prepared, args.output, args.model))


if __name__ == "__main__":
    main()

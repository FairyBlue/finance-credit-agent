"""CPU-only, endpoint-preserving API recovery SFT and independent Train checks.

Recovery targets use current API semantics, never future returns. Positions are
created by fixed legal next-open trades. The old financial demonstrations retain
their original labels and priors; they are rehearsal data, not new API teachers.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent_r1.env.base import Action

from src.agents import order_recovery_probe as probe
from src.agents.coldstart_data import parse_state, render_prompt, tool_call
from src.agents.coldstart_sft import read_examples
from src.env.order_feedback import ORDER_DIRECTION_TOLERANCE
from src.env.trading import TargetArgs, TradingEnv

ROOT = Path(os.environ.get("FINANCE_CREDIT_AGENT_ROOT", Path(__file__).resolve().parents[2]))
TRAIN_IDS = tuple(f"formal-train-{n:04d}" for n in (1, 65, 9, 73, 17, 149, 25, 157))
ACCEPTANCE_IDS = tuple(f"formal-train-{n:04d}" for n in (105, 113, 121))
PROFILE = "history_features_recovery_v2"
PROTOCOL = "single_tool_portfolio_v1"
REPLAY_QUOTAS = {
    ("decision", "buy"): 64, ("decision", "sell"): 56, ("decision", "hold"): 8,
    ("api", "buy"): 20, ("api", "sell"): 20, ("api", "hold"): 21, ("api", "abstain"): 3,
}


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def normalize(value: Any) -> Any:
    return json.loads(probe.canonical(value))


def kwargs_for(row: dict[str, Any], profile: str = PROFILE) -> dict[str, Any]:
    result = (json.loads(row["env_kwargs"]) if isinstance(row["env_kwargs"], str)
              else dict(row["env_kwargs"]))
    result.update(observation_profile=profile, reward_mode="finance_trajectory",
                  capture_interactions=True)
    return result


def row_dates(row: dict[str, Any]) -> set[str]:
    return {str(stamp)[:10] for stamp in kwargs_for(row)["episode_dates"]}


def guard_source(row: dict[str, Any], forbidden: set[str]) -> None:
    if row["extra_info"].get("source_split") != "train":
        raise ValueError("Only original Train source rows are eligible")
    dates = row_dates(row)
    if not dates or min(dates) < "2023-01-03" or max(dates) > "2024-10-31":
        raise ValueError("Source dates escape original Train")
    if dates & forbidden:
        raise ValueError("Source dates overlap a held-out recovery bank")


def reserved_dates(manifest: dict[str, Any]) -> set[str]:
    cases = [case for case in manifest["cases"] if case["kind"] == "rejection"]
    if {case["id"] for case in cases} != set(probe.FAILURE_IDS):
        raise ValueError("The six known failure cases must remain reserved")
    return {stamp[:10] for case in cases for stamp in case["env_kwargs"]["episode_dates"]}


def direction_for_target(current: float, target: float) -> str:
    if not all(math.isfinite(value) for value in (current, target)) or not 0 <= target <= 1:
        raise ValueError("Endpoint and current weight must be finite fractions")
    if target > current + ORDER_DIRECTION_TOLERANCE:
        return "buy"
    if target < current - ORDER_DIRECTION_TOLERANCE:
        return "sell"
    return "no_trade"


def recovery_specs() -> list[dict[str, Any]]:
    weights = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.75, 0.90)
    result = []
    for mode in ("decrease", "increase"):
        for index, weight in enumerate(weights):
            result.append({"mode": mode, "weight": weight, "grid_index": index,
                           "history": ("single", "repeat4", "ABAB")[len(result) % 3]})
    for index, weight in enumerate((0.12, 0.27, 0.43, 0.68)):
        result.append({"mode": ("lower_exact", "upper_exact", "lower_inside", "upper_inside")
                       [index], "weight": weight, "grid_index": index,
                       "history": ("single", "repeat4", "ABAB", "single")[index]})
    return result


def acceptance_specs() -> list[dict[str, Any]]:
    return [{"mode": mode, "weight": weight, "grid_index": index,
             "history": ("single", "ABAB", "repeat4")[index % 3], "acceptance": True}
            for index, (weight, mode) in enumerate(zip(
                (0.07, 0.18, 0.29, 0.47, 0.58, 0.71, 0.83, 0.94),
                ("decrease", "increase", "decrease", "increase", "lower_exact",
                 "upper_inside", "decrease", "increase"), strict=True))]


def target_for(spec: dict[str, Any], current: float) -> float:
    mode, index = spec["mode"], spec["grid_index"]
    if mode.startswith("lower") or mode.startswith("upper"):
        threshold = current + (ORDER_DIRECTION_TOLERANCE if mode.startswith("upper")
                               else -ORDER_DIRECTION_TOLERANCE)
        return math.nextafter(threshold, current) if mode.endswith("inside") else threshold
    if spec.get("acceptance"):
        return (max(0.0, current - (0.045, 0.07, 0.18, 0.15, 0.1, 0.1, 0.26, 0.1)[index])
                if mode == "decrease" else min(0.99, current + (0.04 if index == 7 else 0.13)))
    if mode == "decrease":
        return (0.0, current * 0.5, max(0.0, current - 0.03),
                math.nextafter(current - ORDER_DIRECTION_TOLERANCE, 0.0), 0.1,
                current * 0.7, 0.05, 0.25, current - 0.2, current - 0.35)[index]
    return (0.12, 0.17, current + 0.05,
            math.nextafter(current + ORDER_DIRECTION_TOLERANCE, 1.0), 0.3,
            0.45, 0.6, 0.75, 0.85, 0.95)[index]


def ledger_signature(env: TradingEnv) -> str:
    return probe.canonical({**probe.ledger_state(env),
                            "pending_orders": [asdict(order)
                                               for order in env.ledger.pending_orders],
                            "equity_curve": env._equity_curve})


async def prepared_state(row: dict[str, Any], context_index: int, ordinal: int,
                         spec: dict[str, Any], *, profile: str = PROFILE,
                         env_class: Any = TradingEnv) -> tuple[Any, Any, dict[str, Any]]:
    env = env_class(**kwargs_for(row, profile))
    obs = env.reset(raw_prompt=list(row["prompt"]))
    primary = env.symbols[(context_index * 7 + ordinal) % len(env.symbols)]
    secondary = env.symbols[(context_index * 7 + ordinal + 3) % len(env.symbols)]
    if primary == secondary:
        raise ValueError("Two distinct symbols are required for legal state prefixes")
    prefix = [tool_call("buy", {"symbol": primary, "target_weight": spec["weight"]}),
              tool_call("buy", {"symbol": secondary,
                                 "target_weight": 0.015 if spec["weight"] > 0.8 else 0.04})]
    for text in prefix:
        obs, _, done, info = await env.step(Action(text=text))
        if done or info.get("invalid_action") or len(info.get("fills", ())) != 1:
            raise ValueError("A fixed legal state prefix did not produce its intended buy fill")
        if any(fill["commission"] <= 0 or fill["slippage_cost"] <= 0 for fill in info["fills"]):
            raise ValueError("Recovery state prefixes must retain real nonzero execution costs")
    current = env._current_weight(primary)
    target = target_for(spec, current)
    expected_direction = direction_for_target(current, target)
    wrong_name = ("buy" if spec["mode"] == "decrease" else "sell")
    if spec["mode"].startswith("lower"):
        wrong_name = "sell"
    elif spec["mode"].startswith("upper"):
        wrong_name = "buy"
    wrong = tool_call(wrong_name, {"symbol": primary, "target_weight": target})
    other_weight = env._current_weight(secondary)
    alternate = tool_call("sell" if wrong_name == "buy" else "buy", {
        "symbol": secondary, "target_weight": other_weight + 0.11 if wrong_name == "buy" else 0.0,
    })
    rejected = {"single": [wrong], "repeat4": [wrong] * 4,
                "ABAB": [alternate, wrong, alternate, wrong]}[spec["history"]]
    before = ledger_signature(env)
    for text in rejected:
        obs, reward, done, info = await env.step(Action(text=text))
        if (not info.get("invalid_action") or info.get("error") not in probe.DIRECTION_ERRORS
                or done or reward != 0 or ledger_signature(env) != before):
            raise ValueError("Synthetic rejection did not preserve the real ledger and session")
    return env, obs, {
        "primary_symbol": primary, "secondary_symbol": secondary,
        "current_weight": current, "requested_target": target,
        "expected_direction": expected_direction, "legal_prefix": prefix,
        "rejected_prefix": rejected, "state_spec": copy.deepcopy(spec),
    }


def label_for(semantics: dict[str, Any]) -> str:
    direction = semantics["expected_direction"]
    if direction == "no_trade":
        return tool_call("hold")
    return tool_call(direction, {"symbol": semantics["primary_symbol"],
                                 "target_weight": semantics["requested_target"]})


def validate_label_without_future(env: TradingEnv, completion: str,
                                  semantics: dict[str, Any]) -> None:
    call = json.loads(completion[11:-12])
    before_index = env.current_session_index
    if call["name"] in {"buy", "sell"}:
        env._queue_directional_target(call["name"], TargetArgs(**call["arguments"]))
        if len(env.ledger.pending_orders) != 1:
            raise ValueError("Current API validator did not queue the endpoint label")
    elif call != {"name": "hold", "arguments": {}} or semantics["expected_direction"] != "no_trade":
        raise ValueError("Recovery hold is permitted only for a tolerance-level no_trade endpoint")
    if env.current_session_index != before_index:
        raise ValueError("CPU label validation advanced into a future session")


async def recovery_records(rows: list[dict[str, Any]], tokenizer: Any, *,
                           env_class: Any = TradingEnv) -> list[dict[str, Any]]:
    records = []
    for context_index, row in enumerate(rows):
        for ordinal, spec in enumerate(recovery_specs()):
            env, obs, semantics = await prepared_state(row, context_index, ordinal, spec,
                                                       env_class=env_class)
            state = parse_state(obs)
            if state["order_rejection"]["direction_for_requested_target"] != (
                semantics["expected_direction"]
            ):
                raise ValueError("Independent endpoint label disagrees with frozen v2 feedback")
            completion = label_for(semantics)
            prompt = render_prompt(tokenizer, env, obs, action_protocol=PROTOCOL)
            metadata = {
                "example_family": "current_api_endpoint_recovery", "action_protocol": PROTOCOL,
                "observation_profile": PROFILE, "source_episode": row["extra_info"]["episode_id"],
                "decision_date": state["current_date"],
                "decision_session_index": state["session_index"],
                "current_state_sha256": digest_text(probe.canonical(state)),
                "prompt_sha256": digest_text(prompt),
                "ledger_before_label": probe.ledger_state(env),
                "label_policy": "preserve last requested TOTAL endpoint "
                                "using current API direction",
                "hold_scope": "within validator tolerance; hold preserves the existing portfolio",
                "future_prices_used_to_choose_label": False,
                "state_source": "real Train market and ledger after fixed legal scripted buys",
                "is_on_policy_sample": False, "cpu_label_validation_advances_session": False,
                **semantics,
            }
            validate_label_without_future(env, completion, semantics)
            records.append({"id": f"recovery-api-{context_index:02d}-{ordinal:02d}", "kind": "api",
                            "prompt": prompt, "completion": completion, "metadata": metadata})
    return records


def select_replay(lines: list[str], source_rows: dict[str, dict[str, Any]],
                  forbidden: set[str]) -> tuple[list[str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
    for line in lines:
        record = json.loads(line)
        meta = record["metadata"]
        if (meta.get("example_family") not in {"clean", "replay"}
                or meta.get("recovery_kind") is not None
                or any(meta.get("prefix_expected_invalid", ()) or ())):
            continue
        source = source_rows.get(meta.get("source_episode"))
        if source is None:
            raise ValueError("Original SFT replay source episode is unavailable")
        guard_source(source, forbidden)
        dates = {str(meta["decision_date"])[:10]}
        dates.update(str(stamp)[:10] for stamp in meta.get("horizon_dates", ()))
        if dates & forbidden or any(not "2023-01-03" <= stamp <= "2023-05-22" for stamp in dates):
            raise ValueError("Original SFT replay dates are not isolated early Train")
        visible_dates = re.findall(r'"current_date"\s*:\s*"([^\"]+)"', record["prompt"])
        if visible_dates != [meta["decision_date"]]:
            raise ValueError("Original replay prompt date disagrees with its metadata")
        action = json.loads(record["completion"][11:-12])["name"]
        groups.setdefault((record["kind"], action), []).append((line, record))
    selected, provenance, seen = [], [], set()
    for key, count in REPLAY_QUOTAS.items():
        group = sorted(groups.get(key, ()), key=lambda item: digest_text(item[0]))
        chosen = 0
        for line, record in group:
            prompt_sha = digest_text(record["prompt"])
            if prompt_sha in seen:
                continue
            selected.append(line)
            provenance.append({"id": record["id"], "kind": key[0], "action": key[1],
                               "source_line_sha256": digest_text(line), "prompt_sha256": prompt_sha,
                               "source_episode": record["metadata"]["source_episode"],
                               "decision_date": record["metadata"]["decision_date"]})
            seen.add(prompt_sha)
            chosen += 1
            if chosen == count:
                break
        if chosen != count:
            raise ValueError(f"Fixed replay quota cannot be filled: {key}: {chosen}/{count}")
    return selected, provenance


async def acceptance_bank(rows: list[dict[str, Any]], tokenizer: Any, *,
                          env_class: Any = TradingEnv) -> dict[str, Any]:
    cases = []
    for context_index, row in enumerate(rows):
        for ordinal, spec in enumerate(acceptance_specs()):
            env, _, semantics = await prepared_state(row, context_index + 13, ordinal + 29, spec,
                                                     profile=probe.OLD, env_class=env_class)
            case = {
                "id": f"recovery-accept-{context_index:02d}-{ordinal:02d}", "kind": "rejection",
                "env_kwargs": kwargs_for(row, probe.OLD), "raw_prompt": list(row["prompt"]),
                "prefix_interactions": copy.deepcopy(env.interaction_trace),
                "prefix_steps": normalize([asdict(step) for step in env.credit_step_trace]),
                "prefix": semantics["legal_prefix"] + semantics["rejected_prefix"],
                "source_base_seed": 44,
                "source_identity": {"step": 0, "sample_index": len(cases),
                                    "rollout_n": 0, "validate": True},
                "source": {"episode_id": row["extra_info"]["episode_id"],
                           "row_inputs_sha256": digest_text(probe.canonical({
                               "env_kwargs": {key: value for key, value in
                                              kwargs_for(row, probe.OLD).items()
                                              if key != "adapter"},
                               "raw_prompt": list(row["prompt"])})),
                           "synthetic_actions": True},
                "expected_start_session_index": env.current_session_index,
                "takeover_ledger": probe.ledger_state(env),
            }
            old, _ = await probe.replay_case(case, probe.OLD, env_class=env_class)
            current, obs = await probe.replay_case(case, PROFILE, env_class=env_class)
            if probe.ledger_state(old) != probe.ledger_state(current):
                raise ValueError("Acceptance v2 intervention changed the initial ledger")
            state = parse_state(obs)
            expected_call = label_for(semantics)
            if state["order_rejection"]["direction_for_requested_target"] != (
                semantics["expected_direction"]
            ):
                raise ValueError("Acceptance CPU direction differs from current API feedback")
            case.update(raw_state=state, production_prompt=render_prompt(
                tokenizer, current, obs, action_protocol=PROTOCOL),
                cpu_expected_direction=semantics["expected_direction"],
                cpu_expected_next_call=expected_call, state_spec=semantics,
                teacher_call_is_submitted=False, training_example=False)
            validate_label_without_future(current, expected_call, semantics)
            cases.append(case)
    return {"purpose": "independent_Train_API_recovery_acceptance", "cases": cases,
            "source_split": "train",
            "excluded_known_case_ids": list(probe.FAILURE_IDS + probe.NORMAL_IDS),
            "observation_profile": PROFILE, "action_protocol": PROTOCOL,
            "budget": {"cases": len(cases), "requests_per_trajectory": 4,
                       "greedy_per_case": 1, "sampled_per_case": 4,
                       "trajectories_per_model": len(cases) * 5},
            "teacher_labels_force_actions": False, "formal_Valid_or_OOS_used": False,
            "financial_evaluation": False}


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


async def build(output: Path, *, train_parquet: Path, original_train_parquet: Path,
                replay_jsonl: Path, recovery_directory: Path, model: Path) -> dict[str, Any]:
    import pandas as pd
    from transformers import AutoTokenizer

    artifact_names = ("train.jsonl", "dataset_manifest.json", "dataset_source_receipt.json",
                      "acceptance_bank.json")
    if any((output / name).exists() for name in artifact_names):
        raise FileExistsError("Fresh SFT data artifacts are required; no overwrite")
    replay_manifest_path = replay_jsonl.parent / "manifest.json"
    replay_manifest = probe.read(replay_manifest_path)
    if probe.sha(replay_jsonl) != replay_manifest["train_jsonl_sha256"]:
        raise ValueError("Original v3 SFT data identity changed")
    if probe.sha(original_train_parquet) != replay_manifest["source_train_sha256"]:
        raise ValueError("Original Train source identity changed")
    recovery_manifest_path = recovery_directory / "manifest.json"
    failure_dates = reserved_dates(probe.read(recovery_manifest_path))
    raw_rows = pd.read_parquet(original_train_parquet).to_dict("records")
    by_id = {row["extra_info"]["episode_id"]: row for row in raw_rows}
    recovery_rows = pd.read_parquet(train_parquet).to_dict("records")
    if tuple(row["extra_info"]["episode_id"] for row in recovery_rows) != TRAIN_IDS:
        raise ValueError("Recovery source must be the fixed eight original R5 Train windows")
    acceptance_rows = [by_id[episode] for episode in ACCEPTANCE_IDS]
    acceptance_dates = set().union(*(row_dates(row) for row in acceptance_rows))
    for row in acceptance_rows:
        guard_source(row, failure_dates)
        if not "2023-06-01" <= min(row_dates(row)) <= max(row_dates(row)) <= "2023-07-31":
            raise ValueError("Independent acceptance source must stay in Jun/Jul2023 Train")
    for row in recovery_rows:
        guard_source(row, failure_dates | acceptance_dates)
    recovery_dates = set().union(*(row_dates(row) for row in recovery_rows))
    replay_lines, replay_sources = select_replay(
        replay_jsonl.read_text().splitlines(keepends=True), by_id, failure_dates | acceptance_dates,
    )
    input_paths = (train_parquet, original_train_parquet, replay_jsonl, replay_manifest_path,
                   recovery_manifest_path, model / "merge_receipt.json")
    inputs = {str(path.resolve()): probe.sha(path) for path in input_paths}
    source_paths = [*ROOT.glob("src/**/*.py"),
                    *ROOT.glob("third_party/agent-r1/agent_r1/**/*.py"), Path(__file__).resolve()]
    sources = {str(path.resolve()): probe.sha(path) for path in source_paths}
    tokenizer_paths = [path for path in model.iterdir()
                       if path.is_file() and ("token" in path.name or "vocab" in path.name
                                             or "merges" in path.name
                                             or path.name == "config.json")]
    tokenizer_sources = {str(path.resolve()): probe.sha(path) for path in tokenizer_paths}
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    records = await recovery_records(recovery_rows, tokenizer)
    bank = await acceptance_bank(acceptance_rows, tokenizer)
    if len(records) != 192 or len(replay_lines) != 192 or len(bank["cases"]) != 24:
        raise ValueError("Fixed recovery/replay/acceptance populations differ")
    all_records = records + [json.loads(line) for line in replay_lines]
    if (len({record["id"] for record in all_records}) != 384
            or len({digest_text(record["prompt"]) for record in all_records}) != 384):
        raise ValueError("Duplicate prompt or record identity in the fixed dataset")
    for path, digest in {**inputs, **sources, **tokenizer_sources}.items():
        if probe.sha(Path(path)) != digest:
            raise ValueError(f"Frozen data-generation input changed: {path}")
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "train.jsonl"
    with data_path.open("x") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        for line in replay_lines:
            handle.write(line if line.endswith("\n") else line + "\n")
    encoded = read_examples(data_path, tokenizer, max_length=8192)
    bank.update(training_dataset_sha256=probe.sha(data_path),
                train_decision_dates=sorted({record["metadata"]["decision_date"][:10]
                                             for record in all_records}),
                recovery_training_source_dates=sorted(recovery_dates),
                acceptance_source_dates=sorted(acceptance_dates),
                known_failure_source_dates=sorted(failure_dates),
                training_acceptance_known_failure_dates_disjoint=True,
                input_sha256=inputs)
    write_json(output / "acceptance_bank.json", bank)
    manifest = {
        "status": "COMPLETE", "profile": "shared_order_recovery_sft_v1",
        "rows": 384, "recovery_rows": 192, "unchanged_replay_rows": 192,
        "acceptance_cases": 24, "acceptance_is_training": False,
        "training_observation_profile": PROFILE, "action_protocol": PROTOCOL,
        "input_sha256": inputs, "runtime_sha256": sources, "tokenizer_sha256": tokenizer_sources,
        "train_jsonl_sha256": probe.sha(data_path),
        "acceptance_bank_sha256": probe.sha(output / "acceptance_bank.json"),
        "recovery_source_ids": list(TRAIN_IDS), "acceptance_source_ids": list(ACCEPTANCE_IDS),
        "known_failure_ids_reserved": list(probe.FAILURE_IDS),
        "known_failure_dates_reserved": sorted(failure_dates),
        "recovery_source_dates": sorted(recovery_dates),
        "acceptance_source_dates": sorted(acceptance_dates),
        "training_acceptance_known_failure_dates_disjoint": True,
        "replay_selection": {"method": "fixed kind/action quotas, SHA256 order, unique prompts",
                             "quotas": {"/".join(key): count
                                        for key, count in REPLAY_QUOTAS.items()},
                             "sources": replay_sources, "original_lines_byte_identical": True},
        "recovery_label_distribution": dict(Counter(
            json.loads(record["completion"][11:-12])["name"] for record in records)),
        "all_label_distribution": dict(Counter(
            json.loads(record["completion"][11:-12])["name"] for record in all_records)),
        "recovery_symbols": sorted({record["metadata"]["primary_symbol"] for record in records}),
        "recovery_history_distribution": dict(Counter(
            record["metadata"]["state_spec"]["history"] for record in records)),
        "max_total_tokens_including_eos": max(len(row["input_ids"]) for row in encoded),
        "max_prompt_tokens": max(row["prompt_tokens"] for row in encoded),
        "target_tokens_per_epoch": sum(row["target_tokens"] for row in encoded),
        "completion_only_compatible": True, "truncated_sequences": 0,
        "suggested_fixed_training_budget": {"epochs": 3, "micro_batch": 1,
                                           "gradient_accumulation": 16, "optimizer_updates": 72,
                                           "learning_rate": 5e-5},
        "gpu_used": False, "future_prices_used_to_choose_recovery_labels": False,
        "replay_retains_existing_offline_financial_teacher_labels": True,
        "limitations": [
            "Recovery assumes the last requested endpoint is still intended; "
            "this teaches API execution.",
            "Scripted legal prefixes are not new on-policy model failures "
            "or financial-alpha labels.",
            "Rehearsal retains the old financial teacher, including its 20% buy/full-clear prior.",
            "The six known failures are debugging regression, not fresh independent validation.",
            "Acceptance is held out from this SFT; selected Train states "
            "are not independent trials.",
        ],
    }
    write_json(output / "dataset_source_receipt.json", {
        "input_sha256": inputs, "runtime_sha256": sources,
        "tokenizer_sha256": tokenizer_sources,
        "train_jsonl_sha256": manifest["train_jsonl_sha256"],
        "acceptance_bank_sha256": manifest["acceptance_bank_sha256"],
    })
    write_json(output / "dataset_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path,
                        default=ROOT / ("runs/m2-parameter-grid-20261008-r5/"
                                        "seed42/prepared/train.parquet"))
    parser.add_argument("--original-train-parquet", type=Path,
                        default=ROOT / "artifacts/m2_screening/train.parquet")
    parser.add_argument("--replay-jsonl", type=Path,
                        default=ROOT / "runs/m2-coldstart-v3-20260914/dataset/train.jsonl")
    parser.add_argument("--recovery-directory", type=Path,
                        default=ROOT / "runs/m2-order-recovery-20261011-v1")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "runs/m2-coldstart-v3-20260914/merged_model")
    args = parser.parse_args()
    result = asyncio.run(build(**vars(args)))
    print(json.dumps({key: result[key] for key in (
        "status", "rows", "recovery_rows", "unchanged_replay_rows", "acceptance_cases",
        "recovery_label_distribution", "all_label_distribution", "max_total_tokens_including_eos",
        "target_tokens_per_epoch", "train_jsonl_sha256", "acceptance_bank_sha256",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()

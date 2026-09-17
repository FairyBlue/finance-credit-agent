"""Prepare bounded autonomous Train collection; no generation or teacher labelling here."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.agents.coldstart_data import TRAIN_IDS, parse_state, render_prompt, tool_call
from src.agents.coldstart_eval import grid, validate_contexts
from src.env.trading import TradingEnv

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "runs/m2-coldstart-v2-20260913"
RUN = ROOT / "runs/m2-coldstart-v3-20260914"
MODEL = V2 / "merged_model"
DEFAULT_OUTPUT = RUN / "collection/prepared"
TRAIN_INDICES = (0, 3, 5, 8, 11, 14, 16, 19)
COLLECTION_IDS = tuple(TRAIN_IDS[index] for index in TRAIN_INDICES)
LAST_TRAIN_DATE = date(2023, 5, 22)
PROTOCOL = "single_tool_portfolio_v1"
PURPOSE = "TRAIN_COLLECTION"
IDENTITY_HELPER = ROOT / "scripts/diagnostics/m2_coldstart_v2_run.py"


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def calendar_day(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("collection timestamps must retain the source timezone")
    return parsed.date()


def context_days(contexts):
    return {
        calendar_day(value)
        for horizon in ("short", "full")
        for context in contexts[horizon]
        for field in ("episode_dates", "episode_open_times")
        for value in context["env_kwargs"][field]
    }


def build_contexts(rows, reserved_contexts):
    """Use fixed source indices and original prompts; never inspect realised outcomes."""
    selected = {}
    for row in rows:
        episode = row["extra_info"]["episode_id"]
        if episode in COLLECTION_IDS:
            if episode in selected:
                raise ValueError(f"duplicate selected Train source episode: {episode}")
            selected[episode] = row
    if set(selected) != set(COLLECTION_IDS):
        raise ValueError("missing one or more fixed Train collection episodes")
    reserved_days = set().union(*(context_days(contexts) for contexts in reserved_contexts))
    full = []
    for episode in COLLECTION_IDS:
        row = selected[episode]
        if row["extra_info"].get("source_split") != "train":
            raise ValueError(f"selected episode is outside original Train: {episode}")
        kwargs = json.loads(row["env_kwargs"])
        for field in ("episode_dates", "episode_open_times", "benchmark_closes"):
            if len(kwargs[field]) != 21:
                raise ValueError(f"collection requires original 21-session episodes: {episode}")
        close_days = [calendar_day(value) for value in kwargs["episode_dates"]]
        open_days = [calendar_day(value) for value in kwargs["episode_open_times"]]
        if close_days != sorted(set(close_days)) or close_days != open_days:
            raise ValueError("source collection calendar is unordered or open/close dates differ")
        if any(day > LAST_TRAIN_DATE for day in close_days):
            raise ValueError("collection dates extend beyond 2023-05-22")
        if set(close_days) & reserved_days:
            raise ValueError("collection dates overlap reserved v2 fresh/regression dates")
        if kwargs.get("screening") is not True:
            raise ValueError("collection must retain the existing screening environment contract")
        prompt = copy.deepcopy(list(row["prompt"]))
        if not prompt or any(
            not isinstance(message, dict) or not isinstance(message.get("content"), str)
            for message in prompt
        ):
            raise ValueError("source prompt must be a nonempty original textual conversation")
        source = {
            "episode_id": episode,
            "source_split": "train",
            "env_kwargs": row["env_kwargs"],
            "raw_prompt": prompt,
        }
        kwargs.update(observation_profile="history_features_v1", reward_mode="finance_trajectory")
        full.append(
            {
                "id": episode,
                "source_episode": episode,
                "source_split": "train",
                "source_row_inputs_sha256": hashlib.sha256(
                    json.dumps(source, sort_keys=True, allow_nan=False).encode()
                ).hexdigest(),
                "purpose": PURPOSE,
                "env_kwargs": kwargs,
                "raw_prompt": prompt,
                "state_kind": "cash",
                "prefix": [],
                "prefix_expected_invalid": [],
                "prefix_expected_errors": [],
                "expected_start_session_index": 0,
                "action_protocol": PROTOCOL,
            }
        )
    return {
        "purpose": PURPOSE,
        "short": [],
        "full": full,
        "teacher_labels_included": False,
        "is_formal_evaluation": False,
        "is_acceptance_evaluation": False,
        "action_protocol": PROTOCOL,
        "full_session_count": 21,
    }


def verify_model_identity():
    """Read the already-tested streaming helper without invoking its controller."""
    spec = importlib.util.spec_from_file_location("_coldstart_v2_identity", IDENTITY_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("existing cold-start model identity helper is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model_identity(MODEL)


async def cpu_preflight(contexts, tokenizer):
    from agent_r1.env.base import Action

    result = []
    for context in contexts["full"]:
        env = TradingEnv(**context["env_kwargs"])
        obs = env.reset(raw_prompt=context["raw_prompt"])
        initial = parse_state(obs)
        if env.current_session_index != 0 or initial["positions"] or env.ledger.fills:
            raise ValueError("collection must start from genuine cash with no prefix")
        counts = []
        for index in range(21):
            prompt = render_prompt(tokenizer, env, obs, action_protocol=PROTOCOL)
            count = len(tokenizer.encode(prompt, add_special_tokens=False))
            if not 0 < count <= 8192:
                raise ValueError("collection prompt exceeds 8192 tokens; no truncation")
            counts.append(count)
            if index < 20:
                obs, _, done, info = await env.step(Action(text=tool_call("hold")))
                if info.get("invalid_action") or done != (index == 19):
                    raise ValueError(
                        "CPU environment preflight cannot complete a legal hold replay"
                    )
        if not env.episode_complete or env.ledger.fills:
            raise ValueError(
                "CPU cash/hold preflight altered the intended collection initialisation"
            )
        result.append(
            {
                "episode": context["id"],
                "rendered_sessions": len(counts),
                "max_prompt_tokens": max(counts),
                "hold_transitions": 20,
                "episode_complete": True,
            }
        )
    return {
        "status": "PASS",
        "purpose": "CPU_ENVIRONMENT_AND_TOKENIZER_PREFLIGHT",
        "model_generated": False,
        "used_as_training_examples": False,
        "policy_dependent_prompt_lengths_not_yet_verified": True,
        "episodes": result,
    }


def collection_command(output):
    return [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "src.agents.coldstart_eval",
        "--model",
        str(MODEL),
        "--contexts",
        str(output / "collection_contexts.json"),
        "--output",
        str(output.parent / "rollout"),
        "--action-protocol",
        PROTOCOL,
        "--expected-short",
        "0",
        "--expected-full",
        "8",
    ]


async def prepare(output):
    import pandas as pd
    from transformers import AutoTokenizer

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / "state.json", {"status": "PREPARING", "purpose": PURPOSE, "at_SGT": now()})
    try:
        artifact = ROOT / "artifacts/m2_screening"
        source = read(artifact / "manifest.json")
        source_hash = sha(artifact / "train.parquet")
        if source_hash != source["datasets"]["train"]["sha256"]:
            raise ValueError("original Train parquet source checksum mismatch")
        v2_manifest = read(V2 / "dataset/manifest.json")
        reserved = []
        reserved_identity = {}
        for name, key in (
            ("eval_contexts.json", "eval_contexts_sha256"),
            ("regression_eval_contexts.json", "regression_eval_contexts_sha256"),
        ):
            path = V2 / "dataset" / name
            digest = sha(path)
            if digest != v2_manifest[key]:
                raise ValueError(f"reserved v2 context identity mismatch: {name}")
            reserved.append(read(path))
            reserved_identity[str(path)] = digest
        contexts = build_contexts(
            pd.read_parquet(artifact / "train.parquet").to_dict("records"), reserved
        )
        validate_contexts(contexts, 0, 8)
        rows = grid(contexts)
        if len(rows) != 40 or any(row["request_limit"] != 128 for row in rows):
            raise ValueError(
                "existing collector grid differs from the fixed 40/128 request contract"
            )
        identity = verify_model_identity()
        save(output / "model_identity.json", identity)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        preflight = await cpu_preflight(contexts, tokenizer)
        save(output / "preflight.json", preflight)
        save(output / "collection_contexts.json", contexts)
        sources = {
            str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT / "src").rglob("*.py"))
        }
        sources[str(IDENTITY_HELPER.relative_to(ROOT))] = sha(IDENTITY_HELPER)
        save(output / "source_identity.json", sources)
        command = collection_command(output)
        save(
            output / "collection_command.json",
            {
                "purpose": PURPOSE,
                "argv": command,
                "cwd": str(ROOT),
                "environment": {"PYTHONPATH": str(ROOT / "third_party/agent-r1") + ":" + str(ROOT)},
                "launched": False,
            },
        )
        days = context_days(contexts)
        manifest = {
            "status": "COMPLETE",
            "prepared_at_SGT": now(),
            "purpose": PURPOSE,
            "actual_collection_started": False,
            "on_policy_states_collected": False,
            "source_train_parquet": str(artifact / "train.parquet"),
            "source_train_sha256": source_hash,
            "source_manifest_sha256": sha(artifact / "manifest.json"),
            "train_ids_pool": TRAIN_IDS,
            "fixed_indices": TRAIN_INDICES,
            "collection_ids": COLLECTION_IDS,
            "source_split": "train",
            "date_min": min(days).isoformat(),
            "date_max": max(days).isoformat(),
            "date_ceiling": LAST_TRAIN_DATE.isoformat(),
            "date_isolation_verified": True,
            "reserved_contexts_sha256": reserved_identity,
            "collection_contexts_sha256": sha(output / "collection_contexts.json"),
            "source_identity_sha256": sha(output / "source_identity.json"),
            "model": str(MODEL),
            "model_identity_sha256": sha(output / "model_identity.json"),
            "model_merge_receipt_sha256": identity["merge_receipt_sha256"],
            "tokenizer_model_path": str(MODEL),
            "action_protocol": PROTOCOL,
            "generation": {
                "thinking": False,
                "greedy_per_episode": 1,
                "sampled_per_episode": 4,
                "seed": 42,
                "prompt_cap": 8192,
                "response_cap": 4096,
                "requests_per_trajectory_cap": 128,
            },
            "full_sessions_per_episode": 21,
            "trajectories_planned": 40,
            "teacher_labels_included": False,
            "artificial_prefixes": False,
            "invalid_requests_preserved": True,
            "sampling_budget_adaptation": False,
            "is_acceptance_evaluation": False,
            "preflight_sha256": sha(output / "preflight.json"),
            "max_cpu_hold_prompt_tokens": max(
                row["max_prompt_tokens"] for row in preflight["episodes"]
            ),
            "limitation": (
                "CPU hold replay is not model collection; autonomous lengths checked live."
            ),
        }
        save(output / "manifest.json", manifest)
        save(output / "state.json", manifest)
        print(json.dumps(manifest, indent=2), flush=True)
    except BaseException as error:
        save(
            output / "state.json",
            {
                "status": "FAILED",
                "purpose": PURPOSE,
                "at_SGT": now(),
                "error": repr(error),
                "actual_collection_started": False,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare",))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    asyncio.run(prepare(args.output))


if __name__ == "__main__":
    main()

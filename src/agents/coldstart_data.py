"""Bounded Train-only hindsight demonstrations and separately tagged API teaching.

Future prices select offline targets only. Model-visible states come from fixed,
legal prefixes, never from a previous hindsight winner. This is not RL evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from agent_r1.env.base import Action

from src.agents.screening_prompt import apply_screening_action_protocol
from src.env.trading import TradingEnv

ROOT = Path(__file__).resolve().parents[2]
TRAIN_IDS = [f"formal-train-{i:04d}" for i in range(1, 78, 4)]
EVAL_IDS = [f"formal-train-{i:04d}" for i in (113, 117, 121, 125)]
TEACHER_SYMBOLS = ("GS", "MSFT", "HD")
MAX_ROWS = 512
MAX_API_ROWS = 96
MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots"
    / "1cfa9a7208912126459214e8b04321603b3df60c"
)


def tool_call(name, arguments=None):
    return (
        "<tool_call>"
        + json.dumps({"name": name, "arguments": arguments or {}}, separators=(",", ":"))
        + "</tool_call>"
    )


def prefix_for(state_kind):
    """Fixed state curriculum; no market values or future labels are accepted."""
    if state_kind == "cash":
        return [tool_call("hold")]
    if state_kind not in TEACHER_SYMBOLS:
        raise ValueError(f"unknown state kind: {state_kind}")
    return [tool_call("buy", {"symbol": state_kind, "target_weight": 0.2})]


def slice_kwargs(kwargs, start, stop):
    result = copy.deepcopy(kwargs)
    for field in ("episode_dates", "episode_open_times", "benchmark_closes"):
        result[field] = result[field][start:stop]
    if len(result["episode_dates"]) != 5:
        raise ValueError("coldstart contexts require one prefix plus three outcome sessions")
    result.update(observation_profile="history_features_v1", reward_mode="finance_trajectory")
    return result


def parse_state(obs):
    text = obs.messages[-1]["content"]
    if text.startswith("<tool_response>"):
        text = text.removeprefix("<tool_response>").removesuffix("</tool_response>").strip()
    return json.loads(text)


async def replay_context(context):
    env = TradingEnv(**context["env_kwargs"])
    obs = env.reset(raw_prompt=context["raw_prompt"])
    for text in context["prefix"]:
        obs, _, done, info = await env.step(Action(text=text))
        if done or info.get("invalid_action"):
            raise ValueError(f"invalid fixed prefix: {info}")
    if env.current_session_index != 1:
        raise ValueError("the coldstart decision must follow exactly one fixed prefix")
    return env, obs


def candidate_calls(state, symbols=TEACHER_SYMBOLS):
    """A disclosed action grid, filtered using current visible positions only."""
    calls = [tool_call("hold")]
    for symbol in symbols:
        weight = state["positions"].get(symbol, {}).get("weight", 0.0)
        for target in (0.1, 0.2):
            if target > weight + 1e-7:
                calls.append(tool_call("buy", {"symbol": symbol, "target_weight": target}))
        for target in (0.0, 0.1):
            if target < weight - 1e-7:
                calls.append(tool_call("sell", {"symbol": symbol, "target_weight": target}))
    return calls


async def score_candidate(context, completion):
    env, _ = await replay_context(context)
    _, _, _, info = await env.step(Action(text=completion))
    if info.get("invalid_action"):
        raise ValueError(f"candidate unexpectedly illegal: {completion}: {info}")
    first_info = copy.deepcopy(info)
    while not env.episode_complete:
        _, _, _, info = await env.step(Action(text=tool_call("hold")))
        if info.get("invalid_action"):
            raise ValueError("continuation hold failed")
    return {
        "completion": completion,
        "terminal_equity": info["equity"],
        "net_return": info["equity"] / context["env_kwargs"]["initial_cash"] - 1.0,
        "candidate_fills": [
            {
                key: value.isoformat() if isinstance(value, datetime) else value
                for key, value in fill.items()
            }
            for fill in first_info["fills"]
        ],
        "candidate_skipped_orders": first_info.get("skipped_orders", []),
        "candidate_cost": sum(
            fill["commission"] + fill["slippage_cost"] for fill in first_info["fills"]
        ),
        "zero_fill": len(first_info["fills"]) == 0,
    }


def select_label(scores):
    """Ordered ties retain hold; economically equal candidates add no samples."""
    if not scores or scores[0]["completion"] != tool_call("hold"):
        raise ValueError("hold must be the first candidate")
    winner = scores[0]
    for score in scores[1:]:
        if score["terminal_equity"] > winner["terminal_equity"] and not math.isclose(
            score["terminal_equity"], winner["terminal_equity"], rel_tol=1e-12, abs_tol=1e-6
        ):
            winner = score
    return winner


def render_prompt(tokenizer, env, obs, *, intent=None, action_protocol="single_tool_v1"):
    messages = copy.deepcopy(obs.messages)
    if intent is not None:
        messages.append({"role": "user", "content": "API execution exercise. " + intent})
    return tokenizer.apply_chat_template(
        apply_screening_action_protocol(messages, action_protocol),
        tools=env.tool_schemas,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )


def contexts_from_rows(rows):
    """Deduplicate overlapping source windows by date before producing states."""
    by_date = {}
    for row in rows:
        kw = json.loads(row["env_kwargs"])
        assert tuple(kw["symbols"][:3]) == TEACHER_SYMBOLS
        assert row["extra_info"]["source_split"] == "train"
        for decision_index in range(1, len(kw["episode_dates"]) - 3):
            day = kw["episode_dates"][decision_index]
            if day not in by_date:
                by_date[day] = {
                    "source_episode": row["extra_info"]["episode_id"],
                    "env_kwargs": slice_kwargs(kw, decision_index - 1, decision_index + 4),
                    "raw_prompt": list(row["prompt"]),
                    "decision_date": day,
                }
    result = []
    for day in sorted(by_date):
        for state_kind in ("cash", *TEACHER_SYMBOLS):
            result.append(
                {
                    **copy.deepcopy(by_date[day]),
                    "id": f"{day[:10]}-{state_kind}",
                    "state_kind": state_kind,
                    "prefix": prefix_for(state_kind),
                }
            )
    return result


def eval_contexts_from_rows(rows):
    short, full = [], []
    for row in rows:
        kw = json.loads(row["env_kwargs"])
        raw_prompt = list(row["prompt"])
        episode_id = row["extra_info"]["episode_id"]
        for state_kind in ("cash", *TEACHER_SYMBOLS):
            short.append(
                {
                    "id": episode_id + "-" + state_kind,
                    "source_episode": episode_id,
                    "env_kwargs": slice_kwargs(kw, 3, 8),
                    "raw_prompt": raw_prompt,
                    "decision_date": kw["episode_dates"][4],
                    "state_kind": state_kind,
                    "prefix": prefix_for(state_kind),
                }
            )
        full_kw = copy.deepcopy(kw)
        full_kw.update(observation_profile="history_features_v1", reward_mode="finance_trajectory")
        full.append({"id": episode_id, "env_kwargs": full_kw, "raw_prompt": raw_prompt})
    return {"short": short, "full": full, "teacher_labels_included": False}


def source_dates(contexts):
    return {day for context in contexts for day in context["env_kwargs"]["episode_dates"]}


def api_examples(state):
    examples = []
    calls = candidate_calls(state)
    buy = next((call for call in calls if '"name":"buy"' in call), None)
    sell = next((call for call in calls if '"name":"sell"' in call), None)
    for call in (buy, sell):
        if call is not None:
            parsed = json.loads(call.removeprefix("<tool_call>").removesuffix("</tool_call>"))
            args = parsed["arguments"]
            verb = "Increase" if parsed["name"] == "buy" else "Reduce"
            examples.append(
                (
                    f"{verb} {args['symbol']} to target portfolio weight {args['target_weight']}. "
                    "Execute this given intent with one tool call.",
                    call,
                )
            )
    examples.extend(
        [
            ("Keep the existing portfolio unchanged and advance one session.", tool_call("hold")),
            (
                "Explicitly abstain from a trading decision and advance one session.",
                tool_call("abstain"),
            ),
            (
                "Compute GS close-price change using lookback 2.",
                tool_call("compute", {"symbol": "GS", "metric": "price_change", "lookback": 2}),
            ),
            (
                "Retrieve up to one currently available GS market record for query price.",
                tool_call("retrieve", {"symbol": "GS", "query": "price", "limit": 1}),
            ),
        ]
    )
    return examples


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def build_dataset(output):
    import pandas as pd
    from transformers import AutoTokenizer

    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "artifacts/m2_screening"
    source_manifest = json.loads((source / "manifest.json").read_text())
    assert sha(source / "train.parquet") == source_manifest["datasets"]["train"]["sha256"]
    assert sha(source / "reward_calibration.json") == source_manifest["calibration"]["sha256"]
    records = pd.read_parquet(source / "train.parquet").to_dict("records")
    by_id = {row["extra_info"]["episode_id"]: row for row in records}
    contexts = contexts_from_rows([by_id[episode] for episode in TRAIN_IDS])
    eval_contexts = eval_contexts_from_rows([by_id[episode] for episode in EVAL_IDS])
    assert not source_dates(contexts) & source_dates(eval_contexts["full"])
    contexts = contexts[: MAX_ROWS - MAX_API_ROWS]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rows = []
    api_count = 0
    for index, context in enumerate(contexts):
        env, obs = await replay_context(context)
        state = parse_state(obs)
        prompt = render_prompt(tokenizer, env, obs)
        scores = [await score_candidate(context, call) for call in candidate_calls(state)]
        label = select_label(scores)
        rows.append(
            {
                "id": "decision-" + context["id"],
                "kind": "decision",
                "prompt": prompt,
                "completion": label["completion"],
                "metadata": {
                    "source_episode": context["source_episode"],
                    "decision_date": context["decision_date"],
                    "state_kind": context["state_kind"],
                    "prefix": context["prefix"],
                    "teacher_label": json.loads(
                        label["completion"].removeprefix("<tool_call>").removesuffix("</tool_call>")
                    ),
                    "candidate_scores": scores,
                    "teacher_target": "net_terminal_wealth_after_three_sessions_and_actual_costs",
                    "horizon_dates": context["env_kwargs"]["episode_dates"][2:],
                },
            }
        )
        for intent, completion in api_examples(state):
            if api_count >= MAX_API_ROWS:
                break
            # Validate every API target on the same fixed state, independently.
            api_env, _ = await replay_context(context)
            _, _, _, info = await api_env.step(Action(text=completion))
            if info.get("invalid_action"):
                raise ValueError(f"API target invalid: {completion}: {info}")
            rows.append(
                {
                    "id": f"api-{api_count:04d}-{context['id']}",
                    "kind": "api",
                    "prompt": render_prompt(tokenizer, env, obs, intent=intent),
                    "completion": completion,
                    "metadata": {
                        "source_episode": context["source_episode"],
                        "decision_date": context["decision_date"],
                        "state_kind": context["state_kind"],
                        "prefix": context["prefix"],
                        "given_intent": intent,
                        "teacher_label": None,
                    },
                }
            )
            api_count += 1
        if index % 10 == 0:
            print(
                json.dumps(
                    {
                        "contexts_completed": index + 1,
                        "total_contexts": len(contexts),
                        "rows": len(rows),
                    }
                ),
                flush=True,
            )
    assert len(rows) <= MAX_ROWS
    prompt_lengths = [
        len(tokenizer.encode(row["prompt"], add_special_tokens=False)) for row in rows
    ]
    assert max(prompt_lengths) <= 8192
    with (output / "train.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "eval_contexts.json").write_text(json.dumps(eval_contexts, indent=2))
    counts = {}
    for kind in ("decision", "api"):
        counts[kind] = dict(
            Counter(
                json.loads(
                    row["completion"].removeprefix("<tool_call>").removesuffix("</tool_call>")
                )["name"]
                for row in rows
                if row["kind"] == kind
            )
        )
    manifest = {
        "status": "COMPLETE",
        "created_at_SGT": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "source_train_sha256": sha(source / "train.parquet"),
        "source_manifest_sha256": sha(source / "manifest.json"),
        "source_calibration_sha256": sha(source / "reward_calibration.json"),
        "generator_sha256": sha(Path(__file__)),
        "tokenizer_snapshot": str(MODEL),
        "train_jsonl_sha256": sha(output / "train.jsonl"),
        "eval_contexts_sha256": sha(output / "eval_contexts.json"),
        "source_train_ids": TRAIN_IDS,
        "reserved_train_eval_ids": EVAL_IDS,
        "source_dates_min": min(source_dates(contexts)),
        "source_dates_max": max(source_dates(contexts)),
        "eval_dates_min": min(source_dates(eval_contexts["full"])),
        "eval_dates_max": max(source_dates(eval_contexts["full"])),
        "train_eval_dates_disjoint": True,
        "row_count": len(rows),
        "decision_rows": len(contexts),
        "api_rows": api_count,
        "unique_decision_dates": len({c["decision_date"] for c in contexts}),
        "label_distribution": counts,
        "max_prompt_tokens": max(prompt_lengths),
        "teacher_priors": {
            "symbols": list(TEACHER_SYMBOLS),
            "buy_targets": [0.1, 0.2],
            "sell_targets": [0, 0.1],
            "prefix": "one fixed hold or 20% buy, independent of future; no oracle chaining",
            "lookahead_sessions": 3,
            "objective": "terminal net wealth; distinct from RL scorer",
            "ties": "hold first, then declared action order; one target per state",
            "risk": "hindsight targets may favor maximum grid exposure; no predictive claim",
            "api": "explicit given intent, separately tagged; not autonomous trading examples",
        },
        "selection": (
            "source episode order, deduplicate date, chronological date/state order; "
            "no score or class balancing"
        ),
        "is_on_policy": False,
        "future_information_in_prompt": False,
        "observation_profile": "history_features_v1",
        "non_thinking": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs/m2-coldstart-20260913/dataset")
    args = parser.parse_args()
    asyncio.run(build_dataset(args.output))


if __name__ == "__main__":
    main()

"""Real-state portfolio and recovery demonstrations for common cold-start v2.

Historical future outcomes select offline labels only. Every visible state is
reached by a declared, future-independent prefix through the unchanged ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from collections import Counter
from datetime import datetime
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
    source_dates,
    tool_call,
)
from src.env.trading import TradingEnv

PROTOCOL = "single_tool_portfolio_v1"
OLD_DATASET = ROOT / "runs/m2-coldstart-20260913/dataset"
MODEL = ROOT / "runs/m2-coldstart-20260913/merged_model"
SCENARIOS = (
    "cash",
    "single_low",
    "single_high",
    "pair_low",
    "pair_mixed",
    "triple_below",
    "triple_above",
    "triple_mixed",
)
RECOVERY_SCENARIOS = ("cash", "single_high", "triple_above", "triple_mixed")
MAX_ROWS = 1024
REPLAY_COUNTS = {"decision": 76, "api": 20}
NEW_EVAL_SPANS = (("2023-05-23", "2023-06-01"), ("2023-06-02", "2023-06-12"))


def fixed_prefix(scenario):
    def buy(symbol, weight):
        return tool_call("buy", {"symbol": symbol, "target_weight": weight})

    hold = tool_call("hold")
    prefixes = {
        "cash": [hold] * 3,
        "single_low": [buy("GS", 0.1), hold, hold],
        "single_high": [buy("MSFT", 0.3), hold, hold],
        "pair_low": [buy("GS", 0.1), buy("MSFT", 0.1), hold],
        "pair_mixed": [buy("GS", 0.25), buy("MSFT", 0.15), hold],
        "triple_below": [buy(symbol, 0.19) for symbol in ("GS", "MSFT", "HD")],
        "triple_above": [buy(symbol, 0.21) for symbol in ("GS", "MSFT", "HD")],
        "triple_mixed": [buy("GS", 0.1), buy("MSFT", 0.2), buy("HD", 0.3)],
    }
    return prefixes[scenario]


def sliced_kwargs(kwargs, start, stop):
    result = copy.deepcopy(kwargs)
    for field in ("episode_dates", "episode_open_times", "benchmark_closes"):
        result[field] = result[field][start:stop]
    result.update(observation_profile="history_features_v1", reward_mode="finance_trajectory")
    return result


def make_context(row, decision_index, scenario):
    kwargs = json.loads(row["env_kwargs"])
    if decision_index < 3 or decision_index + 3 >= len(kwargs["episode_dates"]):
        raise ValueError("a context needs three prefix sessions and three outcome sessions")
    day = kwargs["episode_dates"][decision_index]
    return {
        "id": f"{day[:10]}-{scenario}",
        "source_episode": row["extra_info"]["episode_id"],
        "env_kwargs": sliced_kwargs(kwargs, decision_index - 3, decision_index + 4),
        "raw_prompt": list(row["prompt"]),
        "decision_date": day,
        "state_kind": scenario,
        "prefix": fixed_prefix(scenario),
        "prefix_expected_invalid": [False] * 3,
        "prefix_expected_errors": [None] * 3,
        "expected_start_session_index": 3,
        "action_protocol": PROTOCOL,
    }


def training_contexts(rows):
    by_date = {}
    for row in rows:
        if row["extra_info"]["source_split"] != "train":
            raise ValueError("demonstrations must use Train")
        kwargs = json.loads(row["env_kwargs"])
        for index in range(3, len(kwargs["episode_dates"]) - 3):
            by_date.setdefault(kwargs["episode_dates"][index], (row, index))
    selected_dates = sorted(by_date)[::4]
    return [
        make_context(*by_date[day], scenario) for day in selected_dates for scenario in SCENARIOS
    ]


async def replay_context(context):
    env = TradingEnv(**context["env_kwargs"])
    obs = env.reset(raw_prompt=context["raw_prompt"])
    flags = context.get("prefix_expected_invalid", [False] * len(context["prefix"]))
    errors = context.get("prefix_expected_errors", [None] * len(context["prefix"]))
    if len(flags) != len(context["prefix"]) or len(errors) != len(flags):
        raise ValueError("prefix expectations must cover every prefix request")
    infos = []
    for call, expect_invalid, expected_error in zip(context["prefix"], flags, errors, strict=True):
        before = parse_state(obs)
        before_fills = len(env.ledger.fills)
        obs, _, done, info = await env.step(Action(text=call))
        if done or bool(info.get("invalid_action")) != expect_invalid:
            raise ValueError(f"prefix outcome differs from declared expectation: {info}")
        if expect_invalid:
            after = parse_state(obs)
            if expected_error not in info.get("error", ""):
                raise ValueError(f"wrong prefix error: {info}")
            for field in ("session_index", "cash", "positions", "portfolio_equity"):
                if before[field] != after[field]:
                    raise ValueError("rejected prefix modified financial state")
            if len(env.ledger.fills) != before_fills:
                raise ValueError("rejected prefix created a fill")
        infos.append(info)
    if env.current_session_index != context["expected_start_session_index"]:
        raise ValueError("prefix reached the wrong session")
    return env, obs, infos


async def recovery_context(clean):
    # Read-only adapters are shared; copying their immutable payload proxies is
    # unnecessary and unsupported. Only the prefix lists are extended here.
    result = {**clean, "prefix": list(clean["prefix"]),
              "prefix_expected_invalid": list(clean["prefix_expected_invalid"]),
              "prefix_expected_errors": list(clean["prefix_expected_errors"])}
    env, obs, _ = await replay_context(clean)
    state = parse_state(obs)
    scenario = clean["state_kind"]
    if scenario == "cash":
        name, symbol, target = "sell", "GS", 0.0
    elif scenario in ("single_high", "triple_above"):
        name, symbol = "buy", "MSFT" if scenario == "single_high" else "GS"
        target = max(0.0, round(state["positions"][symbol]["weight"] - 0.05, 8))
    elif scenario == "triple_mixed":
        name, symbol = "sell", "HD"
        target = min(1.0, round(state["positions"][symbol]["weight"] + 0.05, 8))
    else:
        raise ValueError("scenario has no declared recovery exercise")
    error = (
        "buy target_weight must be above the current weight"
        if name == "buy"
        else "sell target_weight must be below the current weight"
    )
    call = tool_call(name, {"symbol": symbol, "target_weight": target})
    _, _, done, info = await env.step(Action(text=call))
    if done or not info.get("invalid_action") or error not in info.get("error", ""):
        raise ValueError(f"intended invalid request was not rejected: {info}")
    result["id"] += "-post-error"
    result["prefix"].append(call)
    result["prefix_expected_invalid"].append(True)
    result["prefix_expected_errors"].append(error)
    result["recovery_kind"] = "zero_position_sell" if scenario == "cash" else "wrong_" + name
    return result


async def score_candidate(context, completion):
    env, _, _ = await replay_context(context)
    _, _, _, info = await env.step(Action(text=completion))
    if info.get("invalid_action"):
        raise ValueError(f"teacher candidate invalid: {completion}: {info}")
    first_info = copy.deepcopy(info)
    while not env.episode_complete:
        _, _, _, info = await env.step(Action(text=tool_call("hold")))
        if info.get("invalid_action"):
            raise ValueError("teacher continuation hold failed")
    return {
        "completion": completion,
        "terminal_equity": info["equity"],
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
        "zero_fill": not first_info["fills"],
    }


def api_targets(state, date_index, scenario_index):
    symbol = ("GS", "MSFT", "HD")[(date_index + scenario_index) % 3]
    target = (0.1, 0.2, 0.3)[(date_index + scenario_index // 3) % 3]
    current = state["positions"].get(symbol, {}).get("weight", 0.0)
    if abs(target - current) <= 1e-7:
        call = tool_call("hold")
    else:
        call = tool_call(
            "buy" if target > current else "sell", {"symbol": symbol, "target_weight": target}
        )
    return [
        ("Keep the existing portfolio unchanged and advance one session.", tool_call("hold")),
        (
            f"Set {symbol}'s total portfolio target weight to {target}. "
            "Execute this given intent with one tool call; keep the portfolio unchanged "
            "if it is already at this target.",
            call,
        ),
    ]


def render(tokenizer, env, obs, intent=None):
    return render_prompt(tokenizer, env, obs, intent=intent, action_protocol=PROTOCOL)


def row_metadata(context):
    return {
        key: copy.deepcopy(context[key])
        for key in (
            "source_episode",
            "decision_date",
            "state_kind",
            "prefix",
            "prefix_expected_invalid",
            "prefix_expected_errors",
            "expected_start_session_index",
        )
    } | {
        "action_protocol": PROTOCOL,
        "recovery_kind": context.get("recovery_kind"),
        "example_family": "recovery" if context.get("recovery_kind") else "clean",
    }


async def decision_row(context, tokenizer, scores=None):
    env, obs, _ = await replay_context(context)
    state = parse_state(obs)
    if scores is None:
        scores = [await score_candidate(context, call) for call in candidate_calls(state)]
    winner = select_label(scores)
    # Even when economics are shared with a clean state, validate the selected
    # target after the real error observation before creating a recovery row.
    _, _, _, info = await env.step(Action(text=winner["completion"]))
    if info.get("invalid_action"):
        raise ValueError("selected completion does not recover legally")
    metadata = row_metadata(context) | {
        "teacher_target": "terminal_net_wealth_after_three_sessions_with_actual_costs",
        "candidate_scores": copy.deepcopy(scores),
        "horizon_dates": context["env_kwargs"]["episode_dates"][4:],
        "visible_weights": {s: p["weight"] for s, p in state["positions"].items()},
        "teacher_label": json.loads(winner["completion"][11:-12]),
    }
    return {
        "id": "decision-v2-" + context["id"],
        "kind": "decision",
        "prompt": render(tokenizer, env, obs),
        "completion": winner["completion"],
        "metadata": metadata,
    }


def evenly_spaced(rows, count):
    if len(rows) <= count:
        return rows
    return [rows[i * (len(rows) - 1) // (count - 1)] for i in range(count)]


async def replay_rows(by_id, tokenizer):
    originals = [
        json.loads(line) for line in (OLD_DATASET / "train.jsonl").read_text().splitlines()
    ]
    selected = [
        row
        for kind, count in REPLAY_COUNTS.items()
        for row in evenly_spaced([r for r in originals if r["kind"] == kind], count)
    ]
    results, dates = [], set()
    for row in selected:
        metadata = copy.deepcopy(row["metadata"])
        source_row = by_id[metadata["source_episode"]]
        kwargs = json.loads(source_row["env_kwargs"])
        index = kwargs["episode_dates"].index(metadata["decision_date"])
        context = {
            "env_kwargs": sliced_kwargs(kwargs, index - 1, index + 4),
            "raw_prompt": list(source_row["prompt"]),
            "prefix": metadata["prefix"],
            "expected_start_session_index": 1,
        }
        env, obs, _ = await replay_context(context)
        prompt = render(tokenizer, env, obs, metadata.get("given_intent"))
        _, _, _, info = await env.step(Action(text=row["completion"]))
        if info.get("invalid_action"):
            raise ValueError("old replay target no longer legal")
        metadata.update(
            replay_original_id=row["id"], action_protocol=PROTOCOL, example_family="replay"
        )
        results.append({**row, "id": "replay-" + row["id"], "prompt": prompt, "metadata": metadata})
        dates.update(context["env_kwargs"]["episode_dates"])
    return results, dates


def find_span(rows, start_date, end_date):
    for row in rows:
        kwargs = json.loads(row["env_kwargs"])
        dates = [day[:10] for day in kwargs["episode_dates"]]
        if start_date in dates and end_date in dates:
            return row, dates.index(start_date), dates.index(end_date) + 1
    raise ValueError(f"no actual Train window contains {start_date} through {end_date}")


async def evaluation_contexts(rows):
    short = []
    for start_day, end_day in NEW_EVAL_SPANS:
        row, start, stop = find_span(rows, start_day, end_day)
        if stop - start != 7:
            raise ValueError("short evaluation span must have seven sessions")
        for scenario in SCENARIOS:
            context = make_context(row, start + 3, scenario)
            if scenario in RECOVERY_SCENARIOS:
                context = await recovery_context(context)
            short.append(context)
    row, start, stop = find_span(rows, "2023-05-23", "2023-06-13")
    if stop - start != 15:
        raise ValueError("new full available span must have fifteen sessions")
    full = [
        {
            "id": "gap-2023-05-23-2023-06-13-15sessions",
            "source_episode": row["extra_info"]["episode_id"],
            "env_kwargs": sliced_kwargs(json.loads(row["env_kwargs"]), start, stop),
            "raw_prompt": list(row["prompt"]),
            "state_kind": "cash",
            "prefix": [],
            "prefix_expected_invalid": [],
            "prefix_expected_errors": [],
            "expected_start_session_index": 0,
            "action_protocol": PROTOCOL,
        }
    ]
    return {
        "short": short,
        "full": full,
        "teacher_labels_included": False,
        "is_formal_evaluation": False,
        "full_session_count": 15,
    }


async def build_dataset(output):
    import pandas as pd
    from transformers import AutoTokenizer

    output.mkdir(parents=True, exist_ok=False)
    artifact = ROOT / "artifacts/m2_screening"
    source_manifest = json.loads((artifact / "manifest.json").read_text())
    if sha(artifact / "train.parquet") != source_manifest["datasets"]["train"]["sha256"]:
        raise ValueError("Train source checksum mismatch")
    all_sources = pd.read_parquet(artifact / "train.parquet").to_dict("records")
    by_id = {row["extra_info"]["episode_id"]: row for row in all_sources}
    contexts = training_contexts([by_id[episode] for episode in TRAIN_IDS])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rows = []
    boundaries = Counter()
    for index, context in enumerate(contexts):
        date_index, scenario_index = divmod(index, len(SCENARIOS))
        env, obs, _ = await replay_context(context)
        state = parse_state(obs)
        for symbol, position in state["positions"].items():
            side = "below_20pct" if position["weight"] < 0.2 else "above_20pct"
            boundaries[context["state_kind"] + ":" + symbol + ":" + side] += 1
        new_row = await decision_row(context, tokenizer)
        rows.append(new_row)
        if context["state_kind"] in RECOVERY_SCENARIOS:
            recovery = await recovery_context(context)
            rows.append(
                await decision_row(recovery, tokenizer, new_row["metadata"]["candidate_scores"])
            )
        for api_index, (intent, completion) in enumerate(
            api_targets(state, date_index, scenario_index)
        ):
            api_env, _, _ = await replay_context(context)
            _, _, _, info = await api_env.step(Action(text=completion))
            if info.get("invalid_action"):
                raise ValueError(f"API target invalid: {completion}: {info}")
            rows.append(
                {
                    "id": f"api-v2-{context['id']}-{api_index}",
                    "kind": "api",
                    "prompt": render(tokenizer, env, obs, intent),
                    "completion": completion,
                    "metadata": row_metadata(context)
                    | {"given_intent": intent, "teacher_label": None},
                }
            )
        if index % 8 == 0:
            print(
                json.dumps(
                    {
                        "contexts_completed": index + 1,
                        "total_contexts": len(contexts),
                        "new_rows": len(rows),
                    }
                ),
                flush=True,
            )
    replay, replay_dates = await replay_rows(by_id, tokenizer)
    rows.extend(replay)
    if len(rows) > MAX_ROWS:
        # This is declared before observing labels; no class or return filtering.
        rows = rows[:MAX_ROWS]
    new_eval = await evaluation_contexts(all_sources)
    regression = json.loads((OLD_DATASET / "eval_contexts.json").read_text())
    train_dates = source_dates(contexts) | replay_dates
    new_eval_dates = source_dates(new_eval["full"])
    old_eval_dates = source_dates(regression["full"])
    if (
        train_dates & new_eval_dates
        or train_dates & old_eval_dates
        or new_eval_dates & old_eval_dates
    ):
        raise ValueError("decision/prefix/outcome calendar dates overlap across splits")
    prompt_lengths = [
        len(tokenizer.encode(row["prompt"], add_special_tokens=False)) for row in rows
    ]
    total_lengths = [
        len(tokenizer.encode(row["prompt"] + row["completion"], add_special_tokens=False)) + 1
        for row in rows
    ]
    if max(prompt_lengths) > 8192 or max(total_lengths) > 8192:
        raise ValueError("dataset exceeds prompt/SFT length contract; no truncation")
    with (output / "train.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "eval_contexts.json").write_text(json.dumps(new_eval, indent=2))
    (output / "regression_eval_contexts.json").write_bytes(
        (OLD_DATASET / "eval_contexts.json").read_bytes()
    )
    distributions = {}
    for kind in sorted({row["kind"] for row in rows}):
        distributions[kind] = dict(
            Counter(
                json.loads(row["completion"][11:-12])["name"] for row in rows if row["kind"] == kind
            )
        )
    manifest = {
        "status": "COMPLETE",
        "created_at_SGT": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "source_train_sha256": sha(artifact / "train.parquet"),
        "generator_sha256": sha(Path(__file__)),
        "train_jsonl_sha256": sha(output / "train.jsonl"),
        "eval_contexts_sha256": sha(output / "eval_contexts.json"),
        "regression_eval_contexts_sha256": sha(output / "regression_eval_contexts.json"),
        "original_dataset_sha256": sha(OLD_DATASET / "train.jsonl"),
        "tokenizer_snapshot": str(MODEL),
        "action_protocol": PROTOCOL,
        "non_thinking": True,
        "source_train_ids": TRAIN_IDS,
        "row_count": len(rows),
        "row_kind_counts": dict(Counter(row["kind"] for row in rows)),
        "example_family_counts": dict(Counter(row["metadata"]["example_family"] for row in rows)),
        "replay_count": sum(row["id"].startswith("replay-") for row in rows),
        "clean_context_count": len(contexts),
        "unique_decision_dates": len({context["decision_date"] for context in contexts}),
        "selection": (
            "deduplicate Train dates, chronological dates[::4], fixed scenario order; "
            "old replay 76 decision and 20 API rows evenly spaced within kind; hardcap1024 prefix"
        ),
        "label_distribution": distributions,
        "actual_weight_boundary_counts": dict(boundaries),
        "max_prompt_tokens": max(prompt_lengths),
        "max_total_tokens_including_eos": max(total_lengths),
        "source_dates_min": min(train_dates),
        "source_dates_max": max(train_dates),
        "new_eval_dates_min": min(new_eval_dates),
        "new_eval_dates_max": max(new_eval_dates),
        "regression_dates_min": min(old_eval_dates),
        "regression_dates_max": max(old_eval_dates),
        "episode_and_label_dates_disjoint": True,
        "historical_feature_overlap_allowed": "visible history may include earlier Train dates",
        "new_eval_short_contexts": len(new_eval["short"]),
        "new_eval_full_sessions": 15,
        "new_eval_limitation": (
            "only 15 isolated Train sessions remain; no new disjoint 21-session window"
        ),
        "teacher_priors": {
            "symbols": ["GS", "MSFT", "HD"],
            "buy_targets": [0.1, 0.2],
            "sell_targets": [0, 0.1],
            "selection": "net wealth after three future sessions; hold first on economic ties",
            "prefixes": "three real orders/holds fixed independently of future; no oracle chaining",
            "recovery": "rejected direction requests generated from visible current state only",
            "api": "explicit API-only intention; target0.1/0.2/0.3 and preserve-portfolio hold",
            "hold": "no autonomous winning label overwritten merely to balance actions",
        },
        "future_information_in_prompt": False,
        "is_on_policy": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/m2-coldstart-v2-20260913/dataset"
    )
    args = parser.parse_args()
    asyncio.run(build_dataset(args.output))


if __name__ == "__main__":
    main()

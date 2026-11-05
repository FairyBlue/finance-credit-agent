#!/usr/bin/env python3
"""Independently audit full-reset order-recovery logs using only the stdlib.

All-worker mode requires all 240 planned identities and writes raw_audit.json
once. --model-key/--self-test only read completed workers and print a result.
Neither the model nor the market snapshots are loaded or replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_KEYS = ("sft", "seed42_m1", "seed42_m2", "seed43_m1", "seed43_m2")
CASE_IDS = tuple(
    f"formal-train-{number:04d}" for number in (210, 252, 336, 399, 420, 441, 231, 273)
)
PROFILE = "r5_order_recovery_fullreset_v1"
OBSERVATION_PROFILE = "history_features_recovery_v2"
ADVANCING = {"buy", "sell", "hold", "abstain"}
REQUEST_CAP, RECOVERY_CAP, RESPONSE_CAP, PROMPT_CAP = 256, 4, 4096, 8192
DIRECTION_ERRORS = {
    "buy target_weight must be above the current weight",
    "sell target_weight must be below the current weight",
}


def require(value: Any, message: str) -> None:
    if not value:
        raise ValueError(message)


def near(actual: Any, expected: Any, label: str, *, atol: float = 1e-9) -> None:
    require(
        type(actual) in (int, float)
        and type(expected) in (int, float)
        and math.isfinite(actual)
        and math.isfinite(expected)
        and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=atol),
        f"{label}: numeric evidence differs ({actual!r} != {expected!r})",
    )


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def lines(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def request_seed(base_seed: int, identity: dict[str, Any], index: int) -> int:
    """Independent implementation of the frozen finance-rollout-seed-v1 contract."""
    payload = {
        "base_seed": base_seed,
        "env_step": index,
        "scheme": "finance-rollout-seed-v1",
        **identity,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF


def expected_grid(manifest: dict[str, Any], model_key: str) -> dict[str, Any]:
    planned = {}
    for identity in manifest["models"][model_key]["identities"]:
        for case in manifest["cases"]:
            for sibling in range(5):
                key = f"{identity['id']}:{case['id']}:{OBSERVATION_PROFILE}:{sibling}"
                require(key not in planned, f"Duplicate planned identity: {key}")
                planned[key] = {
                    "id": key,
                    "model_identity": identity["id"],
                    "base_seed": identity["base_seed"],
                    "case_id": case["id"],
                    "case_kind": case["kind"],
                    "profile": OBSERVATION_PROFILE,
                    "start_mode": "fullreset",
                    "sampling": "greedy" if sibling == 0 else "sampled",
                    "rollout_identity": {
                        **case["source_identity"],
                        "rollout_n": sibling,
                        "validate": True,
                    },
                    "request_limit": REQUEST_CAP,
                }
    return planned


def audit_manifest(manifest: dict[str, Any]) -> None:
    require(
        manifest.get("profile") == PROFILE
        and manifest.get("start_mode") == "fullreset"
        and manifest.get("observation_profile") == OBSERVATION_PROFILE
        and manifest.get("action_protocol") == "single_tool_portfolio_v1"
        and manifest.get("enable_thinking") is False,
        "Unexpected full-reset protocol",
    )
    require(set(manifest["models"]) == set(MODEL_KEYS), "Model population differs")
    for key in MODEL_KEYS:
        seeds = (42, 43) if key == "sft" else (int(key[4:6]),)
        expected = [
            {"id": f"seed{seed}_sft" if key == "sft" else key, "base_seed": seed} for seed in seeds
        ]
        require(
            manifest["models"][key]["identities"] == expected,
            f"{key}: frozen model/seed identities differ",
        )
    require([case["id"] for case in manifest["cases"]] == list(CASE_IDS), "Case population differs")
    budget = manifest["budget"]
    frozen = {
        "cases": 8,
        "historical_rejection_cases": 6,
        "normal_cases": 2,
        "model_identities": 6,
        "model_engines": 5,
        "observation_profile": OBSERVATION_PROFILE,
        "start_mode": "fullreset",
        "greedy_per_case": 1,
        "sampled_per_case": 4,
        "planned_trajectories": 240,
        "requests_per_trajectory": REQUEST_CAP,
        "recovery_followup_requests": RECOVERY_CAP,
        "maximum_requests": 240 * REQUEST_CAP,
        "maximum_response_tokens": 240 * REQUEST_CAP * RESPONSE_CAP,
        "prompt_cap": PROMPT_CAP,
        "response_cap": RESPONSE_CAP,
        "training_updates": 0,
        "full_episode_evaluation": True,
        "automatic_retry": False,
        "automatic_actions": False,
    }
    require(budget == frozen, "Frozen full-reset budget differs")
    for case in manifest["cases"]:
        kwargs, ledger = case["env_kwargs"], case["initial_ledger"]
        require(
            case["kind"] == ("rejection" if case["id"] in CASE_IDS[:6] else "normal"),
            f"{case['id']}: case category differs",
        )
        require(
            "prefix" not in case
            and ledger
            == {
                "cash": kwargs["initial_cash"],
                "positions": {},
                "fills": [],
                "session_index": 0,
            },
            f"{case['id']}: fullreset does not begin with cash and empty positions/fills",
        )
        require(
            kwargs.get("observation_profile") == OBSERVATION_PROFILE
            and kwargs.get("reward_mode") == "finance_trajectory"
            and kwargs.get("capture_interactions") is True
            and kwargs.get("screening") is True
            and kwargs.get("finance_scorer_id") == "benchmark_scale_v2"
            and len(kwargs["episode_dates"])
            == len(kwargs["episode_open_times"])
            == len(kwargs["benchmark_closes"])
            == 21,
            f"{case['id']}: initial environment/scorer contract differs",
        )
    require(
        sum(len(expected_grid(manifest, key)) for key in MODEL_KEYS) == 240,
        "Frozen population is not 240 trajectories",
    )


def parse_state(messages: list[dict[str, Any]]) -> dict[str, Any]:
    text = messages[-1]["content"]
    if text.startswith("<tool_response>"):
        text = text.removeprefix("<tool_response>").removesuffix("</tool_response>").strip()
    return json.loads(text)


def diagnostic(document: dict[str, Any], text: str, label: str) -> None:
    require(
        document.get("truncated") is False
        and document.get("text") == text
        and document.get("char_count") == len(text)
        and document.get("sha256") == hashlib.sha256(text.encode()).hexdigest(),
        f"{label}: diagnostic text is truncated, missing or inconsistent",
    )


def legal_call(record: dict[str, Any], label: str) -> dict[str, Any]:
    # Match the pinned Hermes parser: malformed blocks are ignored and JSON
    # strings in arguments are decoded before TradingEnv validates the call.
    calls = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", record["response"], re.DOTALL):
        try:
            call = json.loads(block)
            name, arguments = call["name"], call["arguments"]
            if not isinstance(arguments, dict):
                arguments = json.loads(arguments) if isinstance(arguments, str) else {}
        except (ValueError, KeyError, TypeError):
            continue
        calls.append({"name": name, "arguments": arguments})
    require(len(calls) == 1, f"{label}: legal trace lacks exactly one model tool call")
    call = calls[0]
    require(call["name"] == record["trace"]["action"], f"{label}: model action was changed")
    return call


def check_accounting(
    before: dict[str, Any],
    after: dict[str, Any],
    record: dict[str, Any],
    kwargs: dict[str, Any],
    label: str,
) -> None:
    old_fills, fills = before["fills"], after["fills"]
    require(fills[: len(old_fills)] == old_fills, f"{label}: existing fills changed")
    new_fills = fills[len(old_fills) :]
    require(new_fills == record["feedback"].get("fills", []), f"{label}: feedback fills differ")
    call = legal_call(record, label)
    require(len(new_fills) <= 1, f"{label}: single tool call produced several fills")
    cash, positions = before["cash"], dict(before["positions"])
    for fill in new_fills:
        require(call["name"] == fill["side"] in {"buy", "sell"}, f"{label}: fill side differs")
        require(
            fill["symbol"] == call["arguments"]["symbol"].strip().upper()
            and fill["target_weight"] == float(call["arguments"]["target_weight"]),
            f"{label}: fill target differs from the model call",
        )
        require(fill["shares"] > 0 and fill["raw_open_price"] > 0, f"{label}: invalid fill size")
        direction = 1 if fill["side"] == "buy" else -1
        near(
            fill["execution_price"],
            fill["raw_open_price"] * (1 + direction * kwargs["slippage_bps"] / 10000),
            f"{label}: execution price",
        )
        near(fill["notional"], fill["shares"] * fill["execution_price"], f"{label}: notional")
        near(
            fill["commission"],
            fill["notional"] * kwargs["commission_bps"] / 10000,
            f"{label}: commission",
        )
        near(
            fill["slippage_cost"],
            fill["shares"] * abs(fill["execution_price"] - fill["raw_open_price"]),
            f"{label}: slippage",
        )
        session = before["session_index"]
        require(
            fill["decision_at"] == kwargs["episode_dates"][session]
            and fill["executed_at"] == kwargs["episode_open_times"][session + 1],
            f"{label}: fill did not execute at the next open",
        )
        cash -= direction * fill["shares"] * fill["execution_price"] + fill["commission"]
        shares = positions.get(fill["symbol"], 0.0) + direction * fill["shares"]
        require(
            cash >= -1e-6 and shares >= -1e-10, f"{label}: fill violates cash/no-short accounting"
        )
        cash = max(0.0, cash)
        positions[fill["symbol"]] = max(0.0, shares)
    near(after["cash"], cash, f"{label}: cash ledger", atol=1e-7)
    require(set(after["positions"]) == set(positions), f"{label}: position keys changed")
    for symbol, shares in positions.items():
        near(after["positions"][symbol], shares, f"{label}: {symbol} shares")


def check_observation(
    state: dict[str, Any], ledger: dict[str, Any], kwargs: dict[str, Any], label: str
) -> None:
    session = ledger["session_index"]
    require(
        state["session_index"] == session
        and state["current_date"] == kwargs["episode_dates"][session]
        and state["sessions_remaining"] == len(kwargs["episode_dates"]) - session - 1,
        f"{label}: observed session/date/remaining count differs",
    )
    near(state["cash"], ledger["cash"], f"{label}: visible cash", atol=1e-7)
    observed = state["positions"]
    active = {
        symbol: shares for symbol, shares in ledger["positions"].items() if abs(shares) > 1e-12
    }
    require(set(observed) == set(active), f"{label}: visible positions differ")
    equity = ledger["cash"]
    for symbol, shares in active.items():
        near(observed[symbol]["shares"], shares, f"{label}: visible shares", atol=1e-8)
        equity += shares * state["market"][symbol]["close"]
    near(state["portfolio_equity"], equity, f"{label}: visible equity", atol=1e-3)
    for symbol, shares in active.items():
        near(
            observed[symbol]["weight"],
            shares * state["market"][symbol]["close"] / equity,
            f"{label}: visible weight",
            atol=1e-7,
        )


def terminal_finance(
    row: dict[str, Any],
    case: dict[str, Any],
    records: list[dict[str, Any]],
    advances: list[dict[str, Any]],
    final_state: dict[str, Any],
) -> None:
    key, kwargs, metrics = row["id"], case["env_kwargs"], row["terminal_metrics"]
    initial = kwargs["initial_cash"]
    equity = [initial, *[record["feedback"]["equity"] for record in advances]]
    returns = [
        following / current - 1 for current, following in zip(equity[:-1], equity[1:], strict=True)
    ]
    benchmark = kwargs["benchmark_closes"]
    benchmark_returns = [b / a - 1 for a, b in zip(benchmark[:-1], benchmark[1:], strict=True)]
    net_return, benchmark_return = equity[-1] / initial - 1, benchmark[-1] / benchmark[0] - 1
    peak, drawdown = initial, 0.0
    for value in equity:
        peak = max(peak, value)
        drawdown = max(drawdown, (peak - value) / peak)
    turnover = [record["feedback"]["turnover"] for record in advances]
    max_position = max(
        (
            position["weight"]
            for record in advances
            for position in parse_state(record["observation_messages"])["positions"].values()
        ),
        default=0.0,
    )
    fills = row["final_ledger"]["fills"]
    commission = math.fsum(fill["commission"] for fill in fills)
    slippage = math.fsum(fill["slippage_cost"] for fill in fills)
    near(metrics["maximum_position_weight"], max_position, f"{key}: maximum position", atol=1e-7)
    risk = max(
        drawdown,
        max(0.0, max(turnover) - kwargs["turnover_limit"]),
        max(0.0, metrics["maximum_position_weight"] - kwargs["position_limit"]),
    )
    denominator = max(statistics.pstdev(benchmark_returns), kwargs["finance_volatility_floor"])
    score = (net_return - benchmark_return) / denominator - kwargs["risk_lambda"] * risk
    expected = {
        "raw_net_return": net_return,
        "benchmark_return": benchmark_return,
        "benchmark_excess": net_return - benchmark_return,
        "agent_volatility": statistics.pstdev(returns),
        "benchmark_volatility": statistics.pstdev(benchmark_returns),
        "score_denominator": denominator,
        "max_drawdown": drawdown,
        "turnover": math.fsum(turnover),
        "max_daily_turnover": max(turnover),
        "risk_penalty": risk,
        "transaction_cost": commission + slippage,
        "commission": commission,
        "slippage_cost": slippage,
        "terminal_score": score,
    }
    for field, value in expected.items():
        near(metrics[field], value, f"{key}: terminal {field}")
    require(
        metrics["scorer_id"] == kwargs["finance_scorer_id"]
        and metrics["calibration_sha256"] == kwargs["screening_calibration_sha256"]
        and metrics["invalid_count"] == row["invalid_requests"]
        and metrics["step_count"] == sum(r["environment_applied"] for r in records)
        and metrics["session_count"] == len(kwargs["episode_dates"]),
        f"{key}: terminal scorer/count contract differs",
    )
    near(row["full_window_net_return"], net_return, f"{key}: full-window return")
    near(records[-1]["feedback"]["episode_return"], net_return, f"{key}: final episode return")
    near(records[-1]["reward"], score, f"{key}: terminal reward")
    near(records[-1]["feedback"]["trajectory_reward"], score, f"{key}: feedback terminal reward")
    require(row["final_state"] == final_state, f"{key}: terminal observation differs")
    near(final_state["portfolio_equity"], equity[-1], f"{key}: final visible equity", atol=1e-7)


def computed_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model_identity"], row["case_kind"], row["sampling"])].append(row)
    details = []
    for key, group in sorted(groups.items()):
        complete = [row for row in group if row["status"] == "COMPLETE"]
        all_complete = len(complete) == len(group)
        events = [event for row in group for event in row["recovery_events"]]
        details.append(
            {
                "model_identity": key[0],
                "case_kind": key[1],
                "sampling": key[2],
                "planned": len(group),
                "complete": len(complete),
                "statuses": dict(Counter(row["status"] for row in group)),
                "requests": sum(row["requests"] for row in group),
                "invalid_requests": sum(row["invalid_requests"] for row in group),
                "recovery_events": len(events),
                "recovered_events": sum(event["status"] == "RECOVERED" for event in events),
                "trajectories_with_later_recovery_events": sum(
                    len(row["recovery_events"]) > 1 for row in group
                ),
                "performance_status": "COMPLETE" if all_complete else "INCOMPLETE",
                "mean_full_window_net_return": (
                    sum(row["full_window_net_return"] for row in complete) / len(complete)
                    if all_complete and complete
                    else None
                ),
                "mean_terminal_score": (
                    sum(row["terminal_metrics"]["terminal_score"] for row in complete)
                    / len(complete)
                    if all_complete and complete
                    else None
                ),
                "transaction_cost": (
                    math.fsum(row["terminal_metrics"]["transaction_cost"] for row in complete)
                    if all_complete
                    else None
                ),
            }
        )
    return {
        "full_episode_evaluation": True,
        "start_mode": "fullreset",
        "planned": len(rows),
        "complete": sum(row["status"] == "COMPLETE" for row in rows),
        "performance_status": "COMPLETE"
        if rows and all(row["status"] == "COMPLETE" for row in rows)
        else "INCOMPLETE",
        "statuses": dict(Counter(row["status"] for row in rows)),
        "requests": sum(row["requests"] for row in rows),
        "groups": details,
    }


def audit_trajectory(
    row: dict[str, Any],
    case: dict[str, Any],
    records: list[dict[str, Any]],
    prompt_failure: list[dict[str, Any]],
) -> dict[str, Any]:
    key, kwargs = row["id"], case["env_kwargs"]
    require(
        type(row["requests"]) is int and 0 <= row["requests"] <= REQUEST_CAP,
        f"{key}: request budget violated",
    )
    require(
        len(records) == row["requests"]
        and [r["request_index"] for r in records] == list(range(row["requests"])),
        f"{key}: request sequence differs",
    )
    require(
        row["response_tokens"] == sum(len(r["response_token_ids"]) for r in records),
        f"{key}: response-token total differs",
    )
    ledger = case["initial_ledger"]
    require(row["initial_ledger"] == ledger, f"{key}: historical ledger was reused at fullreset")
    check_observation(row["initial_state"], ledger, kwargs, f"{key}: initial")
    require(
        "order_rejection" not in row["initial_state"]
        and row["initial_state"]["recent_events"] == [],
        f"{key}: historical rejection/events reused",
    )
    traces, interactions, events, advances = [], row["interaction_trace"], [], []
    invalid, unapplied, terminal = 0, 0, 0
    last_state = row["initial_state"]
    expected_failure = None
    for index, record in enumerate(records):
        label = f"{key}:{index}"
        require(expected_failure is None and terminal == 0, f"{label}: request after stop")
        active = events[-1] if events and events[-1]["status"] == "ACTIVE" else None
        require(
            record["recovery_event_index"] == (active["event_index"] if active else None)
            and record["recovery_followup_index"]
            == (active["followup_requests"] + 1 if active else None),
            f"{label}: recovery request mapping differs",
        )
        if active:
            require(
                active["followup_requests"] < RECOVERY_CAP, f"{label}: fifth followup requested"
            )
            active["followup_requests"] += 1
        require(record["ledger_before"] == ledger, f"{label}: ledger chain differs")
        state_text = json.dumps(
            last_state, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        require(
            state_text in record["prompt"]
            or json.dumps(state_text, ensure_ascii=True)[1:-1] in record["prompt"],
            f"{label}: model prompt does not contain the current recorded observation",
        )
        require(
            record["base_seed"] == row["base_seed"]
            and record["rollout_identity"] == row["rollout_identity"]
            and record["seed_env_step"] == index
            and record["decision_session_index"] == ledger["session_index"],
            f"{label}: seed/request/session identity differs",
        )
        params = {
            "n": 1,
            "temperature": 0.0 if row["sampling"] == "greedy" else 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "max_tokens": RESPONSE_CAP,
            "seed": request_seed(row["base_seed"], row["rollout_identity"], index),
        }
        require(record["sampling_params"] == params, f"{label}: sampling/derived seed differs")
        require(
            0 < len(record["prompt_token_ids"]) <= PROMPT_CAP,
            f"{label}: over-limit/empty prompt was submitted",
        )
        tokens = record["response_token_ids"]
        require(len(tokens) <= RESPONSE_CAP, f"{label}: response escaped token cap")
        require(
            all(type(t) is int and t >= 0 for t in tokens + record["prompt_token_ids"]),
            f"{label}: token IDs malformed",
        )
        response_complete = (
            bool(tokens) and len(tokens) < RESPONSE_CAP and record["finish_reason"] == "stop"
        )
        if not record["environment_applied"]:
            require(not response_complete, f"{label}: complete response not applied")
            require(
                not {"feedback", "ledger_after", "trace", "reward", "terminal_reached"}
                & set(record),
                f"{label}: unapplied response has environment effects",
            )
            unapplied += 1
            expected_failure = "RESPONSE_INCOMPLETE"
            continue
        require(
            response_complete and record["stop_reason"] not in {"length", "abort", "aborted"},
            f"{label}: truncated/aborted response was applied",
        )
        trace, feedback, after = record["trace"], record["feedback"], record["ledger_after"]
        rejected, done = bool(feedback.get("invalid_action")), record["terminal_reached"]
        require(
            type(done) is bool
            and trace["step_index"] == len(traces)
            and trace["decision_session_index"] == ledger["session_index"]
            and trace["decision_at"] == kwargs["episode_dates"][ledger["session_index"]]
            and trace["result_session_index"] == after["session_index"]
            and trace["invalid_action"] == rejected
            and trace["done"] == done
            and trace["reward"] == record["reward"],
            f"{label}: trace/flags/reward differ",
        )
        interaction = interactions[len(traces)]
        require(
            all(
                interaction[field] == trace[field]
                for field in (
                    "step_index",
                    "decision_session_index",
                    "result_session_index",
                    "reward",
                    "done",
                )
            )
            and interaction["info"] == feedback,
            f"{label}: interaction mapping differs",
        )
        diagnostic(interaction["response"], record["response"], f"{label}: response")
        diagnostic(
            interaction["observation"],
            json.dumps(
                record["observation_messages"], ensure_ascii=True, sort_keys=True, allow_nan=False
            ),
            f"{label}: observation",
        )
        traces.append(trace)
        last_state = parse_state(record["observation_messages"])
        check_observation(last_state, after, kwargs, label)
        advance = after["session_index"] - ledger["session_index"]
        require(advance in {0, 1}, f"{label}: session advanced by more than one")
        if rejected:
            require(
                after == ledger and trace["action"] == "invalid" and not done,
                f"{label}: rejected action changed ledger or was terminal",
            )
            invalid += 1
            if active is None:
                active = {
                    "event_index": len(events),
                    "session_index": ledger["session_index"],
                    "trigger_request_index": index,
                    "trigger_error": feedback.get("error"),
                    "followup_requests": 0,
                    "invalid_requests": 0,
                    "status": "ACTIVE",
                    "direction_rejection": feedback.get("error") in DIRECTION_ERRORS,
                }
                events.append(active)
        elif advance:
            require(
                trace["action"] in ADVANCING and feedback["action"] == trace["action"],
                f"{label}: legal advance action differs",
            )
            check_accounting(ledger, after, record, kwargs, label)
            equity_before = (
                advances[-1]["feedback"]["equity"] if advances else kwargs["initial_cash"]
            )
            fills = after["fills"][len(ledger["fills"]) :]
            near(
                feedback["turnover"],
                math.fsum(fill["notional"] for fill in fills) / equity_before,
                f"{label}: session turnover",
            )
            near(
                feedback["cost_fraction"],
                math.fsum(fill["commission"] + fill["slippage_cost"] for fill in fills)
                / equity_before,
                f"{label}: session cost fraction",
            )
            benchmark = kwargs["benchmark_closes"]
            near(
                feedback["benchmark_return"],
                benchmark[after["session_index"]] / benchmark[ledger["session_index"]] - 1,
                f"{label}: session benchmark return",
            )
            high_water = max(
                kwargs["initial_cash"],
                feedback["equity"],
                *[item["feedback"]["equity"] for item in advances],
            )
            near(
                feedback["drawdown"],
                max(0.0, (high_water - feedback["equity"]) / high_water),
                f"{label}: session drawdown",
            )
            near(
                feedback["portfolio_return_net"],
                feedback["equity"]
                / (advances[-1]["feedback"]["equity"] if advances else kwargs["initial_cash"])
                - 1,
                f"{label}: session net return",
            )
            near(
                last_state["portfolio_equity"],
                feedback["equity"],
                f"{label}: close equity",
                atol=1e-7,
            )
            advances.append(record)
        else:
            require(
                after == ledger and trace["action"] in {"retrieve", "compute"} and not done,
                f"{label}: non-advance changed ledger or bypassed trading",
            )
            legal_call(record, label)
        if active:
            active["invalid_requests"] += int(rejected)
            if advance:
                active.update(
                    status="RECOVERED",
                    advance_request_index=index,
                    advanced_to_session=after["session_index"],
                    advance_action=trace["action"],
                )
            elif active["followup_requests"] >= RECOVERY_CAP:
                expected_failure = "RECOVERY_CAP"
        if done:
            require(
                advance == 1
                and not rejected
                and after["session_index"] == len(kwargs["episode_dates"]) - 1,
                f"{label}: false terminal/invalid done",
            )
            terminal += 1
        else:
            require(
                after["session_index"] < len(kwargs["episode_dates"]) - 1,
                f"{label}: final session not declared terminal",
            )
            near(record["reward"], 0.0, f"{label}: nonterminal finance reward")
        ledger = after
    require(
        len(interactions) == len(traces)
        and row["episode_trace"] == traces
        and row["final_ledger"] == ledger
        and row["invalid_requests"] == invalid,
        f"{key}: final ledger/episode/interaction/invalid totals differ",
    )
    if prompt_failure:
        require(
            len(prompt_failure) == 1 and not terminal and expected_failure is None,
            f"{key}: spurious prompt-cap evidence",
        )
        failure = prompt_failure[0]
        require(
            failure["request_index"] == row["requests"]
            and failure["failure"] == "PROMPT_CAP"
            and failure["submitted"] is False
            and len(failure["prompt_token_ids"]) > PROMPT_CAP,
            f"{key}: prompt-cap marker differs",
        )
        expected_failure = "PROMPT_CAP"
    if terminal:
        require(
            row["status"] == "COMPLETE"
            and expected_failure is None
            and len(advances) == len(kwargs["episode_dates"]) - 1
            and "failure" not in row,
            f"{key}: invalid COMPLETE classification",
        )
        terminal_finance(row, case, records, advances, last_state)
    else:
        if expected_failure is None and row["requests"] == REQUEST_CAP:
            expected_failure = "REQUEST_CAP"
        require(
            row["status"] == "INCOMPLETE"
            and expected_failure is not None
            and row.get("failure") == expected_failure,
            f"{key}: invalid INCOMPLETE classification",
        )
        require(
            not {
                "terminal_metrics",
                "full_window_net_return",
                "final_state",
                "terminal_score",
                "net_return",
                "episode_return",
            }
            & set(row),
            f"{key}: incomplete trajectory has finance",
        )
        if events and events[-1]["status"] == "ACTIVE":
            events[-1].update(status="INCOMPLETE", failure=expected_failure)
    require(row["recovery_events"] == events, f"{key}: recovery-event evidence differs")
    return {
        "invalid": invalid,
        "unapplied": unapplied,
        "advances": len(advances),
        "events": len(events),
        "recovered": sum(e["status"] == "RECOVERED" for e in events),
    }


def audit_worker(directory: Path, manifest: dict[str, Any], model_key: str) -> dict[str, Any]:
    folder = directory / "workers" / model_key
    files = {
        name: folder / name
        for name in (
            "state.json",
            "result.json",
            "config.json",
            "requests.jsonl",
            "submissions.jsonl",
            "raw.jsonl",
        )
    }
    before = {str(path): sha(path) for path in files.values()}
    state, result, config = (
        read(files[name]) for name in ("state.json", "result.json", "config.json")
    )
    require(
        state == result and state["status"] == "OBSERVED",
        f"{model_key}: state/result not OBSERVED/equal",
    )
    require(
        state.get("profile") == config.get("profile") == PROFILE
        and state.get("model_key") == config.get("model_key") == model_key
        and state.get("start_mode") == config.get("start_mode") == "fullreset"
        and state.get("full_episode_evaluation") is True
        and config.get("observation_profile") == OBSERVATION_PROFILE
        and config.get("manifest_sha256") == sha(directory / "manifest.json")
        and config.get("action_protocol") == "single_tool_portfolio_v1"
        and config.get("enable_thinking") is False,
        f"{model_key}: worker protocol differs",
    )
    planned, rows = expected_grid(manifest, model_key), state["grid"]
    by_id = {row["id"]: row for row in rows}
    require(
        len(rows) == len(by_id) == len(planned) and set(by_id) == set(planned),
        f"{model_key}: missing, duplicate or substituted planned identity",
    )
    for key, expected in planned.items():
        require(
            all(by_id[key].get(field) == value for field, value in expected.items()),
            f"{key}: immutable identity differs",
        )
    evidence, submissions, raw = (
        lines(files[name]) for name in ("requests.jsonl", "submissions.jsonl", "raw.jsonl")
    )
    requests = [record for record in evidence if record.get("submitted") is not False]
    markers = [record for record in evidence if record.get("submitted") is False]
    count = sum(row["requests"] for row in rows)
    require(
        count
        == len(requests)
        == sum(len(batch["requests"]) for batch in submissions)
        == sum(len(batch["outputs"]) for batch in raw)
        and len(submissions) == len(raw),
        f"{model_key}: request/submission/raw/state counts differ",
    )
    recorded, grouped, prompt_failures = {}, defaultdict(list), defaultdict(list)
    for record in requests:
        key = (record["trajectory_id"], record["request_index"])
        require(key[0] in planned and key not in recorded, f"{model_key}: orphan/duplicate request")
        recorded[key] = record
        grouped[key[0]].append(record)
    for marker in markers:
        require(marker["trajectory_id"] in planned, f"{model_key}: orphan prompt marker")
        prompt_failures[marker["trajectory_id"]].append(marker)
    seen, engine_ids = set(), set()
    for index, (submission, returned) in enumerate(zip(submissions, raw, strict=True)):
        require(
            0 < len(submission["requests"]) == len(returned["outputs"]) <= 8,
            f"{model_key}: bad batch size {index}",
        )
        for sent, received in zip(submission["requests"], returned["outputs"], strict=True):
            key = (sent["trajectory_id"], sent["request_index"])
            require(
                key in recorded and key not in seen,
                f"{model_key}: orphan/duplicate submission {key}",
            )
            seen.add(key)
            record = recorded[key]
            require(
                all(record.get(field) == value for field, value in sent.items()),
                f"{key}: submission fields differ from request",
            )
            require(
                sent["prompt_token_ids"] == received["prompt_token_ids"]
                and len(received["outputs"]) == 1,
                f"{key}: batch prompt/response mapping differs",
            )
            response = received["outputs"][0]
            require(
                all(
                    response[field] == record[target]
                    for field, target in (
                        ("token_ids", "response_token_ids"),
                        ("finish_reason", "finish_reason"),
                        ("stop_reason", "stop_reason"),
                    )
                ),
                f"{key}: raw response evidence differs",
            )
            require(
                received["request_id"] == record["vllm_request_id"]
                and received["request_id"] not in engine_ids,
                f"{key}: engine ID repeated/different",
            )
            engine_ids.add(received["request_id"])
    require(seen == set(recorded), f"{model_key}: requests without raw/submission evidence")
    cases = {case["id"]: case for case in manifest["cases"]}
    totals = Counter()
    for row in rows:
        totals.update(
            audit_trajectory(
                row, cases[row["case_id"]], grouped[row["id"]], prompt_failures[row["id"]]
            )
        )
    require(state["summary"] == computed_summary(rows), f"{model_key}: financial summary differs")
    require(
        before == {str(path): sha(path) for path in files.values()},
        f"{model_key}: worker logs changed during audit",
    )
    return {
        "status": "PASS",
        "model_key": model_key,
        "trajectories": len(rows),
        "request_count": count,
        "submitted_count": count,
        "raw_output_count": count,
        "batches": len(raw),
        "prompt_cap_markers": len(markers),
        "response_tokens": sum(row["response_tokens"] for row in rows),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "failure_reasons": dict(
            Counter(row.get("failure") for row in rows if row["status"] == "INCOMPLETE")
        ),
        "rejections_with_unchanged_ledger": totals["invalid"],
        "unapplied_incomplete_responses": totals["unapplied"],
        "legal_session_advances": totals["advances"],
        "recovery_events": totals["events"],
        "recovered_events": totals["recovered"],
        "summary": state["summary"],
        "failures": [
            {
                "id": row["id"],
                "case_id": row["case_id"],
                "sampling": row["sampling"],
                "failure": row["failure"],
                "requests": row["requests"],
                "final_session": row["final_ledger"]["session_index"],
                "recovery_events": row["recovery_events"],
            }
            for row in rows
            if row["status"] == "INCOMPLETE"
        ],
        "source_sha256": before,
    }


def audit(directory: Path, selected: tuple[str, ...]) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    manifest_hash, manifest = sha(manifest_path), read(manifest_path)
    audit_manifest(manifest)
    workers = [audit_worker(directory, manifest, key) for key in selected]
    require(sha(manifest_path) == manifest_hash, "Manifest changed during audit")
    result = {
        "profile": "r5_order_recovery_episode_independent_raw_audit_v1",
        "status": "PASS",
        "mode": "all_workers" if selected == MODEL_KEYS else "read_only_self_test",
        "audited_at_UTC": datetime.now(timezone.utc).isoformat(),
        "directory": str(directory),
        "planned_trajectories": 240,
        "worker_count": len(workers),
        "trajectories": sum(worker["trajectories"] for worker in workers),
        "requests": sum(worker["request_count"] for worker in workers),
        "workers": workers,
        "source_sha256": {
            str(manifest_path): manifest_hash,
            str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
        },
        "limitations": [
            "Eight fixed error-selected Train windows do not estimate failure prevalence.",
            "SFT seed identities share weights; model identity counts are not independent "
            "trained seeds.",
            "Audit validates recorded token mappings and finance, without tokenizer "
            "re-decoding or market replay.",
            "PASS means evidence is internally valid; INCOMPLETE outcomes remain failures, "
            "without financial means.",
        ],
    }
    readout_path = directory / "readout.json"
    if readout_path.exists():
        digest, readout = sha(readout_path), read(readout_path)
        planned = {
            key: value
            for model in MODEL_KEYS
            for key, value in expected_grid(manifest, model).items()
        }
        rows = readout["grid"]
        require(
            len(rows) == len({row["id"] for row in rows}) == 240
            and {row["id"] for row in rows} == set(planned),
            "Readout drops planned/missing identities",
        )
        require(
            readout["summary"] == computed_summary(rows),
            "Readout summary or incomplete finance differs",
        )
        for row in rows:
            require(
                all(row.get(field) == value for field, value in planned[row["id"]].items()),
                f"{row['id']}: readout substituted an identity",
            )
        if selected == MODEL_KEYS:
            require(
                readout["status"] == "OBSERVED"
                and readout["workers"] == dict.fromkeys(MODEL_KEYS, "OBSERVED"),
                "Formal readout contains missing/unobserved workers",
            )
            observed = {
                row["id"]: row
                for key in MODEL_KEYS
                for row in read(directory / "workers" / key / "result.json")["grid"]
            }
            require(
                all(row == observed[row["id"]] for row in rows),
                "Readout rows differ from worker results",
            )
        require(sha(readout_path) == digest, "Readout changed during audit")
        result["source_sha256"][str(readout_path)] = digest
    if selected == MODEL_KEYS:
        require(
            result["worker_count"] == 5
            and result["trajectories"] == 240
            and result["requests"] <= 240 * REQUEST_CAP,
            "Formal full-reset coverage/budget differs",
        )
    for worker in workers:
        require(
            all(
                sha(Path(filename)) == digest
                for filename, digest in worker["source_sha256"].items()
            ),
            f"{worker['model_key']}: artifacts changed after worker audit",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--model-key", choices=MODEL_KEYS)
    parser.add_argument(
        "--self-test", action="store_true", help="Read only the SFT worker by default"
    )
    args = parser.parse_args()
    directory = args.directory.resolve()
    partial = args.model_key is not None or args.self_test
    output = directory / "raw_audit.json"
    if not partial and output.exists():
        raise FileExistsError(f"Refusing to overwrite formal raw audit: {output}")
    result = audit(directory, (args.model_key or "sft",) if partial else MODEL_KEYS)
    if not partial:
        with output.open("x") as handle:
            json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "mode", "worker_count", "trajectories", "requests")
            }
            | {"output": None if partial else str(output)},
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

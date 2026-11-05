#!/usr/bin/env python3
"""CPU environment driver for a fixed same-prefix stochastic-continuation pilot.

Preparation is stdlib-only. The run process speaks JSONL over inherited pipes;
it never loads a model, initializes CUDA, or launches a GPU worker.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, timezone
from datetime import time as wall_time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/m2-same-prefix-sampling-20261018-v1"
MENU = ROOT / "runs/m2-same-prefix-menu-20261018-v1"
DEFAULT = ROOT / "runs/m2-credit-comparison-20261018-v1"
PROFILE = "same_prefix_fresh_estimate_reference_credit_comparison_v1"
HOLD = '<tool_call>{"name":"hold","arguments":{}}</tool_call>'
EOS = {151645, 151643}
MAX_BATCH = 32
MAX_ATTEMPTS = 4
MAX_REQUESTS = 39024


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, default=lambda v: v.isoformat())


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_new(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        stream.write(canonical(value) + "\n")


def append(stream: Any, value: Any) -> None:
    stream.write(canonical(value) + "\n")
    stream.flush()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def request_seed(trajectory_key: str, session: int, attempt: int) -> int:
    namespace = "EST" if trajectory_key.startswith("est/") else "REF"
    require(trajectory_key.startswith(("est/", "ref/")), "explicit EST/REF trajectory namespace")
    payload = ["m2-credit-comparison-v1", namespace, trajectory_key, session, attempt]
    return int.from_bytes(hashlib.sha256(canonical(payload).encode()).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def request_key(trajectory_key: str, session: int, attempt: int) -> str:
    return f"{trajectory_key}/d{session}/a{attempt}"


def prepare_manifest(output: Path | str) -> dict:
    """Freeze all 128 heads and fresh248 estimate/616 reference paths, without envs."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "continuation_manifest.json").exists(), "manifest exists; no retry")
    require(not (output / "request_seed_table.jsonl").exists(), "seed table exists; no retry")
    pilot = ROOT / "runs/m2-stochastic-continuation-20261018-v3"
    old_manifest = read(pilot / "continuation_manifest.json")
    cells = old_manifest["cells"]
    require(len(cells) == 77, "77 fixed reference cells")
    evaluations = {r["request_key"]: r for r in rows(SOURCE / "evaluations.jsonl")}
    generations = {r["request_key"]: r for r in rows(SOURCE / "generations.jsonl")}
    native = rows(SOURCE / "inputs.jsonl")
    require(len(native) == len(evaluations) == len(generations) == 128, "128 original heads")
    by_order = {(c["anchor_id"], c["economic_order_key"]): c for c in cells}
    require(len(by_order) == 77, "unique frozen reference anchor/order cells")
    duplicate_states = {}
    comparisons = 0
    heads = []
    for item in native:
        key = item["request_key"]
        ev, gen = evaluations[key], generations[key]
        legal = (
            gen["generation_status"] == "COMPLETE"
            and ev["env_status"] == "ADVANCING"
            and ev["terminal_label_status"] == "COMPLETE_FIXED_HOLD"
        )
        cell = by_order[item["anchor_id"], ev["economic_order_key"]] if legal else None
        if legal:
            order = (item["anchor_id"], ev["economic_order_key"])
            state = (ev["first_observation_messages"], ev["first_state_after"], ev["first_info"])
            if order in duplicate_states:
                require(
                    state == duplicate_states[order],
                    "same-order source slots have different continuation states",
                )
                comparisons += 1
            else:
                duplicate_states[order] = state
        heads.append(
            {
                "source_request_key": key,
                "slot_id": item["slot"],
                "anchor_id": item["anchor_id"],
                "cell_id": cell["cell_id"] if cell else None,
                "source_group": item["source_group"],
                "legal_head": legal,
                "head_status": ev["env_status"],
                "generation_status": gen["generation_status"],
                "economic_order_key": ev["economic_order_key"],
                "first_action_text": gen["response_text"],
                "expected_first_info": ev["first_info"],
                "expected_first_state_after": ev.get("first_state_after"),
                "expected_first_observation_messages": ev.get("first_observation_messages"),
                "financial_state_sha256": ev["financial_state_sha256"],
                "observation_messages_sha256": ev["observation_messages_sha256"],
            }
        )
    require(
        sum(h["legal_head"] for h in heads) == 124 and len(duplicate_states) == 48,
        "124 legal original heads/48 reference orders",
    )
    require(comparisons == 76, "all76 duplicate-state comparisons")
    cells_by_id = {c["cell_id"]: c for c in cells}
    trajectories = []
    for head in heads:
        if not head["legal_head"]:
            continue
        cell = cells_by_id[head["cell_id"]]
        for rep in range(2):
            trajectories.append(
                {
                    "trajectory_key": f"est/{head['source_request_key']}/r{rep}",
                    "role": "estimate",
                    "cell_id": cell["cell_id"],
                    "anchor_id": head["anchor_id"],
                    "source_request_key": head["source_request_key"],
                    "slot_id": head["slot_id"],
                    "replicate": rep,
                    "decision_session_index": cell["decision_session_index"],
                    "remaining_sessions": 19 - cell["decision_session_index"],
                }
            )
    for cell in cells:
        for rep in range(8):
            trajectories.append(
                {
                    "trajectory_key": f"ref/{cell['cell_id']}/r{rep}",
                    "role": "reference",
                    "cell_id": cell["cell_id"],
                    "anchor_id": cell["anchor_id"],
                    "source_request_key": cell["source_request_key"],
                    "slot_id": cell["source_slot"],
                    "replicate": rep,
                    "decision_session_index": cell["decision_session_index"],
                    "remaining_sessions": 19 - cell["decision_session_index"],
                }
            )
    require(len(trajectories) == 864, "864 fresh trajectories")
    seeds = [
        {
            "request_key": request_key(t["trajectory_key"], session, attempt),
            "trajectory_key": t["trajectory_key"],
            "role": t["role"],
            "decision_session_index": session,
            "attempt": attempt,
            "request_seed": request_seed(t["trajectory_key"], session, attempt),
        }
        for t in trajectories
        for session in range(t["decision_session_index"] + 1, 20)
        for attempt in range(MAX_ATTEMPTS)
    ]
    require(
        len(seeds) == MAX_REQUESTS and len({s["request_seed"] for s in seeds}) == MAX_REQUESTS,
        "39024 unique possible request seeds",
    )
    seed_counts = Counter(s["role"] for s in seeds)
    require(seed_counts == {"estimate": 11408, "reference": 27616}, "fixed per-role budgets")
    pilot_seeds = {r["request_seed"] for r in rows(pilot / "request_seed_table.jsonl")}
    require(
        not pilot_seeds.intersection(s["request_seed"] for s in seeds),
        "fresh seeds never reuse pilot seeds",
    )
    with (output / "request_seed_table.jsonl").open("x") as stream:
        for row in seeds:
            append(stream, row)
    names = [
        SOURCE / n
        for n in (
            "contract.json",
            "inputs.jsonl",
            "generations.jsonl",
            "evaluations.jsonl",
            "readout.json",
            "model_preflight_receipt.json",
        )
    ]
    names += [MENU / n for n in ("anchors.jsonl", "prompt_packets.jsonl", "source_manifest.json")]
    names += [pilot / "continuation_manifest.json", pilot / "request_seed_table.jsonl"]
    manifest = {
        "profile": PROFILE,
        "status": "PREPARED_NOT_STARTED",
        "prepared_at_UTC": datetime.now(timezone.utc).isoformat(),
        "source_sampling_run": str(SOURCE),
        "source_menu_run": str(MENU),
        "source_pilot_run": str(pilot),
        "model_path": old_manifest["model_path"],
        "policy_snapshot_sha256": old_manifest["policy_snapshot_sha256"],
        "source_file_sha256": {str(p): sha(p) for p in names},
        "request_seed_table_sha256": sha(output / "request_seed_table.jsonl"),
        "cells": cells,
        "head_slots": heads,
        "trajectories": trajectories,
        "unique_order_cells": 48,
        "added_hold_cells": 29,
        "expected_cells": 77,
        "expected_trajectories": 864,
        "expected_estimate_paths": 248,
        "expected_reference_paths": 616,
        "head_slots_total": 128,
        "head_slots_legal": 124,
        "head_slots_invalid_NA": 4,
        "source_same_order_continuation_state_comparisons": comparisons,
        "source_same_order_continuation_states_identical": True,
        "max_model_requests": MAX_REQUESTS,
        "max_estimate_requests": 11408,
        "max_reference_requests": 27616,
        "seed_namespaces": ["m2-credit-comparison-v1", "EST", "REF"],
        "max_attempts_per_session": MAX_ATTEMPTS,
        "max_active_trajectories": MAX_BATCH,
        "terminal_session_index": 20,
        "optimizer_updates": 0,
        "formal_valid_oos_access": False,
        "automatic_retry": False,
        "fallback_hold": False,
        "estimate_continuations_deduplicated": False,
        "continuation_policy": "common recovered SFT; temperature1/top_p1/native free tool calls",
    }
    write_new(output / "continuation_manifest.json", manifest)
    return manifest


def generation_complete(record: dict) -> bool:
    tokens = record["response_token_ids"]
    stop = record["stop_reason"]
    return bool(
        0 < len(tokens) < 4096
        and record["returned_finished"] is True
        and record["finish_reason"] == "stop"
        and tokens[-1] in EOS
        and not any(t in EOS for t in tokens[:-1])
        and (stop is None or (type(stop) is int and stop == tokens[-1]))
    )


def install_fast_adapter(replay: Any, output: Path) -> None:
    """Use the production loader in parallel, trusting prior raw-data verification.

    No old source is edited. The adapter binding affects this CPU process only;
    exact original prefix observations and financial states remain mandatory.
    """
    from concurrent.futures import ThreadPoolExecutor

    from src.data import AvailabilityPolicy, StockBenchAdapter
    from src.env import trading

    kwargs = next(iter(replay.data.values()))["kwargs"]
    fields = (
        "manifest_root",
        "storage_root",
        "symbols",
        "data_version",
        "data_start",
        "data_end",
        "input_manifest_sha256",
        "input_file_count",
        "input_sessions_per_symbol",
        "exchange_calendar",
        "market_timezone",
        "market_close_time",
    )
    expected = {
        key: kwargs.get(key, "XNYS" if key == "exchange_calendar" else None) for key in fields
    }
    expected["manifest_root"] = str(Path(expected["manifest_root"]).resolve())
    expected["storage_root"] = str(Path(expected["storage_root"]).resolve())
    expected["symbols"] = tuple(kwargs["symbols"])
    require(len(expected["symbols"]) == 20, "frozen 20-symbol adapter")
    policy = AvailabilityPolicy(
        kwargs["market_timezone"],
        wall_time.fromisoformat(kwargs["market_close_time"]),
        kwargs.get("exchange_calendar", "XNYS"),
    )

    def load(symbol: str) -> Any:
        return StockBenchAdapter.from_storage(
            kwargs["storage_root"],
            symbols=[symbol],
            data_version=kwargs["data_version"],
            availability_policy=policy,
            verify_market_checksums=False,
            expected_raw_sha256_by_path=None,
            start_date=date.fromisoformat(kwargs["data_start"]),
            end_date=date.fromisoformat(kwargs["data_end"]),
        )

    print("CPU fast preprocessing: parallel production loader, prior checksums trusted", flush=True)
    with ThreadPoolExecutor(max_workers=min(20, len(expected["symbols"]))) as pool:
        pieces = list(pool.map(load, expected["symbols"]))
    observations = [observation for piece in pieces for observation in piece._observations]
    adapter = StockBenchAdapter(observations, data_version=kwargs["data_version"])

    def bound_adapter(**actual: Any) -> Any:
        require(actual == expected, "fast adapter configuration differs from frozen original")
        return adapter

    trading._cached_storage_adapter = bound_adapter
    write_new(
        output / "fast_preprocessing_receipt.json",
        {
            "status": "READY",
            "at_UTC": datetime.now(timezone.utc).isoformat(),
            "implementation": "production StockBenchAdapter.from_storage per symbol; "
            "process-local cache binding",
            "max_workers": min(20, len(expected["symbols"])),
            "configuration": expected,
            "observations": len(observations),
            "verify_market_checksums": False,
            "expected_raw_sha256_by_path": None,
            "full_raw_inventory_rehashed": False,
            "trusted_previous_verification": str(MENU / "final_verification.json"),
            "trusted_previous_verification_sha256": sha(MENU / "final_verification.json"),
            "actual_prefix_financial_and_observation_identity_still_required": True,
            "old_source_files_modified": False,
        },
    )


async def run(output: Path, request_fd: int, response_fd: int) -> None:
    # These backend flags affect this separate CPU process, never the GPU worker.
    os.environ.update(
        USE_TORCH="0",
        USE_TF="0",
        USE_FLAX="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "third_party/agent-r1"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from agent_r1.env.base import Action
    from m2_same_prefix_menu import PROTOCOL, TEMPLATE_KWARGS, Replay, financial_state
    from transformers import AutoTokenizer

    from src.agents.screening_prompt import apply_screening_action_protocol

    manifest = read(output / "continuation_manifest.json")
    require(manifest["profile"] == PROFILE, "fixed continuation profile")
    require(
        sha(output / "request_seed_table.jsonl") == manifest["request_seed_table_sha256"],
        "seed table SHA",
    )
    for name, digest in manifest["source_file_sha256"].items():
        require(sha(Path(name)) == digest, f"source input changed: {name}")
    require(not (output / "driver_completion.json").exists(), "no driver retry")
    seed_table = {r["request_key"]: r for r in rows(output / "request_seed_table.jsonl")}
    require(len(seed_table) == MAX_REQUESTS, "all fixed potential request keys")
    cells = {c["cell_id"]: c for c in manifest["cells"]}
    heads = {h["source_request_key"]: h for h in manifest["head_slots"]}
    specs = iter(manifest["trajectories"])
    packets = {p["anchor_id"]: p for p in rows(MENU / "prompt_packets.jsonl")}
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model_path"], local_files_only=True, trust_remote_code=False
    )
    replay = Replay(read(MENU / "source_manifest.json"))
    install_fast_adapter(replay, output)
    active: list[dict] = []
    post_first_identity_by_cell: dict[str, tuple[str, str]] = {}
    classified = model_calls = batch_id = fixed_actions = 0
    counts: Counter = Counter()
    outputs = ("driver_requests.jsonl", "driver_evaluations.jsonl", "trajectory_results.jsonl")

    def prompt_ids(env: Any, obs: Any) -> list[int]:
        return tokenizer.apply_chat_template(
            apply_screening_action_protocol(obs.messages, PROTOCOL),
            tools=env.tool_schemas,
            tokenize=True,
            **TEMPLATE_KWARGS,
        )

    with (
        os.fdopen(request_fd, "w", buffering=1) as requests_pipe,
        os.fdopen(response_fd, "r", buffering=1) as responses_pipe,
        (output / outputs[0]).open("x") as request_log,
        (output / outputs[1]).open("x") as evaluation_log,
        (output / outputs[2]).open("x") as terminal_log,
    ):

        def receive() -> dict:
            line = responses_pipe.readline()
            require(bool(line), "GPU RPC EOF before expected response")
            message = json.loads(line)
            require(message.get("event") != "ERROR", f"GPU RPC worker error: {message}")
            return message

        def finish(state: dict, status: str, reason: str | None = None) -> None:
            nonlocal classified
            env, cell = state["env"], state["cell"]
            metrics = env.screening_terminal_metrics() if status == "COMPLETE" else None
            score = metrics["terminal_score"] if metrics else None
            if status == "COMPLETE":
                require(
                    env.episode_complete and env.current_session_index == 20, "original terminal"
                )
                require(
                    type(score) in (int, float) and math.isfinite(score), "finite actual terminal Y"
                )
            append(
                terminal_log,
                {
                    **state["spec"],
                    "source_group": cell["source_group"],
                    "status": status,
                    "failure_reason": reason,
                    "Y": score,
                    "terminal_metrics": metrics,
                    "model_calls": state["model_calls"],
                    "session_advances": state["session_advances"],
                    "last_session_index": env.current_session_index,
                    "first_action_text": cell["first_action_text"],
                    "economic_order_key": cell["economic_order_key"],
                    "source_slot": cell["source_slot"],
                    "added_hold": cell["added_hold"],
                    "first_info": state["first_info"],
                    "first_state_after": state["first_state_after"],
                    "first_observation_messages_sha256": state["first_observation_messages_sha256"],
                    "financial_state_sha256": cell["financial_state_sha256"],
                    "observation_messages_sha256": cell["observation_messages_sha256"],
                    "prefix_requests": cell["prefix_requests"],
                    "policy_snapshot_sha256": manifest["policy_snapshot_sha256"],
                    "last_observation_messages": state["obs"].messages,
                },
            )
            classified += 1
            counts[status] += 1

        async def refill() -> None:
            nonlocal fixed_actions
            while len(active) < MAX_BATCH:
                spec = next(specs, None)
                if spec is None:
                    break
                cell = dict(cells[spec["cell_id"]])
                if spec["role"] == "estimate":
                    head = heads[spec["source_request_key"]]
                    cell.update(
                        first_action_text=head["first_action_text"],
                        expected_first_info=head["expected_first_info"],
                        expected_first_state_after=head["expected_first_state_after"],
                        source_slot=head["slot_id"],
                        source_request_key=head["source_request_key"],
                        added_hold=False,
                    )
                env, obs, config_sha = await replay.prefix(
                    cell["record_index"], cell["prefix_requests"]
                )
                require(
                    object_sha(financial_state(env, config_sha)) == cell["financial_state_sha256"],
                    "exact prefix finance",
                )
                require(
                    obs.messages == packets[cell["anchor_id"]]["messages"],
                    "exact actual prefix messages",
                )
                ids = prompt_ids(env, obs)
                require(
                    ids == packets[cell["anchor_id"]]["prompt_token_ids"], "exact prefix prompt IDs"
                )
                index = env.current_session_index
                obs, _, _, info = await env.step(Action(text=cell["first_action_text"]))
                replay.requests += 1
                fixed_actions += 1
                require(
                    not info.get("invalid_action") and env.current_session_index == index + 1,
                    "fixed first order must legally advance",
                )
                after = json.loads(canonical(financial_state(env, config_sha)))
                if not cell["added_hold"]:
                    require(
                        json.loads(canonical(info)) == cell["expected_first_info"],
                        "original first info exact",
                    )
                    require(
                        after == cell["expected_first_state_after"], "original first ledger exact"
                    )
                    source_head = heads[cell["source_request_key"]]
                    require(
                        obs.messages == source_head["expected_first_observation_messages"],
                        "original post-first model-visible messages exact",
                    )
                post_identity = (object_sha(after), object_sha(obs.messages))
                if cell["cell_id"] in post_first_identity_by_cell:
                    require(
                        post_identity == post_first_identity_by_cell[cell["cell_id"]],
                        "estimate/reference same-cell post-first finance and messages differ",
                    )
                else:
                    post_first_identity_by_cell[cell["cell_id"]] = post_identity
                active.append(
                    {
                        "spec": spec,
                        "cell": cell,
                        "env": env,
                        "obs": obs,
                        "config_sha": config_sha,
                        "attempt": 0,
                        "model_calls": 0,
                        "session_advances": 0,
                        "first_info": info,
                        "first_state_after": after,
                        "first_observation_messages_sha256": post_identity[1],
                    }
                )

        await refill()
        require(receive().get("event") == "READY", "worker READY required")
        while active:
            pending = []
            for state in list(active):
                ids = prompt_ids(state["env"], state["obs"])
                if len(ids) > 8192:
                    finish(state, "PROMPT_LIMIT", "prompt exceeds8192; no truncation or retry")
                    active.remove(state)
                    continue
                spec = state["spec"]
                session, attempt = state["env"].current_session_index, state["attempt"]
                key = request_key(spec["trajectory_key"], session, attempt)
                seed = seed_table[key]["request_seed"]
                require(
                    seed == request_seed(spec["trajectory_key"], session, attempt),
                    "fixed request seed",
                )
                request = {
                    "request_key": key,
                    "trajectory_key": spec["trajectory_key"],
                    "cell_id": spec["cell_id"],
                    "anchor_id": spec["anchor_id"],
                    "replicate": spec["replicate"],
                    "role": spec["role"],
                    "source_request_key": spec["source_request_key"],
                    "slot_id": spec["slot_id"],
                    "decision_session_index": session,
                    "attempt": attempt,
                    "request_seed": seed,
                    "prompt_token_ids": ids,
                    "prompt_sha256": hashlib.sha256(json.dumps(ids).encode("ascii")).hexdigest(),
                    "prompt_token_count": len(ids),
                    "max_tokens": 4096,
                }
                append(request_log, request)
                pending.append((state, request))
            if not pending:
                await refill()
                continue
            require(model_calls + len(pending) <= MAX_REQUESTS, "global model request bound")
            append(
                requests_pipe,
                {"op": "generate", "batch_id": batch_id, "requests": [r for _, r in pending]},
            )
            message = receive()
            require(
                message.get("event") == "BATCH" and message["batch_id"] == batch_id,
                "exact RPC batch",
            )
            records = message["records"]
            require(len(records) == len(pending), "exact RPC result count")
            for (state, request), record in zip(pending, records, strict=True):
                require(
                    all(record[k] == v for k, v in request.items()),
                    "returned request metadata exact",
                )
                require(
                    record["returned_prompt_token_ids"] == request["prompt_token_ids"],
                    "returned native prompt exact",
                )
                require(
                    (record["generation_status"] == "COMPLETE") is generation_complete(record),
                    "native finish/EOS integrity",
                )
                state["model_calls"] += 1
                model_calls += 1
                env = state["env"]
                before = env.current_session_index
                evaluation = {
                    **request,
                    "batch_id": batch_id,
                    "generation_status": record["generation_status"],
                    "response_text": record["response_text"],
                    "response_token_ids": record["response_token_ids"],
                    "failure_reason": record.get("failure_reason"),
                    "session_before": before,
                    "session_after": before,
                    "info": None,
                    "trace": None,
                    "env_status": "NOT_APPLIED",
                }
                if record["generation_status"] != "COMPLETE":
                    append(evaluation_log, evaluation)
                    finish(state, "INCOMPLETE", record.get("failure_reason"))
                    active.remove(state)
                    continue
                obs, _, _, info = await env.step(
                    Action(text=record["response_text"], token_ids=record["response_token_ids"])
                )
                replay.requests += 1
                state["obs"] = obs
                after = env.current_session_index
                require(after in (before, before + 1), "at most one session advance")
                evaluation.update(
                    session_after=after,
                    info=info,
                    trace=asdict(env.credit_step_trace[-1]),
                    observation_messages=obs.messages,
                    env_status="INVALID"
                    if info.get("invalid_action")
                    else ("ADVANCING" if after == before + 1 else "NONADVANCING"),
                )
                append(evaluation_log, evaluation)
                if after == before + 1:
                    require(not info.get("invalid_action"), "invalid cannot advance")
                    state["session_advances"] += 1
                    state["attempt"] = 0
                    if env.episode_complete:
                        finish(state, "COMPLETE")
                        active.remove(state)
                else:
                    state["attempt"] += 1
                    if state["attempt"] >= MAX_ATTEMPTS:
                        finish(
                            state,
                            "REQUEST_LIMIT",
                            "four model requests without session advancement",
                        )
                        active.remove(state)
            batch_id += 1
            if batch_id % 10 == 0:
                print("continuation", classified, "/864", "model_calls", model_calls, flush=True)
            await refill()
        require(
            classified == manifest["expected_trajectories"]
            and fixed_actions == manifest["expected_trajectories"],
            "all fixed864 trajectories classified",
        )
        append(requests_pipe, {"op": "close"})
        closed = receive()
        require(
            closed.get("event") == "CLOSED" and closed["model_requests"] == model_calls,
            "worker close accounting",
        )
    names = [*outputs, "fast_preprocessing_receipt.json"]
    write_new(
        output / "driver_completion.json",
        {
            "status": "COMPLETE",
            "finished_at_UTC": datetime.now(timezone.utc).isoformat(),
            "trajectories_classified": classified,
            "expected_trajectories": manifest["expected_trajectories"],
            "trajectory_status_counts": dict(counts),
            "model_requests": model_calls,
            "request_limit": MAX_REQUESTS,
            "environment_requests": replay.requests,
            "fixed_first_actions": fixed_actions,
            "environments_constructed": replay.environments,
            "optimizer_updates": 0,
            "formal_valid_oos_access": False,
            "automatic_retry": False,
            "output_sha256": {name: sha(output / name) for name in names},
            "continuation_manifest_sha256": sha(output / "continuation_manifest.json"),
        },
    )
    print(
        canonical(
            {
                "status": "COMPLETE",
                "trajectories": classified,
                "model_requests": model_calls,
                "trajectory_status_counts": dict(counts),
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--request-fd", type=int)
    parser.add_argument("--response-fd", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        manifest = prepare_manifest(args.output)
        print(
            canonical(
                {
                    k: manifest[k]
                    for k in ("expected_cells", "expected_trajectories", "max_model_requests")
                }
            )
        )
    else:
        require(
            args.request_fd is not None and args.response_fd is not None, "two inherited IPC fds"
        )
        asyncio.run(run(args.output.resolve(), args.request_fd, args.response_fd))


if __name__ == "__main__":
    main()

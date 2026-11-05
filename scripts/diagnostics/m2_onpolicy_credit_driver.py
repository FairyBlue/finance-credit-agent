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
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date, datetime, timezone
from datetime import time as wall_time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/m2-same-prefix-sampling-20261018-v1"
MENU = ROOT / "runs/m2-same-prefix-menu-20261018-v1"
DEFAULT = ROOT / "runs/m2-onpolicy-credit-20261018-v1"
PROFILE = "fresh_own_policy_state_credit_validation_v1"
HOLD = '<tool_call>{"name":"hold","arguments":{}}</tool_call>'
EOS = {151645, 151643}
MAX_BATCH = 32
MAX_ATTEMPTS = 4
MAX_REQUESTS = 23744


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
    namespace = {"parent": "PARENT", "head": "HEAD", "est": "EST", "ref": "REF"}.get(
        trajectory_key.split("/", 1)[0]
    )
    require(namespace is not None, "explicit PARENT/HEAD/EST/REF namespace")
    payload = ["m2-onpolicy-credit-v1", namespace, trajectory_key, session, attempt]
    return int.from_bytes(hashlib.sha256(canonical(payload).encode()).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def request_key(trajectory_key: str, session: int, attempt: int) -> str:
    return f"{trajectory_key}/d{session}/a{attempt}"


def prepare_manifest(output: Path | str) -> dict:
    """Freeze fresh-parent windows and every potential request before new outcomes."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "continuation_manifest.json").exists(), "manifest exists; no retry")
    require(not (output / "request_seed_table.jsonl").exists(), "seed table exists; no retry")
    previous = ROOT / "runs/m2-stochastic-continuation-20261018-v3"
    prior = read(previous / "continuation_manifest.json")
    by_episode = {}
    for cell in prior["cells"]:
        if cell["decision_session_index"] == 0:
            episode = cell["source_group"]["source_episode_id"]
            by_episode.setdefault(episode, cell)
    require(len(by_episode) == 8, "exactly the eight previously fixed Train windows")
    parents = []
    anchors = []
    heads = []
    potential = []
    for index, (episode, cell) in enumerate(sorted(by_episode.items())):
        parent_id = f"p{index:03d}"
        parents.append(
            {
                "parent_id": parent_id,
                "trajectory_key": f"parent/{parent_id}",
                "episode_id": episode,
                "record_index": cell["record_index"],
                "source_group": cell["source_group"],
                "initial_financial_state_sha256": cell["financial_state_sha256"],
                "initial_observation_messages_sha256": cell["observation_messages_sha256"],
                "initial_prompt_sha256": cell["historical_prompt_sha256"],
                "terminal_session_index": 20,
            }
        )
        for j in (5, 15):
            aid = f"{parent_id}/j{j}"
            anchors.append(
                {
                    "anchor_id": aid,
                    "parent_id": parent_id,
                    "episode_id": episode,
                    "decision_session_index": j,
                    "record_index": cell["record_index"],
                }
            )
            for slot in range(4):
                head_key = f"head/{aid}/s{slot}"
                heads.append(
                    {
                        "head_key": head_key,
                        "trajectory_key": head_key,
                        "anchor_id": aid,
                        "parent_id": parent_id,
                        "slot_id": slot,
                        "decision_session_index": j,
                    }
                )
                for role, nrep, prefix in (("estimate", 2, "est"), ("reference", 8, "ref")):
                    for rep in range(nrep):
                        potential.append(
                            {
                                "trajectory_key": f"{prefix}/{aid}/s{slot}/r{rep}",
                                "role": role,
                                "anchor_id": aid,
                                "parent_id": parent_id,
                                "head_key": head_key,
                                "slot_id": slot,
                                "replicate": rep,
                                "decision_session_index": j,
                                "remaining_sessions": 19 - j,
                            }
                        )
    seeds = []
    for parent in parents:
        for session in range(20):
            for attempt in range(4):
                key = parent["trajectory_key"]
                seeds.append(
                    {
                        "request_key": request_key(key, session, attempt),
                        "trajectory_key": key,
                        "role": "parent",
                        "decision_session_index": session,
                        "attempt": attempt,
                        "request_seed": request_seed(key, session, attempt),
                    }
                )
    for head in heads:
        key = head["trajectory_key"]
        session = head["decision_session_index"]
        seeds.append(
            {
                "request_key": request_key(key, session, 0),
                "trajectory_key": key,
                "role": "head",
                "decision_session_index": session,
                "attempt": 0,
                "request_seed": request_seed(key, session, 0),
            }
        )
    for trajectory in potential:
        key = trajectory["trajectory_key"]
        for session in range(trajectory["decision_session_index"] + 1, 20):
            for attempt in range(4):
                seeds.append(
                    {
                        "request_key": request_key(key, session, attempt),
                        "trajectory_key": key,
                        "role": trajectory["role"],
                        "decision_session_index": session,
                        "attempt": attempt,
                        "request_seed": request_seed(key, session, attempt),
                    }
                )
    budgets = dict(Counter(s["role"] for s in seeds))
    require(
        budgets == {"parent": 640, "head": 64, "estimate": 4608, "reference": 18432},
        "fixed stage budgets",
    )
    require(
        len(seeds) == MAX_REQUESTS and len({s["request_seed"] for s in seeds}) == MAX_REQUESTS,
        "23744 unique fixed potential seeds",
    )
    with (output / "request_seed_table.jsonl").open("x") as stream:
        for item in seeds:
            append(stream, item)
    sources = [
        previous / "continuation_manifest.json",
        MENU / "source_manifest.json",
        SOURCE / "contract.json",
        SOURCE / "model_preflight_receipt.json",
    ]
    manifest = {
        "profile": PROFILE,
        "status": "PREPARED_NOT_STARTED",
        "prepared_at_UTC": datetime.now(timezone.utc).isoformat(),
        "model_path": prior["model_path"],
        "policy_snapshot_sha256": prior["policy_snapshot_sha256"],
        "source_sampling_run": str(SOURCE),
        "source_menu_run": str(MENU),
        "fixed_window_source": str(previous),
        "source_file_sha256": {str(p): sha(p) for p in sources},
        "request_seed_table_sha256": sha(output / "request_seed_table.jsonl"),
        "parents": parents,
        "anchors": anchors,
        "head_slots": heads,
        "potential_trajectories": potential,
        "expected_parents": 8,
        "planned_anchors": 16,
        "planned_head_slots": 64,
        "maximum_estimate_paths": 128,
        "maximum_reference_paths": 512,
        "max_model_requests": MAX_REQUESTS,
        "stage_request_limits": budgets,
        "max_attempts_per_session": 4,
        "head_requests_per_slot": 1,
        "max_active_trajectories": 32,
        "terminal_session_index": 20,
        "anchor_sessions": [5, 15],
        "anchor_capture": "first entry before any request at session",
        "retain_reached_anchors_if_parent_later_fails": True,
        "reference_share_rule": "same anchor/economic order only when post-first "
        "financial state and full messages identical",
        "conditional_state_mismatch": "preserve heads; whole anchor unsupported/all methods NA; "
        "no REF or replacement",
        "no_extra_hold_reference": True,
        "seed_namespaces": ["PARENT", "HEAD", "EST", "REF"],
        "optimizer_updates": 0,
        "formal_valid_oos_access": False,
        "automatic_retry": False,
        "fallback_hold": False,
        "state_outcome_filtering": False,
        "continuation_policy": "same frozen common recovered SFT; T1/top_p1/native/free tool calls",
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
    require(manifest["profile"] == PROFILE, "onpolicy profile")
    require(
        sha(output / "request_seed_table.jsonl") == manifest["request_seed_table_sha256"],
        "seed table SHA",
    )
    for path, digest in manifest["source_file_sha256"].items():
        require(sha(Path(path)) == digest, "source input changed")
    seed_table = {r["request_key"]: r for r in rows(output / "request_seed_table.jsonl")}
    require(len(seed_table) == MAX_REQUESTS, "all fixed potential requests")
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model_path"], local_files_only=True, trust_remote_code=False
    )
    replay = Replay(read(MENU / "source_manifest.json"))
    install_fast_adapter(replay, output)
    parent_specs = {p["parent_id"]: p for p in manifest["parents"]}
    anchor_specs = {a["anchor_id"]: a for a in manifest["anchors"]}
    snapshots = {}
    head_records = []
    anchor_records = []
    parent_records = []
    branch_records = []
    model_calls = 0
    batch_id = 0
    role_calls = Counter()
    fixed_first_actions = 0
    logs = {
        n: (output / n).open("x")
        for n in (
            "driver_requests.jsonl",
            "driver_evaluations.jsonl",
            "parent_results.jsonl",
            "anchor_snapshots.jsonl",
            "head_results.jsonl",
            "trajectory_results.jsonl",
        )
    }

    def ids(env, obs):
        return tokenizer.apply_chat_template(
            apply_screening_action_protocol(obs.messages, PROTOCOL),
            tools=env.tool_schemas,
            tokenize=True,
            **TEMPLATE_KWARGS,
        )

    def finance(env, config):
        return json.loads(canonical(financial_state(env, config)))

    def snapshot(state):
        session = state["env"].current_session_index
        if session not in (5, 15):
            return
        aid = f"{state['spec']['parent_id']}/j{session}"
        if aid in snapshots:
            return
        prompt = ids(state["env"], state["obs"])
        financial = finance(state["env"], state["config_sha"])
        record = {
            **anchor_specs[aid],
            "status": "CAPTURED",
            "prefix": list(state["history"]),
            "prefix_requests": len(state["history"]),
            "financial_state": financial,
            "financial_state_sha256": object_sha(financial),
            "observation_messages": state["obs"].messages,
            "observation_messages_sha256": object_sha(state["obs"].messages),
            "prompt_token_ids": prompt,
            "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode("ascii")).hexdigest(),
            "policy_snapshot_sha256": manifest["policy_snapshot_sha256"],
            "captured_before_first_request_at_session": True,
        }
        snapshots[aid] = record
        append(logs["anchor_snapshots.jsonl"], record)

    def terminal(state, status, reason=None):
        env = state["env"]
        metrics = env.screening_terminal_metrics() if status == "COMPLETE" else None
        score = metrics["terminal_score"] if metrics else None
        if status == "COMPLETE":
            require(
                env.episode_complete and env.current_session_index == 20 and math.isfinite(score),
                "original finite terminal",
            )
        return {
            **state["spec"],
            "status": status,
            "failure_reason": reason,
            "Y": score,
            "terminal_metrics": metrics,
            "model_calls": state["model_calls"],
            "session_advances": state["session_advances"],
            "last_session_index": env.current_session_index,
            "policy_snapshot_sha256": manifest["policy_snapshot_sha256"],
        }

    async def clone_anchor(aid):
        snap = snapshots[aid]
        parent = parent_specs[snap["parent_id"]]
        env, obs, config = await replay.prefix(parent["record_index"], 0)
        for item in snap["prefix"]:
            obs, _, _, info = await env.step(
                Action(text=item["response_text"], token_ids=item["response_token_ids"])
            )
            replay.requests += 1
            require(
                json.loads(canonical(info)) == item["info"]
                and obs.messages == item["observation_messages"],
                "new parent full-prefix replay exact",
            )
            require(
                asdict(env.credit_step_trace[-1]) == item["trace"], "new parent credit trace exact"
            )
        require(
            finance(env, config) == snap["financial_state"]
            and obs.messages == snap["observation_messages"],
            "fresh parent anchor finance/full messages exact",
        )
        require(ids(env, obs) == snap["prompt_token_ids"], "fresh parent anchor prompt IDs exact")
        return env, obs, config

    def state_for(spec, env, obs, config):
        return {
            "spec": spec,
            "env": env,
            "obs": obs,
            "config_sha": config,
            "attempt": 0,
            "model_calls": 0,
            "session_advances": 0,
            "history": [],
        }

    with (
        os.fdopen(request_fd, "w", buffering=1) as request_pipe,
        os.fdopen(response_fd, "r", buffering=1) as response_pipe,
    ):

        def receive():
            line = response_pipe.readline()
            require(bool(line), "GPU RPC EOF")
            message = json.loads(line)
            require(message.get("event") != "ERROR", f"GPU error: {message}")
            return message

        def rpc(states):
            nonlocal model_calls, batch_id
            pending = []
            out = []
            for state in states:
                spec = state["spec"]
                prompt = ids(state["env"], state["obs"])
                if len(prompt) > 8192:
                    out.append((state, {"driver_status": "PROMPT_LIMIT"}))
                    continue
                session = state["env"].current_session_index
                attempt = state["attempt"]
                key = request_key(spec["trajectory_key"], session, attempt)
                expected = seed_table[key]
                seed = request_seed(spec["trajectory_key"], session, attempt)
                require(
                    seed == expected["request_seed"] and expected["role"] == spec["role"],
                    "fixed per-role request seed",
                )
                request = {
                    "request_key": key,
                    "trajectory_key": spec["trajectory_key"],
                    "role": spec["role"],
                    "parent_id": spec["parent_id"],
                    "anchor_id": spec.get("anchor_id"),
                    "cell_id": spec.get("cell_id"),
                    "head_key": spec.get("head_key"),
                    "slot_id": spec.get("slot_id"),
                    "replicate": spec.get("replicate"),
                    "decision_session_index": session,
                    "attempt": attempt,
                    "request_seed": seed,
                    "prompt_token_ids": prompt,
                    "prompt_token_count": len(prompt),
                    "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode("ascii")).hexdigest(),
                    "max_tokens": 4096,
                }
                append(logs["driver_requests.jsonl"], request)
                pending.append((state, request))
            if pending:
                require(
                    len(pending) <= 32 and model_calls + len(pending) <= MAX_REQUESTS,
                    "native batch/global budget",
                )
                append(
                    request_pipe,
                    {"op": "generate", "batch_id": batch_id, "requests": [q for _, q in pending]},
                )
                message = receive()
                require(
                    message.get("event") == "BATCH" and message["batch_id"] == batch_id,
                    "exact RPC batch",
                )
                require(len(message["records"]) == len(pending), "exact native result count")
                for (state, request), record in zip(pending, message["records"], strict=True):
                    require(
                        all(record[k] == v for k, v in request.items())
                        and record["returned_prompt_token_ids"] == request["prompt_token_ids"],
                        "native identity/prompt mapping",
                    )
                    require(
                        (record["generation_status"] == "COMPLETE") is generation_complete(record),
                        "native EOS/finish integrity",
                    )
                    state["model_calls"] += 1
                    model_calls += 1
                    role_calls[state["spec"]["role"]] += 1
                    require(
                        role_calls[state["spec"]["role"]]
                        <= manifest["stage_request_limits"][state["spec"]["role"]],
                        "per-role budget",
                    )
                    out.append((state, record))
                batch_id += 1
            return out

        async def apply(state, record, remember=False):
            before = state["env"].current_session_index
            evaluation = {
                k: record[k]
                for k in (
                    "request_key",
                    "trajectory_key",
                    "role",
                    "parent_id",
                    "anchor_id",
                    "cell_id",
                    "head_key",
                    "slot_id",
                    "replicate",
                    "decision_session_index",
                    "attempt",
                    "request_seed",
                    "generation_status",
                    "response_text",
                    "response_token_ids",
                )
            }
            evaluation.update(
                session_before=before,
                session_after=before,
                env_status="NOT_APPLIED",
                info=None,
                trace=None,
            )
            if record["generation_status"] != "COMPLETE":
                append(logs["driver_evaluations.jsonl"], evaluation)
                return None
            obs, _, _, info = await state["env"].step(
                Action(text=record["response_text"], token_ids=record["response_token_ids"])
            )
            replay.requests += 1
            state["obs"] = obs
            after = state["env"].current_session_index
            require(after in (before, before + 1), "at most one session advance")
            status = (
                "INVALID"
                if info.get("invalid_action")
                else ("ADVANCING" if after == before + 1 else "NONADVANCING")
            )
            trace = asdict(state["env"].credit_step_trace[-1])
            normalized = json.loads(canonical(info))
            evaluation.update(
                session_after=after,
                env_status=status,
                info=normalized,
                trace=trace,
                observation_messages=obs.messages,
            )
            append(logs["driver_evaluations.jsonl"], evaluation)
            if remember:
                state["history"].append(
                    {
                        "response_text": record["response_text"],
                        "response_token_ids": record["response_token_ids"],
                        "info": normalized,
                        "trace": trace,
                        "observation_messages": obs.messages,
                    }
                )
            if after == before + 1:
                require(not info.get("invalid_action"), "invalid cannot advance")
                state["session_advances"] += 1
                state["attempt"] = 0
            else:
                state["attempt"] += 1
            return evaluation

        # Fresh parents start at reset: no historical M2 policy interaction is replayed.
        parents = []
        for p in manifest["parents"]:
            env, obs, config = await replay.prefix(p["record_index"], 0)
            require(
                object_sha(finance(env, config)) == p["initial_financial_state_sha256"]
                and object_sha(obs.messages) == p["initial_observation_messages_sha256"],
                "fixed window initial identity",
            )
            require(
                hashlib.sha256(json.dumps(ids(env, obs)).encode("ascii")).hexdigest()
                == p["initial_prompt_sha256"],
                "fixed window initial tokens",
            )
            parents.append(state_for({**p, "role": "parent"}, env, obs, config))
        require(receive().get("event") == "READY", "GPU READY")
        while parents:
            for state, record in rpc(parents):
                if record.get("driver_status") == "PROMPT_LIMIT":
                    status = "PROMPT_LIMIT"
                elif await apply(state, record, True) is None:
                    status = "INCOMPLETE"
                else:
                    snapshot(state)
                    status = (
                        "COMPLETE"
                        if state["env"].episode_complete
                        else ("REQUEST_LIMIT" if state["attempt"] >= 4 else None)
                    )
                if status:
                    row = terminal(state, status)
                    row["reached_anchor_ids"] = [
                        a
                        for a in snapshots
                        if snapshots[a]["parent_id"] == state["spec"]["parent_id"]
                    ]
                    parent_records.append(row)
                    append(logs["parent_results.jsonl"], row)
                    parents.remove(state)
        print(
            "PARENTS_COMPLETE",
            len(parent_records),
            "anchors",
            len(snapshots),
            "requests",
            model_calls,
            flush=True,
        )
        # Every planned head gets exactly one first request or explicit NOT_RUN.
        for offset in range(0, len(manifest["head_slots"]), 32):
            states = []
            for spec in manifest["head_slots"][offset : offset + 32]:
                if spec["anchor_id"] not in snapshots:
                    head_records.append(
                        {
                            **spec,
                            "role": "head",
                            "status": "NOT_RUN_ANCHOR",
                            "legal_head": False,
                            "economic_order_key": None,
                            "cell_id": None,
                            "model_calls": 0,
                        }
                    )
                    continue
                env, obs, config = await clone_anchor(spec["anchor_id"])
                states.append(state_for({**spec, "role": "head"}, env, obs, config))
            for state, record in rpc(states):
                row = {
                    **state["spec"],
                    "model_calls": state["model_calls"],
                    "legal_head": False,
                    "cell_id": None,
                    "economic_order_key": None,
                    "first_action_text": record.get("response_text"),
                    "response_token_ids": record.get("response_token_ids"),
                    "first_info": None,
                    "first_state_after": None,
                    "first_observation_messages": None,
                }
                if record.get("driver_status") == "PROMPT_LIMIT":
                    row["status"] = "PROMPT_LIMIT"
                else:
                    evaluation = await apply(state, record)
                    if evaluation is None:
                        row["status"] = "INCOMPLETE"
                    else:
                        row.update(
                            status=evaluation["env_status"],
                            first_info=evaluation["info"],
                            first_trace=evaluation["trace"],
                            first_state_after=finance(state["env"], state["config_sha"]),
                            first_observation_messages=state["obs"].messages,
                        )
                        if evaluation["env_status"] == "ADVANCING":
                            _, calls = state["env"].format_wrapper.parse_response(
                                record["response_text"]
                            )
                            require(len(calls) == 1, "legal advancing single API call")
                            call = calls[0]
                            family = evaluation["trace"]["action"]
                            key = (
                                [
                                    family,
                                    call.arguments["symbol"].strip().upper(),
                                    float(call.arguments["target_weight"]),
                                ]
                                if family in ("buy", "sell")
                                else [family]
                            )
                            require(
                                family in ("buy", "sell", "hold", "abstain"), "advancing family"
                            )
                            row.update(
                                legal_head=True,
                                economic_order_key=json.dumps(key, allow_nan=False),
                                family=family,
                            )
                head_records.append(row)
        require(len(head_records) == 64, "all64 head slots classified")
        heads = {h["head_key"]: h for h in head_records}
        cells = []
        branches = []
        potential = {
            (t["role"], t["head_key"], t["replicate"]): t
            for t in manifest["potential_trajectories"]
        }
        for aid, spec in anchor_specs.items():
            hs = sorted(
                [h for h in head_records if h["anchor_id"] == aid], key=lambda h: h["slot_id"]
            )
            groups = defaultdict(list)
            for h in hs:
                if h["legal_head"]:
                    groups[h["economic_order_key"]].append(h)
            mismatch = any(
                len(
                    {
                        (
                            object_sha(h["first_state_after"]),
                            object_sha(h["first_observation_messages"]),
                        )
                        for h in members
                    }
                )
                > 1
                for members in groups.values()
            )
            status = (
                "MISSING_PARENT_STATE"
                if aid not in snapshots
                else ("CONDITIONAL_STATE_MISMATCH" if mismatch else "CAPTURED")
            )
            ar = {
                **spec,
                "status": status,
                "conditional_state_match": None if aid not in snapshots else not mismatch,
                "legal_heads": sum(h["legal_head"] for h in hs),
                "parent_final_status": next(
                    p["status"] for p in parent_records if p["parent_id"] == spec["parent_id"]
                ),
            }
            anchor_records.append(ar)
            if status != "CAPTURED":
                continue
            for key, members in groups.items():
                representative = min(members, key=lambda h: h["slot_id"])
                cell_id = f"{aid}/s{representative['slot_id']}"
                cell = {
                    "cell_id": cell_id,
                    "anchor_id": aid,
                    "parent_id": spec["parent_id"],
                    "economic_order_key": key,
                    "representative_head_key": representative["head_key"],
                    "source_slot": representative["slot_id"],
                    "post_first_financial_state_sha256": object_sha(
                        representative["first_state_after"]
                    ),
                    "post_first_observation_messages_sha256": object_sha(
                        representative["first_observation_messages"]
                    ),
                }
                cells.append(cell)
                for head in members:
                    head["cell_id"] = cell_id
                    for rep in range(2):
                        branches.append(
                            {**potential["estimate", head["head_key"], rep], "cell_id": cell_id}
                        )
                for rep in range(8):
                    branches.append(
                        {
                            **potential["reference", representative["head_key"], rep],
                            "cell_id": cell_id,
                        }
                    )
        for row in sorted(head_records, key=lambda h: h["head_key"]):
            append(logs["head_results.jsonl"], row)
        with (output / "anchor_results.jsonl").open("x") as f:
            for row in anchor_records:
                append(f, row)
        branch_manifest = {
            "status": "PREPARED_AFTER_FRESH_HEADS_BEFORE_CONTINUATIONS",
            "cells": cells,
            "trajectories": branches,
            "anchors": anchor_records,
            "actual_estimate_paths": sum(t["role"] == "estimate" for t in branches),
            "actual_reference_paths": sum(t["role"] == "reference" for t in branches),
            "potential_paths_not_started": 640 - len(branches),
            "whole_anchor_mismatch_skipped": sum(
                a["status"] == "CONDITIONAL_STATE_MISMATCH" for a in anchor_records
            ),
            "head_results_sha256": sha(output / "head_results.jsonl"),
        }
        write_new(output / "branch_manifest.json", branch_manifest)
        print(
            "HEADS_COMPLETE",
            64,
            "legal",
            sum(h["legal_head"] for h in head_records),
            "cells",
            len(cells),
            "branches",
            len(branches),
            flush=True,
        )
        queue = iter(branches)
        active = []
        cells_by_id = {c["cell_id"]: c for c in cells}

        async def refill():
            nonlocal fixed_first_actions
            while len(active) < 32:
                spec = next(queue, None)
                if spec is None:
                    break
                env, obs, config = await clone_anchor(spec["anchor_id"])
                head = heads[spec["head_key"]]
                obs, _, _, info = await env.step(
                    Action(text=head["first_action_text"], token_ids=head["response_token_ids"])
                )
                replay.requests += 1
                fixed_first_actions += 1
                require(
                    not info.get("invalid_action")
                    and env.current_session_index == spec["decision_session_index"] + 1,
                    "actual head replay legal",
                )
                require(
                    json.loads(canonical(info)) == head["first_info"]
                    and finance(env, config) == head["first_state_after"]
                    and obs.messages == head["first_observation_messages"],
                    "actual head replay info/financial/full messages exact",
                )
                cell = cells_by_id[spec["cell_id"]]
                require(
                    object_sha(finance(env, config)) == cell["post_first_financial_state_sha256"]
                    and object_sha(obs.messages) == cell["post_first_observation_messages_sha256"],
                    "shared REF exact conditional state",
                )
                active.append(state_for(spec, env, obs, config))

        await refill()
        while active:
            for state, record in rpc(active):
                if record.get("driver_status") == "PROMPT_LIMIT":
                    status = "PROMPT_LIMIT"
                elif await apply(state, record) is None:
                    status = "INCOMPLETE"
                else:
                    status = (
                        "COMPLETE"
                        if state["env"].episode_complete
                        else ("REQUEST_LIMIT" if state["attempt"] >= 4 else None)
                    )
                if status:
                    row = terminal(state, status)
                    branch_records.append(row)
                    append(logs["trajectory_results.jsonl"], row)
                    active.remove(state)
            if batch_id % 10 == 0:
                print(
                    "BRANCHES",
                    len(branch_records),
                    "/",
                    len(branches),
                    "requests",
                    model_calls,
                    flush=True,
                )
            await refill()
        require(len(branch_records) == len(branches), "all actual branches classified")
        append(request_pipe, {"op": "close"})
        closed = receive()
        require(
            closed.get("event") == "CLOSED" and closed["model_requests"] == model_calls,
            "worker close accounting",
        )
    for stream in logs.values():
        stream.close()
    names = [
        *logs,
        "anchor_results.jsonl",
        "branch_manifest.json",
        "fast_preprocessing_receipt.json",
    ]
    write_new(
        output / "driver_completion.json",
        {
            "status": "COMPLETE",
            "finished_at_UTC": datetime.now(timezone.utc).isoformat(),
            "parents_classified": len(parent_records),
            "anchors_planned": 16,
            "anchors_captured": len(snapshots),
            "head_slots_classified": len(head_records),
            "legal_head_slots": sum(h["legal_head"] for h in head_records),
            "trajectories_classified": len(branch_records),
            "trajectory_status_counts": dict(Counter(r["status"] for r in branch_records)),
            "model_requests": model_calls,
            "model_requests_by_role": dict(role_calls),
            "request_limit": MAX_REQUESTS,
            "environment_requests": replay.requests,
            "fixed_first_actions": fixed_first_actions,
            "environments_constructed": replay.environments,
            "optimizer_updates": 0,
            "formal_valid_oos_access": False,
            "automatic_retry": False,
            "continuation_manifest_sha256": sha(output / "continuation_manifest.json"),
            "output_sha256": {name: sha(output / name) for name in names},
        },
    )
    print(
        canonical(
            {"status": "COMPLETE", "model_requests": model_calls, "branches": len(branch_records)}
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
        m = prepare_manifest(args.output)
        print(
            canonical(
                {
                    "status": "PREPARED",
                    "parents": len(m["parents"]),
                    "anchors": len(m["anchors"]),
                    "request_limit": MAX_REQUESTS,
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

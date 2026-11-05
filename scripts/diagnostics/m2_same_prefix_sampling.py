#!/usr/bin/env python3
"""Sealed first-request generation and separate CPU labels for fixed Train inputs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/agent-r1"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
SOURCE = ROOT / "runs/m2-same-prefix-menu-20261018-v1"
DEFAULT = ROOT / "runs/m2-same-prefix-sampling-20261018-v1"
PLAN = ROOT / "docs/2026-10-18_M2_SAME_PREFIX_SAMPLING_PLAN.md"
READER = ROOT / "scripts/diagnostics/m2_same_prefix_sampling_readout.py"
EOS = (151645, 151643)
GIB = 1024**3
MAX_RESPONSE = 4096


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, default=lambda v: v.isoformat())


def object_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def write_new(path: Path, value: Any) -> None:
    with path.open("x") as f:
        f.write(canonical(value) + "\n")


def append(f: Any, value: Any) -> None:
    f.write(canonical(value) + "\n")
    f.flush()
    os.fsync(f.fileno())


def live(out: Path, **fields: Any) -> None:
    tmp = out / "live_status.tmp"
    tmp.write_text(canonical({"updated_at_UTC": datetime.now(timezone.utc), **fields}) + "\n")
    tmp.replace(out / "live_status.json")


def verify(out: Path) -> dict[str, Any]:
    c = read(out / "contract.json")
    for path, digest in c["source_sha256"].items():
        assert sha(Path(path)) == digest, f"sealed source changed: {path}"
    assert sha(out / "proposal.json") == c["proposal_sha256"]
    assert sha(out / "inputs.jsonl") == c["inputs_sha256"]
    return c


def safe_metadata(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return {
            "__nonfinite_float__": "NaN"
            if math.isnan(value)
            else ("Infinity" if value > 0 else "-Infinity")
        }
    if isinstance(value, list):
        return [safe_metadata(v) for v in value]
    if isinstance(value, dict):
        return {k: safe_metadata(v) for k, v in value.items()}
    return value


def generation_integrity(raw: dict[str, Any]) -> tuple[str, str | None]:
    tokens = raw["response_token_ids"]
    if not tokens:
        return "INCOMPLETE", "EMPTY_RESPONSE"
    if len(tokens) >= MAX_RESPONSE:
        return "INCOMPLETE", "RESPONSE_TOKEN_CAP"
    if raw["returned_finished"] is not True or raw["finish_reason"] != "stop":
        return "INCOMPLETE", "UNKNOWN_OR_NONSTOP_FINISH"
    if tokens[-1] not in EOS or any(t in EOS for t in tokens[:-1]):
        return "INCOMPLETE", "EOS_IDENTITY_INCONSISTENT"
    stop = raw["stop_reason"]
    if stop is not None and (type(stop) is not int or stop != tokens[-1]):
        return "INCOMPLETE", "STOP_REASON_INCONSISTENT"
    return "COMPLETE", None


def prepare(out: Path, model_receipt: Path, holder: Path) -> None:
    from src.agents.gpu_memory_coordinator import PHASES, config_for_holder, freeze_config
    from src.agents.order_recovery_probe import engine_settings

    proposal = read(SOURCE / "sampling_proposal.json")
    assert proposal["status"] == "PREPARED_NOT_STARTED" and len(proposal["groups"]) == 32
    assert read(SOURCE / "readout.json")["status"] == "PASS"
    previous = read(SOURCE / "contract.json")
    inventory = dict(previous["source_sha256"])
    for name, digest in proposal["source_files_sha256"].items():
        assert sha(SOURCE / name) == digest
        inventory[str(SOURCE / name)] = digest
    for name in ("sampling_proposal.json", "readout.json", "final_verification.json"):
        inventory[str(SOURCE / name)] = sha(SOURCE / name)
    model = Path(proposal["intended_model_path"])
    merge = read(model / "merge_receipt.json")
    assert sha(model / "merge_receipt.json") == proposal["merge_receipt_sha256"]
    assert merge["status"] == "COMPLETE"
    for name, item in merge["files"].items():
        path = model / name
        assert path.is_file() and not path.is_symlink() and path.stat().st_size == item["bytes"]
        assert sha(path) == item["sha256"], f"model differs: {name}"
        inventory[str(path)] = item["sha256"]
    inventory[str(model / "merge_receipt.json")] = sha(model / "merge_receipt.json")
    model_check = read(model_receipt)
    for path, digest in model_check["relevant_source_sha256"].items():
        assert sha(Path(path)) == digest
        inventory[path] = digest
    files = [Path(__file__).resolve(), READER, PLAN]
    files += [
        ROOT / "src/agents" / name
        for name in (
            "gpu_memory_coordinator.py",
            "gpu_reservation_service.py",
            "order_recovery_controller.py",
            "gpu_handoff.py",
            "launcher.py",
            "order_recovery_probe.py",
        )
    ]
    for path in files:
        inventory[str(path)] = sha(path)
    packets = {v["anchor_id"]: v for v in rows(SOURCE / "prompt_packets.jsonl")}
    requests = []
    for group in proposal["groups"]:
        packet = packets[group["anchor_id"]]
        assert packet["historical_prompt_sha256"] == group["historical_prompt_sha256"]
        assert len(packet["prompt_token_ids"]) == group["prompt_token_count"] <= 8192
        for slot in group["slots"]:
            requests.append(
                {
                    "request_key": f"{group['anchor_id']}/s{slot['slot']}",
                    "anchor_id": group["anchor_id"],
                    "slot": slot["slot"],
                    "request_seed": slot["request_seed"],
                    "prompt_token_ids": packet["prompt_token_ids"],
                    "prompt_sha256": slot["prompt_sha256"],
                    "prompt_token_count": group["prompt_token_count"],
                    "source_group": {
                        k: group[k]
                        for k in (
                            "source_seed",
                            "source_sibling_id",
                            "source_episode_id",
                            "source_policy_step",
                            "decision_session_index",
                        )
                    },
                }
            )
    assert len(requests) == 128 and len({v["request_seed"] for v in requests}) == 128
    assert len({v["request_key"] for v in requests}) == 128
    assert not out.exists(), "new execution directory required"
    out.mkdir(parents=True)
    with (out / "proposal.json").open("xb") as f:
        f.write((SOURCE / "sampling_proposal.json").read_bytes())
    with (out / "model_preflight_receipt.json").open("xb") as f:
        f.write(model_receipt.read_bytes())
    inventory[str(out / "model_preflight_receipt.json")] = sha(out / "model_preflight_receipt.json")
    with (out / "inputs.jsonl").open("x") as f:
        for item in requests:
            append(f, item)
    cfg = config_for_holder(
        holder,
        lease_token=f"same-prefix-{out.name}",
        evidence_directory=out / "gpu_handoff",
        portfolio_target_bytes=76 * GIB,
        phase_peak_bytes={p: 40 * GIB for p in PHASES},
        minimum_free_bytes=2 * GIB,
        wait_timeout_seconds=120,
    )
    config_path = out / "gpu_handoff/config.json"
    freeze_config(config_path, cfg)
    write_new(
        out / "contract.json",
        {
            "profile": "same_prefix_native_first_request_v1",
            "status": "SEALED_BEFORE_GENERATION",
            "sealed_at_UTC": datetime.now(timezone.utc),
            "source_menu_run": str(SOURCE),
            "source_sha256": inventory,
            "proposal_sha256": sha(out / "proposal.json"),
            "inputs_sha256": sha(out / "inputs.jsonl"),
            "model_path": str(model),
            "engine": engine_settings(model),
            "batch_size": 8,
            "expected_requests": 128,
            "sampling_params": {
                **proposal["sampling_params"],
                "top_k": -1,
                "n": 1,
                "ignore_eos": False,
            },
            "eos_token_ids": EOS,
            "skip_special_tokens": True,
            "resource_config_sha256": sha(config_path),
            "gpu_outer_limit_bytes": 95 * GIB,
            "gpu_minimum_free_bytes": 2 * GIB,
            "model_requests_limit": 128,
            "followup_model_requests": 0,
            "optimizer_updates": 0,
            "formal_valid_oos_access": False,
            "automatic_retry": False,
            "versions": {
                n: importlib.metadata.version(n)
                for n in (
                    "torch",
                    "vllm",
                    "transformers",
                    "tokenizers",
                    "ray",
                    "pyarrow",
                    "numpy",
                    "flash-attn",
                )
            },
        },
    )
    live(out, status="PREPARED", phase="SEALED_NOT_STARTED", model_requests=0)
    print("SEALED", out / "contract.json", flush=True)


def worker(out: Path) -> None:
    c = verify(out)
    for name in ("USE_TORCH", "USE_TF", "USE_FLAX"):
        os.environ.pop(name, None)
    os.environ.update(
        USE_TORCH="1",
        USE_TF="0",
        USE_FLAX="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_USE_V1="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        TOKENIZERS_PARALLELISM="false",
    )
    assert not any(
        k.startswith(("FINANCE_SCREENING_", "FINANCE_LEARNING_", "FINANCE_CRITIC_"))
        for k in os.environ
    )
    from src.agents.launcher import _resolve_c_compiler

    compiler, version = _resolve_c_compiler(parent=dict(os.environ))
    os.environ["CC"] = compiler
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        c["model_path"], local_files_only=True, trust_remote_code=False
    )
    assert tokenizer.eos_token_id == 151645
    write_new(
        out / "worker_config.json",
        {
            "engine": c["engine"],
            "compiler": {"CC": compiler, "version": version},
            "contract_sha256": sha(out / "contract.json"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "skip_special_tokens": True,
            "versions": c["versions"],
        },
    )
    live(out, status="RUNNING", phase="MODEL_LOADING", model_requests=0)
    llm = LLM(**c["engine"])
    inputs = rows(out / "inputs.jsonl")
    ids_seen: set[int] = set()
    completed = 0
    with (
        (out / "submissions.jsonl").open("x") as sh,
        (out / "raw_generations.jsonl").open("x") as rh,
        (out / "generations.jsonl").open("x") as gh,
    ):
        for start in range(0, len(inputs), 8):
            batch = inputs[start : start + 8]
            params = [{**c["sampling_params"], "seed": v["request_seed"]} for v in batch]
            objects = [SamplingParams(**v) for v in params]
            assert all(
                {k: getattr(obj, k) for k in expected} == expected
                for obj, expected in zip(objects, params, strict=True)
            )
            append(
                sh,
                {
                    "batch_index": start // 8,
                    "requests": [
                        {**v, "actual_sampling_params": q}
                        for v, q in zip(batch, params, strict=True)
                    ],
                },
            )
            outputs = llm.generate(
                [{"prompt_token_ids": v["prompt_token_ids"]} for v in batch],
                objects,
                use_tqdm=False,
            )
            raw = [
                {
                    "request_id": v.request_id,
                    "finished": v.finished,
                    "prompt_token_ids": list(v.prompt_token_ids),
                    "outputs": [
                        {
                            "token_ids": list(o.token_ids),
                            "engine_text": o.text,
                            "finish_reason": o.finish_reason,
                            "stop_reason": o.stop_reason,
                        }
                        for o in v.outputs
                    ],
                }
                for v in outputs
            ]
            append(rh, {"batch_index": start // 8, "outputs": raw})
            assert len(raw) == len(batch)
            request_ids = [int(v["request_id"]) for v in raw]
            assert request_ids == list(range(start, start + len(batch)))
            assert request_ids == sorted(request_ids) and len(set(request_ids)) == len(batch)
            assert not ids_seen.intersection(request_ids)
            if ids_seen:
                assert min(request_ids) > max(ids_seen)
            ids_seen.update(request_ids)
            for inp, output, actual_params in zip(batch, raw, params, strict=True):
                assert (
                    output["prompt_token_ids"] == inp["prompt_token_ids"]
                    and len(output["outputs"]) == 1
                )
                answer = output["outputs"][0]
                tokens = answer["token_ids"]
                record = {
                    **inp,
                    "response_text": tokenizer.decode(tokens, skip_special_tokens=True),
                    "engine_text": answer["engine_text"],
                    "response_token_ids": tokens,
                    "response_token_count": len(tokens),
                    "response_last_token_id": tokens[-1] if tokens else None,
                    "primary_eos_token_id": tokenizer.eos_token_id,
                    "eos_token_ids": EOS,
                    "eos_positions": [i for i, t in enumerate(tokens) if t in EOS],
                    "last_token_is_eos": bool(tokens) and tokens[-1] in EOS,
                    "returned_finished": output["finished"],
                    "returned_prompt_token_ids": output["prompt_token_ids"],
                    "finish_reason": answer["finish_reason"],
                    "stop_reason": answer["stop_reason"],
                    "vllm_request_id": output["request_id"],
                    "actual_sampling_params": actual_params,
                }
                record["generation_status"], record["failure_reason"] = generation_integrity(record)
                append(gh, record)
                completed += 1
            live(out, status="RUNNING", phase="NATIVE_FIRST_REQUESTS", model_requests=completed)
            print("generated", completed, "/128", flush=True)
    assert completed == 128
    verify(out)
    write_new(
        out / "generation_completion.json",
        {
            "status": "COMPLETE",
            "finished_at_UTC": datetime.now(timezone.utc),
            "model_requests": 128,
            "followup_model_requests": 0,
            "optimizer_updates": 0,
            "output_sha256": {
                name: sha(out / name)
                for name in (
                    "generations.jsonl",
                    "raw_generations.jsonl",
                    "submissions.jsonl",
                    "worker_config.json",
                )
            },
        },
    )
    live(out, status="GENERATION_COMPLETE", phase="AWAITING_CPU_LABELS", model_requests=128)


def run_controller(out: Path) -> None:
    from src.agents.gpu_memory_coordinator import (
        _driver_snapshot,
        controller_idle,
        load_config,
        reservation_status,
    )
    from src.agents.gpu_reservation_service import process_identity
    from src.agents.order_recovery_controller import release_for_inference, require_live_claim

    c = verify(out)
    config = load_config(out / "gpu_handoff/config.json", c["resource_config_sha256"])
    write_new(
        out / "controller_start.json",
        {
            "started_at_UTC": datetime.now(timezone.utc),
            "contract_sha256": sha(out / "contract.json"),
            "owner": process_identity(os.getpid()),
            "automatic_retry": False,
        },
    )
    admission = release_for_inference(config)
    write_new(out / "gpu_admission.json", admission)
    command = [
        sys.executable,
        "-u",
        "-B",
        str(Path(__file__).resolve()),
        "worker",
        "--output",
        str(out),
    ]
    started = time.monotonic()
    process = None
    peak = 0
    try:
        with (
            (out / "worker.log").open("x") as log,
            (out / "gpu_observations.jsonl").open("x") as evidence,
        ):
            while True:
                state = reservation_status(config)
                require_live_claim(config, state, process_identity(os.getpid()))
                assert state["paused"] is True and all(
                    state[k] == 0
                    for k in (
                        "payload_bytes",
                        "allocator_allocated_bytes",
                        "allocator_reserved_bytes",
                    )
                )
                sample = _driver_snapshot()
                assert sample["gpu_uuid"] == config["gpu_uuid"]
                used = sample["driver_total_bytes"] - sample["driver_free_bytes"]
                peak = max(peak, used)
                limit = min(
                    c["gpu_outer_limit_bytes"],
                    sample["driver_total_bytes"] - c["gpu_minimum_free_bytes"],
                )
                append(
                    evidence,
                    {
                        **sample,
                        "at_UTC": datetime.now(timezone.utc),
                        "aggregate_used_bytes": used,
                        "sampled_peak_bytes": peak,
                        "outer_limit_bytes": c["gpu_outer_limit_bytes"],
                        "stop_threshold_bytes": limit,
                        "physical_headroom_bytes": c["gpu_minimum_free_bytes"],
                        "process_attribution": False,
                    },
                )
                assert used <= limit, "aggregate GPU bound/headroom exceeded"
                assert time.monotonic() - started < 1800, "GPU worker timeout"
                if process is None:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                code = process.poll()
                if code is not None:
                    break
                time.sleep(1)
        write_new(
            out / "worker_exit_receipt.json",
            {
                "actual_returncode": code,
                "command": command,
                "finished_at_UTC": datetime.now(timezone.utc),
                "sampled_peak_gpu_bytes": peak,
            },
        )
        assert code == 0, f"GPU worker exited {code}"
        assert read(out / "generation_completion.json")["status"] == "COMPLETE"
        deadline = time.monotonic() + 30
        while True:
            try:
                idle = controller_idle(config)
                break
            except RuntimeError as error:
                if "child processes remain" not in str(error) or time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        write_new(out / "holder_idle_request.json", idle)
        write_new(
            out / "controller_completion.json",
            {
                "status": "COMPLETE",
                "finished_at_UTC": datetime.now(timezone.utc),
                "model_requests": 128,
                "actual_worker_exit_code": 0,
                "sampled_peak_gpu_bytes": peak,
                "holder_idle_requested": True,
            },
        )
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


async def evaluate(out: Path) -> None:
    # CPU helper changes backend flags only in this separate label process.
    from agent_r1.env.base import Action
    from m2_same_prefix_menu import HOLD, Replay, financial_state, first_step_checks

    verify(out)
    done = read(out / "generation_completion.json")
    assert done["status"] == "COMPLETE" and done["model_requests"] == 128
    assert all(sha(out / k) == v for k, v in done["output_sha256"].items())
    generated = rows(out / "generations.jsonl")
    planned = rows(out / "inputs.jsonl")
    assert len(generated) == len(planned) == 128
    assert [v["request_key"] for v in generated] == [v["request_key"] for v in planned]
    anchors = {v["anchor_id"]: v for v in rows(SOURCE / "anchors.jsonl")}
    packets = {v["anchor_id"]: v for v in rows(SOURCE / "prompt_packets.jsonl")}
    replay = Replay(read(SOURCE / "source_manifest.json"))
    label_count = 0
    live(
        out, status="RUNNING", phase="CPU_PREFIX_AND_LABELS", model_requests=128, slots_evaluated=0
    )
    with (out / "evaluations.jsonl").open("x") as f:
        for index, generation in enumerate(generated):
            anchor = anchors[generation["anchor_id"]]
            item = {
                "request_key": generation["request_key"],
                "anchor_id": anchor["anchor_id"],
                "slot": generation["slot"],
                "generation_status": generation["generation_status"],
                "env_status": "NOT_APPLIED",
                "terminal_label_status": "NA",
                "first_info": None,
                "first_trace": None,
                "first_observation_messages": None,
                "parsed_calls": [],
                "api_action_key": None,
                "economic_order_key": None,
                "family": None,
                "off_grid": None,
                "in_anchor_menu": None,
                "financial_state_sha256": anchor["financial_state_sha256"],
                "observation_messages_sha256": anchor["observation_messages_sha256"],
                "prefix_requests": anchor["prefix_requests"],
                "hold_Y": anchor["old_hold_Y"],
                "Y": None,
                "q": None,
                "terminal_metrics": None,
                "first_step_checks": None,
            }
            assert generation_integrity(generation)[0] == generation["generation_status"]
            if generation["generation_status"] == "COMPLETE":
                env, obs, config_sha = await replay.prefix(
                    anchor["record_index"], anchor["prefix_requests"]
                )
                before = json.loads(canonical(financial_state(env, config_sha)))
                assert object_sha(before) == anchor["financial_state_sha256"]
                assert obs.messages == packets[anchor["anchor_id"]]["messages"]
                assert object_sha(obs.messages) == anchor["observation_messages_sha256"]
                _, calls = env.format_wrapper.parse_response(generation["response_text"])
                raw_calls = [{"name": v.name, "arguments": v.arguments} for v in calls]
                item["parsed_calls"] = safe_metadata(raw_calls)
                if len(raw_calls) == 1:
                    with suppress(ValueError):
                        item["api_action_key"] = canonical(raw_calls[0])
                session = env.current_session_index
                obs, _, _, info = await env.step(
                    Action(
                        text=generation["response_text"], token_ids=generation["response_token_ids"]
                    )
                )
                replay.requests += 1
                item["first_info"] = safe_metadata(json.loads(canonical(info)))
                item["first_trace"] = asdict(env.credit_step_trace[-1])
                item["first_observation_messages"] = obs.messages
                after = json.loads(canonical(financial_state(env, config_sha)))
                item["first_state_after"] = after
                if info.get("invalid_action", False):
                    assert env.current_session_index == session and after == before
                    item["env_status"] = "INVALID"
                elif env.current_session_index == session:
                    assert after == before
                    item.update(env_status="NONADVANCING", family=item["first_trace"]["action"])
                else:
                    assert env.current_session_index == session + 1 and len(raw_calls) == 1
                    first_step_checks(before, after, item["first_info"], env)
                    name, args = raw_calls[0]["name"], raw_calls[0]["arguments"]
                    if name in ("buy", "sell"):
                        target = float(args["target_weight"])
                        assert math.isfinite(target)
                        key = [name, args["symbol"].strip().upper(), 0.0 if target == 0 else target]
                        item["off_grid"] = target not in (0, 0.1, 0.2)
                    else:
                        assert name in ("hold", "abstain")
                        key = [name]
                    menu_keys = [
                        [m["family"]]
                        if m["family"] == "hold"
                        else [m["family"], m["symbol"], m["target_weight"]]
                        for m in anchor["menu"]
                    ]
                    item.update(
                        env_status="ADVANCING",
                        family=name,
                        economic_order_key=json.dumps(key, allow_nan=False),
                        in_anchor_menu=key in menu_keys,
                        first_step_checks={
                            "before_identity": True,
                            "history_append": True,
                            "cash_position_fees": True,
                        },
                    )
                    holds = 0
                    while not env.episode_complete:
                        _, _, _, next_info = await env.step(Action(text=HOLD))
                        replay.requests += 1
                        holds += 1
                        assert (
                            not next_info["invalid_action"]
                            and not next_info["fills"]
                            and holds <= 20
                        )
                    metrics = env.screening_terminal_metrics()
                    assert holds == 19 - anchor["decision_session_index"]
                    item.update(
                        terminal_label_status="COMPLETE_FIXED_HOLD",
                        terminal_metrics=metrics,
                        Y=metrics["terminal_score"],
                        q=metrics["terminal_score"] - anchor["old_hold_Y"],
                        suffix_hold_requests=holds,
                    )
                    label_count += 1
            append(f, item)
            live(
                out,
                status="RUNNING",
                phase="CPU_LABELS",
                model_requests=128,
                slots_evaluated=index + 1,
                fixed_hold_labels=label_count,
                environment_requests=replay.requests,
            )
            if (index + 1) % 16 == 0:
                print("evaluated", index + 1, "/128", "labels", label_count, flush=True)
    verify(out)
    write_new(
        out / "evaluation_completion.json",
        {
            "status": "COMPLETE",
            "finished_at_UTC": datetime.now(timezone.utc),
            "slots_evaluated": 128,
            "fixed_hold_labels": label_count,
            "environment_requests": replay.requests,
            "environments_constructed": replay.environments,
            "model_requests": 128,
            "optimizer_updates": 0,
            "formal_valid_oos_access": False,
            "output_sha256": {"evaluations.jsonl": sha(out / "evaluations.jsonl")},
        },
    )
    live(
        out,
        status="EVALUATION_COMPLETE",
        phase="AWAITING_READOUT",
        model_requests=128,
        slots_evaluated=128,
    )


def record_exit(out: Path) -> None:
    write_new(
        out / "service_exit_receipt.json",
        {
            "recorded_at_UTC": datetime.now(timezone.utc),
            "SERVICE_RESULT": os.environ.get("SERVICE_RESULT"),
            "EXIT_CODE": os.environ.get("EXIT_CODE"),
            "EXIT_STATUS": os.environ.get("EXIT_STATUS"),
            "status_source": "systemd ExecStopPost environment",
            "contract_sha256": sha(out / "contract.json"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("prepare", "controller", "worker", "evaluate", "record-exit")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--model-receipt", type=Path)
    parser.add_argument("--holder", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            assert args.model_receipt and args.holder
            prepare(args.output.resolve(), args.model_receipt, args.holder)
        elif args.mode == "worker":
            worker(args.output.resolve())
        elif args.mode == "controller":
            run_controller(args.output.resolve())
        elif args.mode == "evaluate":
            asyncio.run(evaluate(args.output.resolve()))
        else:
            record_exit(args.output.resolve())
    except BaseException as error:
        if (
            args.mode != "prepare"
            and args.output.is_dir()
            and not (args.output / f"{args.mode}_failure.json").exists()
        ):
            write_new(
                args.output / f"{args.mode}_failure.json",
                {
                    "status": "FAILED",
                    "mode": args.mode,
                    "error": repr(error),
                    "at_UTC": datetime.now(timezone.utc),
                    "automatic_retry": False,
                },
            )
            live(args.output, status="FAILED", phase=args.mode, error=repr(error))
        raise


if __name__ == "__main__":
    main()

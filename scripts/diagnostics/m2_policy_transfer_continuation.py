#!/usr/bin/env python3
"""Observe only the three unstarted arms after the original SFT request-cap result."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "runs/m2-policy-transfer-eval-20260914"
DEFAULT_DIRECTORY = ROOT / "runs/m2-policy-transfer-eval-20260914-continuation"
PLAN = ROOT / "docs/2026-09-14_M2_POLICY_TRANSFER_EVAL_CONTINUATION.md"
RUNNER = ROOT / "scripts/diagnostics/m2_policy_transfer_eval.py"
CELL_IDS = ("m1", "m2_eta1", "m2_eta025")
PROFILE = "policy_transfer_remaining_arms_v1"
PROTOCOL = "single_tool_portfolio_v1"
INPUTS = (
    "contexts.json",
    "settings.json",
    "models.json",
    "continuation_settings.json",
    "source_identity.json",
    "original_evidence.json",
    "reference.json",
    "amendment.md",
)


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def original_runner(original):
    sources = read(original / "source_identity.json")
    relative = str(RUNNER.relative_to(ROOT))
    if sha(RUNNER) != sources.get(relative):
        raise RuntimeError("original evaluation runner changed after its seal")
    spec = importlib.util.spec_from_file_location("fixed_policy_transfer_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_grid(contexts):
    full = contexts.get("full", [])
    if contexts.get("short") != [] or contexts.get("teacher_labels_included") is not False:
        raise RuntimeError("continuation contexts must exclude short states and teacher labels")
    if len(full) != 12 or len({context["id"] for context in full}) != 12:
        raise RuntimeError("continuation requires the original twelve distinct contexts")
    expected = {}
    for index, context in enumerate(full):
        if context.get("prefix") != [] or context.get("expected_start_session_index") != 0:
            raise RuntimeError("continuation must start from the original empty cash prefix")
        for sibling in range(5):
            lane = "greedy" if sibling == 0 else f"sample{sibling - 1}"
            expected[f"full-{context['id']}-{lane}"] = (index, sibling, context)
    return expected


def validate_observed(directory, contexts):
    """Accept native REQUEST_CAP as incomplete behavior; never manufacture an outcome."""
    from src.agents.rollout_seed import derive_request_seed

    directory = Path(directory)
    value = read(directory / "result.json")
    if value != read(directory / "state.json"):
        raise RuntimeError("native evaluator final state and result disagree")
    if value.get("status") != "OBSERVED" or value.get("action_protocol") != PROTOCOL:
        raise RuntimeError("native evaluator did not finish observing the fixed grid")
    expected = expected_grid(contexts)
    rows = value.get("grid", [])
    if len(rows) != 60 or Counter(row.get("id") for row in rows) != Counter(expected.keys()):
        raise RuntimeError("observed grid is missing or duplicates planned trajectories")
    by_id = {row["id"]: row for row in rows}
    counts = Counter()
    for row in rows:
        index, sibling, context = expected[row["id"]]
        identity = {
            "step": 1,
            "sample_index": index,
            "rollout_n": max(0, sibling - 1),
            "validate": sibling == 0,
        }
        if (
            row.get("context") != context
            or row.get("context_id") != context["id"]
            or row.get("horizon") != "full"
            or row.get("state_kind") != "cash"
            or row.get("sampling") != ("greedy" if sibling == 0 else "sample")
            or row.get("identity") != identity
            or row.get("request_limit") != 128
            or type(row.get("requests")) is not int
            or not 1 <= row["requests"] <= 128
            or not 1 <= row.get("max_response_tokens", 0) < 4096
            or not 1 <= row.get("max_prompt_tokens", 0) <= 8192
            or row.get("prefix_fill_count") != 0
            or row.get("prefix_info") != []
            or row.get("takeover_state", {}).get("session_index") != 0
            or row.get("takeover_state", {}).get("positions") != {}
            or row.get("takeover_equity") != context["env_kwargs"]["initial_cash"]
        ):
            raise RuntimeError("trajectory differs from the original cash-state/sampling contract")
        steps = row.get("steps_including_prefix", [])
        if len(steps) != row["requests"] or any(
            step.get("step_index") != index for index, step in enumerate(steps)
        ):
            raise RuntimeError("trajectory lacks its exact chronological request trace")
        if any(step.get("done") is not False for step in steps[:-1]):
            raise RuntimeError("trajectory continued after an earlier terminal step")
        if row.get("status") == "COMPLETE":
            metrics = row.get("terminal_metrics", {})
            if (
                row.get("failure")
                or steps[-1].get("done") is not True
                or row.get("final_state", {}).get("session_index") != 20
                or metrics.get("session_count") != 21
                or metrics.get("calibration_sha256")
                != context["env_kwargs"]["screening_calibration_sha256"]
            ):
                raise RuntimeError("completed trajectory lacks the original terminal state")
            for number in (row.get("takeover_net_return"), metrics.get("terminal_score")):
                if type(number) not in (int, float) or not math.isfinite(number):
                    raise RuntimeError("completed trajectory lacks finite outcomes")
            counts["complete"] += 1
        elif row.get("status") == "INCOMPLETE" and row.get("failure") == "REQUEST_CAP":
            if (
                row["requests"] != 128
                or steps[-1].get("done") is not False
                or not 0 <= steps[-1].get("result_session_index", -1) < 20
                or any(
                    key in row for key in ("terminal_metrics", "takeover_net_return", "final_state")
                )
            ):
                raise RuntimeError("REQUEST_CAP has a fabricated outcome or unexhausted budget")
            counts["request_cap"] += 1
        else:
            raise RuntimeError(
                "only native REQUEST_CAP may continue; infrastructure/other caps stop"
            )
    observed = {key: [] for key in expected}
    token_totals, max_responses, max_prompts = Counter(), Counter(), Counter()
    with (directory / "requests.jsonl").open() as stream:
        for raw in stream:
            if not raw.endswith("\n"):
                raise RuntimeError("request evidence contains an uncommitted final record")
            record = json.loads(raw)
            trajectory = record.get("trajectory_id")
            if trajectory not in observed:
                raise RuntimeError("request evidence includes an unplanned trajectory")
            row = by_id[trajectory]
            index = len(observed[trajectory])
            tokens, prompt = (
                record.get("response_token_ids", []),
                record.get("prompt_token_ids", []),
            )
            expected_params = {
                "temperature": 0.0 if row["sampling"] == "greedy" else 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.0,
                "max_tokens": 4096,
                "seed": derive_request_seed(42, row["identity"], index),
                "n": 1,
            }
            if (
                record.get("request_index") != index
                or index >= row["requests"]
                or record.get("sampling_params") != expected_params
                or record.get("finish_reason") != "stop"
                or not 0 < len(tokens) < 4096
                or not 0 < len(prompt) <= 8192
                or record.get("trace") != row["steps_including_prefix"][index]
            ):
                raise RuntimeError(
                    "raw request evidence differs from limits, seeds or exact step trace"
                )
            observed[trajectory].append(record["trace"])
            token_totals[trajectory] += len(tokens)
            max_responses[trajectory] = max(max_responses[trajectory], len(tokens))
            max_prompts[trajectory] = max(max_prompts[trajectory], len(prompt))
    for row in rows:
        key = row["id"]
        trace = observed[key]
        if (
            len(trace) != row["requests"]
            or token_totals[key] != row.get("response_tokens")
            or max_responses[key] != row["max_response_tokens"]
            or max_prompts[key] != row["max_prompt_tokens"]
            or dict(Counter(step["action"] for step in trace)) != row.get("actions")
            or sum(bool(step["invalid_action"]) for step in trace) != row.get("invalid")
        ):
            raise RuntimeError("raw requests and final native trajectory accounting disagree")
    return {
        "status": "OBSERVED_WITH_INCOMPLETE" if counts["request_cap"] else "COMPLETE",
        "planned": 60,
        "observed": 60,
        "complete": counts["complete"],
        "incomplete": counts["request_cap"],
        "request_cap": counts["request_cap"],
        "requests": sum(row["requests"] for row in rows),
        "result_sha256": sha(directory / "result.json"),
        "requests_sha256": sha(directory / "requests.jsonl"),
        "terminal_outcomes_imputed": False,
    }


def prepare(base, *, plan=PLAN, original=ORIGINAL):
    base, original = Path(base).resolve(), Path(original).resolve()
    if base.exists():
        raise RuntimeError("new continuation directory required; no overwrite")
    runner = original_runner(original)
    runner.check_contract(original)
    state = read(original / "state.json")
    if (
        state.get("status") != "FAILED"
        or len(state.get("cells", [])) != 1
        or state["cells"][0].get("cell_id") != "sft0"
        or state["cells"][0].get("returncode") != 0
        or any((original / cell).exists() for cell in CELL_IDS)
    ):
        raise RuntimeError("continuation only covers the three genuinely unstarted original arms")
    reference = validate_observed(original / "sft0", read(original / "contexts.json"))
    if reference["complete"] != 58 or reference["request_cap"] != 2:
        raise RuntimeError("the authorized SFT 58-complete/two-request-cap reference differs")
    if not Path(plan).is_file():
        raise RuntimeError("explicit post-observation amendment is required")
    base.mkdir(parents=True, exist_ok=False)
    for name in ("contexts.json", "settings.json", "models.json"):
        shutil.copy2(original / name, base / name)
    shutil.copy2(plan, base / "amendment.md")
    evidence_names = (
        "run_contract.json",
        "source_identity.json",
        "state.json",
        "settings.json",
        "contexts.json",
        "models.json",
        "sft0/result.json",
        "sft0/state.json",
        "sft0/config.json",
        "sft0/requests.jsonl",
    )
    save(
        base / "original_evidence.json",
        {str(original / name): sha(original / name) for name in evidence_names},
    )
    save(base / "source_identity.json", {str(Path(__file__).resolve()): sha(__file__)})
    save(
        base / "reference.json",
        {
            **reference,
            "cell_id": "sft0",
            "directory": str(original / "sft0"),
            "read_only_reference": True,
            "rerun": False,
        },
    )
    save(
        base / "continuation_settings.json",
        {
            "profile": PROFILE,
            "original_directory": str(original),
            "cell_ids": list(CELL_IDS),
            "own_trajectories": 180,
            "original_reference_trajectories": 60,
            "combined_trajectories": 240,
            "automatic_retry": False,
            "training_updates": 0,
            "known_request_cap_policy": "observe_incomplete_and_continue",
            "other_failure_policy": "stop",
            "original_sft_rerun": False,
        },
    )
    save(
        base / "run_contract.json",
        {
            "profile": PROFILE,
            "sealed_at_SGT": now(),
            "inputs": {name: sha(base / name) for name in INPUTS},
        },
    )
    check_contract(base)
    return {"status": "PREPARED", "own_trajectories": 180, "reference": reference}


def check_contract(base):
    contract = read(base / "run_contract.json")
    if contract.get("profile") != PROFILE or set(contract.get("inputs", {})) != set(INPUTS):
        raise RuntimeError("continuation contract is incomplete")
    for name, digest in contract["inputs"].items():
        if sha(base / name) != digest:
            raise RuntimeError("continuation sealed input changed: " + name)
    for path, digest in read(base / "source_identity.json").items():
        if sha(path) != digest:
            raise RuntimeError("continuation source changed after freeze")
    for path, digest in read(base / "original_evidence.json").items():
        if sha(path) != digest:
            raise RuntimeError("original evidence changed after continuation freeze")
    settings = read(base / "continuation_settings.json")
    if settings["cell_ids"] != list(CELL_IDS) or settings["own_trajectories"] != 180:
        raise RuntimeError("continuation may not extend or reorder the three remaining arms")
    original = Path(settings["original_directory"])
    runner = original_runner(original)
    runner.check_contract(original)
    for name in ("settings.json", "contexts.json", "models.json"):
        if sha(original / name) != sha(base / name):
            raise RuntimeError("continuation changed original model/context/sampling settings")
    return runner


def run(base):
    base = Path(base).resolve()
    if (base / "state.json").exists():
        raise RuntimeError("continuation already has state; no retry or resume")
    runner = check_contract(base)
    state = {
        "status": "RUNNING",
        "started_at_SGT": now(),
        "planned_trajectories": 180,
        "combined_planned_trajectories": 240,
        "cells": [],
        "reference": read(base / "reference.json"),
        "automatic_retry": False,
    }
    with (base / "state.json").open("x") as stream:
        json.dump(state, stream, indent=2)
    contexts, models = read(base / "contexts.json"), read(base / "models.json")
    try:
        for folder in ("commands", "processes", "logs"):
            (base / folder).mkdir(exist_ok=False)
        for cell_id in CELL_IDS:
            runner = check_contract(base)
            if (base / cell_id).exists():
                raise RuntimeError("continuation child directory already exists")
            if shutil.disk_usage(base).free < 5 * 1024**3:
                raise RuntimeError("continuation has less than the fixed 5 GiB reserve")
            cell = {"cell_id": cell_id, "status": "RUNNING", "started_at_SGT": now()}
            state["cells"].append(cell)
            state["current_cell"] = cell_id
            save(base / "state.json", state)
            cell["gpu_before"] = runner.wait_gpu_idle()
            command = runner.cell_command(base, cell_id, models[cell_id])
            save(base / "commands" / f"{cell_id}.json", command)
            save(base / "state.json", state)
            environment = dict(os.environ)
            for key in list(environment):
                if key.startswith(("FINANCE_SCREENING_", "FINANCE_LEARNING_", "FINANCE_CRITIC_")):
                    environment.pop(key)
            environment.update(
                PYTHONPATH=f"{ROOT}:{ROOT / 'third_party/agent-r1'}",
                PYTHONUNBUFFERED="1",
                TOKENIZERS_PARALLELISM="false",
            )
            with (base / "logs" / f"{cell_id}.log").open("x") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
                save(
                    base / "processes" / f"{cell_id}.json",
                    {
                        "pid": process.pid,
                        "started_at_SGT": now(),
                        "command": command,
                    },
                )
                cell["returncode"] = process.wait()
            if cell["returncode"] != 0:
                raise RuntimeError(f"{cell_id} native evaluator exited {cell['returncode']}")
            evidence = validate_observed(base / cell_id, contexts)
            check_contract(base)
            cell.update(status=evidence["status"], evidence=evidence, finished_at_SGT=now())
            save(base / "state.json", state)
        complete = sum(cell["evidence"]["complete"] for cell in state["cells"])
        incomplete = 180 - complete
        combined_incomplete = incomplete + state["reference"]["incomplete"]
        state.update(
            status="ALL_CELLS_OBSERVED_WITH_INCOMPLETE" if combined_incomplete else "COMPLETE",
            observed_trajectories=180,
            complete_trajectories=complete,
            incomplete_trajectories=incomplete,
            combined_observed_trajectories=240,
            combined_complete_trajectories=complete + state["reference"]["complete"],
            combined_incomplete_trajectories=combined_incomplete,
        )
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        if state["cells"] and state["cells"][-1]["status"] == "RUNNING":
            state["cells"][-1].update(status="FAILED", error=state["error"], finished_at_SGT=now())
        raise
    finally:
        state["finished_at_SGT"] = now()
        save(base / "state.json", state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "run"))
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--original", type=Path, default=ORIGINAL)
    parser.add_argument("--plan", type=Path, default=PLAN)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    base = args.directory.resolve()
    if args.command == "prepare":
        result = prepare(base, plan=args.plan, original=args.original)
    elif args.command == "check":
        check_contract(base)
        result = {"status": "PASS"}
    else:
        result = run(base)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run and monitor the fixed four-model, two-profile sampling comparison."""

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
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "runs/m2-policy-transfer-eval-20260914"
DEFAULT = ROOT / "runs/m2-sampling-profile-eval-20260915"
PLAN = ROOT / "docs/2026-09-15_M2_SAMPLING_PROFILE_PLAN.md"
EVALUATOR = ROOT / "scripts/diagnostics/m2_sampling_profile_eval.py"
READER = ROOT / "scripts/diagnostics/m2_sampling_profile_readout.py"
CELL_IDS = ("sft0", "m1", "m2_eta1", "m2_eta025")
PROFILE = "qwen_nonthinking_sampling_comparison_v1"
INPUTS = (
    "contexts.json",
    "models.json",
    "settings.json",
    "plan.md",
    "source_identity.json",
    "parent_evidence.json",
)


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def original_runner():
    relative = "scripts/diagnostics/m2_policy_transfer_eval.py"
    if sha(ROOT / relative) != read(ORIGINAL / "source_identity.json")[relative]:
        raise RuntimeError("original evaluation source identity changed")
    return module(ROOT / relative, "original_policy_transfer")


def settings():
    return dict(
        profile=PROFILE,
        cell_ids=list(CELL_IDS),
        trajectories_per_model=96,
        planned_trajectories=384,
        contexts=12,
        siblings_per_profile=4,
        profiles={
            "current": dict(temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0),
            "qwen_recommended": dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0),
        },
        enable_thinking=False,
        seed=42,
        request_limit=128,
        response_limit=4096,
        prompt_limit=8192,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        training_updates=0,
        new_model_exports=0,
        automatic_retry=False,
        request_cap_policy="retain_incomplete_and_continue",
        other_failure_policy="stop",
        original_reference=str(ORIGINAL),
    )


def source_files():
    files = original_runner().source_files()
    for path in (Path(__file__).resolve(), EVALUATOR, READER):
        files[str(path.relative_to(ROOT))] = path
    return files


def check_contract(base):
    contract = read(base / "run_contract.json")
    if contract.get("profile") != PROFILE or set(contract.get("inputs", {})) != set(INPUTS):
        raise RuntimeError("sampling comparison seal is incomplete")
    for name, digest in contract["inputs"].items():
        if sha(base / name) != digest:
            raise RuntimeError("sealed sampling input changed: " + name)
    if read(base / "settings.json") != settings():
        raise RuntimeError("sampling settings differ from fixed comparison")
    sources = read(base / "source_identity.json")
    if set(sources) != set(source_files()):
        raise RuntimeError("sampling source inventory changed")
    for name, path in source_files().items():
        if sha(path) != sources[name]:
            raise RuntimeError("sampling source changed: " + name)
    for name, digest in read(base / "parent_evidence.json").items():
        if sha(name) != digest:
            raise RuntimeError("original input identity changed: " + name)
    old = original_runner()
    old.check_contract(ORIGINAL)
    if read(base / "models.json") != old.model_sources():
        raise RuntimeError("existing actor export identity changed")
    if sha(base / "contexts.json") != sha(ORIGINAL / "contexts.json"):
        raise RuntimeError("original twelve contexts changed")


def prepare(base):
    if base.exists():
        raise RuntimeError("new run directory required; no overwrite")
    old = original_runner()
    old.check_contract(ORIGINAL)
    evaluator = module(EVALUATOR, "sampling_eval_prepare")
    contexts = read(ORIGINAL / "contexts.json")
    evaluator.validate_contexts(contexts)
    if settings()["profiles"] != evaluator.PROFILES or len(evaluator.grid(contexts)) != 96:
        raise RuntimeError("evaluator profile grid differs")
    sources = {name: sha(path) for name, path in source_files().items()}
    base.mkdir(parents=True, exist_ok=False)
    for name in ("contexts.json", "models.json"):
        shutil.copy2(ORIGINAL / name, base / name)
    shutil.copy2(PLAN, base / "plan.md")
    save(base / "settings.json", settings())
    save(base / "source_identity.json", sources)
    save(
        base / "parent_evidence.json",
        {
            str(ORIGINAL / name): sha(ORIGINAL / name)
            for name in (
                "run_contract.json",
                "source_identity.json",
                "models.json",
                "contexts.json",
            )
        },
    )
    save(
        base / "run_contract.json",
        dict(
            profile=PROFILE, sealed_at_SGT=now(), inputs={name: sha(base / name) for name in INPUTS}
        ),
    )
    check_contract(base)
    return dict(status="PREPARED", planned=384, directory=str(base))


def validate_result(directory, contexts):
    evaluator = module(EVALUATOR, "sampling_eval_validate")
    result = read(directory / "result.json")
    if result != read(directory / "state.json") or result.get("status") != "OBSERVED":
        raise RuntimeError("evaluator did not finish the planned grid")
    if (
        result.get("action_protocol") != evaluator.PROTOCOL
        or result.get("enable_thinking") is not False
        or result.get("sampling_profiles") != evaluator.PROFILES
    ):
        raise RuntimeError("evaluator changed the action or sampling profile contract")
    expected = {row["id"]: row for row in evaluator.grid(contexts)}
    rows = result.get("grid", [])
    if len(rows) != 96 or Counter(row.get("id") for row in rows) != Counter(expected.keys()):
        raise RuntimeError("missing or duplicate profile trajectories")
    rowmap, counts, raw_counts, token_counts = {}, Counter(), Counter(), Counter()
    for row in rows:
        contract = expected[row["id"]]
        for key in (
            "id",
            "context_id",
            "context",
            "horizon",
            "state_kind",
            "sampling",
            "profile",
            "identity",
            "request_limit",
        ):
            if row.get(key) != contract[key]:
                raise RuntimeError("trajectory identity/profile differs: " + row["id"])
        steps = row.get("steps_including_prefix", [])
        if (
            type(row.get("requests")) is not int
            or not 1 <= row["requests"] <= 128
            or len(steps) != row["requests"]
            or [step["step_index"] for step in steps] != list(range(row["requests"]))
            or any(step.get("done") is not False for step in steps[:-1])
            or row.get("takeover_state", {}).get("positions") != {}
            or row.get("takeover_state", {}).get("session_index") != 0
            or row.get("takeover_equity") != row["context"]["env_kwargs"]["initial_cash"]
            or row.get("prefix_fill_count") != 0
            or row.get("prefix_info") != []
        ):
            raise RuntimeError("trajectory is truncated or has changed start/trace evidence")
        if row["status"] == "COMPLETE":
            metrics = row.get("terminal_metrics", {})
            if (
                row.get("failure")
                or steps[-1]["done"] is not True
                or row.get("final_state", {}).get("session_index") != 20
                or metrics.get("session_count") != 21
                or metrics.get("calibration_sha256")
                != row["context"]["env_kwargs"]["screening_calibration_sha256"]
            ):
                raise RuntimeError("complete trajectory lacks actual terminal evidence")
            for number in (row.get("takeover_net_return"), metrics.get("terminal_score")):
                if type(number) not in (int, float) or not math.isfinite(number):
                    raise RuntimeError("nonfinite terminal outcome")
            counts["complete"] += 1
        elif row["status"] == "INCOMPLETE" and row.get("failure") == "REQUEST_CAP":
            if (
                row["requests"] != 128
                or steps[-1]["done"] is not False
                or not 0 <= steps[-1].get("result_session_index", 20) < 20
                or any(
                    key in row for key in ("terminal_metrics", "takeover_net_return", "final_state")
                )
            ):
                raise RuntimeError("request cap contains invented terminal outcomes")
            counts["incomplete"] += 1
        else:
            raise RuntimeError("response/prompt truncation or other failure stops later models")
        rowmap[row["id"]] = row
    with (directory / "requests.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            row = rowmap.get(record.get("trajectory_id"))
            if row is None:
                raise RuntimeError("unregistered request trajectory")
            index = raw_counts[row["id"]]
            params = evaluator.sampling_parameters(row["profile"], row["identity"], index)
            native = record.get("native_sampling_params", {})
            if (
                not line.endswith("\n")
                or record.get("profile") != row["profile"]
                or record.get("request_index") != index
                or index >= row["requests"]
                or record.get("sampling_params") != params
                or any(native.get(k) != v for k, v in params.items())
                or record.get("finish_reason") != "stop"
                or not 0 < len(record.get("response_token_ids", [])) < 4096
                or not 0 < len(record.get("prompt_token_ids", [])) <= 8192
                or record.get("trace") != row["steps_including_prefix"][index]
            ):
                raise RuntimeError("raw request differs from fixed params, seed or trace")
            raw_counts[row["id"]] += 1
            token_counts[row["id"]] += len(record["response_token_ids"])
    for row in rows:
        if (
            raw_counts[row["id"]] != row["requests"]
            or token_counts[row["id"]] != row["response_tokens"]
            or dict(Counter(step["action"] for step in row["steps_including_prefix"]))
            != row["actions"]
            or sum(step["invalid_action"] for step in row["steps_including_prefix"])
            != row["invalid"]
        ):
            raise RuntimeError("raw/native request accounting differs")
    return dict(
        observed=96,
        complete=counts["complete"],
        incomplete=counts["incomplete"],
        requests=sum(raw_counts.values()),
        result_sha256=sha(directory / "result.json"),
        requests_sha256=sha(directory / "requests.jsonl"),
    )


def wait_gpu_capacity(timeout=120):
    """Require 60 GiB free for the unchanged 0.5-utilization engine; never stop other jobs."""
    started = time.monotonic()
    while True:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        lines = result.stdout.strip().splitlines()
        if len(lines) != 1:
            raise RuntimeError("sampling comparison expects one visible GPU")
        total, used, free, utilization = (int(value.strip()) for value in lines[0].split(","))
        if free >= 60 * 1024:
            return dict(
                total_mib=total,
                used_mib=used,
                free_mib=free,
                utilization_percent=utilization,
                minimum_free_mib=60 * 1024,
                checked_at_SGT=now(),
            )
        if time.monotonic() - started >= timeout:
            raise RuntimeError("insufficient GPU capacity; no other job was stopped")
        time.sleep(5)


def run(base):
    if (base / "state.json").exists():
        raise RuntimeError("run already has state; no retry or resume")
    check_contract(base)
    state = dict(
        status="RUNNING",
        started_at_SGT=now(),
        planned_trajectories=384,
        cells=[],
        automatic_retry=False,
        training_updates=0,
    )
    with (base / "state.json").open("x") as stream:
        json.dump(state, stream)
    try:
        for folder in ("logs", "commands", "processes"):
            (base / folder).mkdir(exist_ok=False)
        models, contexts = read(base / "models.json"), read(base / "contexts.json")
        for cell_id in CELL_IDS:
            check_contract(base)
            if (base / cell_id).exists() or shutil.disk_usage(base).free < 5 * 1024**3:
                raise RuntimeError("existing cell or insufficient fixed disk reserve")
            cell = dict(cell_id=cell_id, status="RUNNING", started_at_SGT=now())
            state["cells"].append(cell)
            state["current_cell"] = cell_id
            save(base / "state.json", state)
            cell["gpu_before"] = wait_gpu_capacity()
            command = [
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(EVALUATOR),
                "--model",
                models[cell_id]["model_path"],
                "--contexts",
                str(base / "contexts.json"),
                "--output",
                str(base / cell_id),
            ]
            save(base / "commands" / (cell_id + ".json"), command)
            env = dict(os.environ)
            for key in list(env):
                if key.startswith(("FINANCE_SCREENING_", "FINANCE_LEARNING_", "FINANCE_CRITIC_")):
                    env.pop(key)
            env.update(
                PYTHONPATH=f"{ROOT}:{ROOT / 'third_party/agent-r1'}",
                PYTHONUNBUFFERED="1",
                TOKENIZERS_PARALLELISM="false",
            )
            with (base / "logs" / (cell_id + ".log")).open("x") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
                save(
                    base / "processes" / (cell_id + ".json"),
                    dict(pid=process.pid, command=command, started_at_SGT=now()),
                )
                cell["returncode"] = process.wait()
            if cell["returncode"] != 0:
                raise RuntimeError(f"{cell_id} evaluator exited {cell['returncode']}")
            cell["evidence"] = validate_result(base / cell_id, contexts)
            check_contract(base)
            cell.update(
                status="OBSERVED_WITH_INCOMPLETE" if cell["evidence"]["incomplete"] else "COMPLETE",
                finished_at_SGT=now(),
            )
            save(base / "state.json", state)
        complete = sum(cell["evidence"]["complete"] for cell in state["cells"])
        state.update(
            status="ALL_CELLS_OBSERVED_WITH_INCOMPLETE" if complete != 384 else "COMPLETE",
            observed_trajectories=384,
            complete_trajectories=complete,
            incomplete_trajectories=384 - complete,
        )
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        if state["cells"] and state["cells"][-1]["status"] == "RUNNING":
            state["cells"][-1].update(status="FAILED", error=state["error"])
        raise
    finally:
        state["finished_at_SGT"] = now()
        save(base / "state.json", state)
    return state


def refresh(base):
    return subprocess.run(
        ["/usr/bin/python3", str(READER), "--directory", str(base), "--write"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )


def monitor(base, unit):
    started = time.monotonic()
    while time.monotonic() - started < 3 * 3600:
        record = dict(at_SGT=now(), status="MONITORING")
        try:
            result = refresh(base)
            record.update(
                readout_returncode=result.returncode,
                readout_output=result.stdout[-2000:],
                readout_error=result.stderr[-2000:],
            )
            service = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
                    "-p",
                    "ActiveState",
                    "-p",
                    "Result",
                    "-p",
                    "ExecMainStatus",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            fields = dict(line.split("=", 1) for line in service.stdout.splitlines() if "=" in line)
            record["service"] = fields
            if fields.get("ActiveState") in ("inactive", "failed"):
                result = refresh(base)
                state, outcome = read(base / "state.json"), read(base / "readout.json")
                okay = (
                    result.returncode == 0
                    and fields.get("Result") == "success"
                    and fields.get("ExecMainStatus") == "0"
                    and state.get("status") in ("COMPLETE", "ALL_CELLS_OBSERVED_WITH_INCOMPLETE")
                    and outcome.get("execution_status") == state["status"]
                    and outcome.get("terminal_observed") == 384
                    and not outcome.get("problems")
                    and (base / "FINAL_RESULTS.md").exists()
                )
                record.update(
                    status=state["status"] if okay else "STOPPED_EARLY",
                    final_readout_returncode=result.returncode,
                    final_readout_output=result.stdout[-2000:],
                    final_readout_error=result.stderr[-2000:],
                )
                save(base / "monitor_state.json", record)
                return 0 if okay else 2
        except Exception as error:
            record.update(status="READOUT_ERROR", error=repr(error))
        save(base / "monitor_state.json", record)
        time.sleep(30)
    save(base / "monitor_state.json", dict(status="MONITOR_TIMEOUT", at_SGT=now()))
    return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "run", "monitor"))
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--unit", default="m2-sampling-profile-eval-20260915.service")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    base = args.directory.resolve()
    if args.command == "monitor":
        raise SystemExit(monitor(base, args.unit))
    if args.command == "prepare":
        result = prepare(base)
    elif args.command == "run":
        result = run(base)
    else:
        check_contract(base)
        result = dict(status="PASS")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

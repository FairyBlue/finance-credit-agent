#!/usr/bin/env python3
"""Disconnect-safe, fixed-budget second common cold-start experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/agent-r1"))

from src.agents.coldstart_eval import now, save, validate_contexts  # noqa: E402
from src.agents.launcher import _resolve_c_compiler  # noqa: E402

RUN = ROOT / "runs/m2-coldstart-v2-20260913"
MODEL = ROOT / "runs/m2-coldstart-20260913/merged_model"
PLAN = ROOT / "docs/2026-09-13_M2_COLDSTART_V2_PLAN.md"
PYTHON = ROOT / ".venv/bin/python"
PROTOCOL = "single_tool_portfolio_v1"
DATA_FILES = {
    "train.jsonl": "train_jsonl_sha256",
    "eval_contexts.json": "eval_contexts_sha256",
    "regression_eval_contexts.json": "regression_eval_contexts_sha256",
}


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def phases(run=RUN, model=MODEL):
    result = []

    def evaluation(name, selected_model, suite):
        contexts = (
            "regression_eval_contexts.json" if suite == "regression" else "eval_contexts.json"
        )
        result.append({
            "name": name,
            "argv": [
                "-m", "src.agents.coldstart_eval", "--model", str(selected_model),
                "--contexts", str(run / "dataset" / contexts),
                "--output", str(run / name), "--action-protocol", PROTOCOL,
                "--expected-short", "16", "--expected-full", "4" if suite == "regression" else "1",
            ],
            "result": str(run / name / "result.json"),
            "expected_status": "OBSERVED",
        })

    evaluation("before_regression", model, "regression")
    evaluation("before_fresh", model, "fresh")
    result.extend([
        {
            "name": "sft",
            "argv": [
                "-m", "src.agents.coldstart_sft", "train",
                "--data", str(run / "dataset/train.jsonl"),
                "--model", str(model), "--output", str(run / "sft"),
                "--epochs", "2", "--micro-batch", "2", "--gradient-accumulation", "8",
                "--lr", "0.00005", "--seed", "42", "--max-length", "8192",
            ],
            "result": str(run / "sft/summary.json"),
            "expected_status": "COMPLETE",
        },
        {
            "name": "merged_model",
            "argv": [
                "-m", "src.agents.coldstart_sft", "merge", "--model", str(model),
                "--adapter", str(run / "sft/adapter"), "--output", str(run / "merged_model"),
            ],
            "result": str(run / "merged_model/merge_receipt.json"),
            "expected_status": "COMPLETE",
        },
    ])
    evaluation("after_regression", run / "merged_model", "regression")
    evaluation("after_fresh", run / "merged_model", "fresh")
    result.append({
        "name": "readout",
        "argv": [str(ROOT / "scripts/diagnostics/m2_coldstart_v2_readout.py"),
                 "--directory", str(run)],
        "result": str(run / "readout.json"),
        "expected_status": "COMPLETE",
    })
    return result


def verify_dataset(run):
    dataset = run / "dataset"
    manifest = read(dataset / "manifest.json")
    if manifest.get("status") != "COMPLETE" or manifest.get("action_protocol") != PROTOCOL:
        raise RuntimeError("dataset is incomplete or uses a different action protocol")
    for name, key in DATA_FILES.items():
        if sha(dataset / name) != manifest[key]:
            raise RuntimeError(f"dataset identity mismatch: {name}")
    validate_contexts(read(dataset / "regression_eval_contexts.json"), 16, 4)
    validate_contexts(read(dataset / "eval_contexts.json"), 16, 1)
    return manifest


def model_identity(model):
    receipt = read(model / "merge_receipt.json")
    if receipt.get("status") != "COMPLETE" or not receipt.get("files"):
        raise RuntimeError("common initial model has no completed merge receipt")
    for name, identity in receipt["files"].items():
        path = model / name
        if path.stat().st_size != identity["bytes"] or sha(path) != identity["sha256"]:
            raise RuntimeError(f"common initial model identity mismatch: {name}")
    return {
        "path": str(model), "merge_receipt_sha256": sha(model / "merge_receipt.json"),
        "files": receipt["files"],
    }


def seal_sources(run):
    paths = [
        *sorted((ROOT / "src").rglob("*.py")), Path(__file__),
        ROOT / "scripts/diagnostics/m2_coldstart_v2_readout.py", PLAN,
    ]
    sources = run / "sources"
    sources.mkdir(exist_ok=False)
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        target = sources / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = sha(path)
    save(run / "source_manifest.json", hashes)
    return hashes


def check_inputs(run, hashes, dataset_manifest_hash):
    for relative, digest in hashes.items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError(f"live source changed: {relative}")
    if sha(run / "dataset/manifest.json") != dataset_manifest_hash:
        raise RuntimeError("dataset manifest changed after freezing")
    verify_dataset(run)


def resources(run, name):
    sample = {"at_SGT": now(), "phase": name, "disk_free_bytes": shutil.disk_usage(run).free}
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        sample["gpu"] = gpu.stdout.strip()
        sample["gpu_query_exit_code"] = gpu.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        sample["gpu_query_error"] = str(error)
    for name in ("memory.current", "memory.peak", "memory.events"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            sample[name] = path.read_text().strip()
    with (run / "resources.jsonl").open("a") as handle:
        handle.write(json.dumps(sample) + "\n")


def execute_phase(run, phase, state, env):
    receipt = run / "controller.json"
    phase.update(status="RUNNING", started_at_SGT=now())
    state["phase"] = phase["name"]
    state["phases"].append(phase)
    save(receipt, state)
    print(json.dumps(phase), flush=True)
    with (run / (phase["name"] + ".log")).open("x") as log:
        process = subprocess.Popen(
            [str(PYTHON), *phase["argv"]], cwd=ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        )
        phase["pid"] = process.pid
        save(receipt, state)
        try:
            while process.poll() is None:
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=30)
                resources(run, phase["name"])
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        phase.update(exit_code=process.returncode, finished_at_SGT=now())
        if process.returncode != 0:
            raise RuntimeError(f"phase {phase['name']} failed with exit {process.returncode}")
    result = read(Path(phase["result"]))
    if result.get("status") != phase["expected_status"]:
        raise RuntimeError(f"phase {phase['name']} has no completed result")
    phase["status"] = "COMPLETE"
    save(receipt, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=RUN)
    args = parser.parse_args()
    run = args.directory.resolve()
    if (run / "controller.json").exists() or (run / "sources").exists():
        raise RuntimeError("controller identity already used; no automatic retry or overwrite")
    manifest = verify_dataset(run)
    if shutil.disk_usage(run).free <= 25 * 1024**3:
        raise RuntimeError("cold-start experiment needs at least 25 GiB free storage")
    initial = model_identity(MODEL)
    hashes = seal_sources(run)
    save(run / "model_identity.json", initial)
    dataset_manifest_hash = sha(run / "dataset/manifest.json")
    env = os.environ.copy()
    compiler, compiler_version = _resolve_c_compiler(parent=env)
    env.update(
        CC=compiler, PYTHONPATH=str(ROOT / "third_party/agent-r1") + ":" + str(ROOT),
        PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_USE_V1="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    save(run / "compiler.json", {"CC": compiler, "version": compiler_version})
    state = {
        "status": "RUNNING", "started_at_SGT": now(), "phases": [],
        "action_protocol": PROTOCOL, "dataset_manifest_sha256": dataset_manifest_hash,
        "dataset_rows": manifest["row_count"], "automatic_retry": False,
        "automatic_rl_launch": False, "sft_epochs": 2,
    }
    save(run / "controller.json", state)
    try:
        for phase in phases(run):
            check_inputs(run, hashes, dataset_manifest_hash)
            execute_phase(run, phase, state, env)
            if phase["name"] == "merged_model":
                merged = read(run / "merged_model/merge_receipt.json")
                if merged.get("model") != str(MODEL) or merged.get("adapter") != str(
                    run / "sft/adapter"
                ):
                    raise RuntimeError("export receipt differs from the selected model and adapter")
                save(run / "exported_model_identity.json", model_identity(run / "merged_model"))
            check_inputs(run, hashes, dataset_manifest_hash)
        report = read(run / "readout.json")
        state.update(
            status="COMPLETE", finished_at_SGT=now(),
            candidate_gate_status=report["gates"]["status"],
        )
    except BaseException as error:
        if state["phases"] and state["phases"][-1]["status"] == "RUNNING":
            state["phases"][-1]["status"] = "FAILED"
        state.update(status="FAILED", failed_at_SGT=now(), error=repr(error))
        save(run / "controller.json", state)
        raise
    save(run / "controller.json", state)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()

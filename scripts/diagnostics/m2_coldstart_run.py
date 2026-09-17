#!/usr/bin/env python3
"""Disconnect-safe sequential bounded cold-start experiment controller."""

from __future__ import annotations

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

from src.agents.coldstart_eval import now, save  # noqa: E402
from src.agents.launcher import _resolve_c_compiler  # noqa: E402

RUN = ROOT / "runs/m2-coldstart-20260913"
MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots"
    / "1cfa9a7208912126459214e8b04321603b3df60c"
)
PYTHON = ROOT / ".venv/bin/python"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = RUN / "controller.json"
    if receipt.exists():
        raise RuntimeError("controller identity already used")
    dataset = RUN / "dataset"
    manifest = json.loads((dataset / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETE"
    assert sha(dataset / "train.jsonl") == manifest["train_jsonl_sha256"]
    assert sha(dataset / "eval_contexts.json") == manifest["eval_contexts_sha256"]
    assert shutil.disk_usage(RUN).free > 25 * 1024**3
    source_paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        Path(__file__),
        ROOT / "docs/2026-09-13_M2_COLDSTART_PLAN.md",
    ]
    sources = RUN / "sources"
    sources.mkdir()
    hashes = {}
    for path in source_paths:
        relative = path.relative_to(ROOT)
        target = sources / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = sha(path)
    save(RUN / "source_manifest.json", hashes)
    env = os.environ.copy()
    compiler, compiler_version = _resolve_c_compiler(parent=env)
    save(RUN / "compiler.json", {"CC": compiler, "version": compiler_version})
    env.update(
        CC=compiler,
        PYTHONPATH=str(ROOT / "third_party/agent-r1") + ":" + str(ROOT),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="8",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_USE_V1="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    contexts = str(dataset / "eval_contexts.json")
    phases = [
        (
            "base_eval",
            [
                "-m",
                "src.agents.coldstart_eval",
                "--model",
                str(MODEL),
                "--contexts",
                contexts,
                "--output",
                str(RUN / "base_eval"),
            ],
            "result.json",
            "OBSERVED",
        ),
        (
            "sft",
            [
                "-m",
                "src.agents.coldstart_sft",
                "train",
                "--data",
                str(dataset / "train.jsonl"),
                "--model",
                str(MODEL),
                "--output",
                str(RUN / "sft"),
                "--epochs",
                "3",
                "--micro-batch",
                "2",
                "--gradient-accumulation",
                "8",
                "--lr",
                "0.0001",
                "--seed",
                "42",
                "--max-length",
                "8192",
            ],
            "summary.json",
            "COMPLETE",
        ),
        (
            "merged_model",
            [
                "-m",
                "src.agents.coldstart_sft",
                "merge",
                "--model",
                str(MODEL),
                "--adapter",
                str(RUN / "sft/adapter"),
                "--output",
                str(RUN / "merged_model"),
            ],
            "merge_receipt.json",
            "COMPLETE",
        ),
        (
            "sft_eval",
            [
                "-m",
                "src.agents.coldstart_eval",
                "--model",
                str(RUN / "merged_model"),
                "--contexts",
                contexts,
                "--output",
                str(RUN / "sft_eval"),
            ],
            "result.json",
            "OBSERVED",
        ),
    ]
    state = {"status": "RUNNING", "started_at_SGT": now(), "phases": []}
    save(receipt, state)
    try:
        for name, command, result_name, expected in phases:
            for relative_name, digest in hashes.items():
                if sha(ROOT / relative_name) != digest:
                    raise RuntimeError(f"live source changed: {relative_name}")
            phase = {"name": name, "started_at_SGT": now(), "status": "RUNNING", "argv": command}
            state["phase"] = name
            state["phases"].append(phase)
            save(receipt, state)
            print(json.dumps(phase), flush=True)
            with (RUN / (name + ".log")).open("x") as log:
                proc = subprocess.Popen(
                    [str(PYTHON), *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                phase["pid"] = proc.pid
                save(receipt, state)
                while proc.poll() is None:
                    with suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=30)
                    gpu = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    with (RUN / "resources.jsonl").open("a") as resources:
                        resources.write(
                            json.dumps(
                                {
                                    "at_SGT": now(),
                                    "phase": name,
                                    "gpu": gpu.stdout.strip(),
                                    "disk_free_bytes": shutil.disk_usage(RUN).free,
                                }
                            )
                            + "\n"
                        )
                phase.update(exit_code=proc.returncode, finished_at_SGT=now())
                if proc.returncode != 0:
                    raise RuntimeError(f"phase {name} failed with exit {proc.returncode}")
            result = json.loads((RUN / name / result_name).read_text())
            if result["status"] != expected:
                raise RuntimeError(f"phase {name} missing completion receipt")
            phase["status"] = "COMPLETE"
            save(receipt, state)
        base = json.loads((RUN / "base_eval/result.json").read_text())
        after = json.loads((RUN / "sft_eval/result.json").read_text())
        sft = json.loads((RUN / "sft/summary.json").read_text())
        save(
            RUN / "readout.json",
            {
                "status": "COMPLETE",
                "finished_at_SGT": now(),
                "dataset": manifest,
                "sft": sft,
                "base": base["summary"],
                "after_sft": after["summary"],
                "interpretation_boundary": "Common SFT initialization; no M1/M2 reward comparison.",
            },
        )
        state.update(status="COMPLETE", finished_at_SGT=now())
    except BaseException as exc:
        if state["phases"] and state["phases"][-1]["status"] == "RUNNING":
            state["phases"][-1]["status"] = "FAILED"
        state.update(status="FAILED", failed_at_SGT=now(), error=repr(exc))
        save(receipt, state)
        raise
    save(receipt, state)


if __name__ == "__main__":
    main()

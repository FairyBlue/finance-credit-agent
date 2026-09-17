#!/usr/bin/env python3
"""One bounded on-policy state aggregation, SFT, and repeated Train acceptance."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/agent-r1"))

from src.agents.launcher import _resolve_c_compiler  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "_coldstart_v3_helper", ROOT / "scripts/diagnostics/m2_coldstart_v2_run.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("v2 controller helper cannot be loaded")
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)
PROTOCOL = HELPER.PROTOCOL
execute_phase = HELPER.execute_phase
model_identity = HELPER.model_identity
now, read, save, sha = HELPER.now, HELPER.read, HELPER.save, HELPER.sha
verify_dataset = HELPER.verify_dataset
training_phases = HELPER.phases

OLD = ROOT / "runs/m2-coldstart-v2-20260913"
RUN = ROOT / "runs/m2-coldstart-v3-20260914"
MODEL = OLD / "merged_model"
PLAN = ROOT / "docs/2026-09-14_M2_COLDSTART_V3_PLAN.md"


def phase(name, argv, result, status="COMPLETE"):
    return {"name": name, "argv": argv, "result": str(result), "expected_status": status}


def phases(run=RUN):
    prepared = run / "collection/prepared"
    rollout = run / "collection/rollout"
    return [
        phase("prepare", [
            "-m", "src.agents.coldstart_onpolicy_collect", "prepare", "--output", str(prepared),
        ], prepared / "manifest.json"),
        phase("collection", [
            "-m", "src.agents.coldstart_eval", "--model", str(MODEL),
            "--contexts", str(prepared / "collection_contexts.json"),
            "--output", str(rollout), "--action-protocol", PROTOCOL,
            "--expected-short", "0", "--expected-full", "8",
        ], rollout / "result.json", "OBSERVED"),
        phase("dataset", [
            "-m", "src.agents.coldstart_onpolicy_data", "--collection", str(rollout),
            "--prepared", str(prepared), "--output", str(run / "dataset"),
        ], run / "dataset/manifest.json"),
        # These existing phases fix two epochs and both acceptance suites. The
        # v2 after-results below supply the before-results without new sampling.
        *training_phases(run, MODEL)[2:],
    ]


def seal_sources(run):
    paths = [*sorted((ROOT / "src").rglob("*.py")), Path(__file__), PLAN,
             ROOT / "scripts/diagnostics/m2_coldstart_v2_run.py",
             ROOT / "scripts/diagnostics/m2_coldstart_v2_readout.py"]
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


def verify_old_runtime(old=OLD):
    if read(old / "controller.json").get("status") != "COMPLETE":
        raise RuntimeError("v2 results are not complete")
    # Reuse requires unchanged evaluation, rendering, environment and model
    # handling. Newly added v3 modules do not modify these old source files.
    for relative, digest in read(old / "source_manifest.json").items():
        if (relative.startswith("src/") or relative.endswith("m2_coldstart_v2_readout.py")) and (
            sha(ROOT / relative) != digest
        ):
            raise RuntimeError(f"cannot reuse baseline after source change: {relative}")


def reuse_baseline(run, initial, old=OLD):
    verify_old_runtime(old)
    exported = read(old / "exported_model_identity.json")
    if initial != exported:
        raise RuntimeError("selected v3 initial model differs from the v2 evaluated export")
    source_readout = read(old / "readout.json")
    evidence = {
        "status": "COMPLETE", "at_SGT": now(), "regenerated": False,
        "source_run": str(old), "model_identity": initial, "files": {},
        "source_artifacts_sha256": {
            name: sha(old / name) for name in (
                "readout.json", "controller.json", "source_manifest.json",
                "exported_model_identity.json",
            )
        },
        "interpretation": "v2 after evaluations reused as v3 before; repeated Train diagnostics",
    }
    for suite in ("regression", "fresh"):
        source = old / f"after_{suite}"
        config = read(source / "config.json")
        if config["engine"]["model"] != str(MODEL) or config["action_protocol"] != PROTOCOL:
            raise RuntimeError("v2 evaluation model/protocol mismatch")
        target = run / f"before_{suite}"
        target.mkdir(exist_ok=False)
        for name in ("result.json", "requests.jsonl", "config.json"):
            path = source / name
            digest = sha(path)
            key = f"after_{suite}/{name}"
            if name != "config.json" and source_readout["inputs_sha256"][key] != digest:
                raise RuntimeError(f"v2 evaluation evidence changed: {key}")
            shutil.copy2(path, target / name)
            if sha(target / name) != digest:
                raise RuntimeError("baseline evidence copy failed")
            evidence["files"][str((target / name).relative_to(run))] = {
                "source": str(path), "sha256": digest,
            }
    save(run / "baseline_reuse.json", evidence)
    return evidence


def check_frozen(run, sources, evidence, prepared=None, dataset=None):
    for relative, digest in sources.items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError(f"live source changed: {relative}")
    for relative, identity in evidence["files"].items():
        if sha(run / relative) != identity["sha256"]:
            raise RuntimeError(f"baseline copy changed: {relative}")
    for path, digest in (prepared or {}).items():
        if sha(run / path) != digest:
            raise RuntimeError(f"collection preparation changed: {path}")
    if dataset:
        if sha(run / "dataset/manifest.json") != dataset:
            raise RuntimeError("dataset manifest changed")
        manifest = verify_dataset(run)
        if not 741 <= manifest["row_count"] <= 1380:
            raise RuntimeError("aggregated dataset violates the fixed row budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=RUN)
    args = parser.parse_args()
    run = args.directory.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if any((run / item).exists() for item in ("controller.json", "sources", "collection", "sft")):
        raise RuntimeError("run identity already used; no retry or overwrite")
    if shutil.disk_usage(run).free <= 25 * 1024**3:
        raise RuntimeError("cold-start experiment needs at least 25 GiB free storage")
    verify_old_runtime()
    initial = model_identity(MODEL)
    sources = seal_sources(run)
    save(run / "model_identity.json", initial)
    evidence = reuse_baseline(run, initial)
    env = os.environ.copy()
    compiler, version = _resolve_c_compiler(parent=env)
    env.update(
        CC=compiler, PYTHONPATH=str(ROOT / "third_party/agent-r1") + ":" + str(ROOT),
        PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_USE_V1="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    save(run / "compiler.json", {"CC": compiler, "version": version})
    state = {
        "status": "RUNNING", "started_at_SGT": now(), "phases": [],
        "action_protocol": PROTOCOL, "automatic_retry": False,
        "automatic_rl_launch": False, "sft_epochs": 2,
        "planned_collection_trajectories": 40, "max_new_demonstrations": 640,
        "baseline_reused_from": str(OLD), "baseline_regenerated": False,
    }
    save(run / "controller.json", state)
    prepared, dataset = {}, None
    try:
        for item in phases(run):
            check_frozen(run, sources, evidence, prepared, dataset)
            if item["name"] in ("collection", "sft") and model_identity(MODEL) != initial:
                raise RuntimeError("v2 initializer changed before collection or training")
            execute_phase(run, item, state, env)
            if item["name"] == "prepare":
                prepared = {
                    str(path.relative_to(run)): sha(path)
                    for path in sorted((run / "collection/prepared").rglob("*"))
                    if path.is_file()
                }
                save(run / "collection_preparation_identity.json", prepared)
            elif item["name"] == "dataset":
                manifest = verify_dataset(run)
                dataset = sha(run / "dataset/manifest.json")
                state.update(dataset_rows=manifest["row_count"], dataset_manifest_sha256=dataset)
            elif item["name"] == "merged_model":
                merged = read(run / "merged_model/merge_receipt.json")
                if merged.get("model") != str(MODEL) or merged.get("adapter") != str(
                    run / "sft/adapter"
                ):
                    raise RuntimeError("export base/adapter differs from this run")
                save(run / "exported_model_identity.json", model_identity(run / "merged_model"))
            check_frozen(run, sources, evidence, prepared, dataset)
        report = read(run / "readout.json")
        state.update(status="COMPLETE", finished_at_SGT=now(),
                     candidate_gate_status=report["gates"]["status"])
    except BaseException as error:
        if state["phases"] and state["phases"][-1]["status"] == "RUNNING":
            state["phases"][-1]["status"] = "FAILED"
        state.update(status="FAILED", failed_at_SGT=now(), error=repr(error))
        save(run / "controller.json", state)
        raise
    save(run / "controller.json", state)
    print(state, flush=True)


if __name__ == "__main__":
    main()

"""Run one frozen common-SFT acceptance suite under the persistent GPU lease."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from src.agents import order_recovery_acceptance as acceptance
from src.agents import order_recovery_probe as probe
from src.agents.gpu_memory_coordinator import (
    GIB,
    PHASES,
    config_for_holder,
    controller_idle,
    freeze_config,
)
from src.agents.order_recovery_controller import release_for_inference
from src.agents.order_recovery_gpu_monitor import run_monitored


def run(directory: Path, holder: Path) -> dict:
    directory, holder = directory.resolve(), holder.resolve()
    state_path = directory / "controller_state.json"
    if state_path.exists():
        raise RuntimeError("Fresh execution required; no retry or overwrite")
    prepared = probe.read(directory / "state.json")
    digest = prepared.get("manifest_sha256")
    if prepared.get("status") != "PREPARED" or not isinstance(digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", digest
    ):
        raise ValueError("A frozen prepared acceptance digest is required")
    contract = acceptance.verify(directory, digest, check_models=False)
    state = {"status": "PREPARING", "manifest_sha256": digest, "models": [],
             "automatic_retry": False, "automatic_actions": False, "training_updates": 0}
    probe.save(state_path, state)
    try:
        config = config_for_holder(
            holder, lease_token=f"order-recovery-{directory.name}",
            evidence_directory=directory / "gpu_handoff",
            phase_peak_bytes={phase: 40 * GIB for phase in PHASES}, wait_timeout_seconds=120,
        )
        freeze_config(directory / "gpu_handoff/config.json", config)
        for key in acceptance.keys_for(contract["mode"]):
            # Hash the current actor before releasing the reservation; CPU work
            # needs no idle GPU. The worker verifies its weights again.
            model = contract["models"][key]
            probe.validate_model(Path(model["model_path"]), model)
            acceptance.verify(directory, digest, check_models=False)
            slot = {"model_key": key, "model": model["model_path"], "status": "PREPARING"}
            state["models"].append(slot)
            slot["admission"] = release_for_inference(config)
            slot["status"] = state["status"] = "RUNNING"
            command = [sys.executable, "-u", "-m", "src.agents.order_recovery_acceptance", "worker",
                       "--directory", str(directory), "--model-key", key,
                       "--expected-manifest-sha256", digest]
            slot["command"] = command
            probe.save(state_path, state)
            with (directory / f"worker-{key}.log").open("x") as log:
                result = run_monitored(
                    command, config=config, evidence_path=directory / f"gpu-{key}.jsonl",
                    cwd=probe.ROOT, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, check=False, timeout=2400,
                )
            slot.update(returncode=result.returncode,
                        status="VERIFYING" if result.returncode == 0 else "FAILED")
            probe.save(state_path, state)
            if result.returncode:
                raise RuntimeError(f"{key}: worker failed; remaining models unstarted")
            partial = acceptance.readout(directory)
            combined = directory / "workers" / key
            observed = probe.read(combined / "result.json")
            if (partial["short_workers"].get(key) != "OBSERVED"
                    or partial["full_workers"].get(key) != "OBSERVED"
                    or probe.read(combined / "state.json") != observed
                    or observed.get("status") != "OBSERVED"
                    or observed.get("model_key") != key
                    or observed.get("profile") != acceptance.PROFILE):
                raise RuntimeError(f"{key}: successful worker lacks complete observed evidence")
            slot["status"] = "COMPLETE"
            probe.save(state_path, state)
            deadline = time.monotonic() + 30
            while True:
                try:
                    controller_idle(config)
                    break
                except RuntimeError as error:
                    if "child processes remain" not in str(error) or time.monotonic() >= deadline:
                        raise
                    time.sleep(1)
        readout = acceptance.readout(directory)
        if readout["status"] != "OBSERVED":
            raise RuntimeError("All planned acceptance workers must be observed before completion")
        state.update(status="COMPLETE", behavioral_acceptance=readout["behavioral_gate"],
                     acceptance=readout["acceptance"],
                     raw_audit_required=True)
    except BaseException as error:
        if state["models"] and state["models"][-1]["status"] in {
            "PREPARING", "RUNNING", "VERIFYING"
        }:
            state["models"][-1].update(status="FAILED", error=f"{type(error).__name__}: {error}")
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        probe.save(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--holder-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.directory, args.holder_directory), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

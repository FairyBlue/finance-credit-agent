"""Run the fixed inference-only recovery probe under the existing GPU holder lease."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from src.agents.gpu_memory_coordinator import (
    GIB,
    PHASES,
    admission_snapshot,
    config_for_holder,
    controller_claim,
    controller_idle,
    freeze_config,
    reservation_status,
)
from src.agents.gpu_reservation_service import process_identity, request
from src.agents.order_recovery_gpu_monitor import run_monitored

MODEL_KEYS = ("sft", "seed42_m1", "seed42_m2", "seed43_m1", "seed43_m2")
ROOT = Path(__file__).resolve().parents[2]
SFT = ROOT / "runs/m2-coldstart-v3-20260914/merged_model"


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_prepared(directory: Path) -> str:
    """Bind execution to the prepared digest before touching the holder."""
    prepared = json.loads((directory / "state.json").read_text())
    digest = prepared.get("manifest_sha256") if isinstance(prepared, dict) else None
    if (
        not isinstance(prepared, dict) or prepared.get("status") != "PREPARED"
        or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise RuntimeError("recovery execution requires a prepared manifest SHA-256")
    from src.agents.order_recovery_probe import verify

    verify(directory, expected_sha256=digest)
    return digest


def require_live_claim(config: dict, state: dict, owner: dict) -> None:
    """Validate the final sampled lease as well as the holder's own identity."""
    claim = state.get("claim")
    if (
        not Path(owner["cgroup_path"]).name.endswith(".service")
        or not isinstance(claim, dict)
        or claim.get("lease_token") != config["lease_token"]
        or claim.get("owner") != owner
        or claim.get("cgroup_populated") is not True
        or claim.get("watchdog_error") is not None
    ):
        raise RuntimeError("inference requires this controller's healthy live service claim")


def release_for_inference(config: dict, *, timeout: float = 120) -> dict:
    controller_claim(config)
    available = admission_snapshot(config)
    if available["available_bytes"] < 40 * GIB:
        raise RuntimeError("less than 40 GiB of reclaimable inference capacity")
    owner = process_identity(os.getpid())
    state = reservation_status(config)
    require_live_claim(config, state, owner)
    if state["payload_bytes"]:
        request(config["directory"], {
            "op": "release", "lease_token": config["lease_token"],
            "bytes": state["payload_bytes"],
            "request_id": f"recovery-release-{os.getpid()}-{time.time_ns()}",
        })
    deadline = time.monotonic() + timeout
    while True:
        available = admission_snapshot(config)
        state = available["holder_status"]
        require_live_claim(config, state, owner)
        if (
            state["paused"] is True
            and all(state[key] == 0 for key in (
                "payload_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes"
            ))
            and available["driver_free_bytes"] >= 40 * GIB
        ):
            return {
                **available,
                "controller_owner": owner,
                "descendant_cleanup_contract": "launcher-enforced systemd KillMode=control-group",
            }
        if time.monotonic() >= deadline:
            raise TimeoutError("inference capacity unavailable after holder release")
        time.sleep(1)


def run(
    directory: Path, model_root: Path, holder: Path, episode_directory: Path | None = None
) -> dict:
    directory, model_root, holder = (path.resolve() for path in (directory, model_root, holder))
    state_path = directory / "controller_state.json"
    if state_path.exists():
        raise RuntimeError("controller requires a fresh execution; no retry or overwrite")
    state = {"status": "PREPARING", "automatic_retry": False, "training": False, "models": []}
    save(state_path, state)
    try:
        digest = verify_prepared(directory)
        state["manifest_sha256"] = digest
        episode_digest = None
        if episode_directory is not None:
            from src.agents import order_recovery_episode as episode

            episode_directory = episode_directory.resolve()
            prepared = json.loads((episode_directory / "state.json").read_text())
            if prepared.get("status") != "PREPARED":
                raise RuntimeError("Complete-window evaluation requires a fresh prepared manifest")
            episode_digest = prepared["manifest_sha256"]
            episode.verify(episode_directory, episode_digest)
            state["episode_manifest_sha256"] = episode_digest
        models = [(key, SFT if key == "sft" else model_root / key) for key in MODEL_KEYS]
        if any(not (model / "config.json").is_file() for _, model in models):
            raise RuntimeError("all five verified model directories must exist before launch")
        config = config_for_holder(
            holder, lease_token=f"order-recovery-{directory.name}",
            evidence_directory=directory / "gpu_handoff",
            phase_peak_bytes={phase: 40 * GIB for phase in PHASES},
            wait_timeout_seconds=120,
        )
        freeze_config(directory / "gpu_handoff/config.json", config)
        state["status"] = "RUNNING"
        save(state_path, state)
        for key, model in models:
            slot = {"model_key": key, "model": str(model), "status": "PREPARING"}
            state["models"].append(slot)
            slot["admission"] = release_for_inference(config)
            slot["status"] = "RUNNING"
            save(state_path, state)
            command = [
                sys.executable, "-u", "-m", "src.agents.order_recovery_probe", "worker",
                "--directory", str(directory), "--model-key", key, "--model", str(model),
                "--expected-manifest-sha256", digest,
            ]
            if episode_directory is not None:
                command.extend([
                    "--episode-directory", str(episode_directory),
                    "--expected-episode-sha256", episode_digest,
                ])
            slot["command"] = command
            with (directory / f"worker-{key}.log").open("x") as log:
                result = run_monitored(
                    command, config=config, evidence_path=directory / f"gpu-{key}.jsonl",
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, check=False, timeout=1800,
                )
            slot["returncode"] = result.returncode
            slot["status"] = "COMPLETE" if result.returncode == 0 else "FAILED"
            save(state_path, state)
            if result.returncode:
                raise RuntimeError(f"{key} inference worker failed; remaining slots unstarted")
            deadline = time.monotonic() + 30
            while True:
                try:
                    controller_idle(config)
                    break
                except RuntimeError as error:
                    if (
                        "child processes remain" not in str(error)
                        or time.monotonic() >= deadline
                    ):
                        raise
                    time.sleep(1)
        subprocess.run(
            [sys.executable, "-m", "src.agents.order_recovery_probe", "readout",
             "--directory", str(directory)], cwd=ROOT, check=True, timeout=120,
        )
        if episode_directory is not None:
            subprocess.run(
                [sys.executable, "-m", "src.agents.order_recovery_episode", "readout",
                 "--directory", str(episode_directory)], cwd=ROOT, check=True, timeout=120,
            )
        state["status"] = "COMPLETE"
    except BaseException as error:
        if state["models"]:
            active = state["models"][-1]
            if active["status"] in {"PREPARING", "RUNNING"}:
                active.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--holder-directory", type=Path, required=True)
    parser.add_argument("--episode-directory", type=Path)
    args = parser.parse_args()
    run(args.directory, args.model_root, args.holder_directory, args.episode_directory)


if __name__ == "__main__":
    main()

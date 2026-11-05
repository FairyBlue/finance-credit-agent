#!/usr/bin/env python3
"""Small controller: start GPU promptly, overlap CPU prep, retain lease guards."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCRIPTS = Path(__file__).resolve().parent
DEFAULT = ROOT / "runs/m2-stochastic-continuation-20261018-v1"
PREVIOUS = ROOT / "runs/m2-same-prefix-sampling-20261018-v1"
HOLDER = ROOT / "runs/gpu-reservation-persistent-20261007"
PLAN = ROOT / "docs/2026-10-18_M2_STOCHASTIC_CONTINUATION_PLAN.md"
GIB = 1024**3


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: object) -> None:
    with path.open("x") as f:
        json.dump(value, f, allow_nan=False, indent=2)
        f.write("\n")


def verify(c: dict) -> None:
    for p, digest in c["startup_source_sha256"].items():
        if sha(Path(p)) != digest:
            raise RuntimeError(f"startup source changed: {p}")


def prepare(out: Path) -> None:
    from src.agents.gpu_memory_coordinator import PHASES, config_for_holder, freeze_config

    if out.exists():
        raise RuntimeError("new output directory required")
    old = read(PREVIOUS / "contract.json")
    model = Path(old["model_path"])
    prior = read(PREVIOUS / "model_preflight_receipt.json")
    if sha(model / "merge_receipt.json") != prior["merge_receipt_sha256"]:
        raise RuntimeError("SFT merge identity changed")
    model_stat = {}
    for relative, entry in read(model / "merge_receipt.json")["files"].items():
        p = model / relative
        st = p.stat()
        if p.is_symlink() or st.st_size != entry["bytes"]:
            raise RuntimeError(f"model export size/type differs: {relative}")
        if st.st_mtime > datetime.fromisoformat(prior["at_UTC"]).timestamp():
            raise RuntimeError(f"model file modified after verified preflight: {relative}")
        model_stat[relative] = {
            "bytes": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "ctime_ns": st.st_ctime_ns,
            "inode": st.st_ino,
        }
    driver = SCRIPTS / "m2_continuation_driver_v2.py"
    spec = importlib.util.spec_from_file_location("continuation_manifest_builder", driver)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    out.mkdir(parents=True)
    module.prepare_manifest(out)
    config = config_for_holder(
        HOLDER,
        lease_token=f"continuation-{out.name}",
        evidence_directory=out / "gpu_handoff",
        portfolio_target_bytes=76 * GIB,
        phase_peak_bytes={p: 40 * GIB for p in PHASES},
        minimum_free_bytes=2 * GIB,
        wait_timeout_seconds=120,
    )
    config_file = out / "gpu_handoff/config.json"
    freeze_config(config_file, config)
    startup_files = [
        Path(__file__).resolve(),
        SCRIPTS / "m2_continuation_gpu_worker.py",
        PLAN,
        ROOT / "src/agents/gpu_memory_coordinator.py",
        ROOT / "src/agents/gpu_reservation_service.py",
        ROOT / "src/agents/order_recovery_controller.py",
    ]
    write(
        out / "contract.json",
        {
            "status": "SEALED_BEFORE_GPU_LOAD",
            "profile": "sft_stochastic_continuation_feasibility_v1",
            "sealed_at_UTC": now(),
            "model_path": str(model),
            "model_fingerprint": prior["model_fingerprint"],
            "merge_receipt_sha256": prior["merge_receipt_sha256"],
            "model_export_stat": model_stat,
            "model_identity_check": (
                "reuse successful full-hash receipt; fresh size/type/mtime plus small merge SHA"
            ),
            "prior_preflight_sha256": sha(PREVIOUS / "model_preflight_receipt.json"),
            "startup_source_sha256": {str(p): sha(p) for p in startup_files},
            "manifest_sha256": sha(out / "continuation_manifest.json"),
            "seed_table_sha256": sha(out / "request_seed_table.jsonl"),
            "resource_config_sha256": sha(config_file),
            "request_limit": 6904,
            "planned_cells": 77,
            "planned_trajectories": 154,
            "gpu_outer_limit_bytes": 95 * GIB,
            "gpu_minimum_free_bytes": 2 * GIB,
            "CPU_driver_seal": (
                "independent seal before CPU spawn and first generation, while GPU initializes"
            ),
            "startup_policy": (
                "GPU load overlaps CPU preparation; repeated full inventory audit deferred"
            ),
            "optimizer_updates": 0,
            "formal_valid_oos_access": False,
            "automatic_restart": False,
        },
    )
    print(json.dumps({"status": "PREPARED", "output": str(out)}), flush=True)


def stop(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def controller(out: Path) -> None:
    from src.agents.gpu_memory_coordinator import (
        _driver_snapshot,
        controller_idle,
        load_config,
        reservation_status,
    )
    from src.agents.gpu_reservation_service import process_identity
    from src.agents.order_recovery_controller import release_for_inference, require_live_claim

    c = read(out / "contract.json")
    verify(c)
    for relative, expected in c["model_export_stat"].items():
        st = (Path(c["model_path"]) / relative).stat()
        if any(
            getattr(st, field) != expected[key]
            for field, key in [
                ("st_size", "bytes"),
                ("st_mtime_ns", "mtime_ns"),
                ("st_ctime_ns", "ctime_ns"),
                ("st_ino", "inode"),
            ]
        ):
            raise RuntimeError(f"model stat changed after seal: {relative}")
    if (
        sha(out / "continuation_manifest.json") != c["manifest_sha256"]
        or sha(out / "request_seed_table.jsonl") != c["seed_table_sha256"]
    ):
        raise RuntimeError("continuation manifest or seeds changed")
    config = load_config((out / "gpu_handoff/config.json").resolve(), c["resource_config_sha256"])
    owner = process_identity(os.getpid())
    write(
        out / "controller_start.json",
        {"at_UTC": now(), "owner": owner, "contract_sha256": sha(out / "contract.json")},
    )
    write(out / "gpu_admission.json", release_for_inference(config))
    worker = driver = None
    request_r, request_w = os.pipe()
    response_r, response_w = os.pipe()
    worker_command = [
        sys.executable,
        "-u",
        "-B",
        str(SCRIPTS / "m2_continuation_gpu_worker.py"),
        "--output",
        str(out),
    ]
    peak, started = 0, time.monotonic()
    try:
        with (
            (out / "worker_bootstrap.log").open("x") as boot,
            (out / "driver.log").open("x") as driver_log,
            (out / "gpu_observations.jsonl").open("x") as evidence,
        ):
            # No CPU replay or full inventory hashing before the GPU child starts.
            worker = subprocess.Popen(
                worker_command,
                cwd=ROOT,
                stdin=request_r,
                stdout=response_w,
                stderr=boot,
                start_new_session=True,
            )
            os.close(request_r)
            os.close(response_w)
            request_r = response_w = -1
            driver_deadline = time.monotonic() + 600
            while True:
                state = reservation_status(config)
                require_live_claim(config, state, owner)
                if not state["paused"] or any(
                    state[k]
                    for k in (
                        "payload_bytes",
                        "allocator_allocated_bytes",
                        "allocator_reserved_bytes",
                    )
                ):
                    raise RuntimeError("holder must remain paused-zero during inference")
                sample = _driver_snapshot()
                if sample["gpu_uuid"] != config["gpu_uuid"]:
                    raise RuntimeError("GPU UUID changed")
                used = sample["driver_total_bytes"] - sample["driver_free_bytes"]
                peak = max(peak, used)
                limit = min(
                    c["gpu_outer_limit_bytes"],
                    sample["driver_total_bytes"] - c["gpu_minimum_free_bytes"],
                )
                evidence.write(
                    json.dumps(
                        {
                            **sample,
                            "at_UTC": now(),
                            "aggregate_used_bytes": used,
                            "sampled_peak_bytes": peak,
                            "stop_threshold_bytes": limit,
                        }
                    )
                    + "\n"
                )
                evidence.flush()
                if used > limit or time.monotonic() - started > 10800:
                    raise RuntimeError("GPU bound/headroom or three-hour runtime limit")
                if driver is None and (out / "driver_ready.json").exists():
                    seal = read(out / "driver_ready.json")
                    for p, digest in seal["source_sha256"].items():
                        if sha(Path(p)) != digest:
                            raise RuntimeError(f"CPU driver changed after ready seal: {p}")
                    write(out / "driver_start_seal.json", {**seal, "at_UTC": now()})
                    driver_command = [
                        sys.executable,
                        "-u",
                        "-B",
                        str(SCRIPTS / "m2_continuation_driver_v2.py"),
                        "run",
                        "--output",
                        str(out),
                        "--request-fd",
                        str(request_w),
                        "--response-fd",
                        str(response_r),
                    ]
                    driver = subprocess.Popen(
                        driver_command,
                        cwd=ROOT,
                        stdout=driver_log,
                        stderr=subprocess.STDOUT,
                        pass_fds=(request_w, response_r),
                        start_new_session=True,
                    )
                    os.close(request_w)
                    os.close(response_r)
                    request_w = response_r = -1
                    write(out / "driver_launch.json", {"at_UTC": now(), "command": driver_command})
                if driver is None and time.monotonic() > driver_deadline:
                    raise RuntimeError("CPU driver readiness timeout")
                wc = worker.poll()
                dc = driver.poll() if driver is not None else None
                if wc is not None and wc != 0:
                    raise RuntimeError(f"GPU worker actual exit {wc}")
                if dc is not None and dc != 0:
                    raise RuntimeError(f"CPU driver actual exit {dc}")
                if wc == 0 and dc == 0:
                    break
                if wc == 0 and driver is None:
                    raise RuntimeError("worker exited before driver")
                time.sleep(1)
        write(
            out / "process_exit_receipt.json",
            {
                "at_UTC": now(),
                "actual_worker_exit_code": wc,
                "actual_driver_exit_code": dc,
                "sampled_peak_gpu_bytes": peak,
            },
        )
        deadline = time.monotonic() + 30
        while True:
            try:
                idle = controller_idle(config)
                break
            except RuntimeError as error:
                if "child processes remain" not in str(error) or time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        write(out / "holder_idle_request.json", idle)
        write(
            out / "controller_completion.json",
            {
                "status": "COMPLETE",
                "at_UTC": now(),
                "actual_worker_exit_code": wc,
                "actual_driver_exit_code": dc,
                "sampled_peak_gpu_bytes": peak,
            },
        )
    except BaseException as error:
        write(
            out / "controller_failure.json",
            {
                "at_UTC": now(),
                "error": repr(error),
                "actual_worker_exit_code": worker.poll() if worker else None,
                "actual_driver_exit_code": driver.poll() if driver else None,
            },
        )
        raise
    finally:
        stop(driver)
        stop(worker)
        for fd in (request_r, request_w, response_r, response_w):
            if fd >= 0:
                os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "controller", "record-exit"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.mode == "prepare":
        prepare(out)
    elif args.mode == "controller":
        controller(out)
    else:
        write(
            out / "service_exit_receipt.json",
            {
                "at_UTC": now(),
                **{k: os.environ.get(k) for k in ("SERVICE_RESULT", "EXIT_CODE", "EXIT_STATUS")},
            },
        )


if __name__ == "__main__":
    main()

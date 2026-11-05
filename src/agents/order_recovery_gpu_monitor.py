"""Sample aggregate GPU occupancy during the inference-only recovery worker.

This is a conservative device-wide stop threshold, not process attribution or
an allocator limit. The dedicated systemd service cleans up all descendants.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from src.agents.gpu_memory_coordinator import GIB, _driver_snapshot

LIMIT_BYTES = 80 * GIB


def run_monitored(
    command, *, config: dict, evidence_path: Path, timeout: float, check: bool = False, **kwargs
) -> subprocess.CompletedProcess:
    started = time.monotonic()
    process = None
    peak = 0
    with evidence_path.open("x") as evidence:
        try:
            while True:
                snapshot = _driver_snapshot()
                if snapshot["gpu_uuid"] != config["gpu_uuid"]:
                    raise RuntimeError("GPU UUID changed during recovery inference")
                used = snapshot["driver_total_bytes"] - snapshot["driver_free_bytes"]
                peak = max(peak, used)
                elapsed = time.monotonic() - started
                evidence.write(
                    json.dumps(
                        {
                            **snapshot,
                            "elapsed_seconds": elapsed,
                            "aggregate_used_bytes": used,
                            "sampled_peak_bytes": peak,
                            "stop_threshold_bytes": LIMIT_BYTES,
                            "process_attribution": False,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                evidence.flush()
                if used > LIMIT_BYTES:
                    raise RuntimeError("aggregate GPU occupancy exceeded 80 GiB stop threshold")
                if elapsed > timeout:
                    raise subprocess.TimeoutExpired(command, timeout)
                if process is None:
                    process = subprocess.Popen(command, **kwargs)
                returncode = process.poll()
                if returncode is not None:
                    result = subprocess.CompletedProcess(command, returncode)
                    if check:
                        result.check_returncode()
                    return result
                time.sleep(1)
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

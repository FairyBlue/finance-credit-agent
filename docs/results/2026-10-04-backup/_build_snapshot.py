"""Copy the completed pilot's small receipts; never copy model or optimizer data."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent
PILOT = "runs/m2-low-memory-handoff-20261004"
FILES = [
    f"{PILOT}/{name}"
    for name in (
        "FINAL_RESULTS.md", "LIVE_STATUS.md", "comparison_summary.csv",
        "completion_readout.json", "state.json", "settings.json", "run_contract.json",
        "model_identity.json", "source_identity.json", "selection.json", "plan.md",
        "implementation_validation.json", "input_preflight.json", "runtime_preflight.json",
        "launch_receipt.json", "gpu_handoff_config.json", "reservation_cleanup.json",
    )
] + [
    f"{PILOT}/{arm}/{name}"
    for arm in ("m1", "m2_eta025")
    for name in (
        "summary.json", "process.json",
        "checkpoints/global_step_2/actor/actor_export_receipt.json",
        "checkpoints/global_step_2/actor_export_complete.json",
    )
] + [
    "runs/m2-low-memory-pilot-20260930/state.json",
    "runs/m2-low-memory-pilot-20260930/FINAL_RESULTS.md",
    "runs/m2-low-memory-pilot-20260930/failure_diagnosis_20261002.json",
    "runs/m2-low-memory-preparation-20260930/implementation_validation.json",
    "runs/m2-low-memory-preparation-20260930/optimizer_cuda_probe_v2.json",
    "runs/gpu-reservation-preparation-20261002/cuda_probe.json",
    "runs/gpu-reservation-20261004/launch_receipt.json",
    "runs/gpu-reservation-20261004/status.json",
]


def main() -> None:
    completed = json.loads((ROOT / PILOT / "completion_readout.json").read_text())
    if completed.get("status") != "PASS":
        raise RuntimeError("snapshot requires the completed engineering pilot")
    payloads = [(name, (ROOT / name).read_bytes()) for name in FILES]
    records = []
    for name, payload in payloads:
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        records.append({
            "path": name, "source_path": name, "size_bytes": len(payload),
            "sha256": digest, "source_sha256": digest, "copy_mode": "byte_for_byte",
        })
    manifest = {
        "schema_version": 1,
        "captured_at_sgt": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "scope": "Small result receipts only; no weights, optimizer, or full trajectories",
        "files": records,
    }
    (DEST / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Copied {len(records)} payloads ({sum(len(p) for _, p in payloads)} bytes)")


if __name__ == "__main__":
    main()

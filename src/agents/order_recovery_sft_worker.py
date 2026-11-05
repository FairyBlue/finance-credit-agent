"""Fresh SFT child with a fixed native CUDA allocator budget before model load.

The 60 GiB cap covers this process's PyTorch allocator, not CUDA context or other
GPU processes. The controller's independent 80 GiB aggregate stop remains needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ALLOCATOR_CONFIG = "backend:native,expandable_segments:True"
ALLOCATOR_CAP_BYTES = 60 * 1024**3


def load_torch():
    import torch

    return torch


def delegate(argv):
    from src.agents.coldstart_sft import main

    main(argv)


def configure_allocator(torch, receipt: Path, output: Path) -> dict:
    if receipt.exists() or output.exists():
        raise RuntimeError("Fresh allocator receipt and training output required; no overwrite")
    if torch.cuda.memory.get_allocator_backend() != "native":
        raise RuntimeError("The fixed 60 GiB allocator cap requires the native backend")
    total = int(torch.cuda.get_device_properties(0).total_memory)
    if total < ALLOCATOR_CAP_BYTES:
        raise RuntimeError("GPU total memory is smaller than the frozen 60 GiB allocator cap")
    fraction = ALLOCATOR_CAP_BYTES / total
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    actual_fraction = float(torch.cuda.get_per_process_memory_fraction(device=0))
    if not math.isclose(actual_fraction, fraction, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("The child allocator did not acknowledge its frozen memory cap")
    payload = {
        "status": "CONFIGURED",
        "allocator_config": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        "allocator_backend": torch.cuda.memory.get_allocator_backend(),
        "allocator_cap_bytes": ALLOCATOR_CAP_BYTES,
        "device_total_bytes": total,
        "requested_fraction": fraction,
        "actual_fraction": actual_fraction,
        "cuda_device": 0,
        "torch_version": torch.__version__,
        "training_output": str(output.resolve()),
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "covers_context_or_other_processes": False,
        "aggregate_controller_stop_bytes": 80 * 1024**3,
    }
    with receipt.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocator-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args, remaining = parser.parse_known_args(argv)
    if args.output.exists() or args.allocator_receipt.exists():
        raise RuntimeError("Fresh allocator receipt and training output required; no overwrite")
    if not args.allocator_receipt.parent.is_dir():
        raise RuntimeError("Allocator receipt parent must already exist")
    if "torch" in sys.modules:
        raise RuntimeError("SFT worker must fix its allocator before importing torch")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ALLOCATOR_CONFIG
    torch = load_torch()
    configure_allocator(torch, args.allocator_receipt, args.output)
    delegate(["train", *remaining, "--output", str(args.output), "--device", "cuda"])


if __name__ == "__main__":
    main()

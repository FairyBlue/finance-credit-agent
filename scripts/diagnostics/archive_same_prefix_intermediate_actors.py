"""Losslessly archive selected preserved seed42 actors to make run space.

Uses the existing tensor XOR archive; experiment records and HF metadata stay
in place. Only verified original safetensors are retired, with a restore pointer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.parameter_grid_archive import pack_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--arms", nargs="+", choices=("zero", "plain", "exact"),
        default=("zero", "plain", "exact"),
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/m2-same-prefix-storage-20261024-v1",
    )
    args = parser.parse_args()
    source = ROOT / "runs/m2-same-prefix-online-demo-20261024-v2"
    baseline = ROOT / "runs/m2-order-recovery-sft-20261011-v1/merged_model"
    destination = args.output.resolve()
    if destination.parent != ROOT / "runs" or not destination.name.startswith(
        "m2-same-prefix-storage-"
    ):
        parser.error("archive output must be a new same-prefix storage directory under runs")
    if len(set(args.arms)) != len(args.arms):
        parser.error("duplicate source arm")
    destination.mkdir(exist_ok=False)
    receipts = []
    for arm in args.arms:
        model = source / arm / f"round{args.round}/train/actor"
        archive = destination / f"seed42-{arm}-round{args.round}"
        print(f"PACKING {model}", flush=True)
        result = pack_model(model, baseline, archive)
        manifest = json.loads((archive / "manifest.json").read_text())
        paths = []
        for name, row in manifest["shards"].items():
            path = model / name
            stat = path.stat()
            if (
                path.is_symlink()
                or stat.st_size != row["original_bytes"]
                or stat.st_dev != row["original_device"]
                or stat.st_ino != row["original_inode"]
            ):
                raise RuntimeError("verified source file changed before retirement")
            paths.append(path)
        receipt = {
            **result,
            "source": str(model),
            "baseline_must_be_kept": str(baseline),
            "status": "PACK_VERIFIED",
            "restore_api": "src.agents.parameter_grid_archive.restore_model",
            "restore_to_new_directory": str(model.with_name("actor_restored")),
            "raw_shards": [str(p) for p in paths],
        }
        pointer = model / "LOSSLESS_ARCHIVE.json"
        with pointer.open("x") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        for path in paths:
            path.unlink()
        receipt["status"] = "PACK_VERIFIED_RAW_RETIRED"
        pointer.write_text(json.dumps(receipt, indent=2) + "\n")
        receipts.append(receipt)
        (destination / "storage_summary.json").write_text(
            json.dumps({"models": receipts}, indent=2) + "\n"
        )
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()

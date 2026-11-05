#!/usr/bin/env python3
"""Run the frozen own-policy diagnostic with independent replication seeds."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = Path(__file__).with_name("m2_onpolicy_credit_driver.py")
BASE_SHA = "99671ef8008e8ec66fa002bd4139bedcec3f91d8f3c697181cddeac463ebe07c"
REPLICATION_ID = "m2-onpolicy-credit-replication-20261019-v1"
DEFAULT = ROOT / "runs" / REPLICATION_ID
REFERENCE = ROOT / "runs/m2-onpolicy-credit-20261018-v1"
SELECTION = ROOT / "runs/m2-onpolicy-credit-replication-selection-20261019-v1.json"


def load_base():
    if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA:
        raise RuntimeError("frozen base driver changed")
    spec = importlib.util.spec_from_file_location("frozen_onpolicy_replication_driver", BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base()


def request_seed(trajectory_key: str, session: int, attempt: int) -> int:
    role = {"parent": "PARENT", "head": "HEAD", "est": "EST", "ref": "REF"}.get(
        trajectory_key.split("/", 1)[0]
    )
    BASE.require(role is not None, "explicit independent replication seed role")
    payload = [REPLICATION_ID, role, trajectory_key, session, attempt]
    return int.from_bytes(hashlib.sha256(BASE.canonical(payload).encode()).digest()[:8], "big") & (
        (1 << 63) - 1
    )


BASE.request_seed = request_seed


def prepare_manifest(output: Path | str) -> dict:
    output = Path(output).resolve()
    selection = BASE.read(SELECTION)
    BASE.require(selection["eligible_new_windows"] < 8, "same-window fallback requires audit")
    manifest = BASE.prepare_manifest(output)
    fresh = {row["request_seed"] for row in BASE.rows(output / "request_seed_table.jsonl")}
    prior_tables = [
        ROOT / "runs/m2-stochastic-continuation-20261018-v3/request_seed_table.jsonl",
        ROOT / "runs/m2-stochastic-credit-comparison-20261018-v1/request_seed_table.jsonl",
        REFERENCE / "request_seed_table.jsonl",
    ]
    for table in prior_tables:
        old = {row["request_seed"] for row in BASE.rows(table)}
        BASE.require(not fresh.intersection(old), "no overlap with any old potential seed")
    manifest["replication"] = {
        "replication_id": REPLICATION_ID,
        "seed_namespace": REPLICATION_ID,
        "window_selection": "same_eight_train_windows_fresh_seeds_no_date_generalization",
        "reference_run": str(REFERENCE),
        "directional_rule": "both alpha025 and alpha050 candidate-minus-zero < -1e-12 "
        "in full8,ref0_3,ref4_7; descriptive only",
        "date_selection_receipt": str(SELECTION),
        "prior_potential_seed_overlap": 0,
        "base_driver_source_sha256": BASE_SHA,
    }
    sources = [Path(__file__).resolve(), BASE_PATH, SELECTION, *prior_tables]
    manifest["source_file_sha256"].update({str(path): BASE.sha(path) for path in sources})
    temporary = output / "continuation_manifest.tmp.json"
    temporary.write_text(BASE.canonical(manifest) + "\n")
    temporary.replace(output / "continuation_manifest.json")
    return manifest


async def run(output: Path, request_fd: int, response_fd: int) -> None:
    manifest = BASE.read(output / "continuation_manifest.json")
    replication = manifest["replication"]
    BASE.require(
        replication["replication_id"] == replication["seed_namespace"] == REPLICATION_ID,
        "fixed replication identity and seed namespace",
    )
    for row in BASE.rows(output / "request_seed_table.jsonl"):
        BASE.require(
            row["request_seed"]
            == request_seed(row["trajectory_key"], row["decision_session_index"], row["attempt"]),
            "every potential request uses the frozen new namespace",
        )
    await BASE.run(output, request_fd, response_fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--request-fd", type=int)
    parser.add_argument("--response-fd", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        manifest = prepare_manifest(args.output)
        print(json.dumps({"status": "PREPARED", "request_limit": manifest["max_model_requests"]}))
    else:
        BASE.require(
            args.request_fd is not None and args.response_fd is not None, "IPC descriptors"
        )
        asyncio.run(run(args.output.resolve(), args.request_fd, args.response_fd))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read an independent-seed replication without changing the frozen nine methods.

The frozen reader writes its original outputs. This wrapper adds a separately
named descriptive replication receipt; no alpha is selected and batches are
neither pooled nor restricted to an intersection of visited states.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
REPLICATION_ID = "m2-onpolicy-credit-replication-20261019-v1"
PRIOR = ROOT / "runs/m2-onpolicy-credit-20261018-v1"
SELECTION = ROOT / "runs/m2-onpolicy-credit-replication-selection-20261019-v1.json"
WINDOW_RULE = "same_eight_train_windows_fresh_seeds_no_date_generalization"
DIRECTION_RULE = (
    "both alpha025 and alpha050 candidate-minus-zero < -1e-12 "
    "in full8,ref0_3,ref4_7; descriptive only"
)
FROZEN_SOURCES = {
    "m2_onpolicy_credit_driver.py": (
        "99671ef8008e8ec66fa002bd4139bedcec3f91d8f3c697181cddeac463ebe07c"
    ),
    "m2_onpolicy_credit_readout.py": (
        "9df2c3c4d31fa100ad607bc2e15a8a78de7173ad0a86a8e2b3b0eb2187f6184c"
    ),
    "m2_credit_comparison_readout.py": (
        "c89ef20baab2bfa346dd715ee8af0d72ed60e5ed2d0dfc91b2b26e6b5bee3b3d"
    ),
}
PANELS = ("full8", "ref0_3", "ref4_7")
ALPHAS = ("exact_pool_shrink025", "exact_pool_shrink050")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_frozen_reader():
    for name, digest in FROZEN_SOURCES.items():
        require(sha(SCRIPTS / name) == digest, f"frozen source: {name}")
    specification = importlib.util.spec_from_file_location(
        "frozen_onpolicy_credit_readout", SCRIPTS / "m2_onpolicy_credit_readout.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def check_replication_inputs(directory: Path) -> tuple[dict, dict]:
    manifest = read(directory / "continuation_manifest.json")
    metadata = manifest["replication"]
    require(
        metadata
        == {
            "replication_id": REPLICATION_ID,
            "seed_namespace": REPLICATION_ID,
            "window_selection": WINDOW_RULE,
            "reference_run": str(PRIOR),
            "directional_rule": DIRECTION_RULE,
            "date_selection_receipt": str(SELECTION),
        },
        "fixed replication scope and joint directional rule",
    )
    prior_manifest = read(PRIOR / "continuation_manifest.json")
    require(manifest["parents"] == prior_manifest["parents"], "same fixed eight Train windows")
    require(
        manifest["policy_snapshot_sha256"] == prior_manifest["policy_snapshot_sha256"],
        "same frozen continuation policy",
    )
    seed_path = directory / "request_seed_table.jsonl"
    prior_seed_path = PRIOR / "request_seed_table.jsonl"
    require(sha(seed_path) == manifest["request_seed_table_sha256"], "new seed table binding")
    require(
        sha(prior_seed_path) == prior_manifest["request_seed_table_sha256"],
        "prior seed table binding",
    )
    require(
        manifest["source_file_sha256"][str(SELECTION)] == sha(SELECTION), "date selection receipt"
    )
    namespace = {"parent": "PARENT", "head": "HEAD", "est": "EST", "ref": "REF"}
    seeds, prior_seeds = set(), set()
    with seed_path.open() as stream:
        for line in stream:
            record = json.loads(line)
            key = record["trajectory_key"]
            payload = [
                REPLICATION_ID,
                namespace[key.split("/", 1)[0]],
                key,
                record["decision_session_index"],
                record["attempt"],
            ]
            encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
            expected = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)
            require(record["request_seed"] == expected, "fixed independent seed namespace")
            seeds.add(expected)
    with prior_seed_path.open() as stream:
        prior_seeds.update(json.loads(line)["request_seed"] for line in stream)
    require(
        len(seeds) == len(prior_seeds) == 23744 and seeds.isdisjoint(prior_seeds),
        "two distinct, disjoint preassigned seed tables",
    )
    return metadata, {
        "new_seed_table_sha256": sha(seed_path),
        "prior_seed_table_sha256": sha(prior_seed_path),
        "date_selection_receipt_sha256": sha(SELECTION),
        "seed_slots_each": len(seeds),
        "seed_sets_disjoint": True,
    }


def directional_readout(report: dict) -> dict:
    """All six predeclared inequalities must pass; no choosing one alpha."""
    pairs = {(row["candidate"], row["comparator"]): row for row in report["paired_comparisons"]}
    supported = report["counts"]["supported_slot_replicates"]
    output = []
    for alpha in ALPHAS:
        row = pairs[alpha, "zero"]
        values = {panel: row[f"candidate_minus_comparator_delta_{panel}"] for panel in PANELS}
        passes = bool(supported) and all(
            value is not None and value < -1e-12 for value in values.values()
        )
        output.append(
            {
                "method": alpha,
                "candidate_minus_zero": values,
                "all_three_predeclared_directions_pass": passes,
            }
        )
    joint = all(row["all_three_predeclared_directions_pass"] for row in output)
    return {
        "rule": DIRECTION_RULE,
        "threshold": -1e-12,
        "fixed_candidates": output,
        "joint_status": "PASS_DESCRIPTIVE_DIRECTION_ONLY"
        if joint
        else "NOT_DIRECTIONALLY_REPLICATED"
        if supported
        else "INSUFFICIENT_SUPPORT",
        "no_alpha_selection": True,
        "precision_or_significance_claim": False,
    }


def run(directory: Path) -> None:
    require(not (directory / "replication_readout.json").exists(), "no receipt overwrite")
    base = load_frozen_reader()
    metadata, inputs = check_replication_inputs(directory)
    base.run(directory)
    report = read(directory / "onpolicy_readout.json")
    prior_report_path = PRIOR / "onpolicy_readout.json"
    prior = read(prior_report_path)
    require(
        prior["readout_source_sha256"]
        == report["readout_source_sha256"]
        == FROZEN_SOURCES["m2_onpolicy_credit_readout.py"],
        "unchanged readout method",
    )
    require(
        report["counts"]["planned_slot_replicates"] == 128
        and report["counts"]["planned_group_replicates"] == 32,
        "own full planned denominator",
    )
    receipt = {
        "profile": "independent_seed_ownpolicy_credit_replication_readout_v1",
        "status": "PASS",
        "at_UTC": datetime.now(timezone.utc).isoformat(),
        "replication": metadata,
        "counts": report["counts"],
        "operational_counts": report["operational_counts"],
        "directional_replication": directional_readout(report),
        "prior_directional_readout": directional_readout(prior),
        "prior_counts": prior["counts"],
        "all_nine_methods_current": report["method_summary"],
        "all_nine_methods_prior": prior["method_summary"],
        "all_paired_comparisons_current": report["paired_comparisons"],
        "strata": report["strata"],
        "limits": [
            "Fresh seed replication on the same eight Train windows; "
            "no date or OOS generalization.",
            "Each batch uses its own full planned denominator and common nine-method support.",
            "No intersection of visited states, pooled batch score, "
            "alpha selection, or actor training.",
            "REF4+4 agreement is descriptive; it does not establish precise Q or significance.",
            "All-same-order groups are zero-target controls; informative diverse anchors stay few.",
            *report["limits"],
        ],
        "input_sha256": {
            **inputs,
            "prior_readout_sha256": sha(prior_report_path),
            "replication_onpolicy_readout_sha256": sha(directory / "onpolicy_readout.json"),
        },
        "frozen_source_sha256": FROZEN_SOURCES,
        "replication_reader_source_sha256": sha(Path(__file__).resolve()),
    }
    with (directory / "replication_readout.json").open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "counts": report["counts"],
                "directional_replication": receipt["directional_replication"]["joint_status"],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run(args.run.resolve())


if __name__ == "__main__":
    main()

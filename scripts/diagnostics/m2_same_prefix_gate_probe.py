#!/usr/bin/env python3
"""CPU-only gate isolation on sealed native first-request evidence.

This validates an isolated candidate. It does not call environments, models or
the trainer, and does not treat deterministic coverage as a learning result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PROFILE = "same_prefix_gate_only_implementation_probe_v1"
SOURCE = ROOT / "runs/m2-same-prefix-sampling-20261018-v1"
DEFAULT = ROOT / "runs/m2-same-prefix-gate-20261018-v1"
PLAN = ROOT / "docs/2026-10-18_M2_SAME_PREFIX_GATE_PLAN.md"
MODULE = ROOT / "src/rewards/same_prefix_gate.py"
TEST = ROOT / "tests/test_same_prefix_gate.py"
RUNNER_TEST = ROOT / "tests/test_m2_same_prefix_gate_probe.py"
METHODS = ("zero", "ungated_plain", "family_gate_plain", "order_gate_plain")
ATOL = RTOL = 1e-12

# Importing src.rewards runs the legacy package __init__, which indirectly
# imports environment accounting. Load only the sealed pure candidate file.
_spec = importlib.util.spec_from_file_location("same_prefix_gate_isolated_candidate", MODULE)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load isolated gate candidate")
_candidate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _candidate
_spec.loader.exec_module(_candidate)
SamePrefixGateSlot = _candidate.SamePrefixGateSlot
compute_same_prefix_gate = _candidate.compute_same_prefix_gate


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def object_sha(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_new(path: Path, value: object) -> None:
    with path.open("x") as stream:
        stream.write(canonical(value) + "\n")


def require(condition: bool, description: str) -> None:
    if not condition:
        raise ValueError(description)


def same(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=ATOL, rel_tol=RTOL)


def finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def indexed(items: list[dict]) -> dict[str, dict]:
    result = {row["request_key"]: row for row in items}
    require(len(result) == len(items), "duplicate source request key")
    return result


def verify_source(source: Path, expected: dict | None = None) -> dict:
    """Verify immutable artifacts and inherited byte identities, without imports."""
    contract = read(source / "contract.json")
    inherited = contract["source_sha256"]
    for name, digest in inherited.items():
        require(sha(Path(name)) == digest, f"inherited source changed: {name}")
    require(sha(source / "proposal.json") == contract["proposal_sha256"], "source proposal SHA")
    require(sha(source / "inputs.jsonl") == contract["inputs_sha256"], "source inputs SHA")
    for name in ("generation_completion.json", "evaluation_completion.json", "readout.json"):
        evidence = read(source / name)
        require(evidence["status"] in {"COMPLETE", "PASS"}, f"source incomplete: {name}")
        for field in ("input_sha256", "output_sha256"):
            for relative, digest in evidence.get(field, {}).items():
                require(
                    not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                    "source relative artifact path",
                )
                require(sha(source / relative) == digest, f"source artifact changed: {relative}")
    final = read(source / "final_verification.json")
    require(final["status"] == "PASS", "source final evidence PASS")
    # Historical documents_sha256 describes mutable docs at the previous finish;
    # it is deliberately not a new constraint on current documentation.
    for relative, digest in final["evidence_sha256"].items():
        require(sha(source / relative) == digest, f"source audit receipt changed: {relative}")
    inventory = {
        str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()
    }
    if expected is not None:
        require(inventory == expected, "source run tree or bytes changed since gate seal")
    return {
        "source_artifact_sha256": inventory,
        "inherited_source_count": len(inherited),
        "inherited_source_bytes": sum(Path(p).stat().st_size for p in inherited),
    }


def make_inputs(source: Path) -> list[dict]:
    inputs = rows(source / "inputs.jsonl")
    generations = indexed(rows(source / "generations.jsonl"))
    evaluations = indexed(rows(source / "evaluations.jsonl"))
    require(len(inputs) == len(generations) == len(evaluations) == 128, "fixed 128 source slots")
    require(set(indexed(inputs)) == set(generations) == set(evaluations), "source key sets")
    snapshot = read(source / "model_preflight_receipt.json")["model_fingerprint"]
    result = []
    for item in inputs:
        key = item["request_key"]
        generation, evaluation = generations[key], evaluations[key]
        require(
            item["anchor_id"] == generation["anchor_id"] == evaluation["anchor_id"],
            "source anchor mapping",
        )
        require(item["slot"] == generation["slot"] == evaluation["slot"], "source slot mapping")
        require(item["prompt_sha256"] == generation["prompt_sha256"], "source prompt mapping")
        require(
            item["prompt_token_ids"] == generation["returned_prompt_token_ids"],
            "actual full source prompt",
        )
        require(
            generation["generation_status"] == evaluation["generation_status"],
            "source generation classification",
        )
        supported = (
            generation["generation_status"] == "COMPLETE"
            and evaluation["env_status"] == "ADVANCING"
            and evaluation["terminal_label_status"] == "COMPLETE_FIXED_HOLD"
        )
        if supported:
            require(all(finite(evaluation[n]) for n in ("Y", "q", "hold_Y")), "finite label")
            require(same(evaluation["q"], evaluation["Y"] - evaluation["hold_Y"]), "q=Y-H")
            require(evaluation["economic_order_key"] is not None, "supported economic order")
        else:
            require(
                evaluation["Y"] is None and evaluation["q"] is None,
                "unsupported source has no invented label",
            )
        result.append(
            {
                "request_key": key,
                "anchor_id": item["anchor_id"],
                "slot": item["slot"],
                "source_group": item["source_group"],
                "request_seed": item["request_seed"],
                "prompt_sha256": item["prompt_sha256"],
                "financial_state_sha256": evaluation["financial_state_sha256"],
                "observation_messages_sha256": evaluation["observation_messages_sha256"],
                "policy_snapshot_sha256": snapshot,
                "generation_status": generation["generation_status"],
                "env_status": evaluation["env_status"],
                "terminal_label_status": evaluation["terminal_label_status"],
                "slot_label_supported": supported,
                "economic_order_key": evaluation["economic_order_key"],
                "family": evaluation["family"],
                "Y": evaluation["Y"],
                "q": evaluation["q"],
                "H": evaluation["hold_Y"],
                "source_error": (evaluation["first_info"] or {}).get("error"),
                "source_evaluation_row_sha256": object_sha(evaluation),
                "source_generation_row_sha256": object_sha(generation),
            }
        )
    return result


def prepare(output: Path, source: Path, fixture: Path, old_baseline: Path) -> None:
    require(not output.exists(), "new output directory must not exist")
    fixture_receipt = read(fixture)
    require(fixture_receipt["status"] == "PASS", "gate fixture must PASS")
    require(fixture_receipt["module_sha256"] == sha(MODULE), "fixture tested this module")
    require(fixture_receipt["tests_sha256"] == sha(TEST), "fixture tested these tests")
    require(
        fixture_receipt["runner_sha256"] == sha(Path(__file__).resolve()),
        "fixture tested this runner",
    )
    require(
        fixture_receipt["runner_tests_sha256"] == sha(RUNNER_TEST),
        "fixture tested these runner tests",
    )
    baseline = read(old_baseline)
    require(baseline["status"] == "PASS", "old-evidence baseline must PASS")
    evidence = verify_source(source)
    inputs = make_inputs(source)
    output.mkdir(parents=True)
    with (output / "gate_inputs.jsonl").open("x") as stream:
        for row in inputs:
            stream.write(canonical(row) + "\n")
    write_new(output / "fixture_receipt.json", fixture_receipt)
    with (output / "old_immutability_baseline.json").open("xb") as stream:
        stream.write(old_baseline.read_bytes())
    sources = [Path(__file__).resolve(), MODULE, TEST, RUNNER_TEST, PLAN]
    contract = {
        "profile": PROFILE,
        "status": "SEALED_BEFORE_OPERATOR_EXECUTION",
        "sealed_at_UTC": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source),
        **evidence,
        "source_sha256": {str(p): sha(p) for p in sources},
        "inputs_sha256": sha(output / "gate_inputs.jsonl"),
        "fixture_receipt_sha256": sha(output / "fixture_receipt.json"),
        "old_immutability_baseline_sha256": sha(output / "old_immutability_baseline.json"),
        "methods": METHODS,
        "eta": 0.25,
        "atol": ATOL,
        "rtol": RTOL,
        "nonzero_threshold": ATOL,
        "model_requests": 0,
        "environment_requests": 0,
        "optimizer_updates": 0,
        "formal_valid_oos_access": False,
        "pooling_enabled": False,
        "production_training_modified": False,
        "known_outcomes_informed_design": True,
    }
    write_new(output / "contract.json", contract)
    print(canonical({"status": "PREPARED", "slots": len(inputs), "output": str(output)}))


def verify(output: Path) -> dict:
    contract = read(output / "contract.json")
    require(contract["profile"] == PROFILE, "gate profile")
    require(contract["methods"] == list(METHODS), "fixed methods")
    require(
        contract["eta"] == 0.25 and contract["atol"] == ATOL and contract["rtol"] == RTOL,
        "fixed scales and tolerances",
    )
    for name, digest in contract["source_sha256"].items():
        require(sha(Path(name)) == digest, f"sealed new source changed: {name}")
    require(sha(output / "gate_inputs.jsonl") == contract["inputs_sha256"], "sealed gate inputs")
    require(
        sha(output / "fixture_receipt.json") == contract["fixture_receipt_sha256"],
        "sealed fixture receipt",
    )
    require(
        sha(output / "old_immutability_baseline.json")
        == contract["old_immutability_baseline_sha256"],
        "sealed old-evidence baseline",
    )
    return contract


def write_csv(path: Path, items: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(items[0]))
        writer.writeheader()
        for row in items:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def run(output: Path) -> None:
    contract = verify(output)
    source = Path(contract["source_run"])
    verify_source(source, contract["source_artifact_sha256"])
    inputs = rows(output / "gate_inputs.jsonl")
    require(inputs == make_inputs(source), "sealed derived inputs exactly reproduce source")
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in inputs:
        groups[item["anchor_id"]].append(item)
    require(len(groups) == 32 and len(inputs) == 128, "all planned groups and slots retained")
    with (source / "group_metrics.csv").open(newline="") as stream:
        source_group_rows = list(csv.DictReader(stream))
    source_groups = {row["anchor_id"]: row for row in source_group_rows}
    require(
        len(source_group_rows) == len(source_groups) == 32 and set(source_groups) == set(groups),
        "all original reader group identities",
    )
    slot_outputs, group_outputs = [], []
    max_sum = max_shift_error = 0.0
    for anchor, members in sorted(groups.items()):
        members.sort(key=lambda row: row["slot"])
        require(
            len(members) == 4 and {r["slot"] for r in members} == set(range(4)),
            "four original slots even in partial group",
        )
        for name in (
            "prompt_sha256",
            "financial_state_sha256",
            "policy_snapshot_sha256",
            "observation_messages_sha256",
            "source_group",
            "H",
        ):
            require(all(r[name] == members[0][name] for r in members), f"shared {name}")
        supported = all(row["slot_label_supported"] for row in members)
        source_group = source_groups[anchor]
        require(source_group["full_n4_support"] == str(supported), "per-anchor source support")
        if not supported:
            require(
                source_group["n4_family_gate"] == source_group["n4_order_gate"] == "",
                "partial source gates must be NA",
            )
        outputs = {method: None for method in METHODS}
        scaled = {method: None for method in METHODS}
        family_gate = order_gate = None
        if supported:
            slots = [
                SamePrefixGateSlot(
                    anchor_id=anchor,
                    slot_id=row["slot"],
                    prompt_sha256=row["prompt_sha256"],
                    financial_state_sha256=row["financial_state_sha256"],
                    policy_snapshot_sha256=row["policy_snapshot_sha256"],
                    economic_order_key=row["economic_order_key"],
                    score=row["Y"],
                    label_status=row["terminal_label_status"],
                )
                for row in members
            ]
            result = compute_same_prefix_gate(slots)
            shifted = compute_same_prefix_gate(
                [replace(s, score=row["q"]) for s, row in zip(slots, members, strict=True)]
            )
            permuted = compute_same_prefix_gate(list(reversed(slots)))
            family_gate, order_gate = result.family_gate, result.order_gate
            independent_family_gate = len({row["family"] for row in members}) > 1
            require(
                family_gate == independent_family_gate, "family gate independently matches families"
            )
            require(
                source_group["n4_family_gate"] == str(family_gate), "per-anchor source family gate"
            )
            require(
                source_group["n4_order_gate"] == str(order_gate), "per-anchor source order gate"
            )
            repeated_labels = {}
            for row, key in zip(members, result.normalized_order_keys, strict=True):
                if key in repeated_labels:
                    require(
                        row["Y"] == repeated_labels[key],
                        "same-state/order deterministic fixed-hold labels identical",
                    )
                repeated_labels[key] = row["Y"]
            for method in METHODS:
                values = list(getattr(result, method))
                outputs[method] = values
                scaled[method] = list(getattr(result.scaled_eta025, method))
                require(
                    all(same(v, 0.25 * s) for s, v in zip(values, scaled[method], strict=True)),
                    "fixed eta only",
                )
                require(same(math.fsum(values), 0.0), "raw four-slot zero sum")
                require(
                    tuple(reversed(getattr(permuted, method))) == tuple(values),
                    "input permutation equivariance",
                )
                for a, b in zip(values, getattr(shifted, method), strict=True):
                    require(same(a, b), "q/Y common-H translation identity")
                    max_shift_error = max(max_shift_error, abs(a - b))
                max_sum = max(max_sum, abs(math.fsum(values)))
            mean_y = math.fsum(row["Y"] for row in members) / 4
            expected = [row["Y"] - mean_y if independent_family_gate else 0.0 for row in members]
            require(outputs["family_gate_plain"] == expected, "legacy plain formula unchanged")
            if family_gate:
                require(
                    outputs["family_gate_plain"] == outputs["order_gate_plain"],
                    "family-gate open rows unchanged",
                )
            if not order_gate:
                require(
                    len({row["Y"] for row in members}) == 1,
                    "closed order gate has identical deterministic labels",
                )
            require(
                outputs["order_gate_plain"] == outputs["ungated_plain"],
                "fixed-hold deterministic order gate/ungated identity only",
            )
            for row, key in zip(members, result.normalized_order_keys, strict=True):
                require(key[0] == row["family"], "normalized key agrees with original family")
        nonzero = {
            m: any(abs(x) > ATOL for x in outputs[m]) if supported else None for m in METHODS
        }
        changed = (
            [
                abs(a - b) > ATOL
                for a, b in zip(
                    outputs["family_gate_plain"], outputs["order_gate_plain"], strict=True
                )
            ]
            if supported
            else None
        )
        group_outputs.append(
            {
                "anchor_id": anchor,
                **members[0]["source_group"],
                "planned_slots": 4,
                "k_label_supported": sum(r["slot_label_supported"] for r in members),
                "full_n4_support": supported,
                "family_gate": family_gate,
                "order_gate": order_gate,
                "nonzero": nonzero,
                "changed_slots_family_to_order": sum(changed) if supported else None,
                "raw_S": outputs,
                "eta025_S": scaled,
            }
        )
        for index, row in enumerate(members):
            slot = {
                **row,
                "full_n4_support": supported,
                "operator_status": "SUPPORTED_N4" if supported else "NA_PARTIAL_GROUP",
            }
            for method in METHODS:
                slot[f"raw_{method}"] = outputs[method][index] if supported else None
                slot[f"eta025_{method}"] = scaled[method][index] if supported else None
            slot_outputs.append(slot)
    supported_groups = [r for r in group_outputs if r["full_n4_support"]]
    supported_slots = [r for r in slot_outputs if r["full_n4_support"]]
    source_counts = read(source / "readout.json")["counts"]
    require(
        len(supported_groups) == source_counts["full_n4_support_groups"], "source support count"
    )
    require(
        sum(r["family_gate"] for r in supported_groups)
        == source_counts["full_n4_family_gate_open_groups"],
        "source family-gate count",
    )
    require(
        sum(r["order_gate"] for r in supported_groups)
        == source_counts["full_n4_order_gate_open_groups"],
        "source order-gate count",
    )
    counts = {
        "planned_groups": 32,
        "planned_slots": 128,
        "full_n4_groups": len(supported_groups),
        "operator_supported_slots": len(supported_slots),
        "partial_groups": 32 - len(supported_groups),
        "operator_NA_slots": 128 - len(supported_slots),
        "source_legal_slots": sum(r["slot_label_supported"] for r in inputs),
        "source_env_counts": dict(Counter(r["env_status"] for r in inputs)),
        "k_histogram": {
            str(k): sum(r["k_label_supported"] == k for r in group_outputs) for k in range(5)
        },
        "family_gate_open_groups": sum(r["family_gate"] for r in supported_groups),
        "order_gate_open_groups": sum(r["order_gate"] for r in supported_groups),
        "changed_groups_family_to_order": sum(
            r["changed_slots_family_to_order"] > 0 for r in supported_groups
        ),
        "changed_slots_family_to_order": sum(
            r["changed_slots_family_to_order"] for r in supported_groups
        ),
        "restored_signal_groups": sum(
            r["nonzero"]["order_gate_plain"] and not r["nonzero"]["family_gate_plain"]
            for r in supported_groups
        ),
        "lost_signal_groups": sum(
            r["nonzero"]["family_gate_plain"] and not r["nonzero"]["order_gate_plain"]
            for r in supported_groups
        ),
    }
    method_summary = [
        {
            "method": method,
            "groups_with_signal": sum(r["nonzero"][method] for r in supported_groups),
            "slots_with_signal": sum(abs(r[f"raw_{method}"]) > ATOL for r in supported_slots),
            "supported_groups": len(supported_groups),
            "supported_slots": len(supported_slots),
            "NA_groups": 32 - len(supported_groups),
            "NA_slots": 128 - len(supported_slots),
        }
        for method in METHODS
    ]
    checks = {
        "status": "PASS",
        "legacy_family_plain_formula_exact": True,
        "per_anchor_gates_match_original_independent_readout": True,
        "family_open_outputs_unchanged": True,
        "deterministic_same_order_labels_identical": True,
        "order_equals_ungated_on_deterministic_fixed_hold_labels": True,
        "partial_all_four_methods_all_four_slots_NA": True,
        "actual_permutation_equivariance": True,
        "raw_max_abs_group_sum": max_sum,
        "q_Y_common_H_max_abs_difference": max_shift_error,
        "pooling_disabled": True,
        "GAE_whitening_actor_not_executed": True,
        "future_H_passed_to_candidate": False,
        "eta": 0.25,
    }
    forbidden = ("torch", "vllm", "transformers", "src.env", "agent_r1.env")
    loaded = [
        name
        for name in sys.modules
        if any(name == item or name.startswith(item + ".") for item in forbidden)
    ]
    require(not loaded, "gate diagnostic must not import model or environment backends")
    checks["model_environment_backend_modules_loaded"] = loaded
    verify(output)
    verify_source(source, contract["source_artifact_sha256"])
    with (output / "slot_signals.jsonl").open("x") as stream:
        for row in slot_outputs:
            stream.write(canonical(row) + "\n")
    write_csv(output / "slot_signals.csv", slot_outputs)
    write_csv(output / "group_signals.csv", group_outputs)
    write_csv(output / "method_summary.csv", method_summary)
    write_new(output / "checks.json", checks)
    readout = {
        "profile": PROFILE,
        "status": "PASS",
        "counts": counts,
        "method_summary": method_summary,
        "checks": checks,
        "scope": "known-data implementation and coverage verification only",
        "limits": [
            "Fixed-hold Y, not Q under the original continuation policy.",
            "No MSE, action ranking, denoising, gradient or return claim.",
            "SFT on historical M2 states; original windows/dates overlap.",
            "Raw S and fixed eta*S only; no production or actor integration.",
        ],
        "input_sha256": {
            n: sha(output / n)
            for n in ("contract.json", "gate_inputs.jsonl", "fixture_receipt.json")
        },
    }
    write_new(output / "readout.json", readout)
    names = [
        "slot_signals.jsonl",
        "slot_signals.csv",
        "group_signals.csv",
        "method_summary.csv",
        "checks.json",
        "readout.json",
    ]
    write_new(
        output / "completion.json",
        {
            "status": "COMPLETE",
            "finished_at_UTC": datetime.now(timezone.utc).isoformat(),
            "slots_classified": 128,
            "model_requests": 0,
            "environment_requests": 0,
            "optimizer_updates": 0,
            "formal_valid_oos_access": False,
            "output_sha256": {n: sha(output / n) for n in names},
        },
    )
    print(canonical({"status": "PASS", "counts": counts, "method_summary": method_summary}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--source", type=Path, default=SOURCE)
    prepare_parser.add_argument("--fixture", type=Path, required=True)
    prepare_parser.add_argument("--old-baseline", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(
            args.output.resolve(),
            args.source.resolve(),
            args.fixture.resolve(),
            args.old_baseline.resolve(),
        )
    else:
        run(args.output.resolve())


if __name__ == "__main__":
    main()

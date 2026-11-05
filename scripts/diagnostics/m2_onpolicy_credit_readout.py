#!/usr/bin/env python3
"""Descriptive credit readout at fresh states visited by the fixed parent policy.

Only standard-library helpers and the frozen previous reader's pure formulas
are used. No training, models, environment replay or alpha selection occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from m2_credit_comparison_readout import METHODS as BASE_METHODS
from m2_credit_comparison_readout import (
    PANELS,
    close,
    direction,
    finite,
    indexed,
    order_key,
    read,
    require,
    rows,
    sha,
    signals,
    write_csv,
)

SHRINK = {
    "exact_pool_shrink025": 0.25,
    "exact_pool_shrink050": 0.5,
    "exact_pool_shrink075": 0.75,
}
METHODS = (*BASE_METHODS, *SHRINK)
ETA = 0.25
MAX_REQUESTS = 23744
HELPER = Path(__file__).with_name("m2_credit_comparison_readout.py")
HELPER_SHA = "c89ef20baab2bfa346dd715ee8af0d72ed60e5ed2d0dfc91b2b26e6b5bee3b3d"


def predict(scores: list[float], keys: list[tuple]) -> tuple[dict, bool, bool]:
    output, family_gate, exact_gate = signals(scores, keys)
    for name, alpha in SHRINK.items():
        output[name] = [alpha * value for value in output["exact_pool_order_gate"]]
    require(all(close(math.fsum(values), 0) for values in output.values()), "all methods zero sum")
    return output, family_gate, exact_gate


def reference_means(cells: list[dict], reference: dict) -> tuple[list, dict]:
    output, means = [], {}
    for cell in cells:
        key = cell["cell_id"]
        samples = [reference[key, rep] for rep in range(8)]
        complete = all(sample["status"] == "COMPLETE" for sample in samples)
        values = [sample["Y"] for sample in samples]
        mu = (
            {
                "full8": math.fsum(values) / 8,
                "ref0_3": math.fsum(values[:4]) / 4,
                "ref4_7": math.fsum(values[4:]) / 4,
            }
            if complete
            else dict.fromkeys(PANELS)
        )
        means[key] = mu
        output.append(
            {
                "cell_id": key,
                "anchor_id": cell["anchor_id"],
                "economic_order_key": cell["economic_order_key"],
                "planned_reference_paths": 8,
                "complete_reference_paths": sum(
                    sample["status"] == "COMPLETE" for sample in samples
                ),
                "full8_supported": complete,
                "reference_Y": values,
                **{f"Qhat_{panel}": mu[panel] for panel in PANELS},
            }
        )
    return output, means


def calculate(anchors: list[dict], heads: list[dict], cells: list[dict], paths: list[dict]) -> dict:
    """Consume normalized driver records; all 16 anchors remain in every table."""
    require(len(anchors) == 16 and len(heads) == 64, "16 planned anchors and 64 planned heads")
    by_anchor = indexed(anchors, lambda row: row["anchor_id"])
    require(len(by_anchor) == 16, "unique planned anchors")
    by_cell = indexed(cells, lambda row: row["cell_id"])
    by_head = indexed(heads, lambda row: row["head_key"])
    require(len(by_head) == 64, "unique planned head slots")
    estimates, references = {}, {}
    for path in paths:
        role, rep = path["role"], path["replicate"]
        require(role in ("estimate", "reference"), "branch role")
        require(type(rep) is int and rep in range(2 if role == "estimate" else 8), "branch replica")
        if path["status"] == "COMPLETE":
            require(finite(path["Y"]) and path["last_session_index"] == 20, "original terminal Y")
        else:
            require(path["Y"] is None, "failed branch keeps Y NA")
        target = estimates if role == "estimate" else references
        identity = (path["head_key"], rep) if role == "estimate" else (path["cell_id"], rep)
        require(identity not in target, "distinct fresh EST and REF paths")
        target[identity] = path
    ref_rows, ref_mu = reference_means(cells, references)
    heads_by_anchor: dict[str, list[dict]] = defaultdict(list)
    for head in heads:
        require(head["anchor_id"] in by_anchor, "planned head anchor")
        heads_by_anchor[head["anchor_id"]].append(head)
    group_rows, slot_rows = [], []
    for anchor_id, anchor in sorted(by_anchor.items()):
        members = sorted(heads_by_anchor[anchor_id], key=lambda row: row["slot_id"])
        require(
            len(members) == 4 and {h["slot_id"] for h in members} == {0, 1, 2, 3}, "original n4"
        )
        keys = (
            [order_key(h["economic_order_key"]) for h in members]
            if all(h["legal_head"] for h in members)
            else None
        )
        structural_stratum = (
            "UNAVAILABLE_ANCHOR"
            if not anchor["available"]
            else "CONDITIONAL_STATE_MISMATCH"
            if not anchor["conditional_state_match"]
            else "PARTIAL_HEAD_GROUP"
            if keys is None
            else "ALL_SAME_ORDER"
            if len(set(keys)) == 1
            else "ORDER_DIVERSE_SAME_FAMILY"
            if len({key[0] for key in keys}) == 1
            else "FAMILY_DIVERSE"
        )
        for rep in range(2):
            observations = [estimates.get((h["head_key"], rep)) for h in members]
            reasons = []
            if not anchor["available"]:
                reasons.append("ANCHOR_NOT_REACHED_BY_PARENT")
            if anchor["available"] and not anchor["conditional_state_match"]:
                reasons.append("CONDITIONAL_STATE_MISMATCH")
            if keys is None:
                reasons.append("PARTIAL_OR_NONADVANCING_HEAD_GROUP")
            if any(row is None or row["status"] != "COMPLETE" for row in observations):
                reasons.append("INCOMPLETE_ESTIMATE_GROUP")
            if any(
                h["cell_id"] is None or ref_mu.get(h["cell_id"], {}).get("full8") is None
                for h in members
            ):
                reasons.append("INCOMPLETE_EIGHT_DRAW_REFERENCE")
            supported = not reasons
            prediction = dict.fromkeys(METHODS)
            target = dict.fromkeys(PANELS)
            losses = {method: dict.fromkeys(PANELS) for method in METHODS}
            if supported:
                require(
                    all(by_cell[h["cell_id"]]["anchor_id"] == anchor_id for h in members),
                    "reference sharing stays within one actual anchor",
                )
                prediction, _, _ = predict([o["Y"] for o in observations], keys)
                for panel in PANELS:
                    q = [ref_mu[h["cell_id"]][panel] for h in members]
                    baseline = math.fsum(q) / 4
                    target[panel] = [value - baseline for value in q]
                    for method in METHODS:
                        losses[method][panel] = [
                            (a - b) ** 2
                            for a, b in zip(prediction[method], target[panel], strict=True)
                        ]
            group_rows.append(
                {
                    "anchor_id": anchor_id,
                    "parent_key": anchor["parent_key"],
                    "parent_final_status": anchor["parent_final_status"],
                    "decision_session_index": anchor["decision_session_index"],
                    "estimate_replicate": rep,
                    "legal_head_slots": sum(h["legal_head"] for h in members),
                    "family_gate": len({key[0] for key in keys}) > 1 if keys else None,
                    "order_gate": len(set(keys)) > 1 if keys else None,
                    "order_stratum": structural_stratum,
                    "main_supported": supported,
                    "NA_reasons": reasons,
                }
            )
            for index, head in enumerate(members):
                row = {
                    **group_rows[-1],
                    "head_key": head["head_key"],
                    "slot_id": head["slot_id"],
                    "head_status": head["head_status"],
                    "legal_head": head["legal_head"],
                    "cell_id": head["cell_id"],
                    "estimate_Y": observations[index]["Y"] if observations[index] else None,
                }
                for panel in PANELS:
                    row[f"reference_tau_{panel}"] = target[panel][index] if supported else None
                for method in METHODS:
                    value = prediction[method][index] if supported else None
                    row[f"raw_{method}"] = value
                    row[f"eta025_{method}"] = ETA * value if supported else None
                    for panel in PANELS:
                        row[f"loss_{method}_{panel}"] = (
                            losses[method][panel][index] if supported else None
                        )
                slot_rows.append(row)
    require(
        len(group_rows) == 32 and len(slot_rows) == 128, "32 group-reps and 128 slot-reps retained"
    )
    return {"groups": group_rows, "slots": slot_rows, "references": ref_rows}


def summarize(groups: list[dict], slots: list[dict]) -> tuple[list, list, dict]:
    supported = [row for row in slots if row["main_supported"]]
    require(len(supported) % 4 == 0, "common complete four-slot support")
    methods, comparisons = [], []
    for method, panel in itertools.product(METHODS, PANELS):
        values = [row[f"loss_{method}_{panel}"] for row in supported]
        methods.append(
            {
                "method": method,
                "alpha": SHRINK.get(method),
                "reference_panel": panel,
                "planned_group_replicates": len(groups),
                "supported_group_replicates": len(supported) // 4,
                "planned_slot_replicates": len(slots),
                "supported_slot_replicates": len(supported),
                "NA_slot_replicates": len(slots) - len(supported),
                "raw_financial_Y_squared_slot_MSE": math.fsum(values) / len(values)
                if values
                else None,
            }
        )
    for comparator, candidate in itertools.combinations(METHODS, 2):
        delta = {
            panel: math.fsum(
                row[f"loss_{candidate}_{panel}"] - row[f"loss_{comparator}_{panel}"]
                for row in supported
            )
            / len(supported)
            if supported
            else None
            for panel in PANELS
        }
        if supported:
            require(
                close(delta["full8"], (delta["ref0_3"] + delta["ref4_7"]) / 2),
                "affine reference delta",
            )
        signs = [direction(delta[panel]) for panel in PANELS]
        stable = bool(supported) and len(set(signs)) == 1
        structural = (
            candidate == "exact_pool_order_gate"
            and comparator in ("ungated_plain", "order_gate_plain")
        ) or (candidate == "order_gate_plain" and comparator == "ungated_plain")
        comparisons.append(
            {
                "candidate": candidate,
                "comparator": comparator,
                "supported_slot_replicates": len(supported),
                **{f"candidate_minus_comparator_delta_{panel}": delta[panel] for panel in PANELS},
                **{f"direction_{panel}": direction(delta[panel]) for panel in PANELS},
                "reference_half_direction_stable": stable,
                "structural_projection_identity": structural,
                "interpretation": "STRUCTURAL_IDENTITY_NOT_REFERENCE_PRECISION_EVIDENCE"
                if structural and supported
                else "DESCRIPTIVE_ONLY_FIXED_ALPHA_NO_SELECTION"
                if stable
                else "UNRESOLVED_REFERENCE_SENSITIVITY_OR_SUPPORT",
            }
        )
    return (
        methods,
        comparisons,
        {
            "planned_group_replicates": len(groups),
            "planned_slot_replicates": len(slots),
            "supported_group_replicates": len(supported) // 4,
            "supported_slot_replicates": len(supported),
            "NA_group_replicates": len(groups) - len(supported) // 4,
            "NA_slot_replicates": len(slots) - len(supported),
            "common_support_all_nine_methods": True,
        },
    )


def normalize(directory: Path) -> tuple[dict, dict]:
    """Check the driver's small result tables; never load prompt/generation logs."""
    manifest = read(directory / "continuation_manifest.json")
    branch = read(directory / "branch_manifest.json")
    require(manifest["profile"] == "fresh_own_policy_state_credit_validation_v1", "profile")
    require(
        manifest["max_model_requests"] == MAX_REQUESTS
        and manifest["stage_request_limits"]
        == {"parent": 640, "head": 64, "estimate": 4608, "reference": 18432}
        and manifest["optimizer_updates"] == 0
        and manifest["formal_valid_oos_access"] is False
        and manifest["automatic_retry"] is False
        and manifest["fallback_hold"] is False
        and manifest["state_outcome_filtering"] is False
        and manifest["no_extra_hold_reference"] is True
        and manifest["retain_reached_anchors_if_parent_later_fails"] is True,
        "fixed scope, quotas, and no outcome-driven replacement",
    )
    parents = indexed(rows(directory / "parent_results.jsonl"), lambda r: r["parent_id"])
    anchors = indexed(rows(directory / "anchor_results.jsonl"), lambda r: r["anchor_id"])
    heads = indexed(rows(directory / "head_results.jsonl"), lambda r: r["head_key"])
    specs = indexed(manifest["parents"], lambda r: r["parent_id"])
    require(len(specs) == 8 and set(parents) == set(specs), "all eight fresh parents")
    for key, spec in specs.items():
        require(all(parents[key][k] == v for k, v in spec.items()), "parent identity")
        require(parents[key]["role"] == "parent", "parent role")
    specs = indexed(manifest["anchors"], lambda r: r["anchor_id"])
    require(len(specs) == 16 and set(anchors) == set(specs), "all sixteen anchors retained")
    for key, spec in specs.items():
        anchor = anchors[key]
        require(all(anchor[k] == v for k, v in spec.items()), "anchor identity")
        require(anchor["decision_session_index"] in (5, 15), "fixed anchor sessions")
        require(
            anchor["status"] in ("CAPTURED", "MISSING_PARENT_STATE", "CONDITIONAL_STATE_MISMATCH"),
            "anchor outcome class",
        )
        parent = parents[anchor["parent_id"]]
        available = anchor["status"] != "MISSING_PARENT_STATE"
        require((key in parent["reached_anchor_ids"]) is available, "reached anchors retained")
        require(anchor["parent_final_status"] == parent["status"], "later parent status recorded")
        require(
            anchor["conditional_state_match"]
            is (None if not available else anchor["status"] == "CAPTURED"),
            "conditional-state flag",
        )
        anchor.update(available=available, parent_key=parent["trajectory_key"])
    specs = indexed(manifest["head_slots"], lambda r: r["head_key"])
    require(len(specs) == 64 and set(heads) == set(specs), "all 64 fresh head requests or NA")
    for key, spec in specs.items():
        head = heads[key]
        require(all(head[k] == v for k, v in spec.items()), "head identity")
        require(head["legal_head"] is (head["status"] == "ADVANCING"), "legal advancing head")
        require(head["model_calls"] in (0, 1), "one raw first response, no replacement")
        require(
            (head["status"] == "NOT_RUN_ANCHOR") is not anchors[head["anchor_id"]]["available"],
            "heads run only at reached anchors",
        )
        if head["legal_head"]:
            order_key(head["economic_order_key"])
        else:
            require(head["economic_order_key"] is None and head["cell_id"] is None, "head NA")
        head["head_status"] = head["status"]
    cells = indexed(branch["cells"], lambda r: r["cell_id"])
    expected = set()
    for aid, anchor in anchors.items():
        members = [h for h in heads.values() if h["anchor_id"] == aid]
        require(len(members) == 4, "four planned raw heads per anchor")
        require(sum(h["legal_head"] for h in members) == anchor["legal_heads"], "head denominator")
        legal = [h for h in members if h["legal_head"]]
        groups = defaultdict(list)
        for head in legal:
            groups[order_key(head["economic_order_key"])].append(head)
        mismatch = any(
            len(
                {
                    (
                        object_sha(h["first_state_after"]),
                        object_sha(h["first_observation_messages"]),
                    )
                    for h in group
                }
            )
            > 1
            for group in groups.values()
        )
        if anchor["available"]:
            require(mismatch is not anchor["conditional_state_match"], "actual post-head identity")
        if anchor["status"] != "CAPTURED":
            require(all(h["cell_id"] is None for h in members), "mismatch keeps all methods NA")
            continue
        for key, group in groups.items():
            representative = min(group, key=lambda h: h["slot_id"])
            cid = representative["cell_id"]
            require(cid in cells, "one reference cell per eligible concrete order")
            cell = cells[cid]
            require(
                cell["anchor_id"] == aid
                and order_key(cell["economic_order_key"]) == key
                and cell["representative_head_key"] == representative["head_key"]
                and all(h["cell_id"] == cid for h in group),
                "reference cell identity and original head multiset",
            )
            require(
                cell["post_first_financial_state_sha256"]
                == object_sha(representative["first_state_after"])
                and cell["post_first_observation_messages_sha256"]
                == object_sha(representative["first_observation_messages"]),
                "shared reference conditional state",
            )
            expected.update(
                ("estimate", h["head_key"], rep, cid) for h in group for rep in range(2)
            )
            expected.update(("reference", representative["head_key"], rep, cid) for rep in range(8))
    specifications = indexed(branch["trajectories"], lambda r: r["trajectory_key"])
    require(
        {(r["role"], r["head_key"], r["replicate"], r["cell_id"]) for r in specifications.values()}
        == expected
        and len(specifications) == len(expected),
        "EST2 per raw legal head, REF8 per cell",
    )
    paths = indexed(rows(directory / "trajectory_results.jsonl"), lambda r: r["trajectory_key"])
    require(set(paths) == set(specifications), "every actual planned branch classified")
    for key, path in paths.items():
        require(all(path[k] == v for k, v in specifications[key].items()), "branch identity")
        head = heads[path["head_key"]]
        require(
            path["anchor_id"] == head["anchor_id"] and path["cell_id"] == head["cell_id"],
            "branch starts at its own raw head",
        )
        require(
            path["remaining_sessions"] == 19 - path["decision_session_index"], "original window"
        )
    for result in [*parents.values(), *paths.values()]:
        require(result["policy_snapshot_sha256"] == manifest["policy_snapshot_sha256"], "fixed pi")
        require(
            result["status"] in ("COMPLETE", "REQUEST_LIMIT", "INCOMPLETE", "PROMPT_LIMIT"),
            "terminal result status",
        )
        if result["status"] == "COMPLETE":
            require(
                finite(result["Y"])
                and result["last_session_index"] == 20
                and result["Y"] == result["terminal_metrics"]["terminal_score"],
                "finite terminal Y",
            )
            advances = 20 if result["role"] == "parent" else result["remaining_sessions"]
            require(result["session_advances"] == advances, "all original sessions advanced")
        else:
            require(result["Y"] is None and result["terminal_metrics"] is None, "failure label NA")
    return manifest, {
        "anchors": list(anchors.values()),
        "heads": list(heads.values()),
        "cells": list(cells.values()),
        "paths": list(paths.values()),
        "parents": list(parents.values()),
    }


def object_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def run(directory: Path) -> None:
    completion_path = directory / "driver_completion.json"
    completion = read(completion_path)
    require(completion["status"] == "COMPLETE", "driver classified every planned stage")
    require(sha(HELPER) == HELPER_SHA, "frozen pure formula helper")
    manifest, data = normalize(directory)
    parents = data.pop("parents")
    roles = {
        "parent": parents,
        "head": data["heads"],
        "estimate": [p for p in data["paths"] if p["role"] == "estimate"],
        "reference": [p for p in data["paths"] if p["role"] == "reference"],
    }
    role_calls = {role: sum(r["model_calls"] for r in records) for role, records in roles.items()}
    require(
        all(
            role_calls[r] == completion["model_requests_by_role"].get(r, 0)
            and role_calls[r] <= manifest["stage_request_limits"][r]
            for r in roles
        ),
        "per-role request counts and fixed budgets",
    )
    require(sum(role_calls.values()) == completion["model_requests"], "total actual request count")
    require(
        completion["parents_classified"] == len(parents) == 8
        and completion["head_slots_classified"] == len(data["heads"]) == 64
        and completion["trajectories_classified"] == len(data["paths"]),
        "completion denominators",
    )
    require(
        completion["optimizer_updates"] == 0
        and completion["formal_valid_oos_access"] is False
        and completion["automatic_retry"] is False,
        "readout scope",
    )
    input_names = (
        "continuation_manifest.json",
        "branch_manifest.json",
        "parent_results.jsonl",
        "anchor_results.jsonl",
        "head_results.jsonl",
        "trajectory_results.jsonl",
        "driver_completion.json",
    )
    input_sha = {name: sha(directory / name) for name in input_names}
    require(
        completion["continuation_manifest_sha256"] == input_sha["continuation_manifest.json"],
        "completed frozen manifest",
    )
    require(
        all(
            completion["output_sha256"][name] == digest
            for name, digest in input_sha.items()
            if name not in ("continuation_manifest.json", "driver_completion.json")
        ),
        "small result files match driver completion",
    )
    output = calculate(**data)
    methods, comparisons, counts = summarize(output["groups"], output["slots"])
    require(completion["model_requests"] <= MAX_REQUESTS, "fixed total model request ceiling")
    strata_methods, strata_comparisons, strata_counts = [], [], []
    for stratum in sorted({row["order_stratum"] for row in output["groups"]}):
        gs = [row for row in output["groups"] if row["order_stratum"] == stratum]
        ss = [row for row in output["slots"] if row["order_stratum"] == stratum]
        ms, cs, ns = summarize(gs, ss)
        strata_methods.extend({"order_stratum": stratum, **row} for row in ms)
        strata_comparisons.extend({"order_stratum": stratum, **row} for row in cs)
        strata_counts.append({"order_stratum": stratum, **ns})
    tables = {
        "onpolicy_groups.csv": output["groups"],
        "onpolicy_slots.csv": output["slots"],
        "onpolicy_methods.csv": methods,
        "onpolicy_comparisons.csv": comparisons,
        "onpolicy_strata_methods.csv": strata_methods,
        "onpolicy_strata_comparisons.csv": strata_comparisons,
    }
    if output["references"]:
        tables["onpolicy_references.csv"] = output["references"]
    for name, records in tables.items():
        write_csv(directory / name, records)
    report = {
        "profile": "fresh_onpolicy_states_fixed_alpha_credit_readout_v1",
        "status": "PASS",
        "at_UTC": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "operational_counts": {
            "parent_statuses": dict(Counter(r["status"] for r in parents)),
            "anchor_statuses": dict(Counter(r["status"] for r in data["anchors"])),
            "head_statuses": dict(Counter(r["head_status"] for r in data["heads"])),
            "planned_parents": 8,
            "planned_anchors": 16,
            "planned_raw_heads": 64,
            "legal_raw_heads": sum(r["legal_head"] for r in data["heads"]),
            "reference_cells": len(data["cells"]),
            "planned_estimate_paths": len(roles["estimate"]),
            "planned_reference_paths": len(roles["reference"]),
            "branch_statuses": dict(Counter(r["status"] for r in data["paths"])),
            "model_requests_by_role": role_calls,
        },
        "model_requests": completion["model_requests"],
        "maximum_model_requests": MAX_REQUESTS,
        "fixed_shrink_alpha": SHRINK,
        "method_summary": methods,
        "paired_comparisons": comparisons,
        "strata": strata_counts,
        "limits": [
            (
                "All alpha values were fixed before this batch; "
                "the reader does not select a best alpha."
            ),
            "Zero is an auxiliary credit signal, not a hold policy or a no-learning agent.",
            (
                "Eight REF draws and two EST draws are finite; "
                "half-sign agreement is not precision or significance."
            ),
            (
                "Exact pooling versus plain includes a projection identity, "
                "not independent denoising confirmation."
            ),
            (
                "The target is centered over four realized valid heads, "
                "not A^pi or financial policy return."
            ),
            (
                "Fresh parent trajectories visit the fixed pi states on eight "
                "already-used Train windows, not formal OOS."
            ),
            (
                "Reached anchors remain eligible even if the parent later fails; "
                "unavailable or conditional-state-mismatched anchors remain NA."
            ),
            "No GAE, whitening, critic fitting, optimizer update or production integration occurs.",
        ],
        "input_sha256": input_sha,
        "output_sha256": {name: sha(directory / name) for name in tables},
        "readout_source_sha256": sha(Path(__file__).resolve()),
        "pure_formula_helper_sha256": sha(HELPER),
    }
    with (directory / "onpolicy_readout.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "counts": counts}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run(args.run.resolve())


if __name__ == "__main__":
    main()

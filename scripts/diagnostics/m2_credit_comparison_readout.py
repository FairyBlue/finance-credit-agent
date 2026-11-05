#!/usr/bin/env python3
"""Finite-reference, descriptive credit comparisons on fresh continuation draws.

No model, environment, candidate module or trainer is imported. All six methods
share one support intersection; partial original four-slot groups stay NA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

METHODS = (
    "zero",
    "ungated_plain",
    "family_gate_plain",
    "order_gate_plain",
    "family_pool_order_gate",
    "exact_pool_order_gate",
)
PANELS = ("full8", "ref0_3", "ref4_7")
ETA = 0.25
ATOL = RTOL = 1e-12
SIGN_ATOL = 1e-12
REQUEST_LIMIT = 39024


def require(condition: bool, description: str) -> None:
    if not condition:
        raise ValueError(description)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, abs_tol=ATOL, rel_tol=RTOL)


def indexed(items: list[dict], key) -> dict:
    result = {key(row): row for row in items}
    require(len(result) == len(items), "duplicate planned or observed identity")
    return result


def write_csv(path: Path, items: list[dict]) -> None:
    require(bool(items), "nonempty output table")
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


def order_key(value: str) -> tuple:
    key = json.loads(value)
    require(isinstance(key, list) and bool(key), "economic order array")
    if key[0] in ("hold", "abstain"):
        require(len(key) == 1, "parameter-free hold/abstain economic identity")
        return (key[0],)
    require(
        len(key) == 3 and key[0] in ("buy", "sell") and isinstance(key[1], str),
        "directional economic order",
    )
    require(bool(key[1].strip()) and finite(key[2]) and 0 <= key[2] <= 1, "accepted order domain")
    target = float(key[2])
    return key[0], key[1].strip().upper(), 0.0 if target == 0 else target


def signals(scores: list[float], keys: list[tuple]) -> tuple[dict, bool, bool]:
    require(len(scores) == len(keys) == 4 and all(finite(x) for x in scores), "full n4 scores")
    mean = math.fsum(scores) / 4
    plain = [score - mean for score in scores]
    family_gate = len({key[0] for key in keys}) > 1
    exact_gate = len(set(keys)) > 1
    family_cells: dict[str, list[int]] = defaultdict(list)
    exact_cells: dict[tuple, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        family_cells[key[0]].append(index)
        exact_cells[key].append(index)

    def pooled(cells: dict, identity) -> list[float]:
        means = {key: math.fsum(scores[i] for i in cell) / len(cell) for key, cell in cells.items()}
        return [means[identity(key)] - mean for key in keys] if exact_gate else [0.0] * 4

    result = {
        "zero": [0.0] * 4,
        "ungated_plain": plain,
        "family_gate_plain": plain if family_gate else [0.0] * 4,
        "order_gate_plain": plain if exact_gate else [0.0] * 4,
        "family_pool_order_gate": pooled(family_cells, lambda key: key[0]),
        "exact_pool_order_gate": pooled(exact_cells, lambda key: key),
    }
    require(all(close(math.fsum(value), 0) for value in result.values()), "raw signal zero sum")
    if family_gate:
        require(result["family_gate_plain"] == result["order_gate_plain"], "unchanged open gate")
    return result, family_gate, exact_gate


def path_tables(manifest: dict, results: list[dict]) -> tuple[list, dict, dict]:
    specifications = indexed(manifest["trajectories"], lambda row: row["trajectory_key"])
    observed = indexed(results, lambda row: row["trajectory_key"])
    require(len(specifications) == 864 and set(observed) == set(specifications), "all 864 paths")
    estimates, references, post_first_identities = {}, {}, {}
    output = []
    for key, spec in specifications.items():
        result = observed[key]
        require(all(result[name] == value for name, value in spec.items()), "exact path metadata")
        require(
            result["policy_snapshot_sha256"] == manifest["policy_snapshot_sha256"],
            "one fixed common continuation policy",
        )
        post_identity = (
            sha_value(result["first_state_after"]),
            result["first_observation_messages_sha256"],
        )
        if spec["cell_id"] in post_first_identities:
            require(
                post_first_identities[spec["cell_id"]] == post_identity,
                "all EST slots and REF draws share full post-first condition",
            )
        post_first_identities[spec["cell_id"]] = post_identity
        role, rep, day = spec["role"], spec["replicate"], spec["decision_session_index"]
        require(role in ("estimate", "reference") and day in (0, 5, 10, 15), "role and date")
        require(type(rep) is int and rep in range(2 if role == "estimate" else 8), "role replicate")
        require(
            type(result["model_calls"]) is int
            and type(result["session_advances"]) is int
            and 0 <= result["session_advances"] <= 19 - day
            and result["session_advances"] <= result["model_calls"] <= 4 * (19 - day)
            and result["last_session_index"] == day + 1 + result["session_advances"],
            "original endpoint and bounded continuation calls",
        )
        complete = result["status"] == "COMPLETE"
        if complete:
            metrics = result["terminal_metrics"]
            require(
                finite(result["Y"])
                and result["last_session_index"] == 20
                and metrics["session_count"] == 21
                and metrics["terminal_score"] == result["Y"],
                "complete original terminal financial Y",
            )
        else:
            require(result["Y"] is None and result["terminal_metrics"] is None, "failed label NA")
        target, identity = (
            (estimates, (spec["source_request_key"], rep))
            if role == "estimate"
            else (references, (spec["cell_id"], rep))
        )
        require(identity not in target, "duplicate role/slot/replicate")
        target[identity] = result
        output.append(
            {
                "trajectory_key": key,
                "role": role,
                "cell_id": spec["cell_id"],
                "anchor_id": spec["anchor_id"],
                "source_request_key": spec["source_request_key"],
                "slot_id": spec["slot_id"],
                "replicate": rep,
                "decision_session_index": day,
                "status": result["status"],
                "failure_reason": result.get("failure_reason"),
                "model_calls": result["model_calls"],
                "session_advances": result["session_advances"],
                "last_session_index": result["last_session_index"],
                "Y": result["Y"],
            }
        )
    require(len(estimates) == 248 and len(references) == 616, "248 EST and 616 REF draws")
    return output, estimates, references


def sha_value(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def reference_tables(cells: list[dict], references: dict) -> tuple[list, dict]:
    output, means = [], {}
    require(len(cells) == len({cell["cell_id"] for cell in cells}) == 77, "all 77 reference cells")
    for cell in cells:
        draws = [references[cell["cell_id"], rep] for rep in range(8)]
        require(
            all(row["anchor_id"] == cell["anchor_id"] for row in draws), "same reference anchor"
        )
        supported = all(row["status"] == "COMPLETE" for row in draws)
        values = [row["Y"] for row in draws]
        panel_means = (
            {
                "full8": math.fsum(values) / 8,
                "ref0_3": math.fsum(values[:4]) / 4,
                "ref4_7": math.fsum(values[4:]) / 4,
            }
            if supported
            else dict.fromkeys(PANELS)
        )
        if supported:
            require(
                close(panel_means["full8"], (panel_means["ref0_3"] + panel_means["ref4_7"]) / 2),
                "reference full mean is the mean of the fixed halves",
            )
        means[cell["cell_id"]] = panel_means
        output.append(
            {
                "cell_id": cell["cell_id"],
                "anchor_id": cell["anchor_id"],
                "decision_session_index": cell["decision_session_index"],
                "economic_order_key": cell["economic_order_key"],
                "added_hold": cell["added_hold"],
                "planned_reference_paths": 8,
                "complete_reference_paths": sum(row["status"] == "COMPLETE" for row in draws),
                "full8_supported": supported,
                "reference_statuses": [row["status"] for row in draws],
                "reference_Y": values,
                **{f"Qhat_{panel}": panel_means[panel] for panel in PANELS},
                "half_mean_difference": panel_means["ref4_7"] - panel_means["ref0_3"]
                if supported
                else None,
            }
        )
    return output, means


def credit_tables(manifest: dict, estimates: dict, references: dict) -> tuple[list, list]:
    heads = manifest["head_slots"]
    require(len(heads) == 128, "all original 128 heads")
    indexed(heads, lambda row: row["source_request_key"])
    cells = indexed(manifest["cells"], lambda row: row["cell_id"])
    groups: dict[str, list[dict]] = defaultdict(list)
    for head in heads:
        groups[head["anchor_id"]].append(head)
    require(len(groups) == 32, "all original 32 groups")
    require(sum(head["legal_head"] for head in heads) == 124, "124 legal and four rejected heads")
    group_output, slot_output = [], []
    for anchor, unsorted in sorted(groups.items()):
        members = sorted(unsorted, key=lambda row: row["slot_id"])
        require(
            len(members) == 4 and {h["slot_id"] for h in members} == {0, 1, 2, 3}, "original n4"
        )
        day = members[0]["source_group"]["decision_session_index"]
        require(
            all(head["source_group"] == members[0]["source_group"] for head in members), "prefix"
        )
        all_legal = all(head["legal_head"] for head in members)
        structural_keys = (
            [order_key(cells[head["cell_id"]]["economic_order_key"]) for head in members]
            if all_legal
            else None
        )
        stratum = (
            "ORIGINAL_PARTIAL"
            if not all_legal
            else "ALL_SAME_ORDER"
            if len(set(structural_keys)) == 1
            else "ORDER_DIVERSE_SAME_FAMILY"
            if len({key[0] for key in structural_keys}) == 1
            else "FAMILY_DIVERSE"
        )
        for rep in range(2):
            est = [
                estimates[head["source_request_key"], rep] if head["legal_head"] else None
                for head in members
            ]
            reasons = []
            if not all_legal:
                reasons.append("PARTIAL_ORIGINAL_HEAD_GROUP")
            if any(row is not None and row["status"] != "COMPLETE" for row in est):
                reasons.append("INCOMPLETE_ESTIMATE_CONTINUATION")
            if any(
                head["legal_head"] and references[head["cell_id"]]["full8"] is None
                for head in members
            ):
                reasons.append("INCOMPLETE_EIGHT_DRAW_REFERENCE")
            supported = not reasons
            for head, row in zip(members, est, strict=True):
                if row is not None:
                    require(
                        row["cell_id"] == head["cell_id"]
                        and row["anchor_id"] == head["anchor_id"]
                        and row["slot_id"] == head["slot_id"],
                        "original slot-specific independent estimate condition",
                    )
            predictions = dict.fromkeys(METHODS)
            targets = dict.fromkeys(PANELS)
            losses = {method: dict.fromkeys(PANELS) for method in METHODS}
            family_gate = exact_gate = None
            if supported:
                keys = structural_keys
                scores = [row["Y"] for row in est]
                predictions, family_gate, exact_gate = signals(scores, keys)
                for panel in PANELS:
                    q_values = [references[head["cell_id"]][panel] for head in members]
                    mean_q = math.fsum(q_values) / 4
                    targets[panel] = [value - mean_q for value in q_values]
                    require(close(math.fsum(targets[panel]), 0), "multiset-centered reference tau")
                    for method in METHODS:
                        losses[method][panel] = [
                            (prediction - target) ** 2
                            for prediction, target in zip(
                                predictions[method], targets[panel], strict=True
                            )
                        ]
                    projection_energy = (
                        math.fsum(
                            (a - b) ** 2
                            for a, b in zip(
                                predictions["order_gate_plain"],
                                predictions["exact_pool_order_gate"],
                                strict=True,
                            )
                        )
                        / 4
                    )
                    actual_delta = (
                        math.fsum(
                            a - b
                            for a, b in zip(
                                losses["exact_pool_order_gate"][panel],
                                losses["order_gate_plain"][panel],
                                strict=True,
                            )
                        )
                        / 4
                    )
                    require(
                        close(actual_delta, -projection_energy),
                        "exact-pool improvement is within-order projection energy identity",
                    )
            group_output.append(
                {
                    "anchor_id": anchor,
                    "estimate_replicate": rep,
                    "decision_session_index": day,
                    "order_stratum": stratum,
                    "planned_slots": 4,
                    "legal_original_heads": sum(head["legal_head"] for head in members),
                    "all_original_heads_legal": all_legal,
                    "main_supported": supported,
                    "NA_reasons": reasons,
                    "family_gate": family_gate,
                    "order_gate": exact_gate,
                    "raw_S": predictions,
                    "reference_tau": targets,
                    "method_slot_losses": losses,
                }
            )
            for index, head in enumerate(members):
                row = est[index]
                item = {
                    "source_request_key": head["source_request_key"],
                    "anchor_id": anchor,
                    "slot_id": head["slot_id"],
                    "estimate_replicate": rep,
                    "decision_session_index": day,
                    "order_stratum": stratum,
                    "cell_id": head["cell_id"],
                    "legal_original_head": head["legal_head"],
                    "head_status": head["head_status"],
                    "estimate_path_status": row["status"]
                    if row is not None
                    else "NO_LEGAL_HEAD_PATH",
                    "estimate_Y": row["Y"] if row is not None else None,
                    "main_supported": supported,
                    "NA_reasons": reasons,
                }
                for panel in PANELS:
                    item[f"reference_tau_{panel}"] = targets[panel][index] if supported else None
                for method in METHODS:
                    prediction = predictions[method][index] if supported else None
                    item[f"raw_{method}"] = prediction
                    item[f"eta025_{method}"] = ETA * prediction if supported else None
                    for panel in PANELS:
                        item[f"loss_{method}_{panel}"] = (
                            losses[method][panel][index] if supported else None
                        )
                slot_output.append(item)
    require(len(group_output) == 64 and len(slot_output) == 256, "64 group-reps and 256 slot-reps")
    return group_output, slot_output


def direction(value: float | None) -> str:
    if value is None:
        return "NA"
    return "negative" if value < -SIGN_ATOL else "positive" if value > SIGN_ATOL else "zero"


def summaries(groups: list[dict], slots: list[dict]) -> tuple[list, list, list, dict]:
    supported = [row for row in slots if row["main_supported"]]
    require(len(supported) % 4 == 0, "common complete n4 support")
    method_output = []
    for method, panel in itertools.product(METHODS, PANELS):
        total = math.fsum(row[f"loss_{method}_{panel}"] for row in supported)
        method_output.append(
            {
                "method": method,
                "reference_panel": panel,
                "planned_group_replicates": len(groups),
                "supported_group_replicates": len(supported) // 4,
                "planned_slot_replicates": len(slots),
                "supported_slot_replicates": len(supported),
                "NA_slot_replicates": len(slots) - len(supported),
                "raw_financial_Y_squared_slot_MSE": total / len(supported) if supported else None,
            }
        )
    comparison_output, group_comparisons = [], []
    for comparator, candidate in itertools.combinations(METHODS, 2):
        deltas = {}
        for panel in PANELS:
            losses = [
                row[f"loss_{candidate}_{panel}"] - row[f"loss_{comparator}_{panel}"]
                for row in supported
            ]
            deltas[panel] = math.fsum(losses) / len(losses) if losses else None
        if supported:
            require(
                close(deltas["full8"], (deltas["ref0_3"] + deltas["ref4_7"]) / 2),
                "common-reference paired loss delta is affine in the target",
            )
        signs = {panel: direction(deltas[panel]) for panel in PANELS}
        stable = bool(supported) and len(set(signs.values())) == 1
        structural = (
            candidate == "exact_pool_order_gate"
            and comparator in ("ungated_plain", "order_gate_plain")
        ) or (candidate == "order_gate_plain" and comparator == "ungated_plain")
        comparison_output.append(
            {
                "candidate": candidate,
                "comparator": comparator,
                "delta_definition": "candidate loss minus comparator loss",
                "supported_slot_replicates": len(supported),
                **{f"mean_paired_delta_{panel}": deltas[panel] for panel in PANELS},
                **{f"direction_{panel}": signs[panel] for panel in PANELS},
                "reference_half_direction_stable": stable,
                "structural_projection_or_same_order_gate_identity": structural,
                "interpretation": "STRUCTURAL_IDENTITY_NOT_REFERENCE_PRECISION_EVIDENCE"
                if structural and supported
                else "DESCRIPTIVE_DIRECTION_ONLY_FINITE_REFERENCE"
                if stable
                else "UNRESOLVED_REFERENCE_SENSITIVITY_OR_SUPPORT",
            }
        )
        for group in groups:
            group_comparisons.append(
                {
                    "anchor_id": group["anchor_id"],
                    "estimate_replicate": group["estimate_replicate"],
                    "decision_session_index": group["decision_session_index"],
                    "main_supported": group["main_supported"],
                    "NA_reasons": group["NA_reasons"],
                    "candidate": candidate,
                    "comparator": comparator,
                    **{
                        f"mean_paired_delta_{panel}": math.fsum(
                            a - b
                            for a, b in zip(
                                group["method_slot_losses"][candidate][panel],
                                group["method_slot_losses"][comparator][panel],
                                strict=True,
                            )
                        )
                        / 4
                        if group["main_supported"]
                        else None
                        for panel in PANELS
                    },
                }
            )
    counts = {
        "planned_group_replicates": len(groups),
        "planned_slot_replicates": len(slots),
        "original_partial_group_replicates": sum(
            not row["all_original_heads_legal"] for row in groups
        ),
        "supported_group_replicates": len(supported) // 4,
        "supported_slot_replicates": len(supported),
        "NA_group_replicates": len(groups) - len(supported) // 4,
        "NA_slot_replicates": len(slots) - len(supported),
        "all_methods_share_support": True,
    }
    return method_output, comparison_output, group_comparisons, counts


def run(directory: Path) -> None:
    manifest_path = directory / "continuation_manifest.json"
    completion_path = directory / "driver_completion.json"
    results_path = directory / "trajectory_results.jsonl"
    manifest, completion = read(manifest_path), read(completion_path)
    require(completion["status"] == "COMPLETE", "driver classified every planned continuation")
    require(completion["continuation_manifest_sha256"] == sha(manifest_path), "manifest identity")
    require(
        manifest["expected_trajectories"] == 864
        and manifest["expected_estimate_paths"] == 248
        and manifest["expected_reference_paths"] == 616
        and manifest["max_model_requests"] == REQUEST_LIMIT
        and manifest["max_attempts_per_session"] == 4
        and manifest["terminal_session_index"] == 20
        and manifest["estimate_continuations_deduplicated"] is False
        and manifest["source_same_order_continuation_state_comparisons"] == 76
        and manifest["source_same_order_continuation_states_identical"] is True
        and manifest["automatic_retry"] is False
        and manifest["fallback_hold"] is False,
        "fixed role-specific fresh comparison design",
    )
    seed_path = directory / "request_seed_table.jsonl"
    require(sha(seed_path) == manifest["request_seed_table_sha256"], "fresh seed table identity")
    seeds = rows(seed_path)
    require(
        len(seeds)
        == len({row["request_key"] for row in seeds})
        == len({row["request_seed"] for row in seeds})
        == REQUEST_LIMIT
        and Counter(row["role"] for row in seeds) == {"estimate": 11408, "reference": 27616},
        "disjoint role and original-slot request seeds",
    )
    for relative, digest in completion["output_sha256"].items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "relative output path")
        require(sha(directory / path) == digest, f"driver output identity: {relative}")
    paths, estimates, reference_draws = path_tables(manifest, rows(results_path))
    references, means = reference_tables(manifest["cells"], reference_draws)
    groups, slots = credit_tables(manifest, estimates, means)
    methods, comparisons, group_comparisons, counts = summaries(groups, slots)
    require(counts["original_partial_group_replicates"] == 6, "six original partial group-reps")
    model_calls = sum(row["model_calls"] for row in paths)
    require(model_calls == completion["model_requests"] <= REQUEST_LIMIT, "actual request cost")
    require(
        sum(row["model_calls"] for row in paths if row["role"] == "estimate")
        + sum(row["model_calls"] for row in paths if row["role"] == "reference")
        == model_calls,
        "disjoint estimate/reference request accounting",
    )
    require(
        completion["trajectories_classified"] == 864
        and completion["fixed_first_actions"] == 864
        and completion["environments_constructed"] == 864
        and completion["optimizer_updates"] == 0
        and completion["formal_valid_oos_access"] is False
        and completion["automatic_retry"] is False,
        "actual execution scope and complete path denominator",
    )
    role_costs = {
        role: {
            "planned_paths": sum(row["role"] == role for row in paths),
            "path_status_counts": dict(
                Counter(row["status"] for row in paths if row["role"] == role)
            ),
            "model_requests": sum(row["model_calls"] for row in paths if row["role"] == role),
        }
        for role in ("estimate", "reference")
    }
    require(
        role_costs["estimate"]["model_requests"] <= 11408
        and role_costs["reference"]["model_requests"] <= 27616,
        "fixed role budgets",
    )
    strata_methods, strata_comparisons, strata_counts = [], [], []
    for stratum in (
        "ALL_SAME_ORDER",
        "ORDER_DIVERSE_SAME_FAMILY",
        "FAMILY_DIVERSE",
        "ORIGINAL_PARTIAL",
    ):
        gs = [group for group in groups if group["order_stratum"] == stratum]
        ss = [slot for slot in slots if slot["order_stratum"] == stratum]
        ms, cs, _, ns = summaries(gs, ss)
        strata_methods.extend({"order_stratum": stratum, **row} for row in ms)
        strata_comparisons.extend({"order_stratum": stratum, **row} for row in cs)
        strata_counts.append({"order_stratum": stratum, **ns})
    names = {
        "credit_paths.csv": paths,
        "credit_references.csv": references,
        "credit_groups.csv": groups,
        "credit_slots.csv": slots,
        "credit_methods.csv": methods,
        "credit_comparisons.csv": comparisons,
        "credit_group_comparisons.csv": group_comparisons,
        "credit_strata_methods.csv": strata_methods,
        "credit_strata_comparisons.csv": strata_comparisons,
    }
    for name, items in names.items():
        write_csv(directory / name, items)
    report = {
        "profile": "fresh_estimate_reference_credit_comparison_v1",
        "status": "PASS",
        "at_UTC": datetime.now(timezone.utc).isoformat(),
        "scope": "descriptive raw credit loss against an independent finite Monte Carlo reference",
        "counts": {
            **counts,
            "planned_paths": 864,
            "estimate_paths": 248,
            "reference_paths": 616,
            "path_status_counts": dict(Counter(row["status"] for row in paths)),
            "reference_cells_full8_supported": sum(row["full8_supported"] for row in references),
            "model_requests": model_calls,
            "maximum_model_requests": REQUEST_LIMIT,
        },
        "method_summary": methods,
        "paired_comparisons": comparisons,
        "order_strata": strata_counts,
        "role_costs": role_costs,
        "definitions": {
            "target": (
                "Qhat per concrete cell, centered with the original four-slot multiset weights"
            ),
            "support": (
                "all four legal original heads, four complete EST paths, "
                "and eight complete REF paths for every required cell"
            ),
            "partial": (
                "every main signal, tau and loss for all four slots stays NA, including zero"
            ),
            "pool": (
                "within-family/exact-order EST mean, minus the same four-slot EST mean; "
                "both pools share order gate"
            ),
            "production_family_pool": (
                "family_pool_order_gate equals the legacy family-pool formula on the same "
                "four inputs; this is not full production-pipeline integration"
            ),
            "units": "raw terminal financial Y squared, never percentage-point squared",
            "eta": "0.25*S is a derived common scaling only; raw S is compared to raw tau",
            "direction": (
                "paired candidate-minus-comparator mean loss; negative is smaller descriptive loss"
            ),
            "sign_tolerance": SIGN_ATOL,
        },
        "limits": [
            "The reference has eight draws per cell, split 0..3 and 4..7; it is not exact Q^pi.",
            (
                "Exact-pool versus plain and the all-same-order gate comparison have structural "
                "projection improvements independent of reference values; their half-sign "
                "agreement is not reference-precision evidence."
            ),
            (
                "Opposing/zero-vs-nonzero reference-half directions are unresolved, "
                "not a method winner."
            ),
            (
                "Two EST draws per original slot and overlapping Train states "
                "do not yield iid market samples or significance."
            ),
            "This target is a realized-four-action centered Q^pi proxy, not A^pi or policy return.",
            (
                "No GAE, whitening, critic fitting, actor update, formal Valid "
                "or Final OOS is performed."
            ),
        ],
        "input_sha256": {
            p.name: sha(p) for p in (manifest_path, completion_path, results_path, seed_path)
        },
        "output_sha256": {name: sha(directory / name) for name in names},
        "readout_source_sha256": sha(Path(__file__).resolve()),
    }
    with (directory / "credit_readout.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "counts": report["counts"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run(args.run.resolve())


if __name__ == "__main__":
    main()

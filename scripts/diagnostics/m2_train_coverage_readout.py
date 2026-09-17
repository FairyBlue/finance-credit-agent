#!/usr/bin/env python3
"""CPU readout of 16-window Train coverage and smaller common actor LR.

The training worker supplies all trajectories. Evaluation lane 0 is greedy;
lanes 1..4 are sampled. Never pool these lanes or reduce an incomplete window's
fixed denominator. Compare window means, not alleged identical sampled requests.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
DEPENDENCY = ROOT / "scripts/diagnostics/m2_training_tuning_readout.py"
DEPENDENCY_SHA256 = "c45b9dab2512e0ff94b8b5b8c06a93942a86af84da91c1099a76765385acc700"
_spec = importlib.util.spec_from_file_location("coverage_pinned_tuning_helpers", DEPENDENCY)
assert _spec is not None and _spec.loader is not None
shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared)
if shared.sha(DEPENDENCY) != DEPENDENCY_SHA256:
    raise RuntimeError("pinned tuning evidence helper SHA-256 differs")
Evidence = shared.Evidence
sha, load_helpers = shared.sha, shared.load_helpers
read_trajectories, critic_fit_summary = shared.read_trajectories, shared.critic_fit_summary
_indexed_records, finite_tree = shared._indexed_records, shared.finite_tree
PROFILE = "common_sft_train_coverage_v1"
TRAIN_IDS = [
    f"formal-train-{i:04d}"
    for i in (1, 65, 9, 73, 17, 149, 25, 157, 33, 165, 41, 173, 49, 181, 57, 189)
]
EVAL_IDS = [f"formal-train-{i:04d}" for i in (113, 117, 121, 125)]
EVAL_STEPS = [2, 4, 6, 8]
ACTOR_LR, CRITIC_LR = 2e-7, 1e-6
OUTCOMES = (
    "Y",
    "net_return",
    "net_return_pct",
    "terminal_cost",
    "drawdown",
    "turnover",
    "max_weight",
    "final_cash",
    "final_cash_pct",
    "final_equity",
    "final_position_count",
    "fill_count",
    "invalid_per_trajectory",
    "invalid_request_rate_pct",
    "hold_count",
)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def mean(values):
    return math.fsum(values) / len(values) if values else None


def cell_specs(settings):
    contract = settings.get("training_coverage", {})
    if contract.get("profile") != PROFILE or contract.get("evaluation_steps") != EVAL_STEPS:
        raise ValueError("directory does not contain the fixed training-coverage profile/schedule")
    training = settings.get("training", {})
    if training.get("learning_rate") != ACTOR_LR or (
        training.get("critic_learning_rate") != CRITIC_LR
    ):
        raise ValueError("common actor/critic learning rates differ from coverage contract")
    return [
        dict(zip(("cell_id", "arm", "eta", "critic_lr", "rounds"), row, strict=True))
        for row in (
            ("sft0", "M1", 0.0, CRITIC_LR, 0),
            ("m1", "M1", 0.0, CRITIC_LR, 8),
            ("m2_eta025", "M2", 0.25, CRITIC_LR, 8),
        )
    ]


def expected_grid(spec, train_ids=TRAIN_IDS, eval_ids=EVAL_IDS, evaluation_steps=EVAL_STEPS):
    if (
        train_ids != TRAIN_IDS
        or eval_ids != EVAL_IDS
        or evaluation_steps not in (EVAL_STEPS, [0] if not spec["rounds"] else EVAL_STEPS)
    ):
        raise ValueError("frozen 16-window ordering/four-window evaluation schedule differs")
    expected = {}
    for step in range(1, spec["rounds"] + 1):
        for index in (2 * (step - 1), 2 * (step - 1) + 1):
            for sibling in range(4):
                expected[(step, index, sibling, False)] = train_ids[index]
    for step in evaluation_steps if spec["rounds"] else [0]:
        for index, episode in enumerate(eval_ids):
            for sibling in range(5):
                expected[(step, index, sibling, True)] = episode
    return expected


def lane_of(row):
    identity = row["identity"]
    if not identity["validate"]:
        return "train"
    return "greedy" if identity["rollout_n"] == 0 else "sampled"


def count_population(helpers, rows, planned):
    """Observable receipts retain partial behavior; no completion-only mean is a headline."""
    result = helpers.aggregate(rows, planned)
    for key in list(result):
        if key.endswith("complete_only"):
            del result[key]
    requests, invalid = result["requests_observed"], result["invalid_observed"]
    result.update(
        invalid_per_observed_request_pct=100 * invalid / requests if requests else None,
        unique_filled_symbols_observed=len(result["fill_symbols_observed"]),
        reported_status_counts=dict(Counter(row["reported_status"] for row in rows)),
        eligibility_problem_counts=dict(Counter(p for row in rows for p in row["problems"])),
        behavior_denominator="all observed receipts, including incomplete trajectories",
    )
    return result


def outcome(row):
    if row is None or not row["terminal_eligible"]:
        return None
    equity, cash = row.get("final_equity"), row.get("final_cash")
    if not finite(equity) or equity <= 0 or not finite(cash) or not row.get("requests"):
        return None
    result = {key: row.get(key) for key in OUTCOMES}
    result.update(
        net_return_pct=row["net_return"] * 100,
        final_cash_pct=cash / equity * 100,
        invalid_per_trajectory=len(row["invalid_errors"]),
        invalid_request_rate_pct=len(row["invalid_errors"]) / row["requests"] * 100,
        hold_count=row["executed_step_actions"].get("hold", 0),
    )
    return result if all(finite(value) for value in result.values()) else None


def fixed_population(helpers, rows, grid, *, step=None, lane):
    """Average each complete fixed sibling group, then each fixed window equally."""
    selected = {
        key: episode
        for key, episode in grid.items()
        if (step is None or key[0] == step)
        and ("train" if not key[3] else "greedy" if key[2] == 0 else "sampled") == lane
    }
    members = {helpers.grid_key(row): row for row in rows if helpers.grid_key(row) in selected}
    groups = defaultdict(list)
    for key, episode in selected.items():
        groups[(key[0], episode)].append(key)
    windows = []
    for (policy_step, episode), keys in groups.items():
        present = [members[key] for key in keys if key in members]
        outcomes = [outcome(members.get(key)) for key in keys]
        complete = all(item is not None for item in outcomes)
        windows.append(
            {
                "policy_step": policy_step,
                "episode_id": episode,
                "planned_siblings": len(keys),
                "expected_rollout_n": [key[2] for key in keys],
                "counts": count_population(helpers, present, len(keys)),
                "complete_outcome_siblings": sum(item is not None for item in outcomes),
                "fixed_sibling_mean": {
                    key: mean([item[key] for item in outcomes]) for key in OUTCOMES
                }
                if complete
                else None,
            }
        )
    complete_windows = [w for w in windows if w["fixed_sibling_mean"] is not None]
    counts = count_population(helpers, list(members.values()), len(selected))
    return {
        **counts,
        "sampling": lane,
        "planned_windows": len(windows),
        "complete_windows": len(complete_windows),
        "fixed_window_mean": {
            key: mean([w["fixed_sibling_mean"][key] for w in windows]) for key in OUTCOMES
        }
        if windows and len(complete_windows) == len(windows)
        else None,
        "windows": windows,
        "mean_rule": "fixed siblings per window, then equal weight for every planned window",
    }


def actor_probe_readout(evidence, cell, spec, identity):
    records = defaultdict(list)
    problems, pending = [], []
    for reference, raw in evidence.jsonl(cell / "artifacts/actor_update"):
        record = {
            "source": reference,
            **{key: value for key, value in raw.items() if key != "rows"},
        }
        record["detail_rows_at_source"] = len(raw.get("rows", []))
        record["token_details_retained_at_source"] = True
        records[record.get("policy_step")].append(record)
    expected = set(range(1, spec["rounds"] + 1))
    if set(records) - expected or any(len(group) != 1 for group in records.values()):
        problems.append("unexpected/duplicate actor_update_probe policy step")
    if set(records) != expected:
        pending.append("actor_update_probe policy steps not yet complete")
    for step, group in records.items():
        if len(group) != 1:
            continue
        r = group[0]
        g = r.get("groups", {})
        all_real, legal, invalid = [g.get(name, {}) for name in ("all_real", "legal", "invalid")]
        valid_groups = all(
            type(s.get(key)) is int and s[key] >= 0
            for s in (all_real, legal, invalid)
            for key in ("step_count", "response_tokens")
        )
        if (
            r.get("schema_version") != 1
            or r.get("status") != "COMPLETE"
            or r.get("phase") != "actor_same_batch_pre_post"
            or r.get("arm") != spec["arm"]
            or r.get("execution_identity_sha256") != identity
            or r.get("update_call_index") != step
            or r.get("optimizer_step_count") != 1
            or r.get("ppo_epochs") != 1
            or r.get("planned_minibatch_count") != 1
            or r.get("actor_learning_rates") != [ACTOR_LR]
            or r.get("extra_post_forward_micro_batches") != r.get("real_step_count")
            or all_real.get("step_count") != r.get("real_step_count")
            or all_real.get("response_tokens") != r.get("response_token_count")
            or r.get("global_response_token_denominator") != r.get("response_token_count")
            or not valid_groups
            or not finite_tree(r)
        ):
            problems.append(f"step {step} actor_update_probe protocol/identity differs")
        if valid_groups and any(
            legal[key] + invalid[key] != all_real[key] for key in ("step_count", "response_tokens")
        ):
            problems.append(f"step {step} actor_update_probe legal/invalid totals differ")
    return {
        "directory": str((cell / "artifacts/actor_update").relative_to(evidence.run)),
        "planned_records": spec["rounds"],
        "observed_records": sum(len(group) for group in records.values()),
        "summary_complete": not problems and not pending,
        "by_step": {str(step): group for step, group in records.items()},
        "problems": problems,
        "pending": pending,
        "interpretation": (
            "same-input teacher-forced response-token logp change, not generated action probability"
        ),
    }


def paired_validation_readout(evidence, cell, spec, identity, grid):
    helpers = load_helpers()
    expected = {key: episode for key, episode in grid.items() if key[3]}
    records = defaultdict(list)
    problems, pending = [], []
    for reference, raw in evidence.jsonl(cell / "artifacts/paired_validation"):
        records[helpers.grid_key(raw)].append({"source": reference, **raw})
    if set(records) - set(expected) or any(len(group) != 1 for group in records.values()):
        problems.append("unexpected/duplicate paired_validation identity")
    if set(records) != set(expected):
        pending.append("paired_validation receipts not yet complete")
    for key, group in records.items():
        if key not in expected or len(group) != 1:
            continue
        record = group[0]
        mode = "greedy" if key[2] == 0 else "sampled"
        forwarded = record.get("forwarded_sampling_params", {})
        upstream = record.get("upstream_sampling_params", {})
        sampling = {
            "temperature": 0.0 if mode == "greedy" else 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }
        if (
            record.get("schema_version") != 1
            or record.get("profile") != "greedy1_sample4_v1"
            or record.get("status") != "COMPLETE"
            or record.get("arm") != spec["arm"]
            or record.get("execution_identity_sha256") != identity
            or record.get("episode_id") != expected[key]
            or record.get("policy_step") != key[0]
            or record.get("validation_rollout_n") != 5
            or record.get("mode") != mode
            or record.get("sampled_sibling") != (None if key[2] == 0 else key[2] - 1)
            or forwarded != {**upstream, **sampling}
            or not finite_tree(record)
        ):
            problems.append(f"paired_validation mode/identity/sampling differs: {key}")
    return {
        "planned_records": len(expected),
        "observed_records": sum(len(group) for group in records.values()),
        "complete": not problems and not pending,
        "records": [row for group in records.values() for row in group],
        "problems": problems,
        "pending": pending,
    }


def cell_readout(evidence, helpers, spec, settings, model, manifest):
    cell = evidence.run / spec["cell_id"]
    summary = evidence.json(cell / "summary.json") or {}
    command = evidence.json(cell / "command.json") or []
    options = {
        value.lstrip("+").split("=", 1)[0]: value.split("=", 1)[1]
        for value in command
        if isinstance(value, str) and "=" in value
    }
    identity = options.get(helpers.IDENTITY_OPTION)
    train_ids = manifest["datasets"]["train"]["episode_ids"]
    eval_ids = manifest["datasets"]["dev"]["episode_ids"]
    eval_steps = settings["training_coverage"]["evaluation_steps"] if spec["rounds"] else [0]
    grid = expected_grid(spec, train_ids, eval_ids, eval_steps)
    rows, excluded, raw_count = read_trajectories(
        evidence, helpers, cell, spec, grid, manifest.get("calibration", {}).get("payload_sha256")
    )
    problems, pending = [], []
    for row in rows:
        if row["terminal_eligible"] and outcome(row) is None:
            row["terminal_eligible"] = False
            row["Y"] = row["net_return"] = None
            row["problems"].append("nonfinite/missing required terminal portfolio outcome")
    if not command:
        pending.append("cell command is missing")
    if excluded:
        problems.append("trajectory identities are duplicate, outside the grid, or uninterpretable")
    if any(row["problems"] for row in rows):
        problems.append("observed trajectory failed eligibility checks")
    if len(rows) != len(grid) or not all(row["terminal_eligible"] for row in rows):
        pending.append("fixed trajectory grid is not completely terminal eligible")
    diagnostics = _indexed_records(evidence, cell / "artifacts/sibling_diagnostics", "policy_step")
    fits = _indexed_records(evidence, cell / "artifacts/critic_fit", "policy_step")
    updates = [
        dict(source=reference, **record)
        for reference, record in evidence.jsonl(cell / "artifacts/update_probe")
    ]
    log_path = cell / "trainer.log"
    before_log = log_path.stat() if log_path.is_file() else None
    logs = helpers.training_log(log_path, evidence)
    if before_log is not None:
        after_log = log_path.stat()
        if (before_log.st_size, before_log.st_mtime_ns) != (
            after_log.st_size,
            after_log.st_mtime_ns,
        ):
            evidence.pending.append(f"appended during read {log_path.relative_to(evidence.run)}")
    logged_by_step = defaultdict(list)
    for row in logs:
        logged_by_step[row["policy_step"]].append(row)
    changed = Counter(row.get("role") for row in updates if row.get("detected") is True)
    update_keys = [
        (row.get("pid"), row.get("optimizer_step_count"), row.get("role")) for row in updates
    ]
    if len(set(update_keys)) != len(update_keys):
        problems.append("duplicate optimizer update identity")
    if len(updates) > spec["rounds"] * 2:
        problems.append("optimizer update count exceeds frozen budget")
    if any(row.get("detected") is not True or not finite_tree(row) for row in updates):
        problems.append("optimizer update is unchanged or contains nonfinite evidence")
    critic_updates = sorted(
        [row for row in updates if row.get("role") == "critic"],
        key=lambda row: row.get("global_update_count", -1),
    )
    if dict(changed) != (
        {"actor": spec["rounds"], "critic": spec["rounds"]} if spec["rounds"] else {}
    ):
        pending.append("changed actor/critic update counts do not yet match budget")
    expected_steps = set(range(1, spec["rounds"] + 1))
    for name, records in (
        ("advantage", diagnostics),
        ("critic_fit", fits),
        ("training_log", logged_by_step),
    ):
        if set(records) - expected_steps or any(len(items) > 1 for items in records.values()):
            problems.append(f"unexpected/duplicate {name} policy step")
        if set(records) != expected_steps:
            pending.append(f"{name} policy steps not yet complete")
    rounds = []
    for step in range(1, spec["rounds"] + 1):
        train_rows = [
            row
            for row in rows
            if not row["identity"]["validate"] and row["identity"]["step"] == step
        ]
        diagnostic = diagnostics[step][0] if len(diagnostics[step]) == 1 else None
        fit = fits[step][0] if len(fits[step]) == 1 else None
        if diagnostic is not None and (
            diagnostic.get("arm") != spec["arm"]
            or diagnostic.get("eta") != spec["eta"]
            or diagnostic.get("run_seed") != 42
            or not finite_tree(diagnostic)
        ):
            problems.append(f"step {step} advantage arm/eta/seed differs")
        if fit is not None and (
            fit.get("status") != "COMPLETE"
            or fit.get("phase") != "critic_same_batch_pre_post"
            or fit.get("arm") != spec["arm"]
            or fit.get("execution_identity_sha256") != identity
            or fit.get("update_call_index") != step
            or fit.get("extra_forward_passes") != 1
            or fit.get("value_token_count") != fit.get("real_step_count")
            or fit.get("pre", {}).get("returns") != fit.get("post", {}).get("returns")
            or fit.get("critic_learning_rates") != [spec["critic_lr"]]
            or fit.get("planned_mini_batch_count") != 1
            or fit.get("ppo_epochs") != 1
            or fit.get("returns_recomputed") is not False
            or fit.get("dropout", {}).get("any_nonzero") is not False
            or fit.get("dropout", {}).get("fit_comparison_uses_same_mode") is not True
            or fit.get("mode_restored") is not True
            or not finite_tree(fit)
        ):
            problems.append(f"step {step} critic fit protocol/identity differs")
        if (
            fit is not None
            and diagnostic is not None
            and (fit.get("real_step_count") != diagnostic.get("real_step_count"))
        ):
            problems.append(f"step {step} critic/advantage row counts differ")
        if fit is not None and len(critic_updates) >= step:
            update = critic_updates[step - 1]
            if fit.get("pid") != update.get("pid") or fit.get(
                "optimizer_grad_norms_before_clip"
            ) != [update.get("grad_norm")]:
                problems.append(f"step {step} critic fit/optimizer gradient differs")
        rounds.append(
            {
                "policy_step": step,
                "episode_ids": sorted(
                    {episode for key, episode in grid.items() if key[0] == step and not key[3]}
                ),
                "population": fixed_population(helpers, train_rows, grid, step=step, lane="train"),
                "advantage_diagnostics_as_recorded": diagnostics[step],
                "critic_fit": critic_fit_summary(fit),
                "actor_critic_metrics_as_logged": logged_by_step[step],
            }
        )
    validation = paired_validation_readout(evidence, cell, spec, identity, grid)
    problems.extend(validation["problems"])
    pending.extend(validation["pending"])
    actor_probes = actor_probe_readout(evidence, cell, spec, identity)
    problems.extend(actor_probes["problems"])
    pending.extend(actor_probes["pending"])
    for record in rounds:
        step = record["policy_step"]
        record["actor_update_as_recorded"] = actor_probes["by_step"].get(str(step), [])
    evaluations = [
        {
            "policy_step": step,
            "sampling": lane,
            "population": fixed_population(helpers, rows, grid, step=step, lane=lane),
        }
        for step in eval_steps
        for lane in ("greedy", "sampled")
    ]
    head = evidence.json(cell / "artifacts/critic_initialization/head-rank0.json") or {}
    if not head:
        pending.append("actual initial critic head evidence is missing")
    elif (
        head.get("seed") != 42
        or head.get("mode") != "cpu_seed_v1"
        or head.get("arm") != spec["arm"]
        or head.get("execution_identity_sha256") != identity
        or head.get("model_path") != model.get("model_snapshot_path")
        or not isinstance(head.get("head_sha256"), str)
        or len(head["head_sha256"]) != 64
    ):
        problems.append("actual critic head identity differs")
    if command and any(
        options.get(key) != model.get("model_snapshot_path") for key in helpers.MODEL_OPTIONS
    ):
        problems.append("common actor/reference/critic/tokenizer model identity differs")
    if summary and (
        summary.get("cell") != spec or summary.get("model_revision") != model.get("model_revision")
    ):
        problems.append("cell summary contract/model identity differs")
    if summary.get("status") != "COMPLETE":
        pending.append(f"cell summary is {summary.get('status', 'MISSING')}")
    if summary.get("status") == "COMPLETE" and summary.get("actual_rounds") != spec["rounds"]:
        problems.append("completed cell reports wrong round count")
    status = (
        "FAILED"
        if summary.get("status") == "FAILED"
        else ("COMPLETE" if not problems and not pending else "INCOMPLETE")
    )
    return {
        "status": status,
        "spec": spec,
        "summary": summary,
        "problems": problems,
        "pending": pending,
        "raw_trajectory_records": raw_count,
        "excluded_identity_records": excluded,
        "groups": {
            "train": fixed_population(helpers, rows, grid, lane="train"),
            "eval_greedy": count_population(
                helpers, [r for r in rows if lane_of(r) == "greedy"], len(eval_steps) * 4
            ),
            "eval_sampled": count_population(
                helpers, [r for r in rows if lane_of(r) == "sampled"], len(eval_steps) * 16
            ),
        },
        "counts": count_population(helpers, rows, len(grid)),
        "actor_update_probe": actor_probes,
        "paired_validation_as_recorded": validation,
        "rounds": rounds,
        "evaluations": evaluations,
        "trajectory_rows": rows,
        "optimizer_updates": {
            "records": len(updates),
            "detected_by_role": dict(changed),
            "raw": updates,
        },
        "initial_critic_head": head,
        "model_options": {key: options.get(key) for key in helpers.MODEL_OPTIONS},
        "execution_identity_sha256": identity,
        "critic_fit_as_recorded": [row for group in fits.values() for row in group],
        "extra_critic_forward_seconds": sum(
            row.get("extra_forward_seconds_synchronized", 0.0)
            for group in fits.values()
            for row in group
        ),
    }


def paired_curves(cells, eval_ids=EVAL_IDS, evaluation_steps=EVAL_STEPS):
    """Pair fixed windows, after independently averaging each model's siblings."""
    curves = []
    for target, reference in (("m1", "sft0"), ("m2_eta025", "sft0"), ("m2_eta025", "m1")):
        for lane in ("greedy", "sampled"):
            points = []
            for step in evaluation_steps:
                reference_step = 0 if reference == "sft0" else step
                populations = {}
                for name, observed_step in ((target, step), (reference, reference_step)):
                    matches = [
                        item["population"]
                        for item in cells[name]["evaluations"]
                        if item["policy_step"] == observed_step and item["sampling"] == lane
                    ]
                    populations[name] = (
                        {w["episode_id"]: w for w in matches[0]["windows"]}
                        if len(matches) == 1
                        else {}
                    )
                pairs = []
                for episode in eval_ids:
                    after = populations[target].get(episode, {}).get("fixed_sibling_mean")
                    before = populations[reference].get(episode, {}).get("fixed_sibling_mean")
                    delta = None
                    if after is not None and before is not None:
                        delta = {key: after[key] - before[key] for key in OUTCOMES}
                        delta["net_return_percentage_points"] = delta["net_return_pct"]
                    pairs.append(
                        {
                            "episode_id": episode,
                            "target_window_mean": after,
                            "reference_window_mean": before,
                            "delta_target_minus_reference": delta,
                        }
                    )
                complete = [
                    p["delta_target_minus_reference"]
                    for p in pairs
                    if p["delta_target_minus_reference"] is not None
                ]
                points.append(
                    {
                        "policy_step": step,
                        "reference_policy_step": reference_step,
                        "planned_pairs": len(eval_ids),
                        "complete_pairs": len(complete),
                        "fixed_four_window_mean_delta": {
                            key: mean([row[key] for row in complete])
                            for key in (*OUTCOMES, "net_return_percentage_points")
                        }
                        if len(complete) == len(eval_ids)
                        else None,
                        "wins_among_complete_windows": sum(
                            row["net_return"] > 0 for row in complete
                        ),
                        "ties_among_complete_windows": sum(
                            row["net_return"] == 0 for row in complete
                        ),
                        "pairs": pairs,
                    }
                )
            curves.append(
                {
                    "target": target,
                    "reference": reference,
                    "sampling": lane,
                    "direction": "target minus reference",
                    "pairing_unit": "same fixed episode window, with independent sibling means",
                    "request_pairing_claimed": False,
                    "points": points,
                }
            )
    return curves


def build(directory):
    helpers = load_helpers()
    evidence = Evidence(Path(directory).resolve())
    settings = evidence.json(evidence.run / "settings.json", required=True) or {}
    specs = cell_specs(settings)
    state = evidence.json(evidence.run / "state.json") or {}
    model = evidence.json(evidence.run / "model_identity.json", required=True) or {}
    manifest = evidence.json(evidence.run / "prepared/manifest.json", required=True) or {}
    contract = evidence.json(evidence.run / "run_contract.json", required=True) or {}
    sources = evidence.json(evidence.run / "source_identity.json", required=True) or {}
    selection = evidence.json(evidence.run / "selection.json")
    if manifest.get("datasets", {}).get("train", {}).get("episode_ids") != TRAIN_IDS or (
        manifest.get("datasets", {}).get("dev", {}).get("episode_ids") != EVAL_IDS
    ):
        raise ValueError(
            "prepared manifest differs from frozen 16 Train/four Train-evaluation windows"
        )
    if state.get("schedule") is not None and state["schedule"] != specs:
        evidence.errors.append("controller schedule differs from coverage cells")
    for name, expected_hash in contract.get("inputs", {}).items():
        path = evidence.run / name
        if not path.is_file() or sha(path) != expected_hash:
            evidence.errors.append(f"sealed input differs: {name}")
    cells = {
        spec["cell_id"]: cell_readout(evidence, helpers, spec, settings, model, manifest)
        for spec in specs
    }
    head_hashes = {
        name: cell["initial_critic_head"].get("head_sha256") for name, cell in cells.items()
    }
    observed_heads = [value for value in head_hashes.values() if value is not None]
    if len(set(observed_heads)) > 1:
        evidence.errors.append("observed common initial critic head hashes differ")
    all_heads_equal = len(observed_heads) == len(cells) and len(set(observed_heads)) == 1
    problems = evidence.errors + [
        f"{name}: {problem}" for name, cell in cells.items() for problem in cell["problems"]
    ]
    pending = evidence.pending + [
        f"{name}: {problem}" for name, cell in cells.items() for problem in cell["pending"]
    ]
    if not all_heads_equal:
        pending.append("common initial critic heads not fully verified")
    if state.get("status") != "COMPLETE":
        pending.append(f"controller is {state.get('status', 'NOT_STARTED')}")
    total = {
        key: sum(cell["counts"][key] for cell in cells.values())
        for key in (
            "planned",
            "observed_unique_expected",
            "terminal_complete",
            "incomplete_observed",
            "missing",
            "requests_observed",
            "invalid_observed",
        )
    }
    status = (
        "FAILED"
        if state.get("status") == "FAILED"
        else "COMPLETE"
        if not problems and not pending
        else "INCOMPLETE"
    )
    now = datetime.now(timezone.utc)
    result = {
        "schema_version": 1,
        "profile": PROFILE,
        "status": status,
        "execution_status": state.get("status", "NOT_STARTED"),
        "all_expected_trajectories_observed": total["observed_unique_expected"] == total["planned"],
        "all_expected_trajectories_complete": total["terminal_complete"] == total["planned"],
        "generated_at_UTC": now.isoformat(),
        "generated_at_SGT": now.astimezone(ZoneInfo("Asia/Singapore")).isoformat(),
        "controller": state,
        "settings": settings,
        "selection": selection,
        "shared_model_identity": model,
        "source_identity": sources,
        "run_contract": contract,
        "totals": total,
        "cells": cells,
        "evaluation_curves": paired_curves(cells),
        "initial_critic_heads": {"hashes": head_hashes, "all_equal": all_heads_equal},
        "problems": problems,
        "pending": pending,
        "input_sha256": evidence.hashes,
        "stream_snapshots": evidence.streams,
        "dependencies": {
            "tuning_evidence_helper": {"path": str(DEPENDENCY), "sha256": DEPENDENCY_SHA256},
            "scalar_helper": {"path": str(shared.HELPER_PATH), "sha256": shared.HELPER_SHA256},
        },
        "readout_script_sha256": sha(Path(__file__)),
        "limits": [
            "All dates belong to Train; no formal Valid/Test or forward-time OOS evidence.",
            "Actor LR and Train date coverage changed together; no single-factor attribution.",
            "Train episodes change each round; Train returns are not a fixed-window curve.",
            "Greedy lane 0 and sampled lanes 1..4 stay separate; all four siblings are required.",
            "Every mean needs all four windows; missing/incomplete/nonfinite outcomes stay null.",
            "SFT step 0 and RL steps 2/4/6/8 have different seed identities; "
            "window comparisons are not identical request pairs.",
            "All evaluations use the training-worker path; old standalone scores are not pooled.",
            "Actor probe delta is teacher-forced logp drift for a captured response, "
            "not generated action probability.",
            "Actor token details remain at the hashed source; original summaries are retained.",
            "Observed request/invalid/action/fill counts include partial trajectories; "
            "these counters are not zero-filled outcomes.",
            "Active reads are not atomic; refresh before final interpretation. "
            "Model tensors are not loaded or hashed.",
        ],
    }
    return helpers.json_safe(result)


def markdown(result):
    updated = datetime.fromisoformat(result["generated_at_SGT"]).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 扩展 Train 日期与共同 actor 小学习率：当前进展",
        "",
        f"更新时间：{updated} SGT（UTC+8）",
        "",
        f"执行状态：**{result['execution_status']}**；证据状态：**{result['status']}**。",
        "",
        "所有日期均在 Train。greedy 与 sampled 分开；sampled 先取每窗四条均值，再固定四窗等权。",
        "",
        "| 实验组 | 训练完整/计划 | greedy 完整/计划 | sampled 完整/计划 | actor probe 完整/计划 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, cell in result["cells"].items():
        groups, probe = cell["groups"], cell["actor_update_probe"]
        counts = [
            f"{groups[k]['terminal_complete']}/{groups[k]['planned']}"
            for k in ("train", "eval_greedy", "eval_sampled")
        ]
        lines.append(
            f"| {name} | {' | '.join(counts)} "
            f"| {probe['observed_records']}/{probe['planned_records']} |"
        )
    lines += [
        "",
        "| 实验组 | 更新轮数 | 解码 | 完整窗口 | 净收益 % | Y | 成本 USD | 现金 % |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for name, cell in result["cells"].items():
        for item in cell["evaluations"]:
            group = item["population"]
            values = group["fixed_window_mean"]
            formatted = [
                f"{values[k]:.6f}" if values else "待齐全"
                for k in ("net_return_pct", "Y", "terminal_cost", "final_cash_pct")
            ]
            lines.append(
                f"| {name} | {item['policy_step']} | {item['sampling']} "
                f"| {group['complete_windows']}/4 | {' | '.join(formatted)} |"
            )
    lines += [
        "",
        "| 实验组 − 参照组 | 更新轮数 | 解码 | 完整窗口 | 净收益 Δpp | 赢窗（完整窗内） |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for curve in result["evaluation_curves"]:
        for point in curve["points"]:
            values = point["fixed_four_window_mean_delta"]
            delta = f"{values['net_return_percentage_points']:+.6f}" if values else "待齐全"
            lines.append(
                f"| {curve['target']} − {curve['reference']} | {point['policy_step']} "
                f"| {curve['sampling']} | {point['complete_pairs']}/4 | {delta} "
                f"| {point['wins_among_complete_windows']}/{point['complete_pairs']} |"
            )
    lines += [
        "",
        "比较单位是同一窗口的独立轨迹均值。SFT 与 RL 的 policy step/种子不同，"
        "不能声称逐请求完全配对。",
        "新增日期与学习率同时改变，本轮结果不单独归因于某一个因素。"
        "收益、成本、现金、非法率与逐窗细节见 readout.json。",
        "actor probe 的原摘要保留在 readout.json；其 Δlogp 是固定输入/已生成响应的更新前后漂移。",
        "",
        f"证据问题 {len(result['problems'])} 项；待齐全 {len(result['pending'])} 项。",
    ]
    if result["problems"]:
        lines += [""] + [f"- {p}" for p in result["problems"]]
    return "\n".join(lines) + "\n"


def summary_csv(result):
    stream = io.StringIO()
    fields = [
        "cell_id",
        "policy_step",
        "sampling",
        "planned",
        "terminal_complete",
        "complete_windows",
        *OUTCOMES,
    ]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for name, cell in result["cells"].items():
        for item in cell["evaluations"]:
            pop = item["population"]
            writer.writerow(
                {
                    "cell_id": name,
                    "policy_step": item["policy_step"],
                    "sampling": item["sampling"],
                    **{
                        key: pop[key]
                        for key in ("planned", "terminal_complete", "complete_windows")
                    },
                    **(pop["fixed_window_mean"] or {}),
                }
            )
    return stream.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    directory = args.directory.expanduser().resolve()
    result = build(directory)
    if args.write:
        shared._atomic_write(
            directory / "readout.json", json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        shared._atomic_write(directory / "LIVE_STATUS.md", markdown(result))
        shared._atomic_write(directory / "summary.csv", summary_csv(result))
        if result["execution_status"] in {"COMPLETE", "FAILED"}:
            shared._atomic_write(directory / "FINAL_RESULTS.md", markdown(result))
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "execution_status": result["execution_status"],
                    "readout": str(directory / "readout.json"),
                    "totals": result["totals"],
                    "problems": len(result["problems"]),
                    "pending": len(result["pending"]),
                }
            )
        )
    else:
        json.dump(result, sys.stdout, indent=2, allow_nan=False)
        print()
    return result


if __name__ == "__main__":
    main()

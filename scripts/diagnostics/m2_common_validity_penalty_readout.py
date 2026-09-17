#!/usr/bin/env python3
"""CPU readout for the 164-trajectory common environment-rejection penalty trial.

Keep original finance Y and net returns separate from penalized training rewards.
Evaluate fixed greedy/sample lanes; use only old step2/4 as historical RL references.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
COVERAGE_PATH = ROOT / "scripts/diagnostics/m2_train_coverage_readout.py"
COVERAGE_SHA256 = "f5b1b27346ca25fb9608464192b92a17d0ca5bc1dd18a6b95b6adb274b62f9a4"
if hashlib.sha256(COVERAGE_PATH.read_bytes()).hexdigest() != COVERAGE_SHA256:
    raise RuntimeError("pinned coverage helper SHA-256 differs")
_spec = importlib.util.spec_from_file_location("validity_pinned_coverage_helpers", COVERAGE_PATH)
assert _spec is not None and _spec.loader is not None
coverage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(coverage)
shared = coverage.shared
Evidence, sha, load_helpers = coverage.Evidence, coverage.sha, coverage.load_helpers
critic_fit_summary, finite_tree = coverage.critic_fit_summary, coverage.finite_tree
_indexed_records = coverage._indexed_records
fixed_population, count_population = coverage.fixed_population, coverage.count_population
lane_of, outcome = coverage.lane_of, coverage.outcome
actor_probe_readout, paired_validation_readout = (
    coverage.actor_probe_readout,
    coverage.paired_validation_readout,
)
PROFILE = "common_sft_validity_penalty_v1"
PENALTY_PROFILE = "environment_rejection_v1"
COEFFICIENT = 0.05
TRAIN_IDS = coverage.TRAIN_IDS[:8]
EVAL_IDS = coverage.EVAL_IDS
EVAL_STEPS = [2, 4]
ACTOR_LR, CRITIC_LR = coverage.ACTOR_LR, coverage.CRITIC_LR
HISTORICAL_RUN = ROOT / "runs/m2-train-coverage-20260916-v2"


def cell_specs(settings):
    contract = settings.get("training_coverage", {})
    if contract.get("profile") != PROFILE or contract.get("evaluation_steps") != EVAL_STEPS:
        raise ValueError("directory does not contain the fixed validity-penalty profile/schedule")
    training = settings.get("training", {})
    if (
        training.get("learning_rate") != ACTOR_LR
        or training.get("critic_learning_rate") != CRITIC_LR
    ):
        raise ValueError("common actor/critic learning rates differ")
    penalty = settings.get("common_validity_penalty", {})
    if penalty != {
        "profile": PENALTY_PROFILE,
        "coefficient": COEFFICIENT,
        "applies_to": "training_only",
        "financial_terminal_score_unchanged": True,
    }:
        raise ValueError("common rejection penalty configuration differs")
    return [
        dict(zip(("cell_id", "arm", "eta", "critic_lr", "rounds"), row, strict=True))
        for row in (
            ("sft0", "M1", 0.0, CRITIC_LR, 0),
            ("m1", "M1", 0.0, CRITIC_LR, 4),
            ("m2_eta025", "M2", 0.25, CRITIC_LR, 4),
        )
    ]


def expected_grid(spec, train_ids=TRAIN_IDS, eval_ids=EVAL_IDS, evaluation_steps=EVAL_STEPS):
    if (
        train_ids != TRAIN_IDS
        or eval_ids != EVAL_IDS
        or (
            evaluation_steps != ([0] if not spec["rounds"] else EVAL_STEPS)
            and evaluation_steps != EVAL_STEPS
        )
    ):
        raise ValueError("fixed eight Train/four evaluation windows or schedule differ")
    expected = {}
    for step in range(1, spec["rounds"] + 1):
        for index in (2 * (step - 1), 2 * (step - 1) + 1):
            for sibling in range(4):
                expected[(step, index, sibling, False)] = train_ids[index]
    for step in EVAL_STEPS if spec["rounds"] else [0]:
        for index, episode in enumerate(eval_ids):
            for sibling in range(5):
                expected[(step, index, sibling, True)] = episode
    return expected


def read_trajectories(evidence, helpers, cell, spec, grid, calibration_sha):
    seen, excluded, raw_count = {}, [], 0
    for reference, raw in evidence.jsonl(cell / "artifacts/screening_evidence"):
        raw_count += 1
        key = helpers.grid_key(raw)
        if key not in grid or raw.get("episode_id") != grid.get(key):
            excluded.append(
                {
                    "source": reference,
                    "identity": key,
                    "reason": "outside frozen grid or wrong episode",
                }
            )
            continue
        if key in seen:
            seen[key]["sources"].append(reference)
            seen[key]["row"] = None
            continue
        slot = seen[key] = {"sources": [reference], "row": None}
        try:
            row = helpers.trajectory(reference, raw)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            excluded.append(
                {
                    "source": reference,
                    "identity": key,
                    "reason": f"uninterpretable trajectory: {error}",
                }
            )
            continue
        if (
            raw.get("arm") != spec["arm"]
            or raw.get("base_seed") != 42
            or (
                calibration_sha is not None
                and raw.get("metrics", {}).get("calibration_sha256") != calibration_sha
            )
        ):
            row["terminal_eligible"] = False
            row["Y"] = row["net_return"] = None
            row["problems"].append("arm/seed/calibration identity differs")
        if not key[3]:
            credits, steps = raw.get("credits", []), raw.get("steps", [])
            if len(credits) != len(steps) or not all(coverage.finite(v) for v in credits):
                row["problems"].append("missing/nonfinite original financial step credits")
                row["financial_step_evidence"] = []
            else:
                row["financial_step_evidence"] = [
                    {
                        "step_index": step.get("step_index"),
                        "action": step.get("action"),
                        "invalid_action": step.get("invalid_action"),
                        "financial_step_reward": credit,
                    }
                    for step, credit in zip(steps, credits, strict=True)
                ]
        # All aggregates use scalar/count fields. Large interactions are discarded
        # as each line is processed; detailed fill records remain at source.
        for field in ("actual_fills", "skipped_orders", "final_positions"):
            row.pop(field, None)
        slot["row"] = row
    rows = []
    for key, slot in seen.items():
        if len(slot["sources"]) != 1:
            excluded.append(
                {
                    "identity": key,
                    "sources": slot["sources"],
                    "reason": "duplicate identity: all copies excluded",
                }
            )
        elif slot["row"] is not None:
            rows.append(slot["row"])
    rows.sort(key=lambda row: helpers.grid_key(row))
    return rows, excluded, raw_count


def close(left, right):
    return (
        coverage.finite(left)
        and coverage.finite(right)
        and math.isclose(left, right, rel_tol=1e-5, abs_tol=5e-5)
    )


def penalty_readout(spec, raw_rows, diagnostics):
    """Join driver rewards to native, unpenalized financial trajectory credits."""
    problems, pending, trajectories, rounds = [], [], [], []
    train = {
        (r["identity"]["step"], r["episode_id"], r["identity"]["rollout_n"]): r
        for r in raw_rows
        if not r["identity"]["validate"]
    }
    for policy_step in range(1, spec["rounds"] + 1):
        records = diagnostics.get(policy_step, [])
        if len(records) != 1:
            pending.append(f"step {policy_step} common reward evidence not uniquely present")
            continue
        record = records[0]
        summary, steps = record.get("common_validity_penalty"), record.get("reward_evidence")
        if not isinstance(summary, dict) or not isinstance(steps, list):
            problems.append(f"step {policy_step} common reward evidence is missing")
            continue
        if any(
            summary.get(k) != v
            for k, v in {
                "profile": PENALTY_PROFILE,
                "coefficient": COEFFICIENT,
                "invalid_source": "trading_invalid_action",
                "sibling_residual_outcome": "original_financial_terminal_Y",
            }.items()
        ):
            problems.append(
                f"step {policy_step} penalty or residual financial outcome contract differs"
            )
        grouped = defaultdict(list)
        for item in steps:
            if not isinstance(item, dict) or not isinstance(item.get("trajectory_uid"), str):
                problems.append(f"step {policy_step} malformed reward row")
                continue
            grouped[item["trajectory_uid"]].append(item)
        if len(steps) != record.get("real_step_count") or len(grouped) != 8:
            problems.append(f"step {policy_step} reward grid is not eight complete trajectories")
        declared = summary.get("trajectories", [])
        declared_by_uid = {t.get("trajectory_uid"): t for t in declared if isinstance(t, dict)}
        if len(declared_by_uid) != len(declared) or set(declared_by_uid) != set(grouped):
            problems.append(f"step {policy_step} reward trajectory declarations differ")
        completed, observed_keys = [], []
        for uid, items in grouped.items():
            items = sorted(items, key=lambda x: x.get("step_index", -1))
            first = items[0]
            key = (policy_step, first.get("episode_id"), first.get("rollout_id"))
            observed_keys.append(key)
            raw = train.get(key)
            prefix = f"step {policy_step} trajectory {uid}"
            if raw is None or not raw["terminal_eligible"]:
                pending.append(f"{prefix} original complete financial trajectory missing")
                continue
            original = raw.get("financial_step_evidence", [])
            if len(items) != len(original) or [x.get("step_index") for x in items] != list(
                range(len(original))
            ):
                problems.append(f"{prefix} reward step coverage differs")
                continue
            numeric = (
                "terminal_Y",
                "financial_step_reward",
                "invalid_penalty",
                "training_step_reward",
                "unwhitened_U",
                "sibling_residual",
                "final_actor_advantage",
                "critic_return",
            )
            if any(not all(coverage.finite(item.get(k)) for k in numeric) for item in items):
                problems.append(f"{prefix} reward evidence contains nonfinite or missing values")
                continue
            invalid = 0
            for item, source in zip(items, original, strict=True):
                bad = (
                    type(item.get("invalid_action")) is not bool
                    or item.get("invalid_action") != source.get("invalid_action")
                    or item.get("action") != source.get("action")
                    or item.get("episode_id") != key[1]
                    or item.get("rollout_id") != key[2]
                    or item.get("policy_step") != policy_step
                    or not close(item["terminal_Y"], raw["Y"])
                    or not close(item["financial_step_reward"], source["financial_step_reward"])
                    or not close(item["invalid_penalty"], COEFFICIENT * source["invalid_action"])
                    or not close(
                        item["training_step_reward"],
                        item["financial_step_reward"] - item["invalid_penalty"],
                    )
                )
                if (spec["arm"] == "M1" or item["invalid_action"]) and not close(
                    item["sibling_residual"], 0
                ):
                    bad = True
                if bad:
                    problems.append(
                        f"{prefix} step {item['step_index']} financial/invalid/reward join differs"
                    )
                invalid += item["invalid_action"]
            totals = {
                "financial_reward_sum": math.fsum(x["financial_step_reward"] for x in items),
                "penalty_sum": math.fsum(x["invalid_penalty"] for x in items),
                "training_reward_sum": math.fsum(x["training_step_reward"] for x in items),
            }
            target = raw["Y"] - COEFFICIENT * invalid
            if not close(totals["financial_reward_sum"], raw["Y"]) or not close(
                totals["training_reward_sum"], target
            ):
                problems.append(f"{prefix} trajectory does not conserve original Y minus penalty")
            # gamma=lambda=1 and terminal completion imply the full suffix return.
            suffix = 0.0
            for item in reversed(items):
                suffix += item["training_step_reward"]
                if not close(item["critic_return"], suffix):
                    problems.append(f"{prefix} critic return differs from training reward suffix")
                    break
            expected = {
                "step_count": len(items),
                "terminal_Y": raw["Y"],
                "invalid_count": invalid,
                "expected_training_reward_sum": target,
                **totals,
            }
            declared_row = declared_by_uid.get(uid, {})
            if any(not close(declared_row.get(k), value) for k, value in expected.items()):
                problems.append(f"{prefix} declared trajectory sums differ")
            completed.append(
                {
                    "trajectory_uid": uid,
                    "policy_step": policy_step,
                    "episode_id": key[1],
                    "rollout_n": key[2],
                    **expected,
                }
            )
        expected_keys = {key for key in train if key[0] == policy_step}
        if len(set(observed_keys)) != len(observed_keys) or set(observed_keys) != expected_keys:
            problems.append(f"step {policy_step} reward trajectory join is duplicated or differs")
        aggregate = {
            k: math.fsum(row[k] for row in completed)
            for k in ("financial_reward_sum", "penalty_sum", "training_reward_sum", "invalid_count")
        }
        if len(completed) == 8 and any(not close(summary.get(k), v) for k, v in aggregate.items()):
            problems.append(f"step {policy_step} declared batch reward sums differ")
        rounds.append(
            {
                "policy_step": policy_step,
                "planned_trajectories": 8,
                "joined_trajectories": len(completed),
                "observed_sums": aggregate,
                "fixed_eight_mean": {k: v / 8 for k, v in aggregate.items()}
                if len(completed) == 8
                else None,
            }
        )
        trajectories.extend(completed)
    planned = spec["rounds"] * 8
    if len(trajectories) != planned:
        pending.append("training reward trajectory joins not yet complete")
    totals = {
        k: math.fsum(t[k] for t in trajectories)
        for k in ("financial_reward_sum", "penalty_sum", "training_reward_sum", "invalid_count")
    }
    return {
        "profile": PENALTY_PROFILE,
        "coefficient": COEFFICIENT,
        "status": "COMPLETE" if not problems and not pending else "INCOMPLETE",
        "planned_trajectories": planned,
        "joined_trajectories": len(trajectories),
        "observed_sums": totals,
        "fixed_training_mean": {k: v / planned for k, v in totals.items()}
        if planned and len(trajectories) == planned
        else None,
        "rounds": rounds,
        "trajectories": trajectories,
        "problems": problems,
        "pending": pending,
        "checks": "Original financial credits/invalid flags joined to driver reward; "
        "r'=r-.05*invalid; complete gamma=lambda=1 suffix returns; original-Y residual marker; "
        "M1 and invalid-row S zero. Actor loss-token component validation is runtime-owned.",
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
    reward_evidence = penalty_readout(spec, rows, diagnostics)
    problems.extend(reward_evidence["problems"])
    pending.extend(reward_evidence["pending"])
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
        "common_validity_penalty": reward_evidence,
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


def historical_comparison(cells, model, directory=None):
    """Read the sealed old scalar snapshot; never re-run old source verification."""
    path = Path(directory or HISTORICAL_RUN) / "readout.json"
    if not path.is_file():
        return {"status": "UNAVAILABLE", "path": str(path), "comparisons": []}
    try:
        old = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return {"status": "UNAVAILABLE", "path": str(path), "error": str(error), "comparisons": []}
    result = {
        "path": str(path),
        "sha256": sha(path),
        "status": "AVAILABLE",
        "reference_profile": old.get("profile"),
        "reference_source_sha256": old.get("input_sha256", {}),
        "scope": "historical no-penalty coverage SFT0 and matching RL step2/4; never step8",
        "request_pairing_claimed": False,
        "comparisons": [],
    }
    if old.get("status") != "COMPLETE" or old.get("shared_model_identity", {}).get(
        "model_revision"
    ) != model.get("model_revision"):
        result.update(
            status="INCOMPATIBLE", error="historical completeness/common SFT identity differs"
        )
        return result
    for name, cell in cells.items():
        old_items = old.get("cells", {}).get(name, {}).get("evaluations", [])
        for current in cell["evaluations"]:
            step, lane = current["policy_step"], current["sampling"]
            matches = [x for x in old_items if x["policy_step"] == step and x["sampling"] == lane]
            previous = matches[0]["population"] if len(matches) == 1 else None
            before = {w["episode_id"]: w for w in previous["windows"]} if previous else {}
            after = {w["episode_id"]: w for w in current["population"]["windows"]}
            pairs = []
            for episode in EVAL_IDS:
                a = after.get(episode, {}).get("fixed_sibling_mean")
                b = before.get(episode, {}).get("fixed_sibling_mean")
                pairs.append(
                    {
                        "episode_id": episode,
                        "current": a,
                        "historical": b,
                        "delta": {k: a[k] - b[k] for k in coverage.OUTCOMES}
                        if a is not None and b is not None
                        else None,
                    }
                )
            complete = [x["delta"] for x in pairs if x["delta"] is not None]
            result["comparisons"].append(
                {
                    "cell_id": name,
                    "policy_step": step,
                    "reference_policy_step": step,
                    "sampling": lane,
                    "planned_windows": 4,
                    "complete_windows": len(complete),
                    "historical_fixed_window_mean": previous["fixed_window_mean"]
                    if previous
                    else None,
                    "current_fixed_window_mean": current["population"]["fixed_window_mean"],
                    "fixed_four_window_mean_delta": {
                        k: math.fsum(x[k] for x in complete) / 4 for k in coverage.OUTCOMES
                    }
                    if len(complete) == 4
                    else None,
                    "wins_among_complete_windows": sum(x["net_return"] > 0 for x in complete),
                    "pairs": pairs,
                }
            )
    return result


def build(directory):
    helpers = load_helpers()
    evidence = Evidence(Path(directory).resolve())
    settings = evidence.json(evidence.run / "settings.json", required=True) or {}
    specs = cell_specs(settings)
    state = evidence.json(evidence.run / "state.json") or {}
    queue = evidence.json(evidence.run / "gpu_queue_state.json") or {}
    execution_status = state.get("status", queue.get("status", "NOT_STARTED"))
    model = evidence.json(evidence.run / "model_identity.json", required=True) or {}
    manifest = evidence.json(evidence.run / "prepared/manifest.json", required=True) or {}
    contract = evidence.json(evidence.run / "run_contract.json", required=True) or {}
    sources = evidence.json(evidence.run / "source_identity.json", required=True) or {}
    selection = evidence.json(evidence.run / "selection.json")
    if manifest.get("datasets", {}).get("train", {}).get("episode_ids") != TRAIN_IDS or (
        manifest.get("datasets", {}).get("dev", {}).get("episode_ids") != EVAL_IDS
    ):
        raise ValueError(
            "prepared manifest differs from frozen eight Train/four Train-evaluation windows"
        )
    if state.get("schedule") is not None and state["schedule"] != specs:
        evidence.errors.append("controller schedule differs from validity-penalty cells")
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
        pending.append(f"controller is {execution_status}")
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
        if execution_status in {"FAILED", "WAIT_TIMEOUT"}
        else "COMPLETE"
        if not problems and not pending
        else "INCOMPLETE"
    )
    now = datetime.now(timezone.utc)
    result = {
        "schema_version": 1,
        "profile": PROFILE,
        "status": status,
        "execution_status": execution_status,
        "gpu_queue": queue,
        "training_clock_note": "GPU waiting is not training runtime; "
        "controller timestamps begin training execution.",
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
        "evaluation_curves": coverage.paired_curves(cells, EVAL_IDS, EVAL_STEPS),
        "historical_comparison": historical_comparison(cells, model),
        "initial_critic_heads": {"hashes": head_hashes, "all_equal": all_heads_equal},
        "problems": problems,
        "pending": pending,
        "input_sha256": evidence.hashes,
        "stream_snapshots": evidence.streams,
        "dependencies": {
            "coverage_helper": {"path": str(COVERAGE_PATH), "sha256": COVERAGE_SHA256},
            "tuning_evidence_helper": {
                "path": str(coverage.DEPENDENCY),
                "sha256": coverage.DEPENDENCY_SHA256,
            },
            "scalar_helper": {"path": str(shared.HELPER_PATH), "sha256": shared.HELPER_SHA256},
        },
        "readout_script_sha256": sha(Path(__file__)),
        "limits": [
            "All dates belong to Train; no formal Valid/Test or forward-time OOS evidence.",
            "The common training reward deducts .05 per environment rejection; "
            "original financial Y and evaluation are unchanged.",
            "Train episodes change each round; Train returns are not a fixed-window curve.",
            "Historical references use successful no-penalty coverage steps 0/2/4 only; "
            "these are exploratory historical comparisons.",
            "Greedy lane 0 and sampled lanes 1..4 stay separate; all four siblings are required.",
            "Every mean needs all four windows; missing/incomplete/nonfinite outcomes stay null.",
            "SFT step 0 and RL steps 2/4 have different seed identities; "
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
    def number(value):
        return "待齐全" if value is None else f"{value:.6f}"

    updated = datetime.fromisoformat(result["generated_at_SGT"]).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 共同非法动作训练罚项：当前进展",
        "",
        f"更新时间：{updated} SGT（UTC+8）",
        "",
        f"执行状态：**{result['execution_status']}**；证据状态：**{result['status']}**。",
        f"完整轨迹 {result['totals']['terminal_complete']}/164；"
        f"已观察 {result['totals']['observed_unique_expected']}/164。",
        "",
        "全部日期属于 Train。greedy 与 sampled 分开；sampled 每窗四条等权，再固定四窗等权。"
        "缺失、不完整或非有限结果使该窗及全组均值为空，不能缩小分母。",
    ]
    if result["execution_status"] in {"WAITING_FOR_GPU", "STARTING", "WAIT_TIMEOUT"}:
        lines += [
            "",
            "GPU 排队与训练计时分开；等待其他任务释放 GPU 时尚无 controller state 属正常情况。",
            f"GPU 队列状态：{result['gpu_queue'].get('status')}。",
        ]
    lines += [
        "",
        "| 实验组 | 训练完整/计划 | greedy 完整/计划 | sampled 完整/计划 | 奖励证据轨迹/计划 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, cell in result["cells"].items():
        groups, reward = cell["groups"], cell["common_validity_penalty"]
        counts = [
            f"{groups[k]['terminal_complete']}/{groups[k]['planned']}"
            for k in ("train", "eval_greedy", "eval_sampled")
        ]
        lines.append(
            f"| {name} | {' | '.join(counts)} "
            f"| {reward['joined_trajectories']}/{reward['planned_trajectories']} |"
        )
    lines += [
        "",
        "验收财务指标（Y 未扣除新训练罚项）：",
        "",
        "| 实验组 | 更新轮数 | 解码 | 完整窗 | 净收益 % | 原 Y "
        "| 成本 USD | 现金 % | 非法请求 %¹ | 合法成交数 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, cell in result["cells"].items():
        for item in cell["evaluations"]:
            pop, means = item["population"], item["population"]["fixed_window_mean"] or {}
            metrics = [
                number(means.get(k))
                for k in (
                    "net_return_pct",
                    "Y",
                    "terminal_cost",
                    "final_cash_pct",
                    "invalid_request_rate_pct",
                    "fill_count",
                )
            ]
            lines.append(
                f"| {name} | {item['policy_step']} | {item['sampling']} "
                f"| {pop['complete_windows']}/4 | {' | '.join(metrics)} |"
            )
    lines += [
        "",
        "¹ 此列先计算每条轨迹的非法请求比例，再四 siblings/四窗等权；"
        "原始非法请求数/请求总数在 readout.json 另列，二者不混用。",
        "",
        "| 同批目标 − 参照 | 更新轮数 | 解码 | 完整窗 | 净收益 Δpp | 赢窗（完整窗内） |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for curve in result["evaluation_curves"]:
        for point in curve["points"]:
            delta = point["fixed_four_window_mean_delta"] or {}
            lines.append(
                f"| {curve['target']} − {curve['reference']} | {point['policy_step']} "
                f"| {curve['sampling']} | {point['complete_pairs']}/4 "
                f"| {number(delta.get('net_return_percentage_points'))} "
                f"| {point['wins_among_complete_windows']}/{point['complete_pairs']} |"
            )
    lines += [
        "",
        "训练奖励单列（32 条固定轨迹均值，非验收收益）：",
        "",
        "| 实验组 | 原财务奖励和 | 非法扣分和 | 罚后训练奖励和 | 每轨迹非法次数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("m1", "m2_eta025"):
        means = result["cells"][name]["common_validity_penalty"]["fixed_training_mean"] or {}
        lines.append(
            f"| {name} | "
            + " | ".join(
                number(means.get(k))
                for k in (
                    "financial_reward_sum",
                    "penalty_sum",
                    "training_reward_sum",
                    "invalid_count",
                )
            )
            + " |"
        )
    history = result["historical_comparison"]
    lines += [
        "",
        f"无罚项历史对照状态：{history['status']}。"
        "只对照旧成功运行的 SFT0 与同臂 step2/4，未使用 step8。",
        "",
        "| 新 − 旧同臂 | 更新轮数 | 解码 | 完整窗 | 净收益 Δpp | 非法率 Δpp¹ | 成本 ΔUSD |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for item in history["comparisons"]:
        delta = item["fixed_four_window_mean_delta"] or {}
        metrics = [
            number(delta.get(k))
            for k in ("net_return_pct", "invalid_request_rate_pct", "terminal_cost")
        ]
        lines.append(
            f"| {item['cell_id']} | {item['policy_step']} | {item['sampling']} "
            f"| {item['complete_windows']}/4 | {' | '.join(metrics)} |"
        )
    lines += [
        "",
        "同窗比较的是轨迹分布均值，SFT 与 RL 的 policy step/种子不同；"
        "历史运行也不是逐请求完全相同的反事实。M2 的 S 仍使用原财务 Y，"
        "新增罚项只作用于两臂共同训练奖励；actor Δlogp 是已生成响应的 teacher-forced 漂移。",
        f"证据问题 {len(result['problems'])} 项；待齐全 {len(result['pending'])} 项。",
    ]
    if result["problems"]:
        lines += [""] + [f"- {p}" for p in result["problems"]]
    return "\n".join(lines) + "\n"


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
        shared._atomic_write(directory / "summary.csv", coverage.summary_csv(result))
        if result["execution_status"] in {"COMPLETE", "FAILED", "WAIT_TIMEOUT"}:
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

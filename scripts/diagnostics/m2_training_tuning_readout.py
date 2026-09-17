#!/usr/bin/env python3
"""Stream existing Train-only tuning evidence into a repeatable scalar readout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "runs/m2-common-sft-short-rl-20260914/summarize.py"
HELPER_SHA256 = "0881a253795be9da4e25f3201ccabf9c21ba5387c465f6b703ae77494b2a3e54"
TRAIN_IDS = [f"formal-train-{index:04d}" for index in range(1, 30, 4)]
EVAL_IDS = [f"formal-train-{index:04d}" for index in range(113, 126, 4)]
PROFILE = "common_sft_training_tuning_v1"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_helpers(path=HELPER_PATH):
    if sha(path) != HELPER_SHA256:
        raise RuntimeError("pinned scalar summarizer SHA-256 differs")
    spec = importlib.util.spec_from_file_location("tuning_pinned_scalar_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned scalar summarizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Evidence:
    """Retain compact rows and exact observed byte hashes, never whole JSONL files."""

    def __init__(self, run):
        self.run = run
        self.hashes, self.streams = {}, {}
        self.errors, self.pending = [], []

    def json(self, path, required=False):
        relative = str(path.relative_to(self.run))
        if not path.is_file():
            if required:
                self.pending.append(f"missing {relative}")
            return None
        try:
            payload = path.read_bytes()
            self.hashes[relative] = hashlib.sha256(payload).hexdigest()
            return json.loads(payload)
        except (OSError, ValueError) as error:
            self.errors.append(f"unreadable {relative}: {error}")
            return None

    def jsonl(self, directory):
        for path in sorted(directory.glob("*.jsonl")):
            relative = str(path.relative_to(self.run))
            before = path.stat()
            digest, count, size, partial = hashlib.sha256(), 0, 0, False
            with path.open("rb") as stream:
                for number, raw in enumerate(stream, 1):
                    digest.update(raw)
                    size += len(raw)
                    reference = f"{relative}:{number}"
                    if not raw.strip():
                        continue
                    if not raw.endswith(b"\n"):
                        partial = True
                        self.pending.append(f"uncommitted JSONL tail {reference}")
                        continue
                    try:
                        record = json.loads(raw)
                        if not isinstance(record, dict):
                            raise ValueError("record is not an object")
                        count += 1
                        yield reference, record
                    except ValueError as error:
                        self.errors.append(f"unreadable {reference}: {error}")
            after = path.stat()
            self.hashes[relative] = digest.hexdigest()
            changed = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            self.streams[relative] = {
                "bytes_read": size,
                "complete_object_records": count,
                "partial_final_line": partial,
                "changed_during_read": changed,
                "sha256_scope": "exact bytes observed in this streaming pass",
            }
            if changed:
                self.pending.append(f"appended during read {relative}; refresh readout")


def cell_specs(settings):
    contract = settings.get("training_tuning", {})
    if contract.get("profile") != PROFILE:
        raise ValueError("directory does not contain the frozen training-tuning profile")
    phase = contract.get("phase")
    lr = contract.get("selected_critic_learning_rate")
    if phase == "diagnostic" and lr is None:
        rows = [
            ("sft0", "M1", 0.0, 1e-5, 0),
            ("critic_lr_1e5", "M1", 0.0, 1e-5, 2),
            ("critic_lr_1e6", "M1", 0.0, 1e-6, 2),
        ]
        evaluation_steps = [2]
    elif phase == "comparison" and lr in (1e-5, 1e-6):
        rows = [
            ("sft0", "M1", 0.0, lr, 0),
            ("m1", "M1", 0.0, lr, 8),
            ("m2_eta1", "M2", 1.0, lr, 8),
            ("m2_eta025", "M2", 0.25, lr, 8),
        ]
        evaluation_steps = [2, 4, 6, 8]
    else:
        raise ValueError("unknown phase or unfrozen critic learning rate")
    if contract.get("evaluation_steps") != evaluation_steps:
        raise ValueError("evaluation checkpoint schedule differs from frozen phase")
    return [
        dict(zip(("cell_id", "arm", "eta", "critic_lr", "rounds"), row, strict=True))
        for row in rows
    ]


def expected_grid(spec, train_ids, eval_ids, evaluation_steps):
    expected = {}
    for step in range(1, spec["rounds"] + 1):
        first_index = ((step - 1) * 2) % len(train_ids)
        for index in (first_index, first_index + 1):
            for sibling in range(4):
                expected[(step, index, sibling, False)] = train_ids[index]
    for step in evaluation_steps if spec["rounds"] else [0]:
        for index, episode in enumerate(eval_ids):
            expected[(step, index, 0, True)] = episode
    return expected


def population(helpers, rows, planned):
    result = helpers.aggregate(rows, planned)
    result.update(
        {
            "complete_fraction_of_planned": result["terminal_complete"] / planned
            if planned
            else None,
            "invalid_per_observed_request": (
                result["invalid_observed"] / result["requests_observed"]
                if result["requests_observed"]
                else None
            ),
            "legal_hold_observed": result["executed_step_actions_observed"].get("hold", 0),
            "max_drawdown_complete_only": max(
                (
                    row["drawdown"]
                    for row in rows
                    if row["terminal_eligible"] and helpers.finite(row["drawdown"])
                ),
                default=None,
            ),
        }
    )
    return result


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


def _indexed_records(evidence, directory, key):
    result = defaultdict(list)
    for reference, record in evidence.jsonl(directory):
        result[record.get(key)].append({"source": reference, **record})
    return result


def critic_fit_summary(record):
    if record is None:
        return None
    pre, post = record.get("pre", {}), record.get("post", {})
    before, after = pre.get("half_mse"), post.get("half_mse")
    return {
        "source": record.get("source"),
        "pre": pre,
        "post": post,
        "half_mse_change_post_minus_pre": (
            after - before if type(before) in (int, float) and type(after) in (int, float) else None
        ),
        "delta_values": record.get("delta_values"),
        "post_delta_exceeds_cliprange_fraction": record.get(
            "post_delta_exceeds_cliprange_fraction"
        ),
        "extra_forward_seconds_synchronized": record.get("extra_forward_seconds_synchronized"),
        "extra_forward_input_tokens": record.get("extra_forward_input_tokens"),
        "dropout": record.get("dropout"),
        "optimizer_grad_norms_before_clip": record.get("optimizer_grad_norms_before_clip"),
        "critic_learning_rates": record.get("critic_learning_rates"),
        "pre_source": record.get("pre_source"),
        "post_source": record.get("post_source"),
    }


def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


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
    eval_steps = settings["training_tuning"]["evaluation_steps"] if spec["rounds"] else [0]
    grid = expected_grid(spec, train_ids, eval_ids, eval_steps)
    rows, excluded, raw_count = read_trajectories(
        evidence, helpers, cell, spec, grid, manifest.get("calibration", {}).get("payload_sha256")
    )
    problems, pending = [], []
    if not command:
        pending.append("cell command is missing")
    if excluded:
        problems.append("trajectory identities are duplicate, outside the grid, or uninterpretable")
    if any(row["problems"] for row in rows):
        problems.append("observed trajectory failed eligibility checks")
    if len(rows) != len(grid) or not all(row["terminal_eligible"] for row in rows):
        pending.append("fixed trajectory grid is not completely observed")
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
                "population": population(helpers, train_rows, 8),
                "advantage_diagnostics_as_recorded": diagnostics[step],
                "critic_fit": critic_fit_summary(fit),
                "actor_critic_metrics_as_logged": logged_by_step[step],
            }
        )
    evaluations = [
        {
            "policy_step": step,
            "population": population(
                helpers,
                [
                    row
                    for row in rows
                    if row["identity"]["validate"] and row["identity"]["step"] == step
                ],
                4,
            ),
        }
        for step in eval_steps
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
            "train": population(
                helpers,
                [row for row in rows if not row["identity"]["validate"]],
                spec["rounds"] * 8,
            ),
            "eval": population(
                helpers, [row for row in rows if row["identity"]["validate"]], len(eval_steps) * 4
            ),
        },
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


def paired_curves(helpers, cells, eval_ids, eval_steps):
    comparisons = [(name, "sft0") for name in cells if name != "sft0"]
    comparisons += [(name, "m1") for name in cells if name.startswith("m2_")]
    curves = []
    for target, reference in comparisons:
        points = []
        for step in eval_steps:
            reference_step = 0 if reference == "sft0" else step
            indexed = {}
            for name, observed_step in ((target, step), (reference, reference_step)):
                indexed[name] = {
                    row["episode_id"]: row
                    for row in cells[name]["trajectory_rows"]
                    if row["identity"]["validate"] and row["identity"]["step"] == observed_step
                }
            pairs = []
            for episode in eval_ids:
                after, before = indexed[target].get(episode), indexed[reference].get(episode)
                delta = helpers.paired_delta(after, before)
                if delta is not None:
                    delta["net_return_percentage_points"] = delta["net_return"] * 100
                pairs.append(
                    {
                        "episode_id": episode,
                        "target": after,
                        "reference": before,
                        "delta_target_minus_reference": delta,
                    }
                )
            complete = [
                pair["delta_target_minus_reference"]
                for pair in pairs
                if pair["delta_target_minus_reference"] is not None
            ]
            metric_keys = (
                "Y",
                "net_return",
                "net_return_percentage_points",
                "terminal_cost",
                "fill_count",
                "max_weight",
                "final_cash",
                "final_position_count",
            )
            observed_means = {
                key: helpers.mean(row[key] for row in complete if key in row) for key in metric_keys
            }
            points.append(
                {
                    "policy_step": step,
                    "reference_policy_step": reference_step,
                    "planned_pairs": len(eval_ids),
                    "complete_pairs": len(complete),
                    "fixed_four_window_mean_delta": observed_means
                    if len(complete) == len(eval_ids)
                    else None,
                    "observed_complete_pairs_mean_delta": observed_means,
                    "pairs": pairs,
                }
            )
        curves.append(
            {
                "target": target,
                "reference": reference,
                "direction": "target minus reference",
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
    selection = evidence.json(evidence.run / "selection.json", required=True) or {}
    contract = evidence.json(evidence.run / "run_contract.json", required=True) or {}
    sources = evidence.json(evidence.run / "source_identity.json", required=True) or {}
    if manifest.get("datasets", {}).get("train", {}).get("episode_ids") != TRAIN_IDS or (
        manifest.get("datasets", {}).get("dev", {}).get("episode_ids") != EVAL_IDS
    ):
        raise ValueError("prepared manifest differs from eight Train/four Train-evaluation windows")
    if state.get("schedule") is not None and state["schedule"] != specs:
        evidence.errors.append("controller schedule differs from settings phase")
    if selection.get("phase") != settings["training_tuning"]["phase"] or (
        selection.get("selected_critic_learning_rate")
        != settings["training_tuning"]["selected_critic_learning_rate"]
    ):
        evidence.errors.append("selection phase/learning rate differs from settings")
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
    status = (
        "FAILED"
        if state.get("status") == "FAILED"
        else ("COMPLETE" if not problems and not pending else "INCOMPLETE")
    )
    generated_at = datetime.now(timezone.utc)
    result = {
        "schema_version": 1,
        "status": status,
        "execution_status": state.get("status", "NOT_STARTED"),
        "generated_at_UTC": generated_at.isoformat(),
        "generated_at_SGT": generated_at.astimezone(ZoneInfo("Asia/Singapore")).isoformat(),
        "phase": settings["training_tuning"]["phase"],
        "controller": state,
        "settings": settings,
        "selection": selection,
        "shared_model_identity": model,
        "source_identity": sources,
        "run_contract": contract,
        "cells": cells,
        "evaluation_curves": paired_curves(
            helpers, cells, EVAL_IDS, settings["training_tuning"]["evaluation_steps"]
        ),
        "initial_critic_heads": {"hashes": head_hashes, "all_equal": all_heads_equal},
        "problems": problems,
        "pending": pending,
        "input_sha256": evidence.hashes,
        "stream_snapshots": evidence.streams,
        "scalar_helper": {"path": str(HELPER_PATH), "sha256": HELPER_SHA256},
        "readout_script_sha256": sha(Path(__file__)),
        "limits": [
            "All windows belong to Train; no validation/Test/OOS superiority claim.",
            "All scheduled evaluations are retained; no best-return checkpoint selection.",
            "Training windows vary by round; training-return changes are not learning gains.",
            "Four-window deltas require four complete pairs; partial-pair means are separate.",
            "Missing, duplicate and incomplete trajectories are not filled with zero or stitched.",
            "Critic pre/post uses the same rows/returns; pre values reuse the driver forward.",
            "Critic worker timing/MFU includes the extra fit forward; its time is separate.",
            "Scalar evidence/metadata only; model tensor files are not loaded or hashed.",
            "Active file reads are not atomic; refresh before terminal acceptance.",
        ],
    }
    return helpers.json_safe(result)


def _execution_label(status):
    return {
        "RUNNING": "运行中",
        "COMPLETE": "已完成",
        "FAILED": "失败",
        "NOT_STARTED": "待启动",
        "INITIALIZING": "初始化中",
    }.get(status, "状态待确认")


def _cell_execution_label(name, cell, controller):
    reported = cell["summary"].get("status")
    if reported:
        label = _execution_label(reported)
        if reported == "COMPLETE" and cell["status"] != "COMPLETE":
            label += "（证据待核对）"
        return label
    if controller.get("current_cell") == name:
        return "失败" if controller.get("status") == "FAILED" else "初始化中"
    if not cell["raw_trajectory_records"] and not cell["optimizer_updates"]["records"]:
        return "待启动"
    return "已有记录（状态待确认）"


def markdown(result):
    controller = result.get("controller", {})
    execution = result.get("execution_status", controller.get("status", "NOT_STARTED"))
    phase = {"diagnostic": "共同 critic 学习率诊断", "comparison": "固定预算 M1/M2 对照"}
    updated = (
        datetime.fromisoformat(result["generated_at_UTC"])
        .astimezone(ZoneInfo("Asia/Singapore"))
        .strftime("%Y-%m-%d %H:%M:%S")
    )
    evidence_status = (
        "证据核对完成。"
        if result["status"] == "COMPLETE"
        else (
            "当前仍在按计划收集轨迹与更新证据。"
            if execution in {"RUNNING", "NOT_STARTED"}
            else "证据尚待核对，请查看下方问题及待核对项。"
        )
    )
    lines = [
        "# 训练调参实时进度",
        "",
        f"执行状态：**{_execution_label(execution)}** · 当前阶段：**{phase[result['phase']]}**",
        "",
        f"更新时间：{updated} SGT（UTC+8）",
        "",
        f"证据状态：{evidence_status}",
        "",
        "全部窗口均属于 Train；保留每个预定评估轮次的结果。",
        "",
        "| 实验组 | 执行状态 | 训练轨迹完整/计划 | 评估轨迹完整/计划 |",
        "|---|---|---:|---:|",
    ]
    for name, cell in result["cells"].items():
        train, evaluate = cell["groups"]["train"], cell["groups"]["eval"]
        lines.append(
            f"| {name} | {_cell_execution_label(name, cell, controller)} "
            f"| {train['terminal_complete']}/{train['planned']} "
            f"| {evaluate['terminal_complete']}/{evaluate['planned']} |"
        )
    lines += [
        "",
        "| 实验组 − 参照组 | 更新轮数 | 完整配对 | 净收益变化（百分点） | Y 变化 |",
        "|---|---:|---:|---:|---:|",
    ]
    for curve in result["evaluation_curves"]:
        for point in curve["points"]:
            mean = point["fixed_four_window_mean_delta"]
            change = (
                f"{mean['net_return_percentage_points']:+.6f}" if mean is not None else "待齐全"
            )
            y_change = f"{mean['Y']:+.6f}" if mean is not None else "待齐全"
            lines.append(
                f"| {curve['target']} − {curve['reference']} | {point['policy_step']} "
                f"| {point['complete_pairs']}/4 | {change} | {y_change} |"
            )
    if result["problems"]:
        lines += ["", "待处理的证据问题（保留原始诊断）：", ""] + [
            f"- {value}" for value in result["problems"]
        ]
    lines += [
        "",
        f"待核对项：{len(result['pending'])} 项。逐条轨迹、诊断及来源见 readout.json。",
        "",
    ]
    return "\n".join(lines)


def _atomic_write(path, text):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument(
        "--write", action="store_true", help="atomically refresh readout.json/LIVE_STATUS.md"
    )
    args = parser.parse_args(argv)
    directory = args.directory.expanduser().resolve()
    result = build(directory)
    if args.write:
        _atomic_write(
            directory / "readout.json", json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        _atomic_write(directory / "LIVE_STATUS.md", markdown(result))
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "phase": result["phase"],
                    "readout": str(directory / "readout.json"),
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

#!/usr/bin/env python3
"""Read existing common-SFT short-RL evidence; write final results only at terminal."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

RUN = Path(__file__).resolve().parent
PROTOCOL = "single_tool_portfolio_v1"
IDENTITY_OPTION = "ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_SCREENING_IDENTITY_SHA256"
MODEL_OPTIONS = (
    "actor_rollout_ref.model.path",
    "actor_rollout_ref.ref.model.path",
    "critic.model.path",
    "critic.model.tokenizer_path",
)


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def json_safe(value):
    """Preserve observed nonfinite diagnostics explicitly instead of serializing NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"observed_nonfinite": str(value)}
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


class Evidence:
    def __init__(self, run):
        self.run, self.hashes, self.errors = run, {}, []

    def json(self, path, *, required=False):
        if not path.exists():
            if required:
                self.errors.append(f"missing: {path.relative_to(self.run)}")
            return None
        self.hashes[str(path.relative_to(self.run))] = sha(path)
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as error:
            self.errors.append(f"unreadable: {path.relative_to(self.run)}: {error}")
            return None

    def jsonl(self, folder):
        result = []
        for path in sorted(folder.glob("*.jsonl")):
            self.hashes[str(path.relative_to(self.run))] = sha(path)
            with path.open("rb") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    reference = f"{path.relative_to(self.run)}:{line_number}"
                    try:
                        if not line.endswith(b"\n"):
                            raise ValueError("unterminated final record")
                        value = json.loads(line)
                        if not isinstance(value, dict):
                            raise ValueError("record is not an object")
                        result.append((reference, value))
                    except ValueError as error:
                        self.errors.append(f"{reference}: {error}")
        return result


def grid_key(record):
    identity = record.get("rollout_identity", record.get("identity"))
    if not isinstance(identity, dict) or type(identity.get("validate")) is not bool:
        return None
    if any(type(identity.get(k)) is not int for k in ("step", "sample_index", "rollout_n")):
        return None
    return tuple(identity[k] for k in ("step", "sample_index", "rollout_n", "validate"))


def expected_grid(name, train_ids, eval_ids):
    result = {}
    if name != "base":
        for step in (1, 2):
            for index in ((step - 1) * 2, (step - 1) * 2 + 1):
                for sibling in range(4):
                    result[(step, index, sibling, False)] = train_ids[index]
    result.update(
        {
            (0 if name == "base" else 2, index, 0, True): episode
            for index, episode in enumerate(eval_ids)
        }
    )
    return result


def visible_state(interaction):
    capture = interaction.get("observation", {})
    if capture.get("truncated") is not False or not isinstance(capture.get("text"), str):
        return None
    try:
        messages = json.loads(capture["text"])
        content = messages[-1]["content"].strip()
        content = content.removeprefix("<tool_response>").removesuffix("</tool_response>").strip()
        state = json.loads(content)
        return state if isinstance(state, dict) else None
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def trajectory(reference, record):
    problems = []
    steps, interactions, requests = (
        record.get(k, []) for k in ("steps", "interactions", "requests")
    )
    complete_flag = record.get("complete") is True and record.get("status") == "COMPLETE"
    if not complete_flag:
        problems.append(f"trajectory status {record.get('status', 'MISSING')}")
    if not len(steps) == len(interactions) == len(requests):
        problems.append("request/step/interaction counts differ")
    if (
        record.get("schema_version") != 1
        or record.get("observation_profile") != "history_features_v1"
    ):
        problems.append("unexpected evidence schema or observation profile")
    if record.get("action_protocol") != PROTOCOL or record.get("chat_template_kwargs") != {
        "enable_thinking": False,
    }:
        problems.append("shared protocol/non-thinking contract differs")
    if any(
        r.get("status") != "COMPLETE" or not 0 < r.get("response_token_count", 0) < 4096
        for r in requests
    ):
        problems.append("uncompleted or capped generation request")
    advancing = [
        s for s in steps if s.get("result_session_index", -1) > s.get("decision_session_index", -1)
    ]
    if [s.get("decision_session_index") for s in advancing] != list(range(20)):
        problems.append("not exactly twenty ordered session advances")
    if (
        any(
            s.get("result_session_index") != s.get("decision_session_index", -2) + 1
            or s.get("invalid_action")
            for s in advancing
        )
        or not steps
        or steps[-1].get("done") is not True
    ):
        problems.append("invalid advance or terminal flag")
    fills, skips, errors, submitted = [], [], [], Counter()
    for interaction in interactions:
        info = interaction.get("info", {})
        fills.extend(info.get("fills", []))
        skips.extend(info.get("skipped_orders", []))
        if info.get("invalid_action"):
            errors.append(info.get("error", "unspecified invalid"))
        response = interaction.get("response", {})
        if (
            response.get("truncated") is not False
            or interaction.get("observation", {}).get("truncated") is not False
        ):
            problems.append("truncated interaction capture")
        try:
            text = response["text"].strip()
            call = json.loads(text.removeprefix("<tool_call>").removesuffix("</tool_call>"))
            submitted[call["name"]] += 1
        except (KeyError, ValueError, TypeError):
            submitted["unparsed"] += 1
    observed_cost = sum(f["commission"] + f["slippage_cost"] for f in fills)
    metrics = record.get("metrics", {})
    score, net = record.get("terminal_score"), metrics.get("raw_net_return")
    if not finite(score) or not finite(net) or metrics.get("terminal_score") != score:
        problems.append("missing/inconsistent finite terminal Y or net return")
    if complete_flag and any(
        not finite(metrics.get(key))
        for key in (
            "transaction_cost",
            "max_drawdown",
            "turnover",
            "maximum_position_weight",
        )
    ):
        problems.append("missing finite terminal cost/risk/position metrics")
    if finite(metrics.get("transaction_cost")) and not math.isclose(
        observed_cost,
        metrics["transaction_cost"],
        rel_tol=1e-8,
        abs_tol=1e-6,
    ):
        problems.append("actual fill costs differ from terminal metric")
    if complete_flag and metrics.get("invalid_count") != len(errors):
        problems.append("invalid count differs from actual receipts")
    final = visible_state(interactions[-1]) if interactions else None
    if complete_flag and (final is None or final.get("session_index") != 20):
        problems.append("missing exact terminal portfolio observation")
    terminal = complete_flag and not problems
    positions = (final or {}).get("positions")
    return {
        "source": reference,
        "identity": record.get("identity", record.get("rollout_identity")),
        "episode_id": record.get("episode_id"),
        "reported_status": record.get("status"),
        "terminal_eligible": terminal,
        "problems": sorted(set(problems)),
        "Y": score if terminal else None,
        "net_return": net if terminal else None,
        "terminal_cost": metrics.get("transaction_cost") if terminal else None,
        "drawdown": metrics.get("max_drawdown") if terminal else None,
        "turnover": metrics.get("turnover") if terminal else None,
        "max_weight": metrics.get("maximum_position_weight") if terminal else None,
        "final_cash": (final or {}).get("cash") if terminal else None,
        "final_equity": (final or {}).get("portfolio_equity") if terminal else None,
        "final_positions": positions if terminal else None,
        "final_position_count": len(positions)
        if terminal and isinstance(positions, dict)
        else None,
        "requests": len(requests),
        "executed_steps": len(steps),
        "executed_step_actions": dict(Counter(s.get("action", "unknown") for s in steps)),
        "submitted_tool_calls": dict(submitted),
        "fill_count": len(fills),
        "fill_sides": dict(Counter(f["side"] for f in fills)),
        "fill_symbols": dict(Counter(f["symbol"] for f in fills)),
        "actual_fills": fills,
        "observed_fill_cost": observed_cost,
        "skipped_orders": skips,
        "invalid_errors": errors,
        "captured_last_state_is_terminal": terminal,
        "generated_response_tokens": record.get("generated_response_tokens"),
        "prompt_tokens": record.get("prompt_tokens"),
    }


def aggregate(rows, planned):
    complete = [r for r in rows if r["terminal_eligible"]]
    actions, submitted, fills, symbols = Counter(), Counter(), Counter(), Counter()
    for row in rows:
        actions.update(row["executed_step_actions"])
        submitted.update(row["submitted_tool_calls"])
        fills.update(row["fill_sides"])
        symbols.update(row["fill_symbols"])
    return {
        "planned": planned,
        "observed_unique_expected": len(rows),
        "terminal_complete": len(complete),
        "incomplete_observed": len(rows) - len(complete),
        "missing": max(0, planned - len(rows)),
        "executed_step_actions_observed": dict(actions),
        "submitted_tool_calls_observed": dict(submitted),
        "fill_sides_observed": dict(fills),
        "fill_symbols_observed": dict(symbols),
        "observed_fill_cost_including_partial": sum(r["observed_fill_cost"] for r in rows),
        "invalid_observed": sum(len(r["invalid_errors"]) for r in rows),
        "requests_observed": sum(r["requests"] for r in rows),
        "mean_Y_complete_only": mean(r["Y"] for r in complete),
        "mean_net_return_complete_only": mean(r["net_return"] for r in complete),
        "mean_cost_complete_only": mean(
            r["terminal_cost"] for r in complete if finite(r["terminal_cost"])
        ),
        "max_position_weight_complete_only": max(
            (r["max_weight"] for r in complete if finite(r["max_weight"])), default=None
        ),
        "mean_final_position_count_complete_only": mean(
            r["final_position_count"] for r in complete if r["final_position_count"] is not None
        ),
    }


def training_log(path, evidence):
    if not path.exists():
        return []
    evidence.hashes[str(path.relative_to(evidence.run))] = sha(path)
    records = []
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    for number, raw in enumerate(path.open(errors="replace"), 1):
        line = ansi.sub("", raw)
        match = re.search(r"\bstep:(\d+) - ", line)
        if not match:
            continue
        values = {}
        for part in line[match.end() :].strip().split(" - "):
            key, separator, value = part.partition(":")
            if separator and key.startswith(("actor/", "critic/")):
                try:
                    number_value = float(value)
                    values[key] = (
                        number_value if math.isfinite(number_value) else {"nonfinite": value}
                    )
                except ValueError:
                    values[key] = {"unparsed": value}
        if values:
            records.append(
                {"source_line": number, "policy_step": int(match.group(1)), "metrics": values}
            )
    return records


def artifact_validation(evidence, cell, expected_identity):
    directory = cell / "checkpoints/global_step_2"
    if cell.name == "base":
        return {
            "status": "NOT_REQUIRED_ZERO_RL",
            "optimizer_included": False,
            "resume_supported": False,
        }
    marker = evidence.json(directory / "research_artifact_complete.json")
    if marker is None:
        return {"status": "MISSING", "path": str(directory.relative_to(evidence.run))}
    module_path = evidence.run / "sources/src/agents/screening_checkpoint.py"
    identity = evidence.json(evidence.run / "source_identity.json", required=True) or {}
    if not module_path.exists() or sha(module_path) != identity.get(
        "src/agents/screening_checkpoint.py"
    ):
        return {"status": "INVALID", "error": "frozen validator source identity mismatch"}
    spec = importlib.util.spec_from_file_location("fixed2_frozen_checkpoint", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        validated = module.validate_research_artifact(directory, identity_sha256=expected_identity)
        return {
            "status": "VALIDATED",
            "artifact_profile": validated["artifact_profile"],
            "global_step": validated["global_step"],
            "file_bytes": validated["files"],
            "optimizer_included": validated["optimizer_included"],
            "resume_supported": validated["resume_supported"],
            "verification": (
                "required file sizes and metadata hashes; "
                "model tensors not loaded or content-hashed"
            ),
        }
    except (OSError, ValueError, RuntimeError) as error:
        return {"status": "INVALID", "error": str(error)}


def cell_readout(evidence, name, model, train_ids, eval_ids, calibration_sha=None):
    cell, problems = evidence.run / name, []
    summary = evidence.json(cell / "summary.json") or {}
    command = evidence.json(cell / "command.json") or []
    options = dict(x.lstrip("+").split("=", 1) for x in command[3:] if "=" in x)
    expected = expected_grid(name, train_ids, eval_ids)
    grouped = defaultdict(list)
    raw = evidence.jsonl(cell / "artifacts/screening_evidence")
    for reference, record in raw:
        grouped[grid_key(record)].append((reference, record))
    trajectories, excluded = [], []
    for key, entries in grouped.items():
        if (
            key not in expected
            or len(entries) != 1
            or entries[0][1].get("episode_id") != expected[key]
        ):
            excluded.append(
                {
                    "identity": key,
                    "sources": [x[0] for x in entries],
                    "reason": (
                        "unknown/wrong grid or duplicate identity; no stitching/dedup selection"
                    ),
                }
            )
            continue
        reference, record = entries[0]
        try:
            row = trajectory(reference, record)
        except (KeyError, TypeError, ValueError) as error:
            excluded.append(
                {
                    "identity": key,
                    "sources": [reference],
                    "reason": f"uninterpretable trajectory evidence: {error}",
                }
            )
            continue
        if (
            record.get("arm") != ("M1" if name == "base" else name.upper())
            or record.get("base_seed") != 42
            or (
                calibration_sha is not None
                and record.get("metrics", {}).get("calibration_sha256") != calibration_sha
            )
        ):
            row["terminal_eligible"] = False
            row["Y"] = row["net_return"] = None
            row["problems"].append("arm/seed/calibration identity differs")
        trajectories.append(row)
    groups = {
        kind: aggregate([r for r in trajectories if r["identity"]["validate"] == evaluate], planned)
        for kind, evaluate, planned in (
            ("train", False, 0 if name == "base" else 16),
            ("eval", True, 4),
        )
    }
    updates = evidence.jsonl(cell / "artifacts/update_probe")
    changed = Counter(r.get("role", "unknown") for _, r in updates if r.get("detected") is True)
    update_keys = [(r.get("pid"), r.get("optimizer_step_count"), r.get("role")) for _, r in updates]
    target = 0 if name == "base" else 2
    if (
        len(updates) != target * 2
        or dict(changed) != ({} if target == 0 else {"actor": 2, "critic": 2})
        or len(set(update_keys)) != len(update_keys)
    ):
        problems.append("optimizer records/changed actor-critic counts differ from fixed budget")
    head = evidence.json(cell / "artifacts/critic_initialization/head-rank0.json") or {}
    if (
        head.get("seed") != 42
        or head.get("mode") != "cpu_seed_v1"
        or head.get("execution_identity_sha256") != options.get(IDENTITY_OPTION)
        or head.get("model_path") != model["model_snapshot_path"]
    ):
        problems.append("actual initial critic head identity missing or mismatched")
    if head.get("phase") != "critic_head_after_hf_load_before_fsdp" or head.get("arm") != (
        "M1" if name == "base" else name.upper()
    ):
        problems.append("critic head phase/arm differs")
    if (
        any(options.get(k) != model["model_snapshot_path"] for k in MODEL_OPTIONS)
        or summary.get("model_revision") != model["model_revision"]
    ):
        problems.append(
            "common actor/reference/critic/tokenizer model identity missing or mismatched"
        )
    diagnostics = [
        dict(source=reference, **record)
        for reference, record in evidence.jsonl(cell / "artifacts/sibling_diagnostics")
    ]
    if sorted(r.get("policy_step", -1) for r in diagnostics) != ([] if name == "base" else [1, 2]):
        problems.append("missing or duplicate policy-step diagnostics")
    artifact = artifact_validation(evidence, cell, options.get(IDENTITY_OPTION))
    if artifact["status"] not in {"VALIDATED", "NOT_REQUIRED_ZERO_RL"}:
        problems.append("final research model/extra artifact missing or invalid")
    if summary.get("status") != "COMPLETE":
        problems.append(f"cell summary is {summary.get('status', 'MISSING')}")
    if excluded or any(g["terminal_complete"] != g["planned"] for g in groups.values()):
        problems.append("planned complete trajectory population not observed exactly")
    logs = training_log(cell / "trainer.log", evidence)
    rounds = []
    for step in () if name == "base" else (1, 2):
        rows = [
            r
            for r in trajectories
            if not r["identity"]["validate"] and r["identity"]["step"] == step
        ]
        siblings = []
        for index in ((step - 1) * 2, (step - 1) * 2 + 1):
            rr = sorted(
                [r for r in rows if r["identity"]["sample_index"] == index],
                key=lambda r: r["identity"]["rollout_n"],
            )
            all_four = len(rr) == 4 and all(r["terminal_eligible"] for r in rr)
            siblings.append(
                {
                    "episode_id": train_ids[index],
                    "observed": len(rr),
                    "all_four_complete": all_four,
                    "siblings": [
                        {
                            "rollout_n": r["identity"]["rollout_n"],
                            "Y": r["Y"],
                            "net_return": r["net_return"],
                        }
                        for r in rr
                    ],
                    "Y_range_all_four_complete": max(r["Y"] for r in rr) - min(r["Y"] for r in rr)
                    if all_four
                    else None,
                }
            )
        rounds.append(
            {
                "policy_step": step,
                "population": aggregate(rows, 8),
                "sibling_Y": siblings,
                "advantage_diagnostics_as_recorded": [
                    r for r in diagnostics if r.get("policy_step") == step
                ],
                "actor_critic_metrics_as_logged": [r for r in logs if r["policy_step"] == step],
            }
        )
    return {
        "summary": summary,
        "status": "COMPLETE" if not problems else "INCOMPLETE",
        "remaining_or_failed_checks": problems,
        "raw_trajectory_records": len(raw),
        "excluded_identity_records": excluded,
        "groups": groups,
        "trajectories": trajectories,
        "rounds": rounds,
        "optimizer_updates": {
            "records": len(updates),
            "detected_by_role": dict(changed),
            "raw": [dict(source=ref, **r) for ref, r in updates],
        },
        "initial_critic_head": head,
        "model_options": {k: options.get(k) for k in MODEL_OPTIONS},
        "artifact": artifact,
        "diagnostics_as_recorded": diagnostics,
        "training_metrics_as_logged": logs,
    }


def paired_delta(a, b):
    if a is None or b is None or not a["terminal_eligible"] or not b["terminal_eligible"]:
        return None
    values = {
        key: a[key] - b[key]
        for key in (
            "Y",
            "net_return",
            "terminal_cost",
            "max_weight",
            "final_cash",
            "final_position_count",
            "fill_count",
        )
        if finite(a[key]) and finite(b[key])
    }
    values["executed_action_counts"] = {
        action: a["executed_step_actions"].get(action, 0)
        - b["executed_step_actions"].get(action, 0)
        for action in ("hold", "buy", "sell")
    }
    values["actual_fill_counts"] = {
        side: a["fill_sides"].get(side, 0) - b["fill_sides"].get(side, 0)
        for side in ("buy", "sell")
    }
    values["filled_symbols_added"] = sorted(set(a["fill_symbols"]) - set(b["fill_symbols"]))
    values["filled_symbols_removed"] = sorted(set(b["fill_symbols"]) - set(a["fill_symbols"]))
    return values


def build(run):
    evidence = Evidence(run)
    state = evidence.json(run / "state.json", required=True) or {}
    model = evidence.json(run / "model_identity.json", required=True) or {}
    manifest = evidence.json(run / "prepared/manifest.json", required=True) or {}
    contract = evidence.json(run / "run_contract.json", required=True) or {}
    if set(contract.get("inputs", {})) != {
        "settings.json",
        "model_identity.json",
        "input_preflight.json",
        "prepared/manifest.json",
    }:
        evidence.errors.append("sealed input contract population differs from fixed plan")
    for name, digest in contract.get("inputs", {}).items():
        path = run / name
        if not path.exists() or sha(path) != digest:
            evidence.errors.append(f"sealed input changed/missing: {name}")
    for item in [*manifest.get("datasets", {}).values(), manifest.get("calibration", {})]:
        path = run / "prepared" / item.get("path", "missing")
        if not path.is_file() or sha(path) != item.get("sha256"):
            evidence.errors.append(f"prepared data/calibration changed/missing: {item.get('path')}")
    train_ids = manifest.get("datasets", {}).get("train", {}).get("episode_ids", [])
    eval_ids = manifest.get("datasets", {}).get("dev", {}).get("episode_ids", [])
    if len(train_ids) < 4 or len(eval_ids) != 4 or not model:
        raise RuntimeError(
            "cannot interpret fixed population without prepared grid and model identity"
        )
    cells = {
        name: cell_readout(
            evidence,
            name,
            model,
            train_ids,
            eval_ids,
            manifest.get("calibration", {}).get("payload_sha256"),
        )
        for name in ("base", "m1", "m2")
    }
    head_hashes = {name: c["initial_critic_head"].get("head_sha256") for name, c in cells.items()}
    heads_equal = (
        all(isinstance(h, str) and len(h) == 64 for h in head_hashes.values())
        and len(set(head_hashes.values())) == 1
    )
    pairs = []
    for episode in eval_ids:
        observed = {}
        for name, cell in cells.items():
            found = [
                r
                for r in cell["trajectories"]
                if r["episode_id"] == episode
                and r["identity"]["validate"]
                and r["terminal_eligible"]
            ]
            observed[name] = found[0] if len(found) == 1 else None
        deltas = {}
        for arm, reference in (("m1", "base"), ("m2", "base"), ("m2", "m1")):
            a, b = observed[arm], observed[reference]
            deltas[f"{arm}_minus_{reference}"] = paired_delta(a, b)
        pairs.append(
            {"episode_id": episode, "observations": observed, "deltas_complete_pairs_only": deltas}
        )
    remaining = [
        f"{name}: {problem}"
        for name, cell in cells.items()
        for problem in cell["remaining_or_failed_checks"]
    ]
    remaining.extend(evidence.errors)
    if not heads_equal:
        remaining.append("initial critic head hashes are missing or unequal")
    status = (
        "FAILED"
        if state.get("status") == "FAILED"
        else "INCOMPLETE"
        if remaining or state.get("status") != "COMPLETE"
        else "COMPLETE"
    )
    return {
        "status": status,
        "at_SGT": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "controller_status": state.get("status", "MISSING"),
        "controller_error": state.get("error"),
        "cells": cells,
        "evaluation_pairs": pairs,
        "shared_model_identity": model,
        "initial_critic_heads": {"hashes": head_hashes, "all_three_equal": heads_equal},
        "remaining_or_failed_checks": remaining,
        "input_sha256": evidence.hashes,
        "limits": [
            "Fixed two-round repeated Train diagnostic; no A1 or OOS superiority conclusion.",
            "Same shared SFT teacher prior; do not attribute that prior to M2 reward.",
            "Missing, duplicate, or incomplete episodes are not zero returns and are not stitched.",
            "Tool submissions, executed actions, and actual ledger fills are separate.",
            "S diagnostics and actor/critic metrics are reported only when recorded.",
            "Research model/extra artifacts omit Adam and cannot resume training.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument(
        "--check", action="store_true", help="read-only preview; never write final artifacts"
    )
    args = parser.parse_args()
    run = args.run.resolve()
    state = json.loads((run / "state.json").read_text())
    if not args.check and state.get("status") not in {"COMPLETE", "FAILED"}:
        raise RuntimeError("refusing final summary before controller terminal COMPLETE/FAILED")
    if not args.check and any(
        (run / name).exists() for name in ("readout.json", "completion_receipt.json")
    ):
        raise RuntimeError("final output already exists; no overwrite or implicit rerun")
    result = build(run)
    if args.check:
        print(
            json.dumps(
                {
                    "check_only": True,
                    "controller_status": result["controller_status"],
                    "observed": {name: cell["groups"] for name, cell in result["cells"].items()},
                    "remaining_or_failed_checks": result["remaining_or_failed_checks"],
                },
                indent=2,
            )
        )
        return
    serialized = json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n"
    with (run / "readout.json").open("x") as handle:
        handle.write(serialized)
    receipt = {
        "status": result["status"],
        "at_SGT": result["at_SGT"],
        "controller_status": result["controller_status"],
        "readout_sha256": sha(run / "readout.json"),
        "summarizer_sha256": sha(Path(__file__)),
        "remaining_or_failed_checks": result["remaining_or_failed_checks"],
        "raw_evidence_preserved": True,
        "new_sampling_or_training_started": False,
    }
    with (run / "completion_receipt.json").open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

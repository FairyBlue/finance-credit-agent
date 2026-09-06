"""Fixed-grid readout over recorded artifacts; never opens prices or runs evaluation."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.agents.rollout_seed import normalize_rollout_identity
from src.agents.screening_launcher import ARMS, atomic_json, read_jsonl
from src.data.screening import SCREEN_DEV_INDICES

EXPECTED_DEV_IDS = tuple(f"formal-train-{index + 1:04d}" for index in SCREEN_DEV_INDICES)
METRIC_FIELDS = (
    "terminal_score",
    "raw_net_return",
    "benchmark_return",
    "benchmark_excess",
    "max_drawdown",
    "turnover",
    "max_daily_turnover",
    "risk_penalty",
    "transaction_cost",
    "agent_volatility",
)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("descriptive statistics require nonempty evidence")
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def paired_summary(left: dict[str, float], right: dict[str, float]) -> dict[str, Any]:
    if set(left) != set(right) or len(left) != 12:
        raise ValueError("paired screening comparison requires the same complete 12-episode grid")
    rows: list[dict[str, Any]] = []
    for episode_id in sorted(left):
        a, b = _number(left[episode_id], "left score"), _number(right[episode_id], "right score")
        rows.append(
            {
                "episode_id": episode_id,
                "left": a,
                "right": b,
                "delta": _number(a - b, "paired difference"),
            }
        )
    differences: list[float] = [row["delta"] for row in rows]
    return {
        "mean": statistics.mean(differences),
        "median": statistics.median(differences),
        "win_rate": sum(delta > 0.0 for delta in differences) / 12,
        "ties": sum(delta == 0.0 for delta in differences),
        "worst_episode": min(rows, key=lambda row: row["delta"]),
        "rows": rows,
    }


def validate_eval_grid(
    records: Sequence[dict[str, Any]],
    *,
    summary: Mapping[str, Any],
    arm: str,
    seed: int,
    stage: str,
) -> list[dict[str, Any]]:
    """Bind evaluation to the exact registered episode, final policy step and scorer."""
    expected_step = 0 if stage == "zero-shot" else 16
    if (summary.get("arm"), summary.get("seed"), summary.get("stage")) != (arm, seed, stage):
        raise ValueError("cell summary arm/seed/stage identity mismatch")
    if summary.get("expected_eval_episode_ids") != list(EXPECTED_DEV_IDS):
        raise ValueError("summary does not bind the exact prepared 12-episode dev grid")
    if _integer(summary.get("expected_eval_step"), "expected_eval_step") != expected_step:
        raise ValueError("summary does not bind the final evaluation policy step")
    calibration = _digest(summary.get("calibration_payload_sha256"), "calibration identity")
    _digest(summary.get("prepared_manifest_sha256"), "prepared manifest identity")
    by_index: dict[int, dict[str, Any]] = {}
    for record in records:
        identity = normalize_rollout_identity(record.get("identity"))
        if not identity.validate:
            continue
        if record.get("complete") is not True or record.get("status") != "COMPLETE":
            raise ValueError("dev grid includes an incomplete/error rollout")
        index = identity.sample_index
        if (
            identity.step != expected_step
            or identity.rollout_n != 0
            or not 0 <= index < len(EXPECTED_DEV_IDS)
            or record.get("episode_id") != EXPECTED_DEV_IDS[index]
            or record.get("rollout_count") != 1
        ):
            raise ValueError("dev trajectory episode/index/final-step/rollout identity mismatch")
        if index in by_index:
            raise ValueError("duplicate dev trajectory identity")
        if record.get("arm") != arm or record.get("base_seed") != seed:
            raise ValueError("dev trajectory belongs to another arm or seed")
        metrics = record.get("metrics")
        if not isinstance(metrics, dict) or metrics.get("scorer_id") != "benchmark_scale_v2":
            raise ValueError("dev trajectory common scorer metrics missing")
        if metrics.get("calibration_sha256") != calibration:
            raise ValueError("dev trajectory calibration identity mismatch")
        if metrics.get("session_count") != 21:
            raise ValueError("dev trajectory must cover 21 sessions")
        for field in METRIC_FIELDS:
            _number(metrics.get(field), field)
        if _number(record.get("terminal_score"), "terminal_score") != metrics["terminal_score"]:
            raise ValueError("dev terminal score differs from recorded common scorer metrics")
        if not math.isclose(
            metrics["benchmark_excess"],
            metrics["raw_net_return"] - metrics["benchmark_return"],
            abs_tol=1e-12,
        ):
            raise ValueError("dev benchmark excess is inconsistent with net return")
        _integer(record.get("prompt_tokens"), "prompt_tokens")
        _integer(record.get("generated_response_tokens"), "generated_response_tokens")
        _integer(metrics.get("invalid_count"), "invalid_count")
        _integer(metrics.get("step_count"), "step_count", 20)
        by_index[index] = record
    if set(by_index) != set(range(12)):
        raise ValueError("incomplete exact 12-episode dev grid")
    return [by_index[index] for index in range(12)]


def _describe_evals(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    invalid = sum(int(row["metrics"]["invalid_count"]) for row in records)
    steps = sum(int(row["metrics"]["step_count"]) for row in records)
    return {
        "episode_count": len(records),
        "completion_rate": 1.0,
        "metrics": {
            field: _distribution([float(row["metrics"][field]) for row in records])
            for field in METRIC_FIELDS
        },
        "invalid_count": invalid,
        "step_count": steps,
        "invalid_step_rate": invalid / steps,
        "tokens": {
            "prompt": sum(int(row["prompt_tokens"]) for row in records),
            "response": sum(int(row["generated_response_tokens"]) for row in records),
        },
        "rows": [
            {
                "episode_id": row["episode_id"],
                "metrics": row["metrics"],
                "prompt_tokens": row["prompt_tokens"],
                "generated_response_tokens": row["generated_response_tokens"],
            }
            for row in records
        ],
    }


def _diagnostic_exposure(
    records: Sequence[dict[str, Any]],
    *,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    steps = sorted(_integer(row.get("policy_step"), "diagnostic policy_step", 1) for row in records)
    if steps != list(range(1, 17)):
        raise ValueError("diagnostics must cover all 16 policy updates exactly once")
    for row in records:
        if (row.get("arm"), row.get("run_seed")) != (arm, seed):
            raise ValueError("diagnostics belong to another arm or seed")
        _integer(row.get("all_real_steps", {}).get("nonzero_S_count"), "nonzero_S_count")
        rate = _number(row.get("pooling_changed_row_rate"), "pooling_changed_row_rate")
        if not 0.0 <= rate <= 1.0:
            raise ValueError("pooling changed-row rate is outside [0,1]")
    return {
        "residual_exercised": any(row["all_real_steps"]["nonzero_S_count"] > 0 for row in records),
        "pooling_exercised": any(row["pooling_changed_row_rate"] > 0.0 for row in records),
        "pooling_changed_row_rate": _distribution(
            [float(row["pooling_changed_row_rate"]) for row in records]
        ),
    }


def _load_complete_cell(
    directory: Path,
    *,
    arm: str,
    seed: int,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = directory / "summary.json"
    if not path.is_file():
        raise ValueError("missing run")
    summary = json.loads(path.read_text())
    if not isinstance(summary, dict):
        raise ValueError("invalid run summary")
    if summary.get("status") != "PASSED":
        raise ValueError(f"run is not PASSED: {summary.get('status')}")
    evals = validate_eval_grid(
        read_jsonl(directory / "artifacts/screening_evidence"),
        summary=summary,
        arm=arm,
        seed=seed,
        stage=stage,
    )
    return summary, evals


def _identity(summary: Mapping[str, Any]) -> tuple[str, str]:
    return (str(summary["prepared_manifest_sha256"]), str(summary["calibration_payload_sha256"]))


def _compute_summary(directory: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    """Sum all trajectory tokens and use per-cell sampled memory.current, never memory.peak."""
    tokens = {
        phase: {"prompt": 0, "response": 0, "trajectory_records": 0} for phase in ("train", "eval")
    }
    for row in read_jsonl(directory / "artifacts/screening_evidence"):
        phase = "eval" if row["identity"]["validate"] else "train"
        tokens[phase]["prompt"] += _integer(row.get("prompt_tokens"), "prompt_tokens")
        tokens[phase]["response"] += _integer(
            row.get("generated_response_tokens"), "generated_response_tokens"
        )
        tokens[phase]["trajectory_records"] += 1
    host_samples: list[int] = []
    gpu_samples: list[float] = []
    malformed = 0
    path = directory / "resources.jsonl"
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                if "memory.current" in row:
                    host_samples.append(_integer(int(row["memory.current"]), "memory.current"))
                if row.get("gpu"):
                    gpu_samples.append(_number(float(row["gpu"].split(",", 1)[0]), "gpu memory"))
            except (ValueError, TypeError, KeyError):
                malformed += 1
    elapsed = summary.get("elapsed_seconds")
    return {
        "elapsed_seconds": None if elapsed is None else _number(elapsed, "elapsed_seconds"),
        "tokens_by_phase": tokens,
        "sampled_host_memory_current_peak_bytes": max(host_samples, default=None),
        "sampled_gpu_memory_peak_mib": max(gpu_samples, default=None),
        "resource_samples": {"host": len(host_samples), "gpu": len(gpu_samples)},
        "malformed_resource_samples": malformed,
        "resource_scope": "per-cell samples; cgroup cumulative memory.peak is not used",
    }


def _read_cpu_baselines(path: Path, *, calibration: str | None) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError("CPU baseline artifact is not available")
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or len(rows) != 24:
        raise ValueError("CPU baselines require exactly 24 recorded trajectories")
    expected = {
        (baseline, episode_id)
        for baseline in ("cash", "buy_and_hold")
        for episode_id in EXPECTED_DEV_IDS
    }
    observed = set()
    groups: dict[str, list[dict[str, Any]]] = {"cash": [], "buy_and_hold": []}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid CPU baseline row")
        key = (row.get("baseline"), row.get("episode_id"))
        if key not in expected or key in observed:
            raise ValueError("CPU baselines do not match their exact 2 x 12 grid")
        observed.add(key)
        if (
            row.get("scorer_id") != "benchmark_scale_v2"
            or row.get("calibration_sha256") != calibration
        ):
            raise ValueError("CPU baseline scorer/calibration identity mismatch")
        for name in (
            "terminal_score",
            "raw_net_return",
            "benchmark_return",
            "max_drawdown",
            "risk_penalty",
        ):
            _number(row.get(name), name)
        normalized = dict(row)
        normalized["benchmark_excess"] = row["raw_net_return"] - row["benchmark_return"]
        groups[row["baseline"]].append(normalized)
    return {
        baseline: {
            "episode_count": 12,
            "completion_rate": 1.0,
            "metrics": {
                field: _distribution([_number(row[field], field) for row in group])
                for field in METRIC_FIELDS
                if all(field in row for row in group)
            },
            "unavailable_metrics": [
                field for field in METRIC_FIELDS if not all(field in row for row in group)
            ],
            "tokens": {"prompt": 0, "response": 0},
            "rows": sorted(group, key=lambda row: row["episode_id"]),
        }
        for baseline, group in groups.items()
    }


def build_readout(experiment_root: Path) -> dict[str, Any]:
    scores: dict[tuple[str, int], dict[str, float]] = {}
    missing: list[str] = []
    cells: list[dict[str, Any]] = []
    descriptive: dict[str, dict[str, Any]] = {}
    exposure: dict[int, bool] = {}
    pooling_by_arm: dict[int, dict[str, bool]] = {42: {}, 43: {}}
    source_identity: tuple[str, str] | None = None
    for seed in (42, 43):
        for arm in ARMS:
            directory = experiment_root / f"screening-{arm.lower()}-s{seed}"
            try:
                summary, evals = _load_complete_cell(
                    directory, arm=arm, seed=seed, stage="screening"
                )
                if source_identity is None:
                    source_identity = _identity(summary)
                elif _identity(summary) != source_identity:
                    raise ValueError(
                        "cell uses a different prepared manifest or scorer calibration"
                    )
                stats = _describe_evals(evals)
                stats["compute"] = _compute_summary(directory, summary)
                if arm in {"M2", "M2-plain"}:
                    diagnostic = _diagnostic_exposure(
                        read_jsonl(directory / "artifacts/sibling_diagnostics"), arm=arm, seed=seed
                    )
                    stats["diagnostics"] = diagnostic
                    pooling_by_arm[seed][arm] = diagnostic["pooling_exercised"]
                    if arm == "M2":
                        exposure[seed] = diagnostic["residual_exercised"]
                cells.append(summary)
                descriptive[f"{arm}/seed={seed}"] = stats
                scores[(arm, seed)] = {
                    row["episode_id"]: float(row["terminal_score"]) for row in evals
                }
            except (OSError, ValueError, TypeError, KeyError) as error:
                missing.append(f"{arm}/seed={seed}: {error}")
    pooling = {
        seed: all(pooling_by_arm[seed].get(arm, False) for arm in ("M2", "M2-plain"))
        for seed in (42, 43)
    }
    report: dict[str, Any] = {
        "schema_version": 2,
        "purpose": "development_screening_only",
        "primary": "mean_paired_M2_minus_M1_both_training_seeds_positive",
        "formal_valid_accessed": False,
        "final_oos_accessed": False,
        "significance_claim": None,
        "cells": cells,
        "missing": missing,
        "descriptive_by_arm_seed": descriptive,
        "residual_exposure": exposure,
        "pooling_exposure": pooling,
        "pooling_exposure_by_arm": pooling_by_arm,
        "pooling_direction_supported": False,
    }
    if missing:
        report["verdict"] = "INCONCLUSIVE"
    elif not all(exposure.get(seed, False) for seed in (42, 43)):
        report.update(verdict="INCONCLUSIVE", reason="NO-RESIDUAL-EXPOSURE")
    else:
        comparisons = {
            str(seed): {
                control: paired_summary(scores[("M2", seed)], scores[(control, seed)])
                for control in ("M1", "M2-plain", "A2")
            }
            for seed in (42, 43)
        }
        report["comparisons"] = comparisons
        promising = all(comparisons[str(seed)]["M1"]["mean"] > 0.0 for seed in (42, 43))
        report["verdict"] = "PROMISING" if promising else "NOT YET PROMISING"
        report["pooling_direction_supported"] = promising and all(
            pooling[seed] and comparisons[str(seed)]["M2-plain"]["mean"] > 0.0 for seed in (42, 43)
        )
        if not all(pooling.values()):
            report["pooling_note"] = "POOLING-NOT-EXERCISED"

    references: dict[str, Any] = {}
    reference_scores: dict[str, dict[str, float]] = {}
    reference_issues: list[str] = []
    try:
        summary, evals = _load_complete_cell(
            experiment_root / "zero-shot-a2-s42", arm="A2", seed=42, stage="zero-shot"
        )
        if source_identity is None or _identity(summary) != source_identity:
            raise ValueError("zero-shot prepared manifest/calibration differs from the main cells")
        references["zero-shot"] = {
            "summary": summary,
            **_describe_evals(evals),
            "compute": _compute_summary(experiment_root / "zero-shot-a2-s42", summary),
        }
        reference_scores["zero-shot"] = {
            row["episode_id"]: float(row["terminal_score"]) for row in evals
        }
    except (OSError, ValueError, TypeError, KeyError) as error:
        reference_issues.append(f"zero-shot: {error}")
    try:
        baseline_stats = _read_cpu_baselines(
            experiment_root / "baselines.json",
            calibration=source_identity[1] if source_identity else None,
        )
        references.update(baseline_stats)
        for baseline, stats in baseline_stats.items():
            reference_scores[baseline] = {
                row["episode_id"]: float(row["terminal_score"]) for row in stats["rows"]
            }
    except (OSError, ValueError, TypeError, KeyError) as error:
        reference_issues.append(f"CPU baselines: {error}")
    report["references"] = references
    report["reference_issues"] = reference_issues
    report["readout_complete"] = not missing and not reference_issues
    report["reference_comparisons"] = {
        str(seed): {
            name: paired_summary(scores[("M2", seed)], values)
            for name, values in reference_scores.items()
        }
        for seed in (42, 43)
        if ("M2", seed) in scores
    }
    report["reference_policy"] = (
        "Missing references make the report incomplete; "
        "they do not alter the registered main-cell verdict."
    )
    atomic_json(experiment_root / "readout.json", report)
    _write_markdown(experiment_root, report)
    return report


def _write_markdown(experiment_root: Path, report: dict[str, Any]) -> None:
    lines = [
        "# M2 screening readout",
        "",
        f"Primary status: **{report['verdict']}**",
        "",
        "This is a development screen; it is not a significance or Final OOS result.",
        "",
    ]
    if not report["readout_complete"]:
        lines.extend(
            [
                "Report incomplete:",
                "",
                *[f"- {message}" for message in report["missing"] + report["reference_issues"]],
                "",
            ]
        )
    if "comparisons" in report:
        lines.extend(
            [
                "| Seed | Comparison | Mean paired delta | Median | Win rate |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for seed, contrasts in report["comparisons"].items():
            for control, values in contrasts.items():
                lines.append(
                    f"| {seed} | M2 − {control} | {values['mean']:.6g} | "
                    f"{values['median']:.6g} | {values['win_rate']:.1%} |"
                )
        lines.extend(
            ["", f"Pooling direction supported: {report['pooling_direction_supported']}.", ""]
        )
        if report.get("pooling_note"):
            lines.extend([report["pooling_note"], ""])
    rows = {**report["descriptive_by_arm_seed"], **report["references"]}
    if rows:
        lines.extend(
            [
                "| Arm / reference | Mean Y | Mean net return | Mean excess | "
                "Mean max drawdown | Prompt / response tokens |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for label, stats in rows.items():
            metrics, tokens = stats["metrics"], stats["tokens"]
            lines.append(
                f"| {label} | {metrics['terminal_score']['mean']:.6g} | "
                f"{metrics['raw_net_return']['mean']:.3%} | "
                f"{metrics['benchmark_excess']['mean']:.3%} | "
                f"{metrics['max_drawdown']['mean']:.3%} | "
                f"{tokens['prompt']} / {tokens['response']} |"
            )
    lines.extend(
        [
            "",
            "See readout.json for paired rows, turnover, costs, volatility and run identities.",
            "",
        ]
    )
    (experiment_root / "READOUT.md").write_text("\n".join(lines), encoding="utf-8")

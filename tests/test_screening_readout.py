"""Readout decisions require exact terminal grids and independent mechanism exposure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.agents.screening_launcher import ARMS
from src.evaluation.screening import EXPECTED_DEV_IDS, build_readout

CALIBRATION = "a" * 64
MANIFEST = "b" * 64


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _eval_rows(arm: str, seed: int, stage: str) -> list[dict[str, Any]]:
    values = {"A2": 0.0, "M1": 1.0, "M2-plain": 1.5, "M2": 2.0}
    rows = []
    for index, episode_id in enumerate(EXPECTED_DEV_IDS):
        score = values[arm] + index / 10
        metrics = {
            "terminal_score": score,
            "raw_net_return": 0.03,
            "benchmark_return": 0.02,
            "benchmark_excess": 0.01,
            "max_drawdown": 0.04,
            "turnover": 0.5,
            "max_daily_turnover": 0.2,
            "risk_penalty": 0.04,
            "transaction_cost": 100.0,
            "agent_volatility": 0.01,
            "invalid_count": 1,
            "step_count": 21,
            "session_count": 21,
            "scorer_id": "benchmark_scale_v2",
            "calibration_sha256": CALIBRATION,
        }
        rows.append(
            {
                "episode_id": episode_id,
                "arm": arm,
                "base_seed": seed,
                "identity": {
                    "step": 0 if stage == "zero-shot" else 16,
                    "sample_index": index,
                    "rollout_n": 0,
                    "validate": True,
                },
                "rollout_count": 1,
                "complete": True,
                "status": "COMPLETE",
                "terminal_score": score,
                "metrics": metrics,
                "prompt_tokens": 100,
                "generated_response_tokens": 20,
            }
        )
    return rows


def _cell(root: Path, arm: str, seed: int, stage: str = "screening") -> Path:
    directory = root / f"{stage}-{arm.lower()}-s{seed}"
    _write_json(
        directory / "summary.json",
        {
            "arm": arm,
            "seed": seed,
            "stage": stage,
            "status": "PASSED",
            "expected_eval_episode_ids": list(EXPECTED_DEV_IDS),
            "expected_eval_step": 0 if stage == "zero-shot" else 16,
            "calibration_payload_sha256": CALIBRATION,
            "prepared_manifest_sha256": MANIFEST,
        },
    )
    _write_jsonl(
        directory / "artifacts/screening_evidence/eval.jsonl", _eval_rows(arm, seed, stage)
    )
    if stage == "screening":
        _write_jsonl(
            directory / "artifacts/sibling_diagnostics/diagnostics.jsonl",
            [
                {
                    "arm": arm,
                    "run_seed": seed,
                    "policy_step": step,
                    "all_real_steps": {"nonzero_S_count": 10},
                    "pooling_changed_row_rate": 0.5,
                }
                for step in range(1, 17)
            ],
        )
    return directory


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    for seed in (42, 43):
        for arm in ARMS:
            _cell(tmp_path, arm, seed)
    _cell(tmp_path, "A2", 42, "zero-shot")
    baselines = []
    for baseline in ("cash", "buy_and_hold"):
        for episode_id in EXPECTED_DEV_IDS:
            baselines.append(
                {
                    "episode_id": episode_id,
                    "baseline": baseline,
                    "terminal_score": 0.0,
                    "raw_net_return": 0.0 if baseline == "cash" else 0.02,
                    "benchmark_return": 0.02,
                    "max_drawdown": 0.0,
                    "risk_penalty": 0.0,
                    "turnover": 0.0,
                    "max_daily_turnover": 0.0,
                    "transaction_cost": 0.0,
                    "agent_volatility": 0.0,
                    "scorer_id": "benchmark_scale_v2",
                    "calibration_sha256": CALIBRATION,
                }
            )
    _write_json(tmp_path / "baselines.json", baselines)
    return tmp_path


def test_complete_readout_includes_paired_metrics_risk_tokens_and_references(
    experiment: Path,
) -> None:
    report = build_readout(experiment)
    assert report["verdict"] == "PROMISING"
    assert report["readout_complete"] is True
    assert report["pooling_direction_supported"] is True
    assert report["comparisons"]["42"]["M1"]["mean"] == pytest.approx(1.0)
    stats = report["descriptive_by_arm_seed"]["M2/seed=42"]
    assert stats["metrics"]["raw_net_return"]["mean"] == pytest.approx(0.03)
    assert stats["metrics"]["benchmark_excess"]["mean"] == pytest.approx(0.01)
    assert stats["metrics"]["transaction_cost"]["mean"] == 100.0
    assert stats["invalid_step_rate"] == pytest.approx(1 / 21)
    assert stats["tokens"] == {"prompt": 1200, "response": 240}
    assert set(report["references"]) == {"zero-shot", "cash", "buy_and_hold"}
    assert set(report["reference_comparisons"]["43"]) == {"zero-shot", "cash", "buy_and_hold"}
    assert "zero-shot" in (experiment / "READOUT.md").read_text()


@pytest.mark.parametrize(
    "mutation", ["wrong_episode", "wrong_final_step", "wrong_index", "incomplete", "duplicate"]
)
def test_counts_alone_cannot_accept_wrong_dev_identity(experiment: Path, mutation: str) -> None:
    rows = _eval_rows("M2", 42, "screening")
    if mutation == "wrong_episode":
        rows[0]["episode_id"] = "formal-train-0001"
    elif mutation == "wrong_final_step":
        rows[0]["identity"]["step"] = 15
    elif mutation == "wrong_index":
        rows[0]["identity"]["sample_index"] = 1
    elif mutation == "incomplete":
        rows[0]["complete"] = False
    else:
        rows.append(rows[0])
    _write_jsonl(experiment / "screening-m2-s42/artifacts/screening_evidence/eval.jsonl", rows)
    report = build_readout(experiment)
    assert report["verdict"] == "INCONCLUSIVE"
    assert report["missing"] and "M2/seed=42" in report["missing"][0]
    assert "comparisons" not in report


def test_plain_exposure_is_required_for_pooling_claim(experiment: Path) -> None:
    path = experiment / "screening-m2-plain-s42/artifacts/sibling_diagnostics/diagnostics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["pooling_changed_row_rate"] = 0.0
    _write_jsonl(path, rows)
    report = build_readout(experiment)
    assert report["verdict"] == "PROMISING"
    assert report["pooling_direction_supported"] is False
    assert report["pooling_exposure_by_arm"][42] == {"M2": True, "M2-plain": False}
    assert report["pooling_note"] == "POOLING-NOT-EXERCISED"


def test_other_seed_cannot_supply_missing_m2_residual_exposure(experiment: Path) -> None:
    path = experiment / "screening-m2-s43/artifacts/sibling_diagnostics/diagnostics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["all_real_steps"]["nonzero_S_count"] = 0
    _write_jsonl(path, rows)
    report = build_readout(experiment)
    assert report["verdict"] == "INCONCLUSIVE"
    assert report["reason"] == "NO-RESIDUAL-EXPOSURE"


def test_wrong_summary_grid_and_cross_cell_calibration_fail_closed(experiment: Path) -> None:
    path = experiment / "screening-a2-s42/summary.json"
    summary = json.loads(path.read_text())
    summary["expected_eval_episode_ids"] = [str(index) for index in range(12)]
    _write_json(path, summary)
    assert build_readout(experiment)["verdict"] == "INCONCLUSIVE"
    _cell(experiment, "A2", 42)
    path = experiment / "screening-m2-s43/summary.json"
    summary = json.loads(path.read_text())
    summary["calibration_payload_sha256"] = "c" * 64
    _write_json(path, summary)
    rows = _eval_rows("M2", 43, "screening")
    for row in rows:
        row["metrics"]["calibration_sha256"] = "c" * 64
    _write_jsonl(experiment / "screening-m2-s43/artifacts/screening_evidence/eval.jsonl", rows)
    report = build_readout(experiment)
    assert report["verdict"] == "INCONCLUSIVE"
    assert any("different prepared manifest or scorer" in item for item in report["missing"])


def test_failed_references_are_reported_without_creating_a_new_primary_gate(
    experiment: Path,
) -> None:
    path = experiment / "zero-shot-a2-s42/summary.json"
    summary = json.loads(path.read_text())
    summary["status"] = "FAILED"
    _write_json(path, summary)
    (experiment / "baselines.json").unlink()
    report = build_readout(experiment)
    assert report["verdict"] == "PROMISING"
    assert report["readout_complete"] is False
    assert len(report["reference_issues"]) == 2
    assert report["references"] == {}


def test_zero_shot_and_cpu_baselines_cannot_silently_use_wrong_identity(experiment: Path) -> None:
    rows = _eval_rows("A2", 42, "zero-shot")
    rows[0]["identity"]["step"] = 16
    _write_jsonl(experiment / "zero-shot-a2-s42/artifacts/screening_evidence/eval.jsonl", rows)
    baselines_path = experiment / "baselines.json"
    baselines = json.loads(baselines_path.read_text())
    baselines[0]["episode_id"] = "formal-train-0001"
    _write_json(baselines_path, baselines)
    report = build_readout(experiment)
    assert report["verdict"] == "PROMISING"
    assert report["readout_complete"] is False
    assert report["references"] == {}


def test_readout_does_not_call_market_preparation_or_baseline_evaluation(
    experiment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.data import screening

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("readout must use existing evidence only")

    monkeypatch.setattr(screening, "evaluate_screening_baselines", forbidden)
    monkeypatch.setattr(screening, "validate_prepared_screening", forbidden)
    assert build_readout(experiment)["readout_complete"] is True


def test_compute_uses_all_tokens_and_per_cell_current_memory_samples(experiment: Path) -> None:
    directory = experiment / "screening-m2-s42"
    path = directory / "artifacts/screening_evidence/eval.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    training_row = _eval_rows("M2", 42, "screening")[0]
    training_row["identity"]["validate"] = False
    training_row["generated_response_tokens"] = 77
    rows.append(training_row)
    _write_jsonl(path, rows)
    _write_jsonl(
        directory / "resources.jsonl",
        [
            {"memory.current": "100", "memory.peak": "99999999", "gpu": "30, 20"},
            {"memory.current": "150", "memory.peak": "99999999", "gpu": "25, 20"},
        ],
    )
    summary = json.loads((directory / "summary.json").read_text())
    summary["elapsed_seconds"] = 12.5
    _write_json(directory / "summary.json", summary)
    report = build_readout(experiment)
    compute = report["descriptive_by_arm_seed"]["M2/seed=42"]["compute"]
    assert compute["sampled_host_memory_current_peak_bytes"] == 150
    assert compute["sampled_gpu_memory_peak_mib"] == 30.0
    assert compute["elapsed_seconds"] == 12.5
    assert compute["tokens_by_phase"]["train"]["response"] == 77
    assert compute["tokens_by_phase"]["eval"]["response"] == 240

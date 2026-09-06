"""Evidence acceptance must prove the frozen grid, not just plausible totals."""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.agents import screening_launcher

CALIBRATION = "a" * 64
EXECUTION = "b" * 64


@dataclass
class EvidenceCase:
    root: Path
    options: dict[str, Any]
    records: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    updates: dict[str, Any]
    checkpoint_calls: list[tuple[Path, str]]

    def write(self) -> None:
        for name, records in (
            ("screening_evidence", self.records),
            ("sibling_diagnostics", self.diagnostics),
        ):
            directory = self.root / "artifacts" / name
            directory.mkdir(parents=True, exist_ok=True)
            # Completion order is deliberately unrelated to episode or step order.
            (directory / "worker.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in reversed(records)), encoding="utf-8"
            )

    def validate(self) -> dict[str, Any]:
        self.write()
        return screening_launcher.validate_cell_evidence(self.root, **self.options)


def _case(tmp_path, monkeypatch, *, stage="screening", arm="M2", seed=42) -> EvidenceCase:
    from agent_r1.agent_flow.agent_flow import get_trajectory_info

    small = stage in {"wiring", "pilot"}
    rounds = {"wiring": 1, "pilot": 2}.get(stage, 16)
    train_ids = [f"formal-train-{1 + 4 * i:04d}" for i in range(2 if small else rounds * 2)]
    eval_ids = train_ids[:2] if small else [f"formal-train-{210 + 21 * i:04d}" for i in range(12)]
    options = {
        "stage": stage,
        "expected_rounds": rounds,
        "arm": arm,
        "seed": seed,
        "train_episode_ids": train_ids,
        "eval_episode_ids": eval_ids,
        "execution_identity_sha256": EXECUTION,
        "calibration_payload_sha256": CALIBRATION,
    }
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    def record(identity: dict[str, Any], episode_id: str) -> dict[str, Any]:
        score = identity["sample_index"] * 0.01 - 0.3
        return {
            "schema_version": 1,
            "status": "COMPLETE",
            "complete": True,
            "arm": arm,
            "base_seed": seed,
            "episode_id": episode_id,
            "identity": identity,
            "group_uid": f"group-{identity['step']}-{identity['sample_index']}",
            "rollout_count": 1 if identity["validate"] else 4,
            "terminal_score": score,
            "metrics": {
                "calibration_sha256": CALIBRATION,
                "terminal_score": score,
                "raw_net_return": 0.0123,
            },
        }

    if stage != "zero-shot":
        for step in range(1, rounds + 1):
            first_index = ((step - 1) * 2) % len(train_ids)
            indices = [first_index] * 4 + [first_index + 1] * 4
            # Exercise the actual upstream meaning of rollout_n (sibling ID).
            identities = asyncio.run(get_trajectory_info(step, indices, False))
            records.extend(
                record(identity, train_ids[identity["sample_index"]]) for identity in identities
            )
            scope = {
                "row_count": 160,
                "std_U": 1.0,
                "std_S": 0.4,
                "corr_U_S": 0.2,
                "mean_abs_S": 0.2,
                "nonzero_S_count": 80,
                "nonzero_S_rate": 0.5,
                "exact_nonzero_S_count": 80,
            }
            diagnostics.append(
                {
                    "schema_version": 1,
                    "phase": "advantage_pre_update",
                    "policy_step": step,
                    "arm": arm,
                    "run_seed": seed,
                    "eta": 1.0,
                    "trajectory_count": 8,
                    "episode_group_count": 2,
                    "episode_ids": train_ids[first_index : first_index + 2],
                    "sibling_group_uids": [
                        f"group-{step}-{i}" for i in (first_index, first_index + 1)
                    ],
                    "session_group_count": 40,
                    "advancing_row_count": 160,
                    "real_step_count": 160,
                    "response_tokens": 320,
                    "padding_row_count": 0,
                    "action_diversity_rate": 0.5,
                    "pooling_opportunity_rate": 0.25,
                    "pooling_changed_row_count": 40,
                    "pooling_changed_row_rate": 0.25,
                    "whitening_scope": "logical_rollout_batch_real_step_rows",
                    "whitening_epsilon": 1e-8,
                    "whitening_correction": 1,
                    "whitening_mean": 0.1,
                    "whitening_std": 1.2,
                    "advancing_steps": copy.deepcopy(scope),
                    "all_real_steps": copy.deepcopy(scope),
                }
            )
    eval_step = 0 if stage == "zero-shot" else rounds
    eval_identities = asyncio.run(get_trajectory_info(eval_step, list(range(len(eval_ids))), True))
    records.extend(
        record(identity, eval_ids[identity["sample_index"]]) for identity in eval_identities
    )
    updates = {
        "optimizer_step_count": 0 if stage == "zero-shot" else 2 * rounds,
        "detected_role_counts": {}
        if stage == "zero-shot"
        else {
            "actor": rounds,
            "critic": rounds,
        },
    }
    checkpoint_calls: list[tuple[Path, str]] = []

    def optimizer_probe(directory: Path) -> dict[str, Any]:
        assert directory == tmp_path / "artifacts/update_probe"
        return updates

    def checkpoint_probe(directory: Path, *, identity_sha256: str) -> dict[str, Any]:
        checkpoint_calls.append((directory, identity_sha256))
        return {"global_step": rounds, "identity_sha256": identity_sha256}

    monkeypatch.setattr(screening_launcher, "summarize_update_evidence", optimizer_probe)
    monkeypatch.setattr(screening_launcher, "validate_checkpoint", checkpoint_probe)
    return EvidenceCase(tmp_path, options, records, diagnostics, updates, checkpoint_calls)


@pytest.mark.parametrize("stage", ["wiring", "pilot", "screening", "zero-shot"])
@pytest.mark.parametrize("arm,seed", [("A2", 42), ("M1", 43), ("M2-plain", 42), ("M2", 43)])
def test_accepts_only_complete_registered_stage_grid(tmp_path, monkeypatch, stage, arm, seed):
    case = _case(tmp_path, monkeypatch, stage=stage, arm=arm, seed=seed)
    result = case.validate()
    rounds = case.options["expected_rounds"]
    assert result["trajectories"] == (
        {"eval": 12}
        if stage == "zero-shot"
        else {
            "train": rounds * 8,
            "eval": 2 if stage in {"wiring", "pilot"} else 12,
        }
    )
    assert case.checkpoint_calls == (
        []
        if stage == "zero-shot"
        else [
            (tmp_path / "checkpoints" / f"global_step_{rounds}", EXECUTION),
        ]
    )
    first_group = [
        row
        for row in case.records
        if not row["identity"]["validate"] and row["identity"]["sample_index"] == 0
    ]
    if first_group:
        assert {row["identity"]["rollout_n"] for row in first_group} == {0, 1, 2, 3}
        assert {row["identity"]["step"] for row in first_group} == (
            {1, 2} if stage == "pilot" else {1}
        )
    assert {row["identity"]["step"] for row in case.records if row["identity"]["validate"]} == {
        0 if stage == "zero-shot" else rounds
    }


@pytest.mark.parametrize(
    "damage", ["swap_episodes", "wrong_policy_step", "source_index", "wrong_sibling"]
)
def test_correct_totals_cannot_hide_the_wrong_training_grid(tmp_path, monkeypatch, damage):
    case = _case(tmp_path, monkeypatch)
    if damage == "swap_episodes":
        case.records[0]["episode_id"], case.records[4]["episode_id"] = (
            case.records[4]["episode_id"],
            case.records[0]["episode_id"],
        )
    elif damage == "wrong_policy_step":
        case.records[0]["identity"]["step"] = 2
    elif damage == "source_index":
        # Dataset index 1 refers to formal-train-0005; source_index=4 is different.
        case.records[4]["identity"]["sample_index"] = 4
    else:
        case.records[0]["identity"]["rollout_n"] = 4
    with pytest.raises(RuntimeError, match="frozen episode/checkpoint grid"):
        case.validate()
    assert not case.checkpoint_calls


@pytest.mark.parametrize("replacement", [0, 15, 17])
def test_final_evaluation_must_be_after_update_16(tmp_path, monkeypatch, replacement):
    case = _case(tmp_path, monkeypatch)
    for row in case.records:
        if row["identity"]["validate"]:
            row["identity"]["step"] = replacement
    with pytest.raises(RuntimeError, match="frozen episode/checkpoint grid"):
        case.validate()


def test_pilot_two_updates_repeat_only_the_first_two_training_episodes(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch, stage="pilot")
    result = case.validate()
    assert result["trajectories"] == {"train": 16, "eval": 2}
    expected_episodes = {"formal-train-0001", "formal-train-0005"}
    assert set(case.options["train_episode_ids"]) == expected_episodes
    for step in (1, 2):
        training = [
            row
            for row in case.records
            if not row["identity"]["validate"] and row["identity"]["step"] == step
        ]
        assert len(training) == 8
        assert {row["episode_id"] for row in training} == expected_episodes
        assert {row["identity"]["sample_index"] for row in training} == {0, 1}
    assert case.updates["detected_role_counts"] == {"actor": 2, "critic": 2}
    assert case.checkpoint_calls == [(tmp_path / "checkpoints/global_step_2", EXECUTION)]


def test_pilot_missing_second_training_round_is_rejected(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch, stage="pilot")
    case.records = [
        row for row in case.records if row["identity"]["validate"] or row["identity"]["step"] != 2
    ]
    with pytest.raises(RuntimeError, match="count mismatch|frozen episode/checkpoint grid"):
        case.validate()
    assert not case.checkpoint_calls


def test_pilot_final_evaluation_at_update_one_is_rejected(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch, stage="pilot")
    for row in case.records:
        if row["identity"]["validate"]:
            row["identity"]["step"] = 1
    with pytest.raises(RuntimeError, match="frozen episode/checkpoint grid"):
        case.validate()
    assert not case.checkpoint_calls


def test_zero_shot_must_evaluate_step_zero(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch, stage="zero-shot")
    for row in case.records:
        row["identity"]["step"] = 16
    with pytest.raises(RuntimeError, match="frozen episode/checkpoint grid"):
        case.validate()


@pytest.mark.parametrize(
    "damage", ["missing_sibling", "duplicate_replaces_sibling", "missing_dev", "extra_dev"]
)
def test_incomplete_or_duplicated_grid_is_rejected(tmp_path, monkeypatch, damage):
    case = _case(tmp_path, monkeypatch)
    if damage == "missing_sibling":
        case.records.pop(3)
    elif damage == "duplicate_replaces_sibling":
        case.records[3] = copy.deepcopy(case.records[2])
    elif damage == "missing_dev":
        case.records.pop()
    else:
        case.records.append(copy.deepcopy(case.records[-1]))
    with pytest.raises(RuntimeError, match="count mismatch|duplicate screening"):
        case.validate()


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("arm", "A2", "arm/seed"),
        ("base_seed", 43, "arm/seed"),
        ("complete", False, "INCOMPLETE"),
        ("terminal_score", float("nan"), "finite common"),
        ("terminal_score", float("inf"), "finite common"),
    ],
)
def test_trajectory_provenance_completion_and_score_are_required(
    tmp_path,
    monkeypatch,
    field,
    value,
    match,
):
    case = _case(tmp_path, monkeypatch)
    case.records[-1][field] = value
    with pytest.raises(RuntimeError, match=match):
        case.validate()


def test_calibration_mismatch_fails_even_with_complete_finite_scores(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    case.records[-1]["metrics"]["calibration_sha256"] = "c" * 64
    with pytest.raises(RuntimeError, match="calibration identity"):
        case.validate()


@pytest.mark.parametrize("bad_score", [True, 0.0123])
def test_top_score_must_be_numeric_common_y_not_a2_raw_return(tmp_path, monkeypatch, bad_score):
    case = _case(tmp_path, monkeypatch, arm="A2")
    case.records[-1]["terminal_score"] = bad_score
    with pytest.raises(RuntimeError):
        case.validate()


@pytest.mark.parametrize("damage", ["missing", "duplicate", "wrong_step"])
def test_each_logical_update_requires_exactly_one_advantage_record(tmp_path, monkeypatch, damage):
    case = _case(tmp_path, monkeypatch)
    if damage == "missing":
        case.diagnostics.pop()
    elif damage == "duplicate":
        case.diagnostics[-1] = copy.deepcopy(case.diagnostics[-2])
    else:
        case.diagnostics[-1]["policy_step"] = 17
    with pytest.raises(RuntimeError, match="logical-batch advantage diagnostics"):
        case.validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("trajectory_count", 4),
        ("arm", "M1"),
        ("run_seed", 43),
        ("whitening_scope", "ppo_minibatch_tokens"),
        ("whitening_epsilon", 1e-6),
        ("whitening_correction", 0),
        ("phase", "after_minibatch_update"),
        ("schema_version", 2),
    ],
)
def test_plausible_diagnostics_from_another_contract_cannot_pass(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    case = _case(tmp_path, monkeypatch)
    case.diagnostics[0][field] = value
    with pytest.raises(RuntimeError, match="hook contract"):
        case.validate()


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_optimizer_roles_must_each_complete_all_rounds(tmp_path, monkeypatch, role):
    case = _case(tmp_path, monkeypatch)
    case.updates["detected_role_counts"][role] -= 1
    with pytest.raises(RuntimeError, match="optimizer update mismatch"):
        case.validate()
    assert not case.checkpoint_calls


def test_extra_undetected_optimizer_step_cannot_hide_behind_correct_role_counts(
    tmp_path,
    monkeypatch,
):
    case = _case(tmp_path, monkeypatch)
    case.updates["optimizer_step_count"] += 1
    with pytest.raises(RuntimeError, match="unexpected total optimizer update count"):
        case.validate()


@pytest.mark.parametrize("damage", ["optimizer", "advantages"])
def test_zero_shot_cannot_hide_training_work(tmp_path, monkeypatch, damage):
    case = _case(tmp_path, monkeypatch, stage="zero-shot")
    if damage == "optimizer":
        case.updates["optimizer_step_count"] = 1
    else:
        case.diagnostics.append({"policy_step": 0})
    with pytest.raises(RuntimeError, match="zero-shot unexpectedly"):
        case.validate()


def test_checkpoint_validator_failure_propagates_after_valid_evidence(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)

    def reject_checkpoint(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("checkpoint is missing data.pt")

    monkeypatch.setattr(screening_launcher, "validate_checkpoint", reject_checkpoint)
    with pytest.raises(RuntimeError, match="checkpoint is missing data.pt"):
        case.validate()

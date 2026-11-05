"""Separate business correctness from citations in actual tool environments."""

import copy
import json

import pytest

from src.agents.mandate_repair_collect import _safe_result
from src.agents.mandate_repair_env import MandateRepairEnv
from src.agents.mandate_repair_metrics import (
    business_score,
    constraint_extraction,
    local_difference,
)
from src.agents.mandate_repair_train import build_batch, recipe
from src.agents.mandate_repair_verifier import solve_explicit


@pytest.fixture
def task():
    constraints = dict(
        cash_min=1000.0,
        exposure_max=0.85,
        symbol_max={"A": 0.6, "B": 0.6},
        sector_max={"Technology": 0.4},
        sectors={"A": "Technology", "B": "Consumer"},
        cannot_sell=[],
        cannot_buy=[],
    )
    row = dict(
        task_id="business-check",
        task_type="mixed",
        as_of="2025-01-02T16:00:00+00:00",
        account=dict(cash=100.0, positions={"A": 80.0, "B": 10.0}, initial_equity=9100.0),
        quotes={"A": 100.0, "B": 100.0},
        execution=dict(commission_bps=10.0, slippage_bps=5.0),
        documents=[dict(
            doc_id="mandate", title="Current mandate", available_at="2025-01-01T00:00:00+00:00",
            effective_at="2025-01-01T00:00:00+00:00", text="Effective client constraints.",
        )],
    )
    row["gold"] = dict(
        constraints=constraints, required_citations=["mandate"],
        oracle=solve_explicit(
            row["account"], row["quotes"], constraints, execution=row["execution"]
        ),
    )
    return row


def outcome(task, *, mask=False, wrong_business=False):
    env = MandateRepairEnv(task)
    env.step(dict(name="read_document", arguments=dict(doc_id="mandate")), intervention=mask)
    constraints = copy.deepcopy(task["gold"]["constraints"])
    if wrong_business:
        constraints.update(cash_min=0.0, exposure_max=1.0, symbol_max={}, sector_max={})
    plan = env.step(dict(name="solve", arguments=dict(constraints=constraints)))
    env.step(dict(name="submit", arguments=dict(plan_id=plan["plan_id"], citations=["mandate"])))
    return _safe_result(dict(
        env=env, task_id=task["task_id"], trajectory_key="t/rep0", replicate_index=0,
        request_keys=[], failure=None,
    ))


def test_masked_evidence_alone_does_not_create_business_credit(task):
    parent, hidden = outcome(task), outcome(task, mask=True)
    assert parent["success"] is True and hidden["success"] is False
    assert parent["business_correct"] is True and hidden["business_correct"] is True
    assert hidden["failure_category"] == "evidence_only_failure"
    assert hidden["verification"]["evidence_errors"] == ["citation_not_observed"]
    assert local_difference(parent, [hidden], "reward")["unscaled_local_difference"] == 1.0
    business = local_difference(parent, [hidden], "business")
    assert business["unscaled_local_difference"] == 0.0
    assert business["main_reward_difference"] == 1.0


def test_real_executed_constraint_violation_produces_business_credit(task):
    parent, wrong = outcome(task), outcome(task, wrong_business=True)
    assert wrong["failure_category"] == "business_or_proof_failure"
    assert "sector_max:Technology" in wrong["verification"]["violations"]
    assert wrong["metrics"]["weights"]["A"] > 0.4
    assert parent["constraint_extraction"]["complete"] is True
    assert wrong["constraint_extraction"]["complete"] is False
    assert local_difference(parent, [wrong], "business")["unscaled_local_difference"] == 1.0


def test_no_submission_is_zero_but_system_failure_remains_missing(task):
    env = MandateRepairEnv(task)
    for _ in range(8):
        env.step(dict(name="calculate", arguments=dict(expression="1+1")))
    row = _safe_result(dict(
        env=env, task_id=task["task_id"], trajectory_key="t/rep0", replicate_index=0,
        request_keys=[], failure=None,
    ))
    assert row["business_score"] == 0.0
    assert row["failure_category"] == "no_submission"
    assert row["constraint_extraction"]["available"] is False
    assert business_score(dict(row, status="MODEL_ERROR")) is None
    assert business_score(dict(row, collector_failure={"status": "MODEL_ERROR"})) is None


def test_lucky_valid_plan_does_not_count_as_complete_clause_extraction(task):
    missing = copy.deepcopy(task["gold"]["constraints"])
    del missing["sector_max"]["Technology"]
    diagnostic = constraint_extraction(task, missing)
    assert diagnostic["available"] and diagnostic["complete"] is False
    assert diagnostic["missing_clauses"] == ["sector_max:Technology"]


def test_trainer_uses_binary_business_values_and_sham_ignores_them(tmp_path, task):
    success = outcome(task)
    parents, requests = [], []
    for i in range(4):
        key = f"t/rep{i}"
        parents.append(dict(success, task_id="t", trajectory_key=key, replicate_index=i))
        requests.append(dict(
            request_key=f"{key}/call0", trajectory_key=key,
            prompt_token_ids=[3, 4], response_token_ids=[5, 2],
            temperature=1.0, decode_mode="native",
        ))
    for name, rows in (("trajectory_results.jsonl", parents), ("requests.jsonl", requests)):
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    local = tmp_path / "local_credit.jsonl"
    item = dict(
        trajectory_key="t/rep0", request_key="t/rep0/call0", status="COMPLETE",
        continuation_policy="current_policy", **local_difference(
            success, [outcome(task, mask=True)], "business"
        ),
    )
    local.write_text(json.dumps(item) + "\n")
    no_effect, _ = build_batch(tmp_path, "m3", local, local_credit_target="business")
    assert all(row["advantage"] == 0 for group in no_effect.values() for row in group["requests"])
    item.update(local_difference(success, [outcome(task, wrong_business=True)], "business"))
    local.write_text(json.dumps(item) + "\n")
    m3, summary = build_batch(tmp_path, "m3", local, local_credit_target="business")
    assert m3["t/rep0"]["requests"][0]["raw_local_credit"] == 1.0
    assert summary["local_credit_targets"] == ["business"]
    sham, _ = build_batch(tmp_path, "sham", local, local_credit_target="business")
    assert all(row["advantage"] == 0 for group in sham.values() for row in group["requests"])
    with pytest.raises(ValueError, match="estimand"):
        build_batch(tmp_path, "m3", local, local_credit_target="reward")


def test_sft_initial_adapter_is_part_of_resume_recipe(tmp_path):
    from types import SimpleNamespace

    settings = SimpleNamespace(
        command="sft", model=tmp_path / "base", seed=43, learning_rate=5e-5,
        gradient_accumulation=16,
    )
    original = recipe(settings)
    assert "initial_adapter" not in original
    settings.adapter = tmp_path / "old_sft"
    assert recipe(settings)["initial_adapter"] == str(settings.adapter.resolve())


def paired_credit_batch(tmp_path, task):
    """A failed parent can have successful fresh factual continuations."""
    parent = outcome(task, wrong_business=True)
    parents, requests = [], []
    for i in range(4):
        key = f"t/rep{i}"
        parents.append(dict(parent, task_id="t", trajectory_key=key, replicate_index=i))
        for call in range(3):
            requests.append(dict(
                request_key=f"{key}/call{call}", trajectory_key=key,
                prompt_token_ids=[3, 4], response_token_ids=[5, 2],
                temperature=1.0, decode_mode="native",
            ))
    for name, rows in (("trajectory_results.jsonl", parents), ("requests.jsonl", requests)):
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    local = tmp_path / "paired_local.jsonl"
    item = dict(
        trajectory_key="t/rep0", request_key="t/rep0/call1", status="COMPLETE",
        continuation_policy="current_policy", local_credit_target="business",
        credit_estimator="paired_factual_mask_mean_v1", parent_target_value=0.0,
        factual_target_values=[1.0, 1.0], masked_target_values=[0.0, 0.0],
        factual_rewards=[1.0, 1.0], counterfactual_rewards=[0.0, 0.0],
        unscaled_local_difference=1.0,
    )
    local.write_text(json.dumps(item) + "\n")
    return local, item


def test_paired_credit_uses_fresh_factual_mean_not_single_failed_parent(tmp_path, task):
    local, _ = paired_credit_batch(tmp_path, task)
    groups, summary = build_batch(tmp_path, "m3", local, local_credit_target="business")
    rows = groups["t/rep0"]["requests"]
    assert [row["raw_local_credit"] for row in rows] == [0.0, 1.0, 0.0]
    assert summary["local_credit_estimators"] == ["paired_factual_mask_mean_v1"]
    sham, _ = build_batch(tmp_path, "sham", local, local_credit_target="business")
    assert all(row["advantage"] == 0.0 for group in sham.values() for row in group["requests"])


def test_shuffle_moves_credit_off_anchor_and_preserves_advantage_distribution(tmp_path, task):
    local, _ = paired_credit_batch(tmp_path, task)
    real, _ = build_batch(tmp_path, "m3", local, local_credit_target="business")
    shuffled, summary = build_batch(
        tmp_path, "shuffled", local, local_credit_target="business", credit_shuffle_seed=42
    )
    real_rows = [row for group in real.values() for row in group["requests"]]
    shuffled_rows = [row for group in shuffled.values() for row in group["requests"]]
    assert sorted(row["raw_combined_advantage"] for row in real_rows) == sorted(
        row["raw_combined_advantage"] for row in shuffled_rows
    )
    assert sorted(row["advantage"] for row in real_rows) == sorted(
        row["advantage"] for row in shuffled_rows
    )
    assert shuffled["t/rep0"]["requests"][1]["raw_local_credit"] == 0.0
    moved = summary["shuffled_local_assignments"]
    assert len(moved) == 1 and moved[0]["assigned_request_key"] != "t/rep0/call1"
    assert moved[0]["assigned_request_key"].startswith("t/rep0/")
    again, _ = build_batch(tmp_path, "shuffled", local, credit_shuffle_seed=42)
    assert [row["raw_local_credit"] for row in again["t/rep0"]["requests"]] == [
        row["raw_local_credit"] for row in shuffled["t/rep0"]["requests"]
    ]


def test_paired_credit_rejects_missing_or_inconsistent_actual_continuations(tmp_path, task):
    local, item = paired_credit_batch(tmp_path, task)
    item["factual_target_values"] = [1.0]
    local.write_text(json.dumps(item) + "\n")
    with pytest.raises(ValueError, match="two actual paired"):
        build_batch(tmp_path, "m3", local, local_credit_target="business")
    item["factual_target_values"] = [1.0, 1.0]
    item["unscaled_local_difference"] = 0.5
    local.write_text(json.dumps(item) + "\n")
    with pytest.raises(ValueError, match="recomputes"):
        build_batch(tmp_path, "m3", local, local_credit_target="business")

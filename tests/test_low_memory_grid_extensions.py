"""Guard date roles and actual optimizer settings when expanding Train evaluation."""

from dataclasses import replace

import pytest

from src.agents import low_memory_pilot as runner
from src.agents.parameter_grid import PROFILE_SEED42


def fixtures():
    rows = [
        dict(episode_id=episode_id, episode_dates=[f"{i}-{day}" for day in range(21)])
        for i, episode_id in enumerate(PROFILE_SEED42.eval_ids)
    ]
    return {"splits": {"train": rows}}, {"selected_episodes": {"dev": rows}}


def test_expanded_eval_has_exact_reserved_date_grid_without_outcome_selection():
    formal, old = fixtures()
    selected, evidence = runner.select_evaluation_episodes(formal, old, profile=PROFILE_SEED42)
    assert selected == formal["splits"]["train"]
    assert evidence["unique_evaluation_sessions"] == 252
    assert evidence["is_untouched_holdout"] is False


def test_eval_cannot_move_a_train_window_into_reserved_dev_silently():
    formal, old = fixtures()
    profile = replace(PROFILE_SEED42, eval_numbers=(1,))
    with pytest.raises(RuntimeError, match="reserved Train screen-dev"):
        runner.select_evaluation_episodes(formal, old, profile=profile)


def test_eval_date_overlap_is_rejected_even_when_episode_ids_are_different():
    formal, old = fixtures()
    formal["splits"]["train"][1]["episode_dates"][0] = "0-0"
    with pytest.raises(RuntimeError, match="disjoint 21-session"):
        runner.select_evaluation_episodes(formal, old, profile=PROFILE_SEED42)


def test_actual_actor_lr_cannot_disagree_with_the_declared_grid(tmp_path, monkeypatch):
    from src.agents import actor_update_probe, paired_validation

    spec = next(s for s in runner.cells(PROFILE_SEED42) if s.cell_id == "m1_lr1e7")
    monkeypatch.setattr(actor_update_probe, "validate_run_evidence", lambda *a, **kw: {})
    monkeypatch.setattr(paired_validation, "validate_run_evidence", lambda *a, **kw: {})
    monkeypatch.setattr(runner.common, "validate_common_penalty_evidence", lambda *a, **kw: {})
    # Counts are correct; only the actual optimizer LR is wrong.
    monkeypatch.setattr(
        runner, "read_jsonl", lambda path: [{"actor_learning_rates": [2e-7]}] * 4,
    )
    with pytest.raises(RuntimeError, match="actual actor optimizer learning rate"):
        runner.validate_extra_evidence(tmp_path, spec, "a" * 64, profile=PROFILE_SEED42)
    monkeypatch.setattr(
        runner, "read_jsonl", lambda path: [{"actor_learning_rates": [1e-7]}] * 4,
    )
    runner.validate_extra_evidence(tmp_path, spec, "a" * 64, profile=PROFILE_SEED42)


@pytest.mark.parametrize("indices", [(), (210, 210), (True,), (0,)])
def test_evaluation_window_identity_is_bounded(indices):
    with pytest.raises(ValueError, match="evaluation window"):
        replace(PROFILE_SEED42, eval_numbers=indices)


def test_incomplete_evaluation_never_becomes_zero_financial_return(tmp_path, monkeypatch):
    profile = replace(PROFILE_SEED42, eval_numbers=(210,))
    rows = [dict(
        identity=dict(validate=True, rollout_n=0, step=0),
        complete=False, status="INCOMPLETE", failure_kind="trajectory_max_steps",
        evaluation_failure_profile="retain_prefix_v1", metrics=None, terminal_score=None,
        steps=[{"invalid_action": True}] * 256,
    )]
    fields = ("raw_net_return", "transaction_cost", "terminal_score", "max_drawdown",
              "agent_volatility", "benchmark_excess", "turnover", "maximum_position_weight")
    rows.extend(dict(
        identity=dict(validate=True, rollout_n=lane, step=0), complete=True,
        status="COMPLETE", metrics=dict.fromkeys(fields, 0.03),
        steps=[{"invalid_action": False}] * 20,
    ) for lane in (1, 2, 3, 4))
    monkeypatch.setattr(runner, "read_jsonl", lambda path: rows)
    result = runner.evaluation_readout(tmp_path, policy_step=0, profile=profile)
    assert result["greedy"]["mean_raw_net_return"] is None
    assert result["greedy"]["trajectories"] == 0
    assert result["greedy"]["incomplete_trajectories"] == 1
    assert result["greedy"]["financial_status"] == "INCOMPLETE"
    assert result["sampled"]["mean_raw_net_return"] == pytest.approx(0.03)
    assert result["sampled"]["trajectories"] == 4
    rows[0]["metrics"] = dict.fromkeys(fields, 0.0)
    with pytest.raises(RuntimeError, match="unscored prefix"):
        runner.evaluation_readout(tmp_path, policy_step=0, profile=profile)

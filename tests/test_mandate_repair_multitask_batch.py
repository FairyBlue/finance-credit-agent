"""Mixed-task PPO batches preserve parent coverage, scale and recovery identity."""

import json
from types import SimpleNamespace

import pytest

from src.agents import mandate_repair_train as train


def arguments(tmp_path, **changes):
    values = dict(
        command="rl", base_model=tmp_path / "base", seed=42, arm="rloo",
        learning_rate=2e-6, clip_epsilon=0.2, local_weight=0.25, device="cpu",
        updates=None, trajectory_minibatch_size=4, mixed_task_minibatches=False,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def parents(task_count=8):
    return {
        f"task{task}/rep{replica}": dict(
            parent=dict(task_id=f"task{task}", replicate_index=replica),
            requests=[dict(advantage=0.0)],
        )
        for task in range(task_count)
        for replica in range(4)
    }


def test_mixed_batch_covers_32_parents_once_in_four_eight_task_updates(tmp_path):
    admitted = parents()
    args = arguments(tmp_path, trajectory_minibatch_size=8, mixed_task_minibatches=True)
    schedule = train.trajectory_schedule(admitted, args, "a" * 64)
    assert len(schedule) == 4
    assert [len(batch) for batch in schedule] == [8] * 4
    assert sorted(key for batch in schedule for key in batch) == sorted(admitted)
    for replica, batch in enumerate(schedule):
        assert {admitted[key]["parent"]["replicate_index"] for key in batch} == {replica}
        assert len({admitted[key]["parent"]["task_id"] for key in batch}) == 8
    assert train.trajectory_schedule(admitted, args, "a" * 64) == schedule


def test_mixed_schedule_rejects_silent_repetition_or_missing_tasks(tmp_path):
    args = arguments(tmp_path, trajectory_minibatch_size=8, mixed_task_minibatches=True)
    with pytest.raises(ValueError, match="task count"):
        train.trajectory_schedule(parents(7), args, "b" * 64)
    args.updates = 8
    with pytest.raises(ValueError, match="exactly once"):
        train.trajectory_schedule(parents(), args, "b" * 64)
    args.updates = None
    incomplete = parents()
    del incomplete["task0/rep0"]
    with pytest.raises(ValueError, match="one genuine parent"):
        train.trajectory_schedule(incomplete, args, "b" * 64)


def test_default_four_parent_schedule_keeps_historical_repeated_updates(tmp_path):
    args = arguments(tmp_path, updates=4)
    schedule = train.trajectory_schedule(parents(1), args, "c" * 64)
    assert schedule == [[f"task0/rep{replica}" for replica in range(4)]] * 4
    old = train.recipe(args)
    assert "trajectory_minibatch_size" not in old
    assert "mixed_task_minibatches" not in old
    args.trajectory_minibatch_size = 8
    args.mixed_task_minibatches = True
    assert train.recipe(args) != old


def test_each_trajectory_has_equal_mass_and_zero_advantages_do_not_rescale_it():
    request_counts = [1, 2, 3, 4, 5, 6, 7, 8]
    masses = [count * train.trajectory_request_weight(8, count) for count in request_counts]
    assert masses == pytest.approx([1 / 8] * 8)
    assert sum(masses) == pytest.approx(1)
    # Seven zero-advantage trajectories still belong in the selected denominator.
    assert train.trajectory_request_weight(8, 2) == 1 / 16
    with pytest.raises(ValueError, match="nonempty"):
        train.trajectory_request_weight(0, 2)


def test_fixed_probability_diagnostic_is_bounded_balanced_and_read_only(tmp_path):
    import copy
    import random

    rows = []
    for tool in ("calculate", "solve", "read_document"):
        for advantage in (-1.0, 0.0, 1.0):
            for index in range(4):
                rows.append(dict(
                    request_key=f"{tool}/{advantage}/{index}",
                    action=dict(name=tool), raw_combined_advantage=advantage,
                    advantage=advantage, hidden_gold="must not determine selection",
                ))
    original = copy.deepcopy(rows)
    rng_state = random.getstate()
    selected = train.diagnostic_requests(rows, 16)
    assert len(selected) == len({row["request_key"] for row in selected}) == 16
    assert {train.diagnostic_phase(row) for row in selected} == {
        "calculation", "execution", "evidence"
    }
    assert {row["raw_combined_advantage"] for row in selected} == {-1.0, 0.0, 1.0}
    assert train.diagnostic_requests(list(reversed(rows)), 16) == selected
    assert rows == original
    assert random.getstate() == rng_state
    assert train.diagnostic_requests(rows, 0) == []
    for limit in (-1, 17):
        with pytest.raises(ValueError, match="0..16"):
            train.diagnostic_requests(rows, limit)
    args = arguments(tmp_path)
    original_recipe = train.recipe(args)
    args.diagnostic_request_limit = 16
    # Read-only diagnostics can be enabled only on a later batch without
    # changing the persistent optimizer's mathematical recovery identity.
    assert train.recipe(args) == original_recipe


def test_probability_delta_reports_recorded_response_change_without_kl_claim():
    row = dict(
        request_key="task/rep0/call3", trajectory_key="task/rep0",
        action=dict(name="calculate"), raw_combined_advantage=0.25, advantage=0.5,
        prompt_token_ids=[1, 2, 3], response_token_ids=[4, 5],
    )
    result = train.probability_delta_record(row, -0.5, -0.4)
    assert result["mean_response_logprob_delta"] == pytest.approx(0.1)
    assert result["phase"] == "calculation"
    assert result["prompt_tokens"] == 3
    assert result["response_tokens"] == 2
    assert "not KL" in result["diagnostic"]


def test_collection_seed_contract_is_separate_from_persistent_optimizer_seed(tmp_path):
    args = arguments(tmp_path)
    collection = dict(seed=42, repetitions=4, temperature=1.0, decode_mode="native")
    assert train.validate_collection_sampling(collection, args) == 42
    collection["seed"] = 4301
    with pytest.raises(ValueError, match="actual collection seed"):
        train.validate_collection_sampling(collection, args)
    original_recipe = train.recipe(args)
    args.collection_seed = 4301
    assert train.validate_collection_sampling(collection, args) == 4301
    assert train.recipe(args) == original_recipe
    assert train.execution(args)["collection_seed"] == 4301
    args.collection_seed = 4302
    with pytest.raises(ValueError, match="actual collection seed"):
        train.validate_collection_sampling(collection, args)
    collection["seed"] = 4302
    assert train.validate_collection_sampling(collection, args) == 4302
    assert train.recipe(args) == original_recipe
    for invalid in (dict(repetitions=3), dict(temperature=0.8), dict(decode_mode="guided")):
        with pytest.raises(ValueError, match="native temperature-one K4"):
            train.validate_collection_sampling(dict(collection, **invalid), args)


def test_mixed_batch_recipe_is_checked_before_checkpoint_restore(tmp_path):
    import torch

    model = torch.nn.Linear(2, 1, bias=False)
    args = arguments(tmp_path, trajectory_minibatch_size=8, mixed_task_minibatches=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    checkpoint = tmp_path / "state.pt"
    train.save_checkpoint(
        checkpoint, model, optimizer, args, dict(completed_optimizer_steps=0), torch
    )
    recovered = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    assert train.restore_checkpoint(checkpoint, model, recovered, args, torch) == {
        "completed_optimizer_steps": 0
    }
    args.trajectory_minibatch_size = 4
    args.mixed_task_minibatches = False
    with pytest.raises(ValueError, match="resume recipe"):
        train.restore_checkpoint(checkpoint, model, recovered, args, torch)


@pytest.mark.parametrize("time_limit,expected_updates", [(4.0, 0), (6.0, 1), (None, 3)])
def test_sft_time_cap_counts_load_and_stops_at_complete_update(
    tmp_path, monkeypatch, time_limit, expected_updates
):
    import torch

    clock = SimpleNamespace(value=0.0)
    monkeypatch.setattr(train.time, "monotonic", lambda: clock.value)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.master = torch.nn.Parameter(torch.ones(1))

        def forward(self, **kwargs):
            clock.value += 2.0
            return SimpleNamespace(loss=self.master.square().sum())

    model = TinyModel()

    def load(*args):
        clock.value += 3.0
        return model, SimpleNamespace(pad_token_id=0), torch

    examples = [
        dict(id=str(i), input_ids=[3, 4, 2], labels=[-100, 4, 2],
             prompt_tokens=1, target_tokens=2)
        for i in range(6)
    ]

    def read(*args):
        clock.value += 2.0
        return examples

    monkeypatch.setattr(train, "load_model", load)
    monkeypatch.setattr(train, "read_examples", read)
    monkeypatch.setattr(train, "completion_logits_model", lambda value: value)
    monkeypatch.setattr(train, "collate_examples", lambda *args: {})
    monkeypatch.setattr(train, "export_adapter", lambda *args: "d" * 64)
    data = tmp_path / "teacher.jsonl"
    data.write_text(json.dumps(dict(id="teacher")) + "\n")
    args = arguments(
        tmp_path, command="sft", model=tmp_path / "base", output=tmp_path / "out",
        data=data, adapter=None, epochs=1, micro_batch=1, max_length=32,
        resume_state=None, state_output=None, actor_output=None, gradient_accumulation=2,
        checkpoint_every=1, max_training_seconds=time_limit,
    )
    summary = train.sft(args)
    assert summary["completed_optimizer_steps"] == expected_updates
    assert summary["cursor"] == expected_updates * 2
    assert summary["actual_supervised_prompt_tokens"] == expected_updates * 2
    assert summary["actual_supervised_target_tokens"] == expected_updates * 4
    if time_limit is None:
        assert summary["status"] == "COMPLETE"
        assert "max_training_seconds" not in summary
    else:
        assert summary["status"] == "COMPLETE_TIME_LIMIT"
        assert summary["elapsed_stage_seconds"] >= time_limit
        assert summary["time_limit_reached"]
        assert summary["time_limit_overrun_seconds"] == max(0, clock.value - time_limit)
    if time_limit == 6.0:
        # Restarting a capped stage must not grant another full time budget.
        args.resume_state = args.output / "training_state.pt"
        recovered = train.sft(args)
        assert recovered["completed_optimizer_steps"] == expected_updates
        assert recovered["cursor"] == expected_updates * 2
        assert len((args.output / "steps.jsonl").read_text().splitlines()) == expected_updates

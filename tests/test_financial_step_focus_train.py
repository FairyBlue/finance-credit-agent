"""Calculation-only actor updates preserve full-group advantage and loss scale."""

import json
from types import SimpleNamespace

import pytest

from src.agents import financial_step_focus_train as focus
from src.agents import mandate_repair_train as base


@pytest.fixture
def parent_batch(tmp_path):
    parents, requests = [], []
    for replica, reward in enumerate((0.98, 0.98, -0.02, -0.02)):
        key = f"task/rep{replica}"
        parents.append(
            dict(
                task_id="task",
                trajectory_key=key,
                replicate_index=replica,
                terminal_reward=reward,
                role="parent",
            )
        )
        tools = (
            ("read_document", "calculate", "calculate", "submit")
            if replica < 3
            else ("search_docs", "submit")
        )
        for call, tool in enumerate(tools):
            requests.append(
                dict(
                    trajectory_key=key,
                    request_key=f"{key}/call{call}",
                    prompt_token_ids=[3, 4],
                    response_token_ids=[5, 2],
                    action=dict(name=tool),
                    role="parent",
                    temperature=1.0,
                    decode_mode="native",
                )
            )
    for name, records in (("trajectory_results.jsonl", parents), ("requests.jsonl", requests)):
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in records))
    return tmp_path


def test_only_actual_calculations_keep_the_original_normalized_advantage(parent_batch):
    original, old_meta = base.build_batch(parent_batch)
    selected, metadata = focus.build_batch(parent_batch)
    assert metadata["advantages_mean"] == old_meta["advantages_mean"]
    assert metadata["advantages_std"] == old_meta["advantages_std"]
    assert selected.keys() == original.keys()
    for key, group in selected.items():
        assert len(group["requests"]) == len(original[key]["requests"])
        for new, old in zip(group["requests"], original[key]["requests"], strict=True):
            assert new["request_key"] == old["request_key"]
            assert new["raw_combined_advantage"] == old["raw_combined_advantage"]
            assert new["unmasked_actor_advantage"] == old["advantage"]
            expected = old["advantage"] if old["action"]["name"] == "calculate" else 0.0
            assert new["advantage"] == expected
    assert metadata["eligible_calculation_requests"] == 6
    assert metadata["suppressed_noncalculation_requests"] == 8
    assert metadata["effective_advantage_requests"] == 6
    assert all(r["advantage"] == 0 for r in selected["task/rep3"]["requests"])


def test_loss_denominator_retains_suppressed_requests_and_zero_update_parents(parent_batch):
    admitted, metadata = focus.build_batch(parent_batch)
    assert len(admitted) == 4
    assert metadata["full_parent_request_counts"] == {
        "task/rep0": 4,
        "task/rep1": 4,
        "task/rep2": 4,
        "task/rep3": 2,
    }
    assert base.trajectory_request_weight(4, len(admitted["task/rep0"]["requests"])) == 1 / 16
    expected = sum(
        abs(row["advantage"]) / (4 * len(group["requests"]))
        for group in admitted.values()
        for row in group["requests"]
    )
    assert metadata["retained_absolute_advantage_request_mass"] == pytest.approx(expected)
    assert (
        metadata["retained_absolute_advantage_request_mass"]
        < (metadata["unmasked_absolute_advantage_request_mass"])
    )


def test_focus_recipe_refuses_old_full_actor_checkpoint_but_resumes_its_own(tmp_path, monkeypatch):
    import torch

    args = SimpleNamespace(
        command="rl",
        base_model=tmp_path / "base",
        seed=42,
        arm="rloo",
        learning_rate=1e-5,
        device="cpu",
        clip_epsilon=0.2,
        local_weight=0.25,
        trajectory_minibatch_size=8,
        mixed_task_minibatches=True,
    )
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    old = tmp_path / "old.pt"
    base.save_checkpoint(old, model, optimizer, args, dict(completed_optimizer_steps=0), torch)
    assert "actor_update_scope" not in base.recipe(args)
    monkeypatch.setattr(base, "recipe", focus.recipe)
    assert base.recipe(args)["actor_update_scope"] == "calculate_only"
    with pytest.raises(ValueError, match="resume recipe"):
        base.restore_checkpoint(old, model, optimizer, args, torch)
    new = tmp_path / "focused.pt"
    base.save_checkpoint(new, model, optimizer, args, dict(completed_optimizer_steps=0), torch)
    assert base.restore_checkpoint(new, model, optimizer, args, torch) == {
        "completed_optimizer_steps": 0
    }

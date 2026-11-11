"""Invalid-output feedback preserves original advantages, loss scale and recipe identity."""

import json
from types import SimpleNamespace

import pytest

from src.agents import financial_invalid_feedback_train as invalid
from src.agents import financial_step_focus_train as focus
from src.agents import mandate_repair_train as base


@pytest.fixture
def parent_batch(tmp_path):
    parents, requests = [], []
    for replica, reward in enumerate((1.0, 0.0, -1.0, 0.0)):
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
        for call, action in enumerate(
            (dict(name="read_document"), dict(name="calculate"), None, dict(name="submit"))
        ):
            requests.append(
                dict(
                    trajectory_key=key,
                    request_key=f"{key}/call{call}",
                    prompt_token_ids=[3, 4],
                    response_token_ids=[5, 2],
                    action=action,
                    action_parse_error="INVALID_TOOL_JSON: malformed" if action is None else None,
                    role="parent",
                    temperature=1.0,
                    decode_mode="native",
                )
            )
    for name, records in (("trajectory_results.jsonl", parents), ("requests.jsonl", requests)):
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in records))
    return tmp_path


def test_invalid_outputs_keep_positive_negative_and_zero_original_advantages(parent_batch):
    original, old_meta = base.build_batch(parent_batch)
    admitted, metadata = invalid.build_batch(parent_batch)
    assert metadata["advantages_mean"] == old_meta["advantages_mean"]
    assert metadata["advantages_std"] == old_meta["advantages_std"]
    for key, group in admitted.items():
        for new, old in zip(group["requests"], original[key]["requests"], strict=True):
            eligible = old["action"] is None or old["action"]["name"] == "calculate"
            assert new["advantage"] == (old["advantage"] if eligible else 0.0)
            assert new["raw_combined_advantage"] == old["raw_combined_advantage"]
            assert new["unmasked_actor_advantage"] == old["advantage"]
    assert metadata["eligible_calculation_requests"] == 4
    assert metadata["eligible_parse_failure_requests"] == 4
    assert metadata["suppressed_other_requests"] == 8
    assert metadata["parse_failure_positive_actor_requests"] == 1
    assert metadata["parse_failure_negative_actor_requests"] == 1
    assert metadata["parse_failure_zero_actor_requests"] == 2
    assert metadata["effective_advantage_requests"] == 4


def test_original_request_denominator_and_full_parent_groups_are_retained(parent_batch):
    admitted, metadata = invalid.build_batch(parent_batch)
    assert len(admitted) == 4
    assert all(len(group["requests"]) == 4 for group in admitted.values())
    assert all(value == 4 for value in metadata["full_parent_request_counts"].values())
    assert base.trajectory_request_weight(4, len(admitted["task/rep0"]["requests"])) == 1 / 16
    expected = sum(
        abs(row["advantage"]) / (4 * len(group["requests"]))
        for group in admitted.values()
        for row in group["requests"]
    )
    assert metadata["retained_absolute_advantage_request_mass"] == pytest.approx(expected)
    assert metadata["parse_failure_absolute_advantage_request_mass"] > 0


def test_checkpoint_scope_rejects_full_actor_and_calculation_only_adam(tmp_path, monkeypatch):
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
    full = tmp_path / "full.pt"
    base.save_checkpoint(full, model, optimizer, args, dict(completed_optimizer_steps=0), torch)
    monkeypatch.setattr(base, "recipe", focus.recipe)
    focused = tmp_path / "focused.pt"
    base.save_checkpoint(focused, model, optimizer, args, dict(completed_optimizer_steps=0), torch)
    monkeypatch.setattr(base, "recipe", invalid.recipe)
    for old in (full, focused):
        with pytest.raises(ValueError, match="resume recipe"):
            base.restore_checkpoint(old, model, optimizer, args, torch)
    current = tmp_path / "current.pt"
    base.save_checkpoint(current, model, optimizer, args, dict(completed_optimizer_steps=0), torch)
    assert base.restore_checkpoint(current, model, optimizer, args, torch) == {
        "completed_optimizer_steps": 0
    }

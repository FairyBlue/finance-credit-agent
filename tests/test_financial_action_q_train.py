"""Fixed-action Q replaces only identified parent requests, with full history weights."""

import json
from types import SimpleNamespace

import pytest

from src.agents import financial_action_q_train as action_q
from src.agents import financial_invalid_feedback_train as invalid
from src.agents import mandate_repair_train as base


def write_rows(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


@pytest.fixture
def batch(tmp_path):
    parents, requests, estimates = [], [], []
    messages = [dict(role="system", content="rules"), dict(role="user", content="same problem")]
    history_sha = action_q._history_identity({"messages": messages})
    for replica, reward in enumerate((1.0, 1.0, 0.0, 0.0)):
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
            (
                dict(name="read_document"),
                dict(name="calculate") if replica != 3 else None,
                None,
                dict(name="submit"),
            )
        ):
            requests.append(
                dict(
                    trajectory_key=key,
                    request_key=f"{key}/call{call}",
                    prompt_token_ids=[3, 4],
                    response_token_ids=[5, 2],
                    action=action,
                    messages=messages,
                    role="parent",
                    temperature=1.0,
                    decode_mode="native",
                )
            )
        auxiliary = 1.0 if replica == 0 else 0.0
        estimates.append(
            dict(
                task_id="task",
                trajectory_key=key,
                parent_request_key=f"{key}/call1",
                status="COMPLETE",
                parent_return=reward,
                auxiliary_return=auxiliary,
                q_mean=(reward + auxiliary) / 2,
                adapter_sha256="a" * 64,
                estimator=action_q.ESTIMATOR,
                m=2,
                auxiliary_only=True,
                parent_action_executed_once=True,
                continuation_policy="current_policy",
                prefix_history_sha256=history_sha,
            )
        )
    write_rows(tmp_path / "trajectory_results.jsonl", parents)
    write_rows(tmp_path / "requests.jsonl", requests)
    (tmp_path / "collection_started.json").write_text(json.dumps({"adapter_sha256": "a" * 64}))
    values = tmp_path / "action_q_values.jsonl"
    write_rows(values, estimates)
    return tmp_path, values, estimates


def test_complete_q_group_replaces_targets_and_rewhitens_full_request_set(batch):
    directory, values, estimates = batch
    original, old_meta = base.build_batch(directory)
    admitted, metadata = action_q.build_batch(directory, action_q_values=values)
    q_total = sum(r["q_mean"] for r in estimates)
    all_rows = []
    for key, group in admitted.items():
        q = next(r["q_mean"] for r in estimates if r["trajectory_key"] == key)
        for new, old in zip(group["requests"], original[key]["requests"], strict=True):
            expected = (
                q - (q_total - q) / 3
                if new["request_key"].endswith("call1")
                else old["raw_combined_advantage"]
            )
            assert new["raw_combined_advantage"] == pytest.approx(expected)
            assert new["action_q_used"] == new["request_key"].endswith("call1")
            assert new["full_parent_request_count"] == 4
            all_rows.append(new)
    mean = sum(r["raw_combined_advantage"] for r in all_rows) / len(all_rows)
    assert metadata["advantages_mean"] == pytest.approx(mean)
    assert metadata["advantages_std"] != old_meta["advantages_std"]
    for row in all_rows:
        unmasked = (row["raw_combined_advantage"] - metadata["advantages_mean"]) / metadata[
            "advantages_std"
        ]
        eligible = row["action"] is None or row["action"].get("name") == "calculate"
        assert row["advantage"] == pytest.approx(unmasked if eligible else 0)
    assert metadata["replaced_target_requests"] == 4
    assert metadata["complete_q_task_groups"] == 1
    assert metadata["auxiliary_actor_training_examples"] == 0
    assert len(admitted) == 4 and len(all_rows) == 16
    assert base.trajectory_request_weight(4, len(admitted["task/rep0"]["requests"])) == 1 / 16
    assert metadata["data_sha256"] != old_meta["data_sha256"]


def test_incomplete_or_nonshared_q_group_falls_back_without_selecting_good_returns(batch):
    directory, values, estimates = batch
    original, _ = base.build_batch(directory)
    estimates[-1]["status"] = "UNSUPPORTED_BUDGET"
    estimates[-1]["q_mean"] = None
    write_rows(values, estimates)
    admitted, meta = action_q.build_batch(directory, action_q_values=values)
    assert meta["complete_q_task_groups"] == 0
    assert meta["unsupported_q_task_ids"] == ["task"]
    assert meta["replaced_target_requests"] == 0
    for key, group in admitted.items():
        assert [r["raw_combined_advantage"] for r in group["requests"]] == [
            r["raw_combined_advantage"] for r in original[key]["requests"]
        ]


@pytest.mark.parametrize("fault", ["adapter", "request", "return", "mean", "duplicate"])
def test_wrong_auxiliary_identity_or_arithmetic_is_rejected(batch, fault):
    directory, values, estimates = batch
    if fault == "adapter":
        estimates[0]["adapter_sha256"] = "b" * 64
    elif fault == "request":
        estimates[0]["parent_request_key"] = "another/request"
    elif fault == "return":
        estimates[0]["parent_return"] += 1
    elif fault == "mean":
        estimates[0]["q_mean"] += 1
    else:
        estimates.append(estimates[0])
    write_rows(values, estimates)
    with pytest.raises(ValueError):
        action_q.build_batch(directory, action_q_values=values)


def test_new_q_signal_can_unfreeze_original_zero_variance(batch):
    directory, values, estimates = batch
    parents = base.read_rows(directory / "trajectory_results.jsonl")
    for parent in parents:
        parent["terminal_reward"] = 0.0
    for row in estimates:
        row["parent_return"] = 0.0
        row["q_mean"] = row["auxiliary_return"] / 2
    write_rows(directory / "trajectory_results.jsonl", parents)
    write_rows(values, estimates)
    _, meta = action_q.build_batch(directory, action_q_values=values)
    assert meta["original_rloo_advantages_std"] == 0
    assert meta["advantages_std"] > 0
    assert meta["zero_raw_to_nonzero_targets"] == 4


def test_checkpoint_estimator_identity_rejects_v5_and_resumes_own_state(tmp_path, monkeypatch):
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
    monkeypatch.setattr(base, "recipe", invalid.recipe)
    old = tmp_path / "v5.pt"
    base.save_checkpoint(old, model, optimizer, args, dict(completed_optimizer_steps=0), torch)
    monkeypatch.setattr(base, "recipe", action_q.recipe)
    with pytest.raises(ValueError, match="resume recipe"):
        base.restore_checkpoint(old, model, optimizer, args, torch)
    current = tmp_path / "action_q.pt"
    base.save_checkpoint(current, model, optimizer, args, dict(completed_optimizer_steps=0), torch)
    assert (
        base.restore_checkpoint(current, model, optimizer, args, torch)["completed_optimizer_steps"]
        == 0
    )

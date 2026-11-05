"""Independent state baselines replace selected advantages, never actor data."""

import copy
import json

import pytest

from src.agents.mandate_repair_train import build_batch


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def batch(tmp_path):
    parents, requests = [], []
    for replica, reward in enumerate((0.9, 0.3, -0.01, 0.5)):
        key = f"t/rep{replica}"
        parents.append(dict(
            task_id="t", trajectory_key=key, replicate_index=replica,
            terminal_reward=reward, success=replica == 0, role="parent",
        ))
        for call in range(2):
            requests.append(dict(
                trajectory_key=key, request_key=f"{key}/call{call}",
                prompt_token_ids=[3, 4], response_token_ids=[5, 2],
                temperature=1.0, decode_mode="native",
            ))
    write_rows(tmp_path / "trajectory_results.jsonl", parents)
    write_rows(tmp_path / "requests.jsonl", requests)
    (tmp_path / "collection_started.json").write_text(json.dumps(dict(adapter_sha256="a" * 64)))
    item = dict(
        record_type="state_value_baseline", status="COMPLETE",
        parent_request_key="t/rep0/call1", request_key="t/rep0/call1",
        trajectory_key="t/rep0", m=2, aux_returns=[0.1, 0.3], mc_baseline=0.2,
        adapter_sha256="a" * 64, continuation_policy="current_policy",
        parent_action_excluded=True, parent_return_excluded=True,
        aux_rng_independent=True, selection_probability=0.25,
        aux_trajectory_keys=["mc/t/rep0/call1/rep0", "mc/t/rep0/call1/rep1"],
    )
    baseline = tmp_path / "state_value_baselines.jsonl"
    write_rows(baseline, [item])
    return tmp_path, baseline, item


def test_mc_replaces_baseline_and_keeps_every_parent_step(batch):
    directory, baseline, _ = batch
    ordinary, _ = build_batch(directory)
    groups, summary = build_batch(directory, "mc_uniform", mc_baselines=baseline)
    assert len(groups) == 4
    assert summary["parent_requests"] == 8
    assert summary["mc_baseline_requests"] == 1
    assert summary["mc_tokens_trained"] == 0
    assert summary["parent_prompt_tokens"] == summary["parent_response_tokens"] == 16
    selected = groups["t/rep0"]["requests"][1]
    assert selected["raw_combined_advantage"] == pytest.approx(0.9 - 0.2)
    assert selected["mc_baseline_used"] is True
    assert selected["raw_local_credit"] == 0
    assert selected["raw_combined_advantage"] != pytest.approx(
        selected["trajectory_advantage"] + 0.9 - 0.2
    )
    for key, group in groups.items():
        assert len(group["requests"]) == 2
        for actual, reference in zip(
            group["requests"], ordinary[key]["requests"], strict=True
        ):
            if actual["request_key"] != selected["request_key"]:
                assert actual["raw_combined_advantage"] == reference["raw_combined_advantage"]
                assert actual["mc_baseline_used"] is False


def test_mc_arms_share_estimator_normalization_and_actor_data(batch):
    directory, baseline, _ = batch
    uniform, a = build_batch(directory, "mc_uniform", mc_baselines=baseline)
    priority, b = build_batch(directory, "mc_priority", mc_baselines=baseline)
    assert uniform == priority
    assert a["mc_estimator"] == b["mc_estimator"]
    assert a["data_sha256"] == b["data_sha256"]


def test_mc_rejects_mismatched_policy_snapshot(batch):
    directory, baseline, item = batch
    item["adapter_sha256"] = "b" * 64
    write_rows(baseline, [item])
    with pytest.raises(ValueError, match="actual parent adapter snapshot"):
        build_batch(directory, "mc_uniform", mc_baselines=baseline)


@pytest.mark.parametrize("flag", [
    "parent_action_excluded", "parent_return_excluded", "aux_rng_independent",
])
def test_mc_requires_parent_action_return_and_rng_separation(batch, flag):
    directory, baseline, item = batch
    item[flag] = False
    write_rows(baseline, [item])
    with pytest.raises(ValueError, match="exclude the parent action/return"):
        build_batch(directory, "mc_uniform", mc_baselines=baseline)


def test_mc_rejects_duplicate_or_fictitious_parent_keys(batch):
    directory, baseline, item = batch
    write_rows(baseline, [item, item])
    with pytest.raises(ValueError, match="unique MC parent request"):
        build_batch(directory, "mc_uniform", mc_baselines=baseline)
    item["parent_request_key"] = item["request_key"] = "not-a-parent"
    write_rows(baseline, [item])
    with pytest.raises(ValueError, match="actual parent request"):
        build_batch(directory, "mc_uniform", mc_baselines=baseline)


def test_mc_rejects_parent_in_auxiliary_trajectory_keys(batch):
    directory, baseline, item = batch
    item["aux_trajectory_keys"][0] = item["trajectory_key"]
    write_rows(baseline, [item])
    with pytest.raises(ValueError, match="distinct and exclude the parent"):
        build_batch(directory, "mc_uniform", mc_baselines=baseline)


@pytest.mark.parametrize("changes, message", [
    ({"aux_returns": [0.1]}, "two actual finite"),
    ({"aux_returns": [0.1, float("nan")]}, "two actual finite"),
    ({"mc_baseline": 0.4}, "recomputes"),
    ({"selection_probability": 0.0}, "selection probability"),
])
def test_mc_validates_actual_returns_mean_and_selection_probability(batch, changes, message):
    directory, baseline, item = batch
    item.update(changes)
    write_rows(baseline, [item])
    with pytest.raises(ValueError, match=message):
        build_batch(directory, "mc_uniform", mc_baselines=baseline)


def test_incomplete_pair_falls_back_to_rloo_without_dropping_parent_data(batch):
    directory, baseline, item = batch
    item.update(status="INCOMPLETE", aux_returns=[0.1], mc_baseline=None)
    write_rows(baseline, [item])
    ordinary, _ = build_batch(directory)
    groups, summary = build_batch(directory, "mc_uniform", mc_baselines=baseline)
    assert summary["mc_baseline_requests"] == 0
    assert summary["unsupported_mc_parent_requests"] == [item["parent_request_key"]]
    assert summary["parent_requests"] == 8
    for key, group in groups.items():
        assert [row["advantage"] for row in group["requests"]] == [
            row["advantage"] for row in ordinary[key]["requests"]
        ]


def test_mc_records_are_part_of_batch_identity_and_cannot_mix_with_local_credit(batch):
    directory, baseline, item = batch
    _, before = build_batch(directory, "mc_uniform", mc_baselines=baseline)
    changed = copy.deepcopy(item)
    changed.update(aux_returns=[0.1, 0.5], mc_baseline=0.3)
    write_rows(baseline, [changed])
    _, after = build_batch(directory, "mc_uniform", mc_baselines=baseline)
    assert before["data_sha256"] != after["data_sha256"]
    with pytest.raises(ValueError, match="cannot mix with local credit"):
        build_batch(directory, "mc_uniform", local_credit=baseline, mc_baselines=baseline)

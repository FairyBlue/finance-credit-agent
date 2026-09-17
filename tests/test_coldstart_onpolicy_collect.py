"""Collection uses fixed original Train episodes, cash starts and isolated dates."""

import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from src.agents import coldstart_onpolicy_collect as collection
from src.agents.coldstart_eval import grid


def source_rows():
    days = [
        (datetime(2023, 1, 3, tzinfo=timezone.utc) + timedelta(days=index)).date().isoformat()
        for index in range(21)
    ]
    kwargs = {
        "episode_dates": [day + "T21:00:00+00:00" for day in days],
        "episode_open_times": [day + "T14:30:00+00:00" for day in days],
        "benchmark_closes": [1000.0] * 21,
        "symbols": ["GS", "MSFT", "HD"],
        "initial_cash": 1000,
        "screening": True,
        "commission_bps": 10,
        "slippage_bps": 5,
    }
    return [
        {
            "env_kwargs": json.dumps(kwargs),
            "extra_info": {"episode_id": episode, "source_split": "train"},
            "prompt": [{"role": "user", "content": "Original source prompt for " + episode}],
        }
        for episode in collection.TRAIN_IDS
    ]


def reserved(day="2023-05-23", time="20:00:00"):
    return [
        {
            "short": [],
            "full": [
                {
                    "env_kwargs": {
                        "episode_dates": [day + "T" + time + "+00:00"],
                        "episode_open_times": [day + "T13:30:00+00:00"],
                    }
                }
            ],
        }
    ]


def test_fixed_indices_full_original_prompt_and_cash_are_preserved_without_mutating_rows():
    rows = source_rows()
    old = copy.deepcopy(rows)
    contexts = collection.build_contexts(list(reversed(rows)), reserved())
    assert collection.COLLECTION_IDS == (
        "formal-train-0001",
        "formal-train-0013",
        "formal-train-0021",
        "formal-train-0033",
        "formal-train-0045",
        "formal-train-0057",
        "formal-train-0065",
        "formal-train-0077",
    )
    assert contexts["short"] == []
    assert len(contexts["full"]) == 8
    assert contexts["purpose"] == "TRAIN_COLLECTION"
    assert contexts["teacher_labels_included"] is False
    assert contexts["is_acceptance_evaluation"] is False
    by_id = {row["extra_info"]["episode_id"]: row for row in rows}
    for context, episode in zip(contexts["full"], collection.COLLECTION_IDS, strict=True):
        assert context["source_episode"] == episode
        assert context["raw_prompt"] == by_id[episode]["prompt"]
        assert context["prefix"] == []
        assert context["state_kind"] == "cash"
        assert context["expected_start_session_index"] == 0
        source = json.loads(by_id[episode]["env_kwargs"])
        assert context["env_kwargs"] == {
            **source,
            "observation_profile": "history_features_v1",
            "reward_mode": "finance_trajectory",
        }
        assert context["purpose"] == "TRAIN_COLLECTION"
    assert rows == old
    trajectories = grid(contexts)
    assert len(trajectories) == 40
    assert all(row["request_limit"] == 128 for row in trajectories)
    assert sum(row["sampling"] == "greedy" for row in trajectories) == 8
    assert sum(row["sampling"] == "sample" for row in trajectories) == 32


def test_train_split_and_exact_selected_population_are_required():
    rows = source_rows()
    rows[0]["extra_info"]["source_split"] = "valid"
    with pytest.raises(ValueError, match="outside original Train"):
        collection.build_contexts(rows, reserved())
    with pytest.raises(ValueError, match="missing"):
        collection.build_contexts(source_rows()[1:], reserved())
    with pytest.raises(ValueError, match="duplicate"):
        collection.build_contexts(source_rows() + [source_rows()[0]], reserved())


def test_calendar_overlap_is_detected_despite_different_close_hours():
    with pytest.raises(ValueError, match="overlap reserved"):
        collection.build_contexts(source_rows(), reserved("2023-01-20", "20:00:00"))


@pytest.mark.parametrize("failure", ["cutoff", "length", "open_date", "unordered", "timezone"])
def test_original_21_session_calendar_and_may_22_ceiling_are_enforced(failure):
    rows = source_rows()
    kwargs = json.loads(rows[0]["env_kwargs"])
    if failure == "cutoff":
        kwargs["episode_dates"][-1] = "2023-05-23T20:00:00+00:00"
        kwargs["episode_open_times"][-1] = "2023-05-23T13:30:00+00:00"
    elif failure == "length":
        kwargs["benchmark_closes"].pop()
    elif failure == "open_date":
        kwargs["episode_open_times"][-1] = "2023-01-30T13:30:00+00:00"
    elif failure == "unordered":
        kwargs["episode_dates"][0], kwargs["episode_dates"][1] = (
            kwargs["episode_dates"][1],
            kwargs["episode_dates"][0],
        )
    else:
        kwargs["episode_dates"][0] = "2023-01-03T21:00:00"
    rows[0]["env_kwargs"] = json.dumps(kwargs)
    with pytest.raises(ValueError):
        collection.build_contexts(rows, reserved())


def test_may_22_itself_is_allowed_and_source_prompt_identity_changes_with_prompt():
    rows = source_rows()
    kwargs = json.loads(rows[0]["env_kwargs"])
    kwargs["episode_dates"][-1] = "2023-05-22T20:00:00+00:00"
    kwargs["episode_open_times"][-1] = "2023-05-22T13:30:00+00:00"
    rows[0]["env_kwargs"] = json.dumps(kwargs)
    before = collection.build_contexts(rows, reserved())
    rows[0]["prompt"][0]["content"] += " source change"
    after = collection.build_contexts(rows, reserved())
    assert (
        before["full"][0]["source_row_inputs_sha256"]
        != (after["full"][0]["source_row_inputs_sha256"])
    )


def test_prepared_command_uses_v2_model_and_collection_population_only(tmp_path):
    command = collection.collection_command(tmp_path / "prepared")
    assert command[command.index("--model") + 1] == str(collection.MODEL)
    assert command[command.index("--expected-short") + 1] == "0"
    assert command[command.index("--expected-full") + 1] == "8"
    assert command[command.index("--contexts") + 1] == str(
        tmp_path / "prepared/collection_contexts.json"
    )
    assert command[command.index("--output") + 1] == str(tmp_path / "rollout")

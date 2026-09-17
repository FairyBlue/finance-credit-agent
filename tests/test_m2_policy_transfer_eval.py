"""Bounded reload contract checks without loading any model or GPU runtime."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "policy_transfer_entry",
    Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_policy_transfer_eval.py",
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def frames():
    result = {}
    for dataset, ids in (("train", entry.TRAIN_IDS), ("dev", entry.EVAL_IDS)):
        result[dataset] = [
            {
                "extra_info": {"episode_id": episode, "source_split": "train"},
                "env_kwargs": json.dumps(
                    {
                        "episode_dates": [f"{dataset}-{episode}-{i}" for i in range(21)],
                        "initial_cash": 1000000,
                        "commission_bps": 10,
                        "screening_calibration_sha256": "a" * 64,
                    }
                ),
                "prompt": [{"role": "user", "content": "Manage the portfolio."}],
            }
            for episode in ids
        ]
    return result


def outcome(contexts):
    rows = []
    for index, context in enumerate(contexts["full"]):
        for sibling in range(5):
            lane = "greedy" if sibling == 0 else f"sample{sibling - 1}"
            rows.append(
                {
                    "id": f"full-{context['id']}-{lane}",
                    "context": context,
                    "horizon": "full",
                    "sampling": "greedy" if sibling == 0 else "sample",
                    "identity": {
                        "step": 1,
                        "sample_index": index,
                        "rollout_n": max(0, sibling - 1),
                        "validate": sibling == 0,
                    },
                    "status": "COMPLETE",
                    "request_limit": 128,
                    "requests": 20,
                    "max_response_tokens": 30,
                    "max_prompt_tokens": 4000,
                    "prefix_fill_count": 0,
                    "prefix_info": [],
                    "takeover_state": {"session_index": 0, "positions": {}},
                    "final_state": {"session_index": 20},
                    "takeover_net_return": -0.01,
                    "terminal_metrics": {
                        "session_count": 21,
                        "terminal_score": -3.0,
                        "calibration_sha256": "a" * 64,
                    },
                }
            )
    return {"status": "OBSERVED", "action_protocol": entry.PROTOCOL, "grid": rows}


def test_exact_context_grid_retains_original_cash_windows_and_costs():
    contexts = entry.make_contexts(frames())
    assert contexts["short"] == []
    assert contexts["teacher_labels_included"] is False
    assert [c["id"] for c in contexts["full"]] == list(entry.TRAIN_IDS + entry.EVAL_IDS)
    assert [c["group"] for c in contexts["full"]] == ["training_windows"] * 8 + [
        "evaluation_windows"
    ] * 4
    for context in contexts["full"]:
        assert context["prefix"] == []
        assert context["state_kind"] == "cash"
        assert context["expected_start_session_index"] == 0
        assert context["env_kwargs"]["commission_bps"] == 10
        assert context["env_kwargs"]["observation_profile"] == "history_features_v1"
        assert context["env_kwargs"]["reward_mode"] == "finance_trajectory"
        assert len(context["env_kwargs"]["episode_dates"]) == 21


@pytest.mark.parametrize("damage", ["order", "split", "sessions", "date_overlap"])
def test_context_rejects_changed_population(damage):
    source = frames()
    if damage == "order":
        source["train"].reverse()
    elif damage == "split":
        source["train"][0]["extra_info"]["source_split"] = "test"
    else:
        kwargs = json.loads(source["dev"][0]["env_kwargs"])
        if damage == "sessions":
            kwargs["episode_dates"].pop()
        else:
            kwargs["episode_dates"] = json.loads(source["train"][0]["env_kwargs"])["episode_dates"]
        source["dev"][0]["env_kwargs"] = json.dumps(kwargs)
    with pytest.raises(RuntimeError):
        entry.make_contexts(source)


def test_fixed_subprocess_command_allows_only_the_registered_read_only_eval(tmp_path):
    model = {"model_path": "/fixed/actor/huggingface"}
    command = entry.cell_command(tmp_path, "m2_eta025", model)
    assert command[2:4] == ["-m", "src.agents.coldstart_eval"]
    options = dict(zip(command[4::2], command[5::2], strict=True))
    assert options == {
        "--model": model["model_path"],
        "--contexts": str(tmp_path / "contexts.json"),
        "--output": str(tmp_path / "m2_eta025"),
        "--action-protocol": entry.PROTOCOL,
        "--expected-short": "0",
        "--expected-full": "12",
    }
    with pytest.raises(ValueError):
        entry.cell_command(tmp_path, "new_model", model)
    assert entry.settings()["total_trajectories"] == 240
    assert entry.settings()["training_updates"] == entry.settings()["model_exports"] == 0


def test_accepts_all_60_complete_negative_outcomes(tmp_path):
    contexts = entry.make_contexts(frames())
    result = tmp_path / "result.json"
    entry.save(result, outcome(contexts))
    evidence = entry.validate_result(result, contexts)
    assert evidence["complete"] == evidence["planned"] == 60
    assert evidence["requests"] == 1200


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "duplicate",
        "incomplete",
        "response_cap",
        "prompt_cap",
        "request_cap",
        "prefix",
        "preexisting_position",
        "seed",
        "terminal_session",
        "nonfinite",
        "calibration",
        "context",
        "failure",
        "sampling",
    ],
)
def test_completion_rejects_partial_or_changed_evidence(tmp_path, damage):
    contexts = entry.make_contexts(frames())
    result = outcome(contexts)
    row = result["grid"][0]
    if damage == "missing":
        result["grid"].pop()
    elif damage == "duplicate":
        result["grid"][-1] = copy.deepcopy(row)
    elif damage == "incomplete":
        row["status"] = "INCOMPLETE"
    elif damage == "response_cap":
        row["max_response_tokens"] = 4096
    elif damage == "prompt_cap":
        row["max_prompt_tokens"] = 8193
    elif damage == "request_cap":
        row["requests"] = 129
    elif damage == "prefix":
        row["prefix_fill_count"] = 1
    elif damage == "preexisting_position":
        row["takeover_state"]["positions"] = {"GS": {"weight": 0.2}}
    elif damage == "seed":
        row["identity"]["step"] = 8
    elif damage == "terminal_session":
        row["final_state"]["session_index"] = 19
    elif damage == "nonfinite":
        row["takeover_net_return"] = "nan"
    elif damage == "calibration":
        row["terminal_metrics"]["calibration_sha256"] = "b" * 64
    elif damage == "context":
        row["context"] = {**row["context"], "prefix": ["hold"]}
    elif damage == "failure":
        row["failure"] = "REQUEST_CAP"
    else:
        row["sampling"] = "sample"
    path = tmp_path / "result.json"
    entry.save(path, result)
    with pytest.raises(RuntimeError):
        entry.validate_result(path, contexts)


def test_completed_directory_never_restarts(tmp_path):
    entry.save(tmp_path / "state.json", {"status": "COMPLETE"})
    with pytest.raises(RuntimeError, match="no retry or resume"):
        entry.run(tmp_path)


def test_prepare_never_overwrites_existing_directory(tmp_path):
    with pytest.raises(RuntimeError, match="no overwrite"):
        entry.prepare(tmp_path)


def test_sft_receipt_and_sizes_verified_without_reading_weight_bytes(tmp_path, monkeypatch):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{}")
    files = {
        p.name: {"bytes": p.stat().st_size, "sha256": entry.sha(p)} for p in tmp_path.iterdir()
    }
    entry.save(tmp_path / "merge_receipt.json", {"status": "COMPLETE", "files": files})
    receipt_sha = entry.sha(tmp_path / "merge_receipt.json")
    old_sha = entry.sha

    def guarded_sha(path):
        assert not str(path).endswith(".safetensors")
        return old_sha(path)

    monkeypatch.setattr(entry, "sha", guarded_sha)
    assert entry.check_sft(tmp_path, receipt_sha)["receipt_sha256"] == receipt_sha
    (tmp_path / "model.safetensors").write_bytes(b"changed-size")
    with pytest.raises(RuntimeError, match="size"):
        entry.check_sft(tmp_path, receipt_sha)


@pytest.mark.parametrize("first", ["4\n", "2048\n4\n"])
def test_waits_for_natural_gpu_cleanup_without_killing(first, monkeypatch):
    clock = [0.0]
    values = iter([first, "4\n4\n"])
    commands = []
    monkeypatch.setattr(entry.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        entry.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )

    def command(argv, **kwargs):
        commands.append(argv)
        assert 0 < kwargs["timeout"] <= 10
        return SimpleNamespace(stdout=next(values))

    monkeypatch.setattr(entry.subprocess, "run", command)
    result = entry.wait_gpu_idle()
    assert result["status"] == "IDLE"
    assert len(commands) == (1 if first == "4\n" else 2)
    assert all(command[0] == "nvidia-smi" for command in commands)


def test_busy_gpu_has_a_fixed_timeout_and_never_kills(monkeypatch):
    clock = [0.0]
    commands = []
    monkeypatch.setattr(entry.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        entry.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )

    def command(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(stdout="1024\n")

    monkeypatch.setattr(entry.subprocess, "run", command)
    with pytest.raises(RuntimeError, match="120 seconds"):
        entry.wait_gpu_idle()
    assert clock[0] == 120
    assert all(command[0] == "nvidia-smi" for command in commands)

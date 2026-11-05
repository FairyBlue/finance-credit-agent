"""Five validation lanes share native workers while training dispatch stays unchanged."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents import paired_validation as paired
from src.agents import rollout_seed


def worker_class():
    class Worker:
        def __init__(self):
            self.config = SimpleNamespace(
                actor_rollout_ref=SimpleNamespace(
                    rollout=SimpleNamespace(
                        n=4,
                        agent=SimpleNamespace(num_workers=1),
                        val_kwargs={
                            "n": 5,
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "top_k": -1,
                            "do_sample": True,
                        },
                    )
                )
            )
            self.calls = []
            self.failure = None
            self.retained_outputs = {}

        async def _run_agent_flow(
            self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs
        ):
            result = {
                "sampling_params": sampling_params,
                "trajectory": trajectory,
                "agent_name": agent_name,
                "trace": trace,
                "kwargs": kwargs,
            }
            self.calls.append(result)
            if self.failure:
                raise self.failure
            return self.retained_outputs.get(trajectory["rollout_n"], result)

    rollout_seed.patch_agent_flow_worker_class(Worker)
    return Worker


def identity(lane=0, *, validate=True, step=2):
    return {"step": step, "sample_index": 0, "rollout_n": lane, "validate": validate}


def kwargs():
    return {
        "agent_name": "trading",
        "index": 0,
        "uid": "fixed-window-group",
        "extra_info": {"episode_id": "formal-train-0113"},
    }


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCE_SCREENING_IDENTITY_SHA256", "e" * 64)
    monkeypatch.setenv("FINANCE_SCREENING_ARM", "M2")
    return tmp_path / "paired"


def records(directory):
    return [
        json.loads(line)
        for path in directory.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]


@pytest.mark.parametrize("arm", ["M1", "M2-plain", "M2"])
def test_five_lanes_only_override_decoding_and_preserve_seed_identity(evidence, monkeypatch, arm):
    monkeypatch.setenv("FINANCE_SCREENING_ARM", arm)
    Worker = worker_class()
    assert paired.patch_worker_class(Worker, evidence)
    worker = Worker()
    params = {"temperature": 1.0, "top_p": 1.0, "repetition_penalty": 1.0, "logprobs": True}
    before = copy.deepcopy(params)

    async def dispatch_all():
        return await asyncio.gather(
            *[worker._run_agent_flow(params, identity(lane), **kwargs()) for lane in range(5)]
        )

    outputs = asyncio.run(dispatch_all())
    assert params == before
    for lane, result in enumerate(outputs):
        forwarded = result["sampling_params"]
        assert forwarded["temperature"] == (0.0 if lane == 0 else 1.0)
        assert forwarded["top_p"] == forwarded["repetition_penalty"] == 1.0
        assert forwarded["top_k"] == -1
        assert (
            forwarded["min_p"]
            == forwarded["presence_penalty"]
            == forwarded["frequency_penalty"]
            == 0.0
        )
        assert forwarded["logprobs"] is True
        assert result["trajectory"] == identity(lane)
        assert result["kwargs"][rollout_seed.ROLLOUT_IDENTITY_KWARG] == identity(lane)
        assert result is worker.calls[lane]
    rows = sorted(records(evidence), key=lambda row: row["identity"]["rollout_n"])
    assert len(rows) == 5
    for lane, row in enumerate(rows):
        assert row["status"] == "COMPLETE" and row["validation_rollout_n"] == 5
        assert row["mode"] == ("greedy" if lane == 0 else "sampled")
        assert row["sampled_sibling"] == (None if lane == 0 else lane - 1)
        assert row["identity"] == identity(lane) and row["policy_step"] == 2
        assert row["execution_identity_sha256"] == "e" * 64
        assert row["arm"] == arm
        assert row["upstream_sampling_params"] == before
        assert row["seed_identity_preserved"] is True
    assert len({rollout_seed.derive_request_seed(42, identity(lane), 0) for lane in range(5)}) == 5
    assert rollout_seed.derive_request_seed(
        42, identity(1, step=0), 0
    ) != rollout_seed.derive_request_seed(42, identity(1, step=2), 0)


def test_training_bypasses_config_sampling_and_evidence(evidence, monkeypatch):
    Worker = worker_class()
    paired.patch_worker_class(Worker, evidence)
    worker = Worker()
    worker.config = None
    monkeypatch.delenv("FINANCE_SCREENING_ARM")
    params = {"temperature": 0.91, "other_training_field": [1, 2]}
    trajectory = identity(3, validate=False)
    result = asyncio.run(worker._run_agent_flow(params, trajectory, **kwargs()))
    assert result["sampling_params"] is params
    assert result["trajectory"] is trajectory
    assert not evidence.exists()


@pytest.mark.parametrize("damage", ["n", "temperature", "do_sample", "workers", "train_n"])
def test_wrong_remote_worker_config_fails_before_dispatch(evidence, damage):
    Worker = worker_class()
    paired.patch_worker_class(Worker, evidence)
    worker = Worker()
    rollout = worker.config.actor_rollout_ref.rollout
    if damage == "workers":
        rollout.agent.num_workers = 2
    elif damage == "train_n":
        rollout.n = 5
    else:
        rollout.val_kwargs[damage] = {"n": 1, "temperature": 0.0, "do_sample": False}[damage]
    with pytest.raises(RuntimeError):
        asyncio.run(
            worker._run_agent_flow({"temperature": 1.0, "top_p": 1.0}, identity(), **kwargs())
        )
    assert not worker.calls and not evidence.exists()


@pytest.mark.parametrize("lane", [-1, 5, True, 1.0, "1"])
def test_only_registered_lane_indices_are_allowed(lane):
    with pytest.raises(ValueError):
        paired.lane_settings(lane)


def test_runtime_failure_is_preserved_once_without_retry(evidence):
    Worker = worker_class()
    paired.patch_worker_class(Worker, evidence)
    worker = Worker()
    worker.failure = RuntimeError("original incomplete trajectory")
    with pytest.raises(RuntimeError, match="original incomplete"):
        asyncio.run(
            worker._run_agent_flow({"temperature": 1.0, "top_p": 1.0}, identity(), **kwargs())
        )
    assert len(worker.calls) == len(records(evidence)) == 1
    row = records(evidence)[0]
    assert row["status"] == "FAILED" and "original incomplete" in row["error"]


def retained_prefix():
    return SimpleNamespace(
        steps=[
            SimpleNamespace(
                reward_score=0.0,
                extra_fields={
                    "finance_grid_evaluation_failure": True,
                    "finance_grid_evaluation_transport_only": True,
                    "finance_grid_evaluation_failure_profile": "retain_prefix_v1",
                    "finance_grid_evaluation_failure_kind": "trajectory_max_steps",
                    "finance_screening_trajectory_complete": False,
                    "finance_screening_terminal_score": None,
                    "finance_screening_net_return": None,
                },
            )
            for _ in range(256)
        ]
    )


def test_paired_retained_prefix_stays_incomplete_and_requires_explicit_validator_opt_in(
    evidence, monkeypatch,
):
    monkeypatch.setenv("FINANCE_GRID_EVALUATION_INCOMPLETE", "retain_prefix_v1")
    Worker = worker_class()
    paired.patch_worker_class(Worker, evidence)
    worker = Worker()
    worker.retained_outputs[0] = retained_prefix()

    async def dispatch():
        for lane in range(5):
            await worker._run_agent_flow(
                {"temperature": 1.0, "top_p": 1.0}, identity(lane, step=0), **kwargs(),
            )

    asyncio.run(dispatch())
    row = next(row for row in records(evidence) if row["identity"]["rollout_n"] == 0)
    assert row["status"] == "INCOMPLETE" and row["terminal_score"] is row["metrics"] is None
    arguments = dict(
        expected_steps=[0], expected_episode_ids=["formal-train-0113"],
        arm="M2", execution_identity="e" * 64,
    )
    with pytest.raises(RuntimeError, match="actual paired"):
        paired.validate_run_evidence(evidence, **arguments)
    result = paired.validate_run_evidence(evidence, allow_incomplete=True, **arguments)
    assert result["execution_status"] == "PASS" and result["status"] == "INCOMPLETE"
    assert result["performance_status"] == "INCOMPLETE"
    assert result["complete_trajectories"] == 4 and result["incomplete_trajectories"] == 1
    assert len(worker.calls) == 5


@pytest.mark.parametrize("damage", ["profile", "score", "mixed", "count", "reward"])
def test_paired_dispatch_rejects_malformed_validation_failure_transport(
    evidence, monkeypatch, damage,
):
    monkeypatch.setenv("FINANCE_GRID_EVALUATION_INCOMPLETE", "retain_prefix_v1")
    Worker = worker_class()
    paired.patch_worker_class(Worker, evidence)
    worker = Worker()
    output = retained_prefix()
    if damage == "profile":
        output.steps[0].extra_fields["finance_grid_evaluation_failure_profile"] = "wrong"
    elif damage == "score":
        output.steps[0].extra_fields["finance_screening_terminal_score"] = 0.0
    elif damage == "mixed":
        output.steps[0].extra_fields["finance_grid_evaluation_failure"] = False
    elif damage == "count":
        output.steps.pop()
    else:
        output.steps[0].reward_score = 1.0
    worker.retained_outputs[0] = output
    with pytest.raises(RuntimeError, match="invalid retained"):
        asyncio.run(
            worker._run_agent_flow(
                {"temperature": 1.0, "top_p": 1.0}, identity(), **kwargs(),
            )
        )
    assert records(evidence)[0]["status"] == "FAILED"


def test_patch_is_idempotent_and_cannot_change_directory(evidence):
    Worker = worker_class()
    assert paired.patch_worker_class(Worker, evidence)
    assert not paired.patch_worker_class(Worker, evidence)
    with pytest.raises(RuntimeError, match="cannot change"):
        paired.patch_worker_class(Worker, evidence / "another")


def test_patch_requires_seed_wrapper_and_absolute_path(evidence):
    class Unseeded:
        async def _run_agent_flow(self, *args, **kwargs):
            pass

    with pytest.raises(RuntimeError, match="deterministic seed dispatch"):
        paired.patch_worker_class(Unseeded, evidence)
    with pytest.raises(ValueError, match="absolute"):
        paired.patch_worker_class(Unseeded, Path("relative"))


def test_environment_hook_is_absent_without_opt_in(monkeypatch):
    monkeypatch.delenv(paired.PROFILE_ENV, raising=False)
    previous = rollout_seed._patch_loaded_agent_flow_module
    assert paired.install_from_environment() is False
    assert rollout_seed._patch_loaded_agent_flow_module is previous


def test_environment_hook_chains_after_seed_import_without_extra_finder(evidence, monkeypatch):
    calls = []
    Worker = worker_class()
    module = SimpleNamespace(
        AgentFlowWorkerBase=Worker,
        AgentFlowWorker=SimpleNamespace(__ray_metadata__=SimpleNamespace(modified_class=Worker)),
    )
    monkeypatch.setenv(paired.PROFILE_ENV, paired.PROFILE)
    monkeypatch.setenv(paired.DIRECTORY_ENV, str(evidence))
    monkeypatch.setattr(rollout_seed, "_configured_evidence_root", evidence.parent / "seed")
    monkeypatch.setattr(
        rollout_seed, "_patch_loaded_agent_flow_module", lambda mod: calls.append(mod)
    )
    monkeypatch.delitem(paired.sys.modules, paired.TARGET_MODULE, raising=False)
    finders = list(paired.sys.meta_path)
    assert paired.install_from_environment()
    assert paired.install_from_environment()
    rollout_seed._patch_loaded_agent_flow_module(module)
    assert calls == [module]
    assert getattr(Worker._run_agent_flow, paired._MARKER) == str(evidence)
    assert paired.sys.meta_path == finders


def test_environment_hook_rejects_missing_seed_setup(evidence, monkeypatch):
    monkeypatch.setenv(paired.PROFILE_ENV, paired.PROFILE)
    monkeypatch.setenv(paired.DIRECTORY_ENV, str(evidence))
    monkeypatch.setattr(rollout_seed, "_configured_evidence_root", None)
    with pytest.raises(RuntimeError, match="rollout seeds"):
        paired.install_from_environment()


@pytest.mark.parametrize("damage", [None, "missing", "duplicate", "params", "episode", "failed"])
def test_run_evidence_requires_all_twenty_correct_actual_lanes(evidence, damage):
    Worker = worker_class()
    paired.patch_worker_class(Worker, evidence)
    worker = Worker()
    episodes = [f"formal-train-{index:04d}" for index in (113, 117, 121, 125)]

    async def dispatch_all():
        for index, episode in enumerate(episodes):
            for lane in range(5):
                trajectory = {**identity(lane, step=0), "sample_index": index}
                arguments = {**kwargs(), "index": index, "extra_info": {"episode_id": episode}}
                await worker._run_agent_flow(
                    {"temperature": 1.0, "top_p": 1.0},
                    trajectory,
                    **arguments,
                )

    asyncio.run(dispatch_all())
    if damage:
        path = next(evidence.glob("*.jsonl"))
        rows = records(evidence)
        if damage == "missing":
            rows.pop()
        elif damage == "duplicate":
            rows[-1] = rows[0]
        elif damage == "params":
            rows[0]["forwarded_sampling_params"]["temperature"] = 1.0
        elif damage == "episode":
            rows[0]["episode_id"] = episodes[-1]
        else:
            rows[0]["status"] = "FAILED"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with pytest.raises(RuntimeError):
            paired.validate_run_evidence(
                evidence,
                expected_steps=[0],
                expected_episode_ids=episodes,
                arm="M2",
                execution_identity="e" * 64,
            )
    else:
        result = paired.validate_run_evidence(
            evidence,
            expected_steps=[0],
            expected_episode_ids=episodes,
            arm="M2",
            execution_identity="e" * 64,
        )
        assert result["status"] == "PASS"
        assert result["trajectories"] == 20
        assert result["greedy_trajectories"] == 4 and result["sampled_trajectories"] == 16


@pytest.mark.parametrize("damage", [None, "missing", "duplicate"])
def test_twelve_windows_require_actual_lanes_at_both_policy_steps(evidence, damage):
    Worker = worker_class()
    paired.patch_worker_class(Worker, evidence)
    worker = Worker()
    episodes = [f"formal-train-{index:04d}" for index in range(190, 442, 21)]

    async def dispatch_all():
        for step in (2, 4):
            for index, episode in enumerate(episodes):
                for lane in range(5):
                    trajectory = {**identity(lane, step=step), "sample_index": index}
                    arguments = {
                        **kwargs(), "index": index, "extra_info": {"episode_id": episode}
                    }
                    await worker._run_agent_flow(
                        {"temperature": 1.0, "top_p": 1.0}, trajectory, **arguments
                    )

    asyncio.run(dispatch_all())
    if damage:
        path = next(evidence.glob("*.jsonl"))
        rows = records(evidence)
        if damage == "missing":
            rows.pop()
        else:
            rows[-1] = rows[0]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    arguments = dict(
        expected_steps=[2, 4], expected_episode_ids=episodes,
        arm="M2", execution_identity="e" * 64,
    )
    if damage:
        with pytest.raises(RuntimeError, match="missing|repeated|unexpected"):
            paired.validate_run_evidence(evidence, **arguments)
    else:
        result = paired.validate_run_evidence(evidence, **arguments)
        assert result["status"] == "PASS"
        assert result["trajectories"] == 120
        assert result["greedy_trajectories"] == 24
        assert result["sampled_trajectories"] == 96
        assert result["policy_steps"] == [2, 4]


@pytest.mark.parametrize("episodes", [[], ["formal-train-0190"] * 12])
def test_paired_grid_rejects_empty_or_duplicate_windows(evidence, episodes):
    with pytest.raises(ValueError, match="invalid expected paired validation grid"):
        paired.validate_run_evidence(
            evidence, expected_steps=[2, 4], expected_episode_ids=episodes,
            arm="M2", execution_identity="e" * 64,
        )

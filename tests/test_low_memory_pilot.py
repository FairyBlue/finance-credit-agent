"""Bounded low-memory pilot command, queue, and existing-evidence contracts."""

from __future__ import annotations

import copy
import json
import math
import subprocess

import pytest

hydra = pytest.importorskip("hydra")

from src.agents import low_memory_pilot as entry  # noqa: E402
from src.agents.metadata import RunHashes  # noqa: E402


@pytest.fixture
def prepared(tmp_path):
    model = {
        "source": "local_export", "name": str(tmp_path / "shared-sft"),
        "merge_receipt_sha256": "a" * 64,
    }
    settings = entry.settings(model)
    manifest = {
        "hashes": {"payload_sha256": "b" * 64},
        "datasets": {
            name: {"path": name + ".parquet", "episode_ids": ids, "row_count": len(ids)}
            for name, ids in (("train", entry.TRAIN_IDS), ("dev", entry.EVAL_IDS))
        },
        "calibration": {
            "path": "reward_calibration.json", "payload_sha256": "c" * 64, "sha256": "f" * 64,
        },
    }
    entry.atomic_json(tmp_path / "settings.json", settings)
    return tmp_path, settings, manifest


def command_for(prepared, cell_id):
    base, settings, manifest = prepared
    (base / cell_id).mkdir(exist_ok=True)
    config = entry.make_config(base, manifest, cell_id)
    command = entry.build_command(
        config=config, run_settings=entry.cell_settings(settings, cell_id),
        hashes=RunHashes("d" * 40, "e" * 40, "fixture", True), cell_dir=base / cell_id,
    )
    return config, command


@pytest.mark.parametrize("cell_id", ["m1", "m2_eta025"])
def test_real_hydra_preserves_training_and_selects_chunked_execution(prepared, cell_id):
    base, settings, _ = prepared
    config, command = command_for(prepared, cell_id)
    with hydra.initialize_config_dir(
        config_dir=str(entry.REPO_ROOT / "third_party/agent-r1/agent_r1/config"), version_base=None,
    ):
        actual = hydra.compose(config_name="agent_ppo_trainer", overrides=command[3:])
    model = settings["models"]["screening"]["name"]
    assert actual.actor_rollout_ref.model.path == actual.actor_rollout_ref.ref.model.path == model
    assert actual.critic.model.path == actual.critic.model.tokenizer_path == model
    assert actual.trainer.total_epochs == 1 and actual.trainer.total_training_steps == 2
    assert actual.trainer.test_freq == actual.trainer.save_freq == 2
    assert actual.trainer.val_only is False and actual.trainer.val_before_train is False
    assert actual.trainer.resume_mode == "disable" and actual.trainer.resume_from_path is None
    assert actual.data.train_max_samples == 4 and actual.data.train_batch_size == 2
    assert not actual.data.shuffle and not actual.data.validation_shuffle
    assert actual.data.apply_chat_template_kwargs.enable_thinking is False
    assert actual.actor_rollout_ref.rollout.n == 4
    assert actual.actor_rollout_ref.rollout.val_kwargs.n == 5
    assert actual.actor_rollout_ref.rollout.val_kwargs.do_sample is True
    assert actual.actor_rollout_ref.rollout.gpu_memory_utilization == 0.20
    assert actual.actor_rollout_ref.actor.optim.lr == 2e-7
    assert actual.critic.optim.lr == 1e-6
    for training in (actual.actor_rollout_ref.actor, actual.critic):
        assert training.ppo_epochs == 1 and training.ppo_mini_batch_size == 2
        assert dict(training.optim.override_optimizer_config) == {"foreach": False, "fused": False}
    for fsdp in (actual.actor_rollout_ref.actor.fsdp_config, actual.critic.model.fsdp_config):
        assert fsdp.param_offload is False and fsdp.optimizer_offload is False
        assert fsdp.seed == 42
    assert list(actual.actor_rollout_ref.actor.checkpoint.save_contents) == [
        "finance_actor_bf16_hf",
    ]
    assert list(actual.critic.checkpoint.save_contents) == []
    env = actual.ray_kwargs.ray_init.runtime_env.env_vars
    assert env.FINANCE_SCREENING_RESOURCE_PROFILE == "chunked_adam_disk_v1"
    assert env.FINANCE_SCREENING_ARTIFACT_PROFILE == "actor_bf16_hf_v1"
    assert str(base / cell_id / "artifacts/placement") == env.FINANCE_SCREENING_PLACEMENT_DIR
    assert env.FINANCE_SCREENING_CRITIC_INIT == "cpu_seed_v1"
    assert env.FINANCE_COMMON_INVALID_PENALTY == "0.05"
    assert env.FINANCE_PAIRED_VALIDATION_PROFILE == "greedy1_sample4_v1"
    assert float(env.FINANCE_SCREENING_RESIDUAL_ETA) == (0 if cell_id == "m1" else 0.25)
    for key, expected in {
        "CHUNK_BYTES": 64 * 1024**2,
        "GPU_RESERVE_BYTES": 2 * 1024**3,
        "HOST_RESERVE_BYTES": 2 * 1024**3,
        "DISK_RESERVE_BYTES": 8 * 1024**3,
    }.items():
        assert env["FINANCE_CHUNKED_ADAM_" + key] == str(expected)
    assert len(env.FINANCE_SCREENING_IDENTITY_SHA256) == 64
    assert config["screening"]["invalid_action_penalty"] == 0.05
    assert not any(name.startswith("FINANCE_LEARNING_") for name in env)


@pytest.mark.parametrize("cell_id", ["m1", "m2_eta025"])
def test_coordinated_workers_enable_vllm_profile_logs_without_changing_learning(
    prepared, monkeypatch, cell_id,
):
    from omegaconf import OmegaConf

    base, _, _ = prepared
    _, original_command = command_for(prepared, cell_id)
    entry.atomic_json(base / "gpu_coordinator_config.json", {"fixture": "coordinator"})
    monkeypatch.setattr(entry, "memory_coordinator_config", lambda path: {"fixture": True})
    _, command = command_for(prepared, cell_id)
    with hydra.initialize_config_dir(
        config_dir=str(entry.REPO_ROOT / "third_party/agent-r1/agent_r1/config"),
        version_base=None,
    ):
        original = hydra.compose(config_name="agent_ppo_trainer", overrides=original_command[3:])
        actual = hydra.compose(config_name="agent_ppo_trainer", overrides=command[3:])
    for group in ("data", "actor_rollout_ref", "critic", "algorithm", "trainer"):
        assert OmegaConf.to_container(actual[group], resolve=True) == OmegaConf.to_container(
            original[group], resolve=True,
        )
    assert actual.ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL == "INFO"
    key = "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL="
    assert [arg for arg in command if arg.startswith(key)] == [key + "INFO"]


def test_fixed_windows_two_arms_and_seventy_two_trajectories(prepared):
    _, settings, _ = prepared
    assert entry.coverage.TRAIN_IDS[:4] == entry.TRAIN_IDS
    assert entry.TRAIN_IDS == ["formal-train-0001", "formal-train-0065",
                               "formal-train-0009", "formal-train-0073"]
    assert entry.EVAL_IDS == ["formal-train-0113", "formal-train-0117",
                            "formal-train-0121", "formal-train-0125"]
    assert entry.EVAL_STEPS == [2]
    assert [(s.cell_id, s.arm, s.rounds) for s in entry.cells()] == [
        ("m1", "M1", 2), ("m2_eta025", "M2", 2),
    ]
    training_per_arm = settings["training"]["total_training_steps"] * 2 * 4
    evaluation_per_arm = len(entry.EVAL_IDS) * 5 * len(entry.EVAL_STEPS)
    assert training_per_arm == 16 and evaluation_per_arm == 20
    assert len(entry.cells()) * (training_per_arm + evaluation_per_arm) == 72
    assert settings["training_coverage"]["planned_trajectories"] == 72
    assert settings["training_coverage"]["automatic_retry"] is False
    assert settings["training_coverage"]["resume_supported"] is False


def test_handoff_is_bound_to_execution_config_and_real_ray_environment(prepared):
    from src.agents import gpu_handoff

    base, settings, _ = prepared
    path = base / "gpu_handoff_config.json"
    entry.atomic_json(path, dict(
        profile=gpu_handoff.PROFILE, directory=str(base / "holder"),
        owner_sha256="a" * 64, pid=123, pid_start_ticks=456,
        gpu_uuid="GPU-fixture", cuda_device=0, minimum_free_bytes=60 * 1024**3,
    ))
    config, command = command_for(prepared, "m1")
    digest = entry.restart.digest(path)
    assert config["screening"]["gpu_handoff_config_sha256"] == digest
    with hydra.initialize_config_dir(
        config_dir=str(entry.REPO_ROOT / "third_party/agent-r1/agent_r1/config"), version_base=None,
    ):
        actual = hydra.compose(config_name="agent_ppo_trainer", overrides=command[3:])
    env = actual.ray_kwargs.ray_init.runtime_env.env_vars
    assert str(path) == env.FINANCE_GPU_HANDOFF_CONFIG
    assert digest == env.FINANCE_GPU_HANDOFF_CONFIG_SHA256
    assert str(base / "m1/artifacts/gpu_handoff") == env.FINANCE_GPU_HANDOFF_EVIDENCE_DIR
    handoff = entry.restart.read(path)
    handoff["pid"] = 999
    entry.atomic_json(path, handoff)
    with pytest.raises(RuntimeError, match="differs from execution identity"):
        entry.build_command(
            config=config, run_settings=entry.cell_settings(settings, "m1"),
            hashes=RunHashes("d" * 40, "e" * 40, "fixture", True), cell_dir=base / "m1",
        )


@pytest.mark.parametrize("where,field,value", [
    ("training", "total_training_steps", 4),
    ("training", "learning_rate", 1e-6),
    ("resources", "chunk_bytes", 32 * 1024**2),
    ("resources", "profile", "pageable_adam_v1"),
])
def test_mutated_resource_or_budget_settings_rejected(prepared, where, field, value):
    base, settings, manifest = prepared
    config = entry.make_config(base, manifest, "m1")
    run_settings = entry.cell_settings(settings, "m1")
    run_settings[where][field] = value
    with pytest.raises(RuntimeError, match="condition differs"):
        entry.build_command(
            config=config, run_settings=run_settings,
            hashes=RunHashes("d" * 40, "e" * 40, "fixture", True), cell_dir=base / "m1",
        )


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_queue_requires_sixty_gib_free_then_admits_once(tmp_path, monkeypatch):
    clock = Clock()
    telemetry = iter([60 * 1024 - 1, 60 * 1024])
    monkeypatch.setattr(entry, "time", clock)
    monkeypatch.setattr(entry, "gpu_free_mib", lambda: next(telemetry))
    entry.wait_gpu(tmp_path, "m1", timeout=100)
    result = entry.restart.read(tmp_path / "gpu_queue_state.json")
    assert result["status"] == "ADMITTED" and result["cell"] == "m1"
    assert result["free_MiB"] == result["minimum_free_MiB"] == 60 * 1024
    assert result["waited_seconds"] == 45
    assert clock.sleeps == [45]
    assert result["automatic_retry"] is False and result["stops_other_processes"] is False


@pytest.mark.parametrize("telemetry_failure", [False, True])
def test_queue_timeout_preserves_reason_without_launching(tmp_path, monkeypatch, telemetry_failure):
    clock = Clock()
    monkeypatch.setattr(entry, "time", clock)

    def reading():
        if telemetry_failure:
            raise subprocess.TimeoutExpired("nvidia-smi", 15)
        return 60 * 1024 - 1

    monkeypatch.setattr(entry, "gpu_free_mib", reading)
    with pytest.raises(RuntimeError, match="queue limit"):
        entry.wait_gpu(tmp_path, "m2_eta025", timeout=50)
    result = entry.restart.read(tmp_path / "gpu_queue_state.json")
    assert result["status"] == "WAIT_TIMEOUT" and result["waited_seconds"] == 50
    assert clock.sleeps == [45, 5]
    assert bool(result["last_telemetry_error"]) == telemetry_failure
    assert not (tmp_path / "m2_eta025").exists()


@pytest.mark.parametrize("reading", ["N/A\n", "1000\n2000\n", "-1\n", ""])
def test_ambiguous_gpu_readings_do_not_admit(monkeypatch, reading):
    monkeypatch.setattr(entry.subprocess, "check_output", lambda *a, **kw: reading)
    with pytest.raises(RuntimeError, match="expected one GPU"):
        entry.gpu_free_mib()


def test_existing_single_step_evidence_readout_preserves_sampled_and_greedy(monkeypatch):
    directory = entry.REPO_ROOT / "runs/m2-train-coverage-20260916-v2/m1"
    paths = list((directory / "artifacts/screening_evidence").glob("*.jsonl"))
    if not paths:
        pytest.skip("requires existing local coverage evidence, not distributed in a clean clone")
    rows = []
    for path in paths:
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    chosen = [row for row in rows if row["identity"]["validate"] and row["identity"]["step"] == 2]
    assert len(chosen) == 20
    monkeypatch.setattr(entry, "read_jsonl", lambda path: copy.deepcopy(chosen))
    readout = entry.evaluation_readout(directory)
    assert readout["greedy"]["trajectories"] == 4
    assert readout["sampled"]["trajectories"] == 16
    for name, lanes in (("greedy", {0}), ("sampled", {1, 2, 3, 4})):
        expected = [row for row in chosen if row["identity"]["rollout_n"] in lanes]
        for field in ("raw_net_return", "transaction_cost", "terminal_score"):
            average = math.fsum(row["metrics"][field] for row in expected) / len(expected)
            assert readout[name]["mean_" + field] == average
        assert readout[name]["invalid_decisions"] == sum(
            step["invalid_action"] for row in expected for step in row["steps"]
        )


def test_status_reports_wait_and_never_claims_finished_result(tmp_path):
    entry.atomic_json(tmp_path / "state.json", {
        "status": "RUNNING", "current_cell": "m1", "planned_trajectories": 72,
    })
    entry.atomic_json(tmp_path / "gpu_queue_state.json", {
        "status": "WAITING_FOR_GPU", "cell": "m1", "free_MiB": 31 * 1024,
    })
    entry.refresh_status(tmp_path)
    text = (tmp_path / "LIVE_STATUS.md").read_text()
    assert "WAITING_FOR_GPU" in text and "31.00 GiB" in text and "72" in text
    assert "新加坡时间" in text and "0/36" in text
    assert not (tmp_path / "FINAL_RESULTS.md").exists()


def test_run_refuses_prior_state_even_when_previous_run_failed(tmp_path, monkeypatch):
    entry.atomic_json(tmp_path / "state.json", {"status": "FAILED"})
    monkeypatch.setattr(entry, "check_contract", lambda base: pytest.fail("must not retry"))
    with pytest.raises(RuntimeError, match="no implicit retry"):
        entry.run(tmp_path)

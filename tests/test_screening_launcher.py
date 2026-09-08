"""Execution contracts for the history-only screening launcher and readout."""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from ray.runtime_env import RuntimeEnv

from src.agents import screening_launcher
from src.agents.metadata import RunHashes
from src.agents.screening_launcher import (
    ARMS,
    REPO_ROOT,
    build_cell_command,
    build_cell_config,
    capture_process,
    set_overrides,
)
from src.evaluation.screening import paired_summary


@pytest.mark.parametrize("trainer_exitcode", (0, 17))
@pytest.mark.parametrize(
    ("gpu_result", "error_type"),
    (
        (subprocess.TimeoutExpired("nvidia-smi", 15), "TimeoutExpired"),
        (subprocess.CompletedProcess("nvidia-smi", 9, "", "NVML unavailable"), "NonzeroExit"),
        (subprocess.CompletedProcess("nvidia-smi", 9, "NVML unavailable", ""), "NonzeroExit"),
        (FileNotFoundError("nvidia-smi is missing"), "FileNotFoundError"),
        (OSError("NVML access failed"), "OSError"),
    ),
)
def test_gpu_telemetry_failure_keeps_trainer_running_and_preserves_its_exitcode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    gpu_result: subprocess.CompletedProcess[str] | Exception,
    error_type: str, trainer_exitcode: int,
):
    """A failed sample must survive a still-running trainer and a later good sample."""
    command = ["fake-trainer"]

    class Trainer:
        pid = 12345
        returncode: int | None = None

        def __init__(self) -> None:
            self.wait_timeouts: list[float | None] = []

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            self.wait_timeouts.append(timeout)
            if len(self.wait_timeouts) == 1:
                assert timeout is not None
                raise subprocess.TimeoutExpired(command, timeout)
            self.returncode = trainer_exitcode
            return self.returncode

    trainer = Trainer()
    popen = Mock(return_value=trainer)
    gpu_query = Mock(side_effect=[
        gpu_result,
        subprocess.CompletedProcess("nvidia-smi", 0, "24000, 97\n", ""),
    ])
    killpg = Mock()
    monkeypatch.setattr(screening_launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(screening_launcher.subprocess, "run", gpu_query)
    monkeypatch.setattr(screening_launcher.os, "killpg", killpg)
    monkeypatch.setattr(screening_launcher, "utc_now", lambda: "fake-clock")

    assert capture_process(command, environment={}, cell_dir=tmp_path) == trainer_exitcode
    assert trainer.wait_timeouts == [10, 10]
    assert gpu_query.call_count == 2
    killpg.assert_not_called()
    rows = [json.loads(line) for line in (tmp_path / "resources.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["gpu_error"]["type"] == error_type
    assert rows[0]["gpu_error"]["message"]
    assert "gpu" not in rows[0]
    if isinstance(gpu_result, subprocess.CompletedProcess):
        assert rows[0]["gpu_error"]["returncode"] == 9
        assert rows[0]["gpu_error"]["message"] == "NVML unavailable"
    assert rows[1]["gpu"] == "24000, 97"
    assert "gpu_error" not in rows[1]
    assert all(row["pid"] == trainer.pid and row["time"] == "fake-clock" for row in rows)


def test_unexpected_capture_error_still_stops_trainer_and_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    trainer = Mock(pid=12345)
    trainer.poll.return_value = None
    trainer.wait.return_value = -signal.SIGTERM
    killpg = Mock()
    monkeypatch.setattr(screening_launcher.subprocess, "Popen", Mock(return_value=trainer))
    monkeypatch.setattr(
        screening_launcher.subprocess, "run", Mock(side_effect=RuntimeError("unexpected bug")),
    )
    monkeypatch.setattr(screening_launcher.os, "killpg", killpg)
    monkeypatch.setattr(screening_launcher, "utc_now", lambda: "fake-clock")

    with pytest.raises(RuntimeError, match="unexpected bug"):
        capture_process(["fake-trainer"], environment={}, cell_dir=tmp_path)
    killpg.assert_called_once_with(trainer.pid, signal.SIGTERM)
    trainer.wait.assert_called_once_with(timeout=30)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("stage", ("wiring", "pilot", "screening", "zero-shot"))
@pytest.mark.parametrize("enable_thinking", (None, False, True))
def test_screening_commands_have_shared_exact_contract(
    tmp_path: Path, arm: str, stage: str, enable_thinking: bool | None,
):
    settings = yaml.safe_load((REPO_ROOT / "configs/m2_screening.yaml").read_text())
    settings.pop("generation", None)
    if enable_thinking is not None:
        settings["generation"] = {"enable_thinking": enable_thinking}
    manifest = {
        "hashes": {"payload_sha256": "a" * 64},
        "datasets": {
            name: {"path": f"{name}.parquet"}
            for name in ("train", "dev", "pilot_train", "pilot_eval")
        },
        "calibration": {"path": "reward_calibration.json", "sha256": "b" * 64},
    }
    config = build_cell_config(
        settings=settings,
        manifest=manifest,
        prepared_dir=tmp_path,
        arm=arm,
        stage=stage,
        seed=42,
        cell_dir=tmp_path,
    )
    command = build_cell_command(
        config=config,
        settings=settings,
        hashes=RunHashes("c" * 40, "d" * 40, "test", False),
        cell_dir=tmp_path,
    )
    options = dict(argument.lstrip("+").split("=", 1) for argument in command[3:])
    template_key = "data.apply_chat_template_kwargs.enable_thinking"
    if enable_thinking is None:
        assert template_key not in options
        assert "apply_chat_template_kwargs" not in config["data"]
    else:
        assert options[template_key] == str(enable_thinking)
        assert config["data"]["apply_chat_template_kwargs"] == {
            "enable_thinking": enable_thinking,
        }
    assert options["actor_rollout_ref.rollout.n"] == "4"
    assert options["actor_rollout_ref.rollout.agent.num_workers"] == "1"
    assert options["actor_rollout_ref.rollout.max_model_len"] == "8192"
    assert options["actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len"] == "8192"
    assert options["actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs"] == "8"
    assert options["actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config"] == (
        "{cudagraph_capture_sizes:[1,2,4,8]}"
    )
    assert options["actor_rollout_ref.rollout.gpu_memory_utilization"] == "0.2"
    for role in ("actor_rollout_ref.actor.fsdp_config", "critic.model.fsdp_config"):
        assert options[f"{role}.param_offload"] == "False"
        assert options[f"{role}.optimizer_offload"] == "False"
    env_prefix = "ray_kwargs.ray_init.runtime_env.env_vars."
    assert options[env_prefix + "FINANCE_SCREENING_RESOURCE_PROFILE"] == "pageable_adam_v1"
    assert options[env_prefix + "FINANCE_SCREENING_PLACEMENT_DIR"] == str(
        tmp_path / "artifacts/placement"
    )
    assert options["algorithm.gamma"] == options["algorithm.lam"] == "1.0"
    assert options["critic.ppo_mini_batch_size"] == "2"
    assert options["actor_rollout_ref.actor.ppo_mini_batch_size"] == "2"
    assert options["actor_rollout_ref.actor.policy_loss.loss_mode"] == "gspo"
    assert options["trainer.test_freq"] == (
        "1" if stage == "wiring" else "2" if stage == "pilot" else "16"
    )
    assert options["trainer.total_epochs"] == ("2" if stage == "pilot" else "1")
    assert options["trainer.save_freq"] == ("1" if stage == "pilot" else "4")
    for role in ("actor_rollout_ref.actor", "critic"):
        assert options[f"{role}.optim.override_optimizer_config"] == "{foreach:false,fused:false}"
    assert options["trainer.val_before_train"] == ("True" if stage == "zero-shot" else "False")
    assert options["trainer.val_only"] == ("True" if stage == "zero-shot" else "False")
    assert options["data.shuffle"] == "False"
    assert "test_file" not in config["data"]["processed"]
    assert "stockbench" not in str(config["data"])
    assert "valid.parquet" not in str(config["data"])
    assert (
        config["model"]["revision"]
        == settings["models"]["wiring" if stage == "wiring" else "screening"]["revision"]
    )
    assert options["ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_SCREENING_ARM"] == arm
    assert (
        json.loads(options["ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_SCREENING_RUN_SEED"])
        == "42"
    )
    flow = yaml.safe_load((tmp_path / "flow.yaml").read_text())[0]
    assert flow["screening_arm"] == arm


@pytest.mark.parametrize("enable_thinking", (None, False, True))
def test_real_hydra_compose_produces_ray_compatible_string_environment(
    tmp_path: Path, enable_thinking: bool | None,
):
    """Numeric-looking argv strings must survive Hydra before Ray validates them."""
    settings = yaml.safe_load((REPO_ROOT / "configs/m2_screening.yaml").read_text())
    settings.pop("generation", None)
    if enable_thinking is not None:
        settings["generation"] = {"enable_thinking": enable_thinking}
    manifest = {
        "hashes": {"payload_sha256": "a" * 64},
        "datasets": {
            name: {"path": f"{name}.parquet"}
            for name in ("train", "dev", "pilot_train", "pilot_eval")
        },
        "calibration": {"path": "reward_calibration.json", "sha256": "b" * 64},
    }
    config = build_cell_config(
        settings=settings,
        manifest=manifest,
        prepared_dir=tmp_path,
        arm="M1",
        stage="wiring",
        seed=42,
        cell_dir=tmp_path,
    )
    command = build_cell_command(
        config=config,
        settings=settings,
        hashes=RunHashes("c" * 40, "d" * 40, "test", False),
        cell_dir=tmp_path,
    )
    config_directory = REPO_ROOT / "third_party/agent-r1/agent_r1/config"
    with initialize_config_dir(config_dir=str(config_directory), version_base=None):
        resolved = compose(config_name="agent_ppo_trainer", overrides=command[3:])
    expected_template = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    assert OmegaConf.to_container(resolved.data.apply_chat_template_kwargs) == expected_template
    if enable_thinking is not None:
        assert type(resolved.data.apply_chat_template_kwargs.enable_thinking) is bool
    for optimizer in (resolved.actor_rollout_ref.actor.optim, resolved.critic.optim):
        assert OmegaConf.to_container(optimizer.override_optimizer_config) == {
            "foreach": False, "fused": False,
        }
    assert OmegaConf.to_container(
        resolved.actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config
    ) == {"cudagraph_capture_sizes": [1, 2, 4, 8]}
    env_vars = OmegaConf.to_container(
        resolved.ray_kwargs.ray_init.runtime_env.env_vars,
        resolve=True,
    )
    assert isinstance(env_vars, dict)
    assert all(isinstance(key, str) and isinstance(value, str) for key, value in env_vars.items())
    assert env_vars["FINANCE_SCREENING_RUN_SEED"] == "42"
    assert env_vars["FINANCE_SCREENING_ARM"] == "M1"
    assert env_vars["FINANCE_SCREENING_DIAGNOSTICS_DIR"] == str(
        tmp_path / "artifacts/sibling_diagnostics"
    )
    # This is the real Ray schema validation, without starting Ray or loading a model.
    runtime_environment = RuntimeEnv(env_vars=cast(dict[str, str], env_vars))
    assert runtime_environment["env_vars"] == env_vars


@pytest.mark.parametrize("generation", (
    None, {}, {"enable_thinking": "false"}, {"enable_thinking": 0},
    {"enable_thinking": False, "temperature": 0.7},
))
def test_generation_settings_reject_ambiguous_or_unregistered_modes(generation):
    with pytest.raises(ValueError, match="boolean enable_thinking"):
        screening_launcher.generation_template_kwargs({"generation": generation})


def test_override_replacement_preserves_literal_add_and_rejects_duplicates():
    assert set_overrides(["x=1", "+y=a"], {"x": 4, "y": "b"}) == ["x=4", "+y=b"]
    with pytest.raises(ValueError, match="duplicate"):
        set_overrides(["x=1", "+x=2"], {"x": 3})


@pytest.mark.parametrize("invalid", (
    "actor_param_offload", "actor_optimizer_offload",
    "critic_param_offload", "critic_optimizer_offload",
    "rollout_gpu_memory_utilization", "profile", "resume",
))
def test_placement_profile_rejects_partial_layout_or_unvalidated_restore(
    tmp_path: Path, invalid: str,
):
    settings = yaml.safe_load((REPO_ROOT / "configs/m2_screening.yaml").read_text())
    manifest = {
        "hashes": {"payload_sha256": "a" * 64},
        "datasets": {key: {"path": key + ".parquet"} for key in ("pilot_train", "pilot_eval")},
        "calibration": {"path": "calibration.json", "sha256": "b" * 64},
    }
    config = build_cell_config(
        settings=settings, manifest=manifest, prepared_dir=tmp_path, arm="M2", stage="pilot",
        seed=42, cell_dir=tmp_path,
    )
    if invalid.endswith("offload"):
        settings["resources"][invalid] = True
    elif invalid == "profile":
        settings["resources"][invalid] = "unknown"
    elif invalid == "rollout_gpu_memory_utilization":
        settings["resources"][invalid] = 0.28
    with pytest.raises(ValueError, match="resource profile|registered placement|restore"):
        build_cell_command(
            config=config, settings=settings, hashes=RunHashes("c" * 40, "d" * 40, "test", False),
            cell_dir=tmp_path, resume_path=tmp_path / "checkpoint" if invalid == "resume" else None,
        )


def test_paired_readout_rejects_complete_case_selection():
    full = {str(i): float(i) for i in range(12)}
    shifted = {key: value + 1 for key, value in full.items()}
    result = paired_summary(shifted, full)
    assert result["mean"] == result["median"] == result["win_rate"] == 1
    with pytest.raises(ValueError, match="complete 12"):
        paired_summary({str(i): float(i) for i in range(11)}, full)

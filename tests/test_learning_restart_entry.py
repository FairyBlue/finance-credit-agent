"""Resolve the real Hydra command for the bounded learning restart."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.agents.learning_restart import build_restart_command, make_config, settings
from src.agents.metadata import RunHashes
from src.agents.screening_launcher import REPO_ROOT


@pytest.mark.parametrize(
    ("arm", "zero", "rounds"),
    [("M1", True, 1), ("M2", False, 4), ("M1", False, 2), ("M1", False, 4)]
)
def test_actual_restart_hydra_contract(tmp_path: Path, arm: str, zero: bool, rounds: int):
    manifest = {
        "hashes": {"payload_sha256": "a" * 64},
        "datasets": {k: {"path": f"{k}.parquet"} for k in ("train", "dev")},
        "calibration": {"path": "reward_calibration.json", "sha256": "b" * 64},
    }
    cfg = make_config(tmp_path, manifest, arm, rounds, zero=zero)
    command = build_restart_command(
        config=cfg,
        run_settings=settings(),
        hashes=RunHashes("c" * 40, "d" * 40, "test", True),
        cell_dir=tmp_path,
    )
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "third_party/agent-r1/agent_r1/config"),
        version_base=None,
    ):
        resolved = compose(config_name="agent_ppo_trainer", overrides=command[3:])
    assert resolved.data.train_max_samples == 8
    assert resolved.data.train_batch_size == 2
    assert resolved.actor_rollout_ref.rollout.n == 4
    assert resolved.trainer.total_training_steps == rounds
    assert resolved.trainer.val_only == zero
    assert resolved.trainer.test_freq == rounds
    assert resolved.trainer.save_freq == (2 if arm == "M2" else rounds)
    assert resolved.data.screening_observation_profile == "history_features_v1"
    assert resolved.data.apply_chat_template_kwargs.enable_thinking is False
    assert resolved.trainer.resume_mode == "disable"
    assert resolved.actor_rollout_ref.rollout.max_model_len == 12288
    assert resolved.actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len == 12288
    assert resolved.actor_rollout_ref.rollout.gpu_memory_utilization == 0.2
    assert all(
        isinstance(v, str) for v in resolved.ray_kwargs.ray_init.runtime_env.env_vars.values()
    )
    runtime_env = resolved.ray_kwargs.ray_init.runtime_env.env_vars
    assert runtime_env.FINANCE_SCREENING_CRITIC_INIT == "cpu_seed_v1"
    assert settings()["critic_initialization"] == cfg["training"]["critic_initialization"]
    assert cfg["training"]["critic_initialization"] == runtime_env.FINANCE_SCREENING_CRITIC_INIT
    assert resolved.critic.model.fsdp_config.seed == cfg["run"]["seed"] == 42
    assert ("FINANCE_LEARNING_GATE_DIR" in runtime_env) == (arm == "M2" and not zero)


@pytest.mark.parametrize("mismatch", ["config", "settings", "resources"])
def test_restart_rejects_unrecorded_critic_initialization(tmp_path: Path, mismatch: str):
    run_settings = settings()
    cfg = {"training": {"critic_initialization": "cpu_seed_v1"}}
    if mismatch == "config":
        cfg["training"]["critic_initialization"] = "legacy"
    elif mismatch == "settings":
        run_settings["critic_initialization"] = "legacy"
    else:
        run_settings["resources"]["profile"] = "native"
    with pytest.raises(ValueError, match="critic.init"):
        build_restart_command(
            config=cfg, run_settings=run_settings,
            hashes=RunHashes("c" * 40, "d" * 40, "test", True), cell_dir=tmp_path,
        )

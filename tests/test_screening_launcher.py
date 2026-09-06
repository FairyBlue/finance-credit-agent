"""Execution contracts for the history-only screening launcher and readout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from ray.runtime_env import RuntimeEnv

from src.agents.metadata import RunHashes
from src.agents.screening_launcher import (
    ARMS,
    REPO_ROOT,
    build_cell_command,
    build_cell_config,
    set_overrides,
)
from src.evaluation.screening import paired_summary


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("stage", ("wiring", "pilot", "screening", "zero-shot"))
def test_screening_commands_have_shared_exact_contract(tmp_path: Path, arm: str, stage: str):
    settings = yaml.safe_load((REPO_ROOT / "configs/m2_screening.yaml").read_text())
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
    assert options["actor_rollout_ref.rollout.n"] == "4"
    assert options["actor_rollout_ref.rollout.agent.num_workers"] == "1"
    assert options["actor_rollout_ref.rollout.max_model_len"] == "6144"
    assert options["algorithm.gamma"] == options["algorithm.lam"] == "1.0"
    assert options["critic.ppo_mini_batch_size"] == "2"
    assert options["actor_rollout_ref.actor.ppo_mini_batch_size"] == "2"
    assert options["actor_rollout_ref.actor.policy_loss.loss_mode"] == "gspo"
    assert options["trainer.test_freq"] == ("1" if stage in {"wiring", "pilot"} else "16")
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


def test_real_hydra_compose_produces_ray_compatible_string_environment(tmp_path: Path):
    """Numeric-looking argv strings must survive Hydra before Ray validates them."""
    settings = yaml.safe_load((REPO_ROOT / "configs/m2_screening.yaml").read_text())
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


def test_override_replacement_preserves_literal_add_and_rejects_duplicates():
    assert set_overrides(["x=1", "+y=a"], {"x": 4, "y": "b"}) == ["x=4", "+y=b"]
    with pytest.raises(ValueError, match="duplicate"):
        set_overrides(["x=1", "+x=2"], {"x": 3})


def test_paired_readout_rejects_complete_case_selection():
    full = {str(i): float(i) for i in range(12)}
    shifted = {key: value + 1 for key, value in full.items()}
    result = paired_summary(shifted, full)
    assert result["mean"] == result["median"] == result["win_rate"] == 1
    with pytest.raises(ValueError, match="complete 12"):
        paired_summary({str(i): float(i) for i in range(11)}, full)

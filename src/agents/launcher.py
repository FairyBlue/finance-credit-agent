"""Build and execute the locked Agent-R1 command for one experimental arm."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import load_experiment_config
from .metadata import RunHashes, collect_run_hashes

REPO_ROOT = Path(__file__).resolve().parents[2]
FLOW_CONFIGS = {
    "A1": REPO_ROOT / "configs" / "agent_flow_a1.yaml",
    "A2": REPO_ROOT / "configs" / "agent_flow_a2.yaml",
    "A3": REPO_ROOT / "configs" / "agent_flow_a3.yaml",
}


def _required(config: dict[str, Any], *path: str) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing config value: {'.'.join(path)}")
        value = value[key]
    return value


def _hydra(key: str, value: object, *, add: bool = False) -> str:
    prefix = "+" if add else ""
    if isinstance(value, bool):
        rendered = "True" if value else "False"
    elif isinstance(value, (list, dict)):
        rendered = json.dumps(value, separators=(",", ":"))
    else:
        rendered = str(value)
    return f"{prefix}{key}={rendered}"


def build_upstream_command(config: dict[str, Any], hashes: RunHashes) -> list[str]:
    arm_id = str(_required(config, "arm", "id"))
    if arm_id not in FLOW_CONFIGS:
        raise ValueError(f"unsupported arm id: {arm_id}")
    training = _required(config, "training")
    data = _required(config, "data")
    processed = _required(data, "processed")
    model_config = _required(config, "model")
    model_name = model_config.get("name") or model_config.get("train")
    if not model_name:
        raise ValueError("model.name or model.train is required")
    is_step_po = arm_id in {"A2", "A3"}
    run_name = config.get("run", {}).get("name", f"{arm_id.lower()}-training")
    logger = ["console", "wandb"]

    command = [
        sys.executable,
        "-m",
        "agent_r1.trainer.main_agent_ppo",
        _hydra("algorithm.adv_estimator", "gae" if is_step_po else "grpo"),
        _hydra("data.train_files", str(REPO_ROOT / processed["train_file"])),
        _hydra("data.val_files", str(REPO_ROOT / processed["valid_file"])),
        _hydra("data.train_batch_size", training["train_batch_size"]),
        _hydra("data.max_prompt_length", training["max_prompt_length"]),
        _hydra("data.max_response_length", training["max_response_length"]),
        _hydra("data.filter_overlong_prompts", True),
        _hydra("data.truncation", "error"),
        _hydra("data.return_raw_chat", True),
        _hydra("actor_rollout_ref.model.path", model_name),
        _hydra("actor_rollout_ref.actor.optim.lr", training["learning_rate"]),
        _hydra("actor_rollout_ref.model.use_remove_padding", True),
        _hydra("actor_rollout_ref.model.enable_gradient_checkpointing", True),
        _hydra("actor_rollout_ref.actor.ppo_mini_batch_size", training["train_batch_size"]),
        _hydra("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", 1),
        _hydra("actor_rollout_ref.actor.use_kl_loss", True),
        _hydra("actor_rollout_ref.actor.kl_loss_coef", 0.001),
        _hydra("actor_rollout_ref.actor.kl_loss_type", "low_var_kl"),
        _hydra("actor_rollout_ref.actor.entropy_coeff", 0),
        _hydra("actor_rollout_ref.actor.fsdp_config.param_offload", True),
        _hydra("actor_rollout_ref.actor.fsdp_config.optimizer_offload", True),
        _hydra("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", 1),
        _hydra("actor_rollout_ref.rollout.tensor_model_parallel_size", 1),
        _hydra("actor_rollout_ref.rollout.name", "vllm"),
        _hydra("actor_rollout_ref.rollout.gpu_memory_utilization", 0.55),
        _hydra("actor_rollout_ref.rollout.n", 1 if is_step_po else training["rollout_n"]),
        _hydra(
            "actor_rollout_ref.rollout.agent.agent_flow_config_path",
            str(FLOW_CONFIGS[arm_id]),
        ),
        _hydra("actor_rollout_ref.rollout.agent.default_agent_flow", "trading"),
        _hydra("actor_rollout_ref.rollout.agent.max_steps", training["max_turns"]),
        _hydra("actor_rollout_ref.rollout.agent.num_workers", 1),
        _hydra("actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", 1),
        _hydra("actor_rollout_ref.ref.fsdp_config.param_offload", True),
        _hydra("reward_model.enable", False),
        _hydra("algorithm.use_kl_in_reward", False),
        _hydra("trainer.critic_warmup", 0),
        _hydra("trainer.logger", logger),
        _hydra("trainer.project_name", _required(config, "wandb", "project")),
        _hydra("trainer.experiment_name", run_name),
        _hydra("trainer.n_gpus_per_node", training["n_gpus_per_node"]),
        _hydra("trainer.nnodes", 1),
        _hydra("trainer.save_freq", -1 if config.get("run", {}).get("smoke") else 50),
        _hydra("trainer.test_freq", 1),
        _hydra("trainer.total_epochs", training["total_epochs"]),
        _hydra("reproducibility.agent_r1_commit", hashes.agent_r1_commit, add=True),
        _hydra("reproducibility.repo_commit", hashes.repo_commit, add=True),
        _hydra("reproducibility.data_version", hashes.data_version, add=True),
        _hydra("reproducibility.repo_dirty", hashes.repo_dirty, add=True),
    ]
    if is_step_po:
        command.extend(
            [
                _hydra("actor_rollout_ref.actor.policy_loss.loss_mode", "gspo"),
                _hydra("critic.model.path", model_name),
                _hydra("critic.optim.lr", 1e-5),
                _hydra("critic.model.use_remove_padding", True),
                _hydra("critic.model.enable_gradient_checkpointing", True),
                _hydra("critic.ppo_micro_batch_size_per_gpu", 1),
                _hydra("critic.model.fsdp_config.param_offload", True),
                _hydra("critic.model.fsdp_config.optimizer_offload", True),
            ]
        )
    else:
        command.append(_hydra("critic.enable", False))
    return command


def launch(config_path: str | Path, *, dry_run: bool, expected_arm: str | None = None) -> int:
    config = load_experiment_config(config_path)
    arm_id = str(_required(config, "arm", "id"))
    if expected_arm is not None and arm_id != expected_arm:
        raise ValueError(f"entrypoint {expected_arm} cannot launch config for {arm_id}")
    agent_path = REPO_ROOT / str(_required(config, "framework", "agent_r1_path"))
    hashes = collect_run_hashes(
        repo_root=REPO_ROOT,
        agent_r1_path=agent_path,
        data_version=str(_required(config, "data", "version")),
        expected_agent_r1_commit=str(_required(config, "framework", "agent_r1_commit")),
        require_clean_repo=(
            bool(_required(config, "training", "require_clean_repo")) and not dry_run
        ),
    )
    command = build_upstream_command(config, hashes)
    print(json.dumps({"command": command, "run_hashes": hashes.as_config()}, indent=2))
    if dry_run:
        return 0
    for path_key in ("train_file", "valid_file"):
        path = REPO_ROOT / str(_required(config, "data", "processed", path_key))
        if not path.is_file():
            raise FileNotFoundError(f"missing processed Agent-R1 dataset: {path}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(agent_path), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment["WANDB_PROJECT"] = str(_required(config, "wandb", "project"))
    environment["WANDB_MODE"] = str(_required(config, "wandb", "mode"))
    environment.setdefault("VLLM_USE_V1", "1")
    environment.setdefault("HYDRA_FULL_ERROR", "1")
    return subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False).returncode


def cli(*, expected_arm: str | None = None, default_config: str | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config, required=default_config is None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return launch(args.config, dry_run=args.dry_run, expected_arm=expected_arm)

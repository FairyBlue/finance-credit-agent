"""Build and execute the locked Agent-R1 command for one experimental arm."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from src.data import AvailabilityPolicy, StockBenchAdapter
from src.data.episodes import load_episode_manifest, verify_episode_benchmarks
from src.data.integrity import (
    OverlapReconciliationAudit,
    RawManifestAudit,
    audit_exchange_sessions,
    verify_overlap_reconciliation,
    verify_raw_daily_manifest,
)
from src.evaluation import run_configured_temporal_leakage_probe

from .config import load_experiment_config
from .metadata import RunHashes, collect_run_hashes, validate_composite_data_version

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


def _canonical_timestamps(values: list[object], *, context: str) -> list[str]:
    rendered: list[str] = []
    for value in values:
        if not isinstance(value, str) or "TODO" in value.upper():
            raise RuntimeError(f"{context} is not frozen")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise RuntimeError(f"{context} lacks timezone")
        rendered.append(parsed.isoformat())
    return rendered


def verify_configured_source_manifests(
    config: dict[str, Any], *, repo_root: str | Path
) -> dict[str, RawManifestAudit]:
    """Bind both configured snapshot hashes to their actual raw Parquet bytes."""

    root = Path(repo_root).expanduser().resolve()
    data = _required(config, "data")
    sources = _required(data, "sources")
    symbols = _required(data, "universe", "symbols")
    calendar = str(_required(data, "availability", "exchange_calendar"))
    if not isinstance(symbols, list) or not all(isinstance(item, str) for item in symbols):
        raise ValueError("data.universe.symbols must be a frozen ticker list")

    audits: dict[str, RawManifestAudit] = {}
    for source_name in ("history", "stockbench"):
        source = _required(sources, source_name)
        window = _required(source, "input_window")
        manifest_root = Path(str(_required(source, "manifest_root")))
        storage_root = Path(str(_required(source, "storage_root")))
        if not manifest_root.is_absolute():
            manifest_root = root / manifest_root
        if not storage_root.is_absolute():
            storage_root = root / storage_root
        if source_name == "stockbench" and "repo_commit" in source:
            observed = _required(data, "upstream_window_observation")
            session_audit = audit_exchange_sessions(
                start=date.fromisoformat(str(_required(window, "start"))),
                end=date.fromisoformat(str(_required(window, "end"))),
                exchange_calendar=calendar,
            )
            if (
                session_audit.session_count
                != int(_required(observed, "calendar_session_count"))
                or len(session_audit.canonical_bytes)
                != int(_required(observed, "calendar_session_list_bytes"))
                or session_audit.sha256
                != str(_required(observed, "calendar_session_list_sha256"))
            ):
                raise RuntimeError("configured XNYS session-list checksum does not match")
            expected_commit = str(_required(source, "repo_commit"))
            expected_tree = str(_required(source, "storage_tree"))
            actual_commit = subprocess.run(
                ["git", "-C", str(manifest_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            actual_tree = subprocess.run(
                ["git", "-C", str(manifest_root), "rev-parse", "HEAD:storage"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            dirty_storage = subprocess.run(
                [
                    "git",
                    "-C",
                    str(manifest_root),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                    "--",
                    "storage",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if actual_commit != expected_commit or actual_tree != expected_tree or dirty_storage:
                raise RuntimeError(
                    "StockBench checkout does not match its frozen commit/storage tree"
                )
        audits[source_name] = verify_raw_daily_manifest(
            manifest_root=manifest_root,
            storage_root=storage_root,
            symbols=symbols,
            start=date.fromisoformat(str(_required(window, "start"))),
            end=date.fromisoformat(str(_required(window, "end"))),
            expected_sha256=str(_required(source, "input_manifest_sha256")),
            expected_file_count=int(_required(source, "files")),
            expected_sessions_per_symbol=int(_required(source, "sessions_per_symbol")),
            exchange_calendar=calendar,
        )
    return audits


def verify_configured_overlap(
    config: dict[str, Any], *, repo_root: str | Path
) -> OverlapReconciliationAudit:
    """Verify the history/StockBench price basis on a pinned pre-OOS session."""

    root = Path(repo_root).expanduser().resolve()
    data = _required(config, "data")
    audit_config = _required(data, "cross_source_overlap_audit")
    session = date.fromisoformat(str(_required(audit_config, "session")))
    availability = _required(data, "availability")
    policy = AvailabilityPolicy(
        str(_required(availability, "timezone")),
        time.fromisoformat(str(_required(availability, "market_close_time"))),
        str(_required(availability, "exchange_calendar")),
    )
    sources = _required(data, "sources")

    def adapter(source_name: str) -> StockBenchAdapter:
        source = _required(sources, source_name)
        storage_root = Path(str(_required(source, "storage_root")))
        if not storage_root.is_absolute():
            storage_root = root / storage_root
        return StockBenchAdapter.from_storage(
            storage_root,
            symbols=_required(data, "universe", "symbols"),
            data_version=str(_required(source, "version")),
            availability_policy=policy,
            start_date=session,
            end_date=session,
        )

    return verify_overlap_reconciliation(
        left=adapter("history"),
        right=adapter("stockbench"),
        as_of=policy.market_available_at(session),
        symbols=_required(data, "universe", "symbols"),
        tolerance_bps=_required(audit_config, "tolerance_bps"),
    )


def verify_processed_dataset_bindings(
    config: dict[str, Any], *, repo_root: str | Path
) -> None:
    """Reject ignored processed rows that were built from stale assumptions."""

    import pandas as pd

    root = Path(repo_root).expanduser().resolve()
    data = _required(config, "data")
    sources = _required(data, "sources")
    processed = _required(data, "processed")
    manifest_path = Path(str(_required(data, "episode_manifest")))
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = load_episode_manifest(config=config, manifest_path=manifest_path)
    split_documents = _required(manifest, "splits")
    execution = _required(config, "execution")
    reward = _required(config, "reward")
    availability = _required(data, "availability")
    symbols = _required(data, "universe", "symbols")
    max_agent_steps = int(_required(config, "training", "max_agent_steps"))
    reserved_information_steps = int(
        _required(config, "training", "reserved_information_steps_per_session")
    )
    for split in ("train", "valid"):
        source_name = str(_required(data, "split_source", split))
        source = _required(sources, source_name)
        window = _required(source, "input_window")
        manifest_root = Path(str(_required(source, "manifest_root")))
        if not manifest_root.is_absolute():
            manifest_root = root / manifest_root
        storage_root = Path(str(_required(source, "storage_root")))
        if not storage_root.is_absolute():
            storage_root = root / storage_root
        expected_common = {
            "symbols": symbols,
            "manifest_root": str(manifest_root.resolve()),
            "storage_root": str(storage_root.resolve()),
            "data_version": _required(source, "version"),
            "data_start": _required(window, "start"),
            "data_end": _required(window, "end"),
            "input_manifest_sha256": _required(source, "input_manifest_sha256"),
            "input_file_count": int(_required(source, "files")),
            "input_sessions_per_symbol": int(_required(source, "sessions_per_symbol")),
            "initial_cash": float(_required(execution, "initial_cash")),
            "commission_bps": float(_required(execution, "commission_bps")),
            "slippage_bps": float(_required(execution, "slippage_bps")),
            "drawdown_coefficient": float(_required(reward, "drawdown_coefficient")),
            "turnover_coefficient": float(_required(reward, "turnover_coefficient")),
            "finance_epsilon": float(_required(reward, "epsilon")),
            "risk_lambda": float(_required(reward, "risk_lambda")),
            "turnover_limit": float(_required(reward, "turnover_limit")),
            "position_limit": float(_required(reward, "position_limit")),
            "market_timezone": str(_required(availability, "timezone")),
            "exchange_calendar": str(_required(availability, "exchange_calendar")),
            "market_close_time": str(_required(availability, "market_close_time")),
        }
        path = root / str(_required(processed, f"{split}_file"))
        if not path.is_file():
            raise FileNotFoundError(f"missing processed Agent-R1 dataset: {path}")
        frame = pd.read_parquet(path)
        required_columns = {
            "agent_name",
            "data_source",
            "prompt",
            "reward_model",
            "env_kwargs",
            "extra_info",
        }
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise RuntimeError(
                f"{split} processed dataset is missing columns: {sorted(missing_columns)}"
            )
        if frame.empty:
            raise RuntimeError(f"{split} processed dataset must not be empty")
        episodes = _required(split_documents, split)
        if not isinstance(episodes, list):
            raise RuntimeError(f"reviewed manifest split {split} must be a list")
        if len(frame) != len(episodes):
            raise RuntimeError(
                f"{split} processed row count does not match reviewed manifest"
            )
        for index, row in frame.iterrows():
            episode = episodes[index]
            if not isinstance(episode, dict):
                raise RuntimeError(f"reviewed manifest {split} row {index} must be a mapping")
            if row["agent_name"] != "trading":
                raise RuntimeError(f"{split} row {index} agent_name mismatch")
            if row["data_source"] != f"{source_name}/{split}":
                raise RuntimeError(f"{split} row {index} data_source mismatch")
            prompt = row["prompt"]
            if hasattr(prompt, "tolist"):
                prompt = prompt.tolist()
            expected_prompt = [
                {
                    "role": "user",
                    "content": "Manage the portfolio using only point-in-time tools.",
                }
            ]
            if prompt != expected_prompt:
                raise RuntimeError(f"{split} row {index} prompt mismatch")
            if row["reward_model"] != {"style": "rule", "ground_truth": 0.0}:
                raise RuntimeError(f"{split} row {index} reward_model mismatch")
            try:
                env_kwargs = json.loads(row["env_kwargs"])
            except (TypeError, json.JSONDecodeError) as error:
                raise RuntimeError(f"{split} row {index} env_kwargs is not valid JSON") from error
            if not isinstance(env_kwargs, dict):
                raise RuntimeError(f"{split} row {index} env_kwargs must be a mapping")
            raw_dates = episode.get("episode_dates")
            raw_opens = episode.get("episode_open_times")
            raw_benchmark = episode.get("benchmark_closes")
            if not all(
                isinstance(values, list)
                for values in (raw_dates, raw_opens, raw_benchmark)
            ):
                raise RuntimeError(f"reviewed manifest {split} row {index} is incomplete")
            assert isinstance(raw_dates, list)
            assert isinstance(raw_opens, list)
            assert isinstance(raw_benchmark, list)
            budgeted_steps = (len(raw_dates) - 1) * (1 + reserved_information_steps)
            if len(raw_dates) < 2 or budgeted_steps > max_agent_steps:
                raise RuntimeError(f"{split} row {index} exceeds max_agent_steps")

            expected = {
                **expected_common,
                "episode_dates": _canonical_timestamps(
                    raw_dates,
                    context=f"reviewed manifest {split} row {index} episode_dates",
                ),
                "episode_open_times": _canonical_timestamps(
                    raw_opens,
                    context=f"reviewed manifest {split} row {index} episode_open_times",
                ),
                "benchmark_closes": [float(value) for value in raw_benchmark],
            }
            if set(env_kwargs) != set(expected):
                raise RuntimeError(f"{split} row {index} env_kwargs key set mismatch")
            for key, expected_value in expected.items():
                if env_kwargs.get(key) != expected_value:
                    raise RuntimeError(f"{split} row {index} {key} mismatch")
            extra_info = row["extra_info"]
            expected_info = {
                "episode_id": episode.get("episode_id"),
                "split": split,
                "index": index,
            }
            if extra_info != expected_info:
                raise RuntimeError(f"{split} row {index} extra_info mismatch")


def verify_configured_t2_gate(config: dict[str, Any], *, repo_root: str | Path) -> None:
    """Block scientific training when the real-data shifted-label gate fails."""

    if bool(config.get("run", {}).get("smoke")):
        return
    gate = _required(config, "evaluation", "shifted_label_probe")
    if not bool(_required(gate, "require_before_main_training")):
        return
    configured = run_configured_temporal_leakage_probe(
        config=config,
        repo_root=repo_root,
    )
    if not configured.passed:
        result = configured.result
        raise RuntimeError(
            "T2 shifted-label gate failed; main training is blocked: "
            f"aligned={result.aligned_accuracy:.6f}, "
            f"shifted={result.shifted_accuracy:.6f}, "
            f"criteria={configured.criteria}"
        )


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
        _hydra("actor_rollout_ref.rollout.agent.max_steps", training["max_agent_steps"]),
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
    data = _required(config, "data")
    sources = _required(data, "sources")
    data_version = validate_composite_data_version(
        configured=str(_required(data, "version")),
        history_manifest_sha256=str(_required(sources, "history", "input_manifest_sha256")),
        stockbench_manifest_sha256=str(_required(sources, "stockbench", "input_manifest_sha256")),
    )
    hashes = collect_run_hashes(
        repo_root=REPO_ROOT,
        agent_r1_path=agent_path,
        data_version=data_version,
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
    verify_configured_source_manifests(config, repo_root=REPO_ROOT)
    verify_configured_overlap(config, repo_root=REPO_ROOT)
    verify_configured_t2_gate(config, repo_root=REPO_ROOT)
    manifest_path = Path(str(_required(config, "data", "episode_manifest")))
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    verify_episode_benchmarks(
        config=config,
        manifest_path=manifest_path,
        repo_root=REPO_ROOT,
    )
    verify_processed_dataset_bindings(config, repo_root=REPO_ROOT)
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

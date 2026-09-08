"""History-Train-only launcher for the separately authorized M2 screening stage."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from contextlib import suppress
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from .config import load_experiment_config
from .launcher import (
    REPO_ROOT,
    _finalize_model_paths,
    _hydra,
    _training_environment,
    build_upstream_command,
    verify_gpu_runtime_and_model,
)
from .metadata import collect_run_hashes, verify_validation_identity
from .rollout_seed import normalize_rollout_identity
from .screening_checkpoint import validate_checkpoint
from .screening_prompt import validate_action_protocol
from .update_probe import summarize_update_evidence

ARMS = ("A2", "M1", "M2-plain", "M2")
STAGES = ("wiring", "pilot", "screening", "zero-shot")


def atomic_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, sort_keys=True, indent=2, allow_nan=False, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(directory: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"invalid evidence row: {path}")
                records.append(record)
    return records


def set_overrides(command: list[str], overrides: dict[str, Any]) -> list[str]:
    """Replace exact Hydra keys, adding only genuinely new structured fields."""
    result = list(command)
    for key, value in overrides.items():
        matches = [
            i for i, arg in enumerate(result)
            if arg.startswith((key + "=", "+" + key + "="))
        ]
        if len(matches) > 1:
            raise ValueError(f"duplicate Hydra override: {key}")
        if matches:
            result[matches[0]] = _hydra(key, value, add=result[matches[0]].startswith("+"))
        else:
            result.append(_hydra(key.lstrip("+"), value, add=key.startswith("+")))
    return result


def generation_template_kwargs(settings: dict[str, Any]) -> dict[str, bool]:
    """A single explicit template mode shared by training and final evaluation.

    Omitted generation settings preserve historical tokenizer defaults. Reject
    string booleans and unknown controls instead of silently changing the prompt.
    """
    if "generation" not in settings:
        return {}
    generation = settings["generation"]
    if (
        not isinstance(generation, dict)
        or set(generation) != {"enable_thinking"}
        or type(generation["enable_thinking"]) is not bool
    ):
        raise ValueError("screening generation requires exactly one boolean enable_thinking")
    return {"enable_thinking": generation["enable_thinking"]}


def build_cell_config(
    *, settings: dict[str, Any], manifest: dict[str, Any], prepared_dir: Path,
    arm: str, stage: str, seed: int, cell_dir: Path,
) -> dict[str, Any]:
    if arm not in ARMS or stage not in STAGES or seed not in settings["seeds"]:
        raise ValueError("unregistered screening arm, stage, or seed")
    if settings.get("stage") != "m2_screening":
        raise ValueError("this launcher only accepts the screening stage")
    config = load_experiment_config(REPO_ROOT / "configs/base.yaml")
    config["run"] = {
        "seed": seed, "name": f"m2-v2-{stage}-{arm.lower()}-s{seed}",
        "kind": "m2_screening_v2", "smoke": False,
    }
    config["screening"] = {"arm": arm, "stage": stage, "authorized": True}
    config["arm"] = {"id": "A2" if arm == "A2" else "A3", "name": arm}
    config["model"] = copy.deepcopy(
        settings["models"]["wiring" if stage == "wiring" else "screening"]
    )
    config["training"].update(settings["training"])
    small = stage in {"wiring", "pilot"}
    episodes = 2 if small else 32
    config["training"].update({
        "episodes": episodes, "total_training_steps": episodes // 2,
        "reference_model_dtype": settings["resources"]["reference_model_dtype"],
        "critic_param_offload": settings["resources"]["critic_param_offload"],
    })
    if stage == "pilot":
        pilot = settings["engineering_pilot"]
        if pilot["train_episodes"] != episodes:
            raise ValueError("resource pilot must retain the same first two Train episodes")
        config["training"].update({
            key: pilot[key] for key in ("total_epochs", "total_training_steps", "save_freq")
        })
    train_key = "pilot_train" if small else "train"
    eval_key = "pilot_eval" if small else "dev"
    config["data"] = {
        "version": "m2-screening-manifest:" + manifest["hashes"]["payload_sha256"],
        "episode_manifest": str(prepared_dir / "manifest.json"),
        "split_source": {"train": "history", "valid": "history"},
        "processed": {
            "train_file": str(prepared_dir / manifest["datasets"][train_key]["path"]),
            "valid_file": str(prepared_dir / manifest["datasets"][eval_key]["path"]),
        },
    }
    template_kwargs = generation_template_kwargs(settings)
    if template_kwargs:
        config["data"]["apply_chat_template_kwargs"] = template_kwargs
    protocol = validate_action_protocol(settings.get("action_protocol", "legacy"))
    if protocol != "legacy":
        config["data"]["screening_action_protocol"] = protocol
    config["tracking"]["log_root"] = str(cell_dir)
    config["reward"] = {
        "scorer_id": "benchmark_scale_v2",
        "calibration": str(prepared_dir / manifest["calibration"]["path"]),
        "calibration_sha256": manifest["calibration"]["sha256"],
    }
    return config


def build_cell_command(
    *, config: dict[str, Any], settings: dict[str, Any], hashes: Any, cell_dir: Path,
    resume_path: Path | None = None,
) -> list[str]:
    arm = config["screening"]["arm"]
    stage = config["screening"]["stage"]
    flow_path = cell_dir / "flow.yaml"
    flow_path.write_text(yaml.safe_dump([{
        "name": "trading", "_target_": "src.agents.agent_flow.ScreeningTradingAgentEnvLoop",
        "env_type": "trading", "screening_arm": arm,
    }], sort_keys=True), encoding="utf-8")
    command = build_upstream_command(
        config, hashes, update_probe_dir=cell_dir / "artifacts/update_probe",
        flow_config_path=flow_path,
    )
    r = settings["resources"]
    t = settings["training"]
    template_kwargs = generation_template_kwargs(settings)
    if config["data"].get("apply_chat_template_kwargs", {}) != template_kwargs:
        raise ValueError("resolved chat template mode differs from screening settings")
    protocol = validate_action_protocol(settings.get("action_protocol", "legacy"))
    if config["data"].get("screening_action_protocol", "legacy") != protocol:
        raise ValueError("resolved action protocol differs from screening settings")
    profile = r.get("profile", "native")
    if profile not in {"native", "pageable_adam_v1"}:
        raise ValueError("unregistered screening resource profile")
    if profile == "pageable_adam_v1":
        if any(r[key] is not False for key in (
            "actor_param_offload", "actor_optimizer_offload",
            "critic_param_offload", "critic_optimizer_offload",
        )) or r["rollout_gpu_memory_utilization"] != 0.20:
            raise ValueError("pageable_adam_v1 requires the complete registered placement")
        if resume_path is not None:
            raise ValueError("pageable_adam_v1 checkpoint restore has not been validated")
    optimizer_options = r["optimizer_overrides"]
    if optimizer_options != {"foreach": False, "fused": False}:
        raise ValueError("screening requires the registered scalar AdamW resource settings")
    # Hydra's dictionary grammar requires unquoted keys; JSON is not valid here.
    optimizer_override = "{foreach:false,fused:false}"
    if r["rollout_cudagraph_capture_sizes"] != [1, 2, 4, 8]:
        raise ValueError("screening requires the registered CUDA graph capture sizes")
    steps = config["training"]["total_training_steps"]
    overrides = {
        "data.shuffle": False,
        "data.dataloader_num_workers": 0,
        "data.val_batch_size": 2,
        "data.filter_overlong_prompts": False,
        "algorithm.gamma": t["gamma"], "algorithm.lam": t["lam"],
        "actor_rollout_ref.rollout.n": 4,
        "actor_rollout_ref.rollout.temperature": t["temperature"],
        "actor_rollout_ref.rollout.top_p": t["top_p"],
        "actor_rollout_ref.rollout.top_k": t["top_k"],
        "actor_rollout_ref.rollout.do_sample": True,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.0,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.top_k": -1,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": False,
        "actor_rollout_ref.rollout.agent.num_workers": 1,
        "actor_rollout_ref.rollout.max_model_len": (
            config["training"]["max_prompt_length"] + config["training"]["max_response_length"]
        ),
        "actor_rollout_ref.rollout.gpu_memory_utilization": r["rollout_gpu_memory_utilization"],
        # Pinned verl overwrites rollout.max_model_len with the HF capacity;
        # engine_kwargs is merged last and preserves our actual engine limit.
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len": (
            config["training"]["max_prompt_length"] + config["training"]["max_response_length"]
        ),
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs": 8,
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config": (
            "{cudagraph_capture_sizes:[1,2,4,8]}"
        ),
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 2,
        "actor_rollout_ref.actor.shuffle": False,
        "actor_rollout_ref.actor.fsdp_config.param_offload": r["actor_param_offload"],
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": r["actor_optimizer_offload"],
        "critic.ppo_epochs": 1, "critic.ppo_mini_batch_size": 2,
        "critic.shuffle": False, "critic.optim.lr": t["critic_learning_rate"],
        "actor_rollout_ref.actor.optim.override_optimizer_config": optimizer_override,
        "critic.optim.override_optimizer_config": optimizer_override,
        "critic.model.fsdp_config.param_offload": r["critic_param_offload"],
        "critic.model.fsdp_config.optimizer_offload": r["critic_optimizer_offload"],
        "actor_rollout_ref.actor.checkpoint.async_save": False,
        "critic.checkpoint.async_save": False,
        "trainer.val_before_train": stage == "zero-shot",
        "trainer.val_only": stage == "zero-shot",
        # Upstream also evaluates on the last step: no intermediate dev access.
        "trainer.test_freq": steps,
        "trainer.save_freq": config["training"]["save_freq"],
        "trainer.default_local_dir": str(cell_dir / "checkpoints"),
        "trainer.max_actor_ckpt_to_keep": t["max_checkpoints"],
        "trainer.max_critic_ckpt_to_keep": t["max_checkpoints"],
        "trainer.resume_mode": "resume_path" if resume_path else "disable",
        "trainer.resume_from_path": str(resume_path) if resume_path else None,
        "ray_kwargs.ray_init.num_cpus": r["ray_num_cpus"],
        "+ray_kwargs.ray_init.object_store_memory": r["ray_object_store_memory"],
    }
    if template_kwargs:
        # This shared dataset setting is used by the real Agent-R1 environment
        # loop for both sampled Train and final greedy/zero-shot observations.
        overrides["+data.apply_chat_template_kwargs.enable_thinking"] = (
            template_kwargs["enable_thinking"]
        )
    if protocol != "legacy":
        overrides["+data.screening_action_protocol"] = protocol
    env_flags = {
        "FINANCE_SCREENING_ARM": arm,
        # Hydra otherwise coerces numeric-looking strings into ints; Ray's
        # runtime_env requires every environment value to remain a string.
        "FINANCE_SCREENING_RUN_SEED": json.dumps(str(config["run"]["seed"])),
        "FINANCE_SCREENING_DIAGNOSTICS_DIR": str(cell_dir / "artifacts/sibling_diagnostics"),
        "FINANCE_SCREENING_EVIDENCE_DIR": str(cell_dir / "artifacts/screening_evidence"),
        "FINANCE_SCREENING_IDENTITY_SHA256": sha256(json.dumps({
            "commit": hashes.repo_commit, "config": config, "settings": settings,
        }, sort_keys=True).encode()).hexdigest(),
    }
    if profile == "pageable_adam_v1":
        env_flags.update({
            "FINANCE_SCREENING_RESOURCE_PROFILE": profile,
            "FINANCE_SCREENING_PLACEMENT_DIR": str(cell_dir / "artifacts/placement"),
        })
    for key, value in env_flags.items():
        overrides["+ray_kwargs.ray_init.runtime_env.env_vars." + key] = value
    return set_overrides(command, overrides)


def capture_process(command: list[str], *, environment: dict[str, str], cell_dir: Path) -> int:
    """Run under the durable supervisor, keeping logs independent of SSH pipes."""
    log_path = cell_dir / "trainer.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as output:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
        )
        atomic_json(cell_dir / "process.json", {
            "pid": process.pid, "pgid": process.pid, "started_at": utc_now(),
        })
        try:
            with (cell_dir / "resources.jsonl").open("a", encoding="utf-8") as resources:
                while process.poll() is None:
                    measurement: dict[str, Any] = {"time": utc_now(), "pid": process.pid}
                    for filename in ("memory.current", "memory.peak", "memory.events"):
                        path = Path("/sys/fs/cgroup") / filename
                        if path.exists():
                            measurement[filename] = path.read_text().strip()
                    try:
                        gpu = subprocess.run([
                            "nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ], capture_output=True, text=True, timeout=15, check=False)
                    except (subprocess.TimeoutExpired, OSError) as error:
                        measurement["gpu_error"] = {
                            "type": type(error).__name__, "message": str(error),
                        }
                    else:
                        if gpu.returncode == 0:
                            measurement["gpu"] = gpu.stdout.strip()
                        else:
                            measurement["gpu_error"] = {
                                "type": "NonzeroExit", "returncode": gpu.returncode,
                                "message": gpu.stderr.strip() or gpu.stdout.strip(),
                            }
                    resources.write(json.dumps(measurement, sort_keys=True) + "\n")
                    resources.flush()
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=10)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    return int(process.returncode)


def validate_cell_evidence(
    cell_dir: Path, *, stage: str, expected_rounds: int, arm: str, seed: int,
    train_episode_ids: list[str], eval_episode_ids: list[str],
    execution_identity_sha256: str, calibration_payload_sha256: str,
) -> dict[str, Any]:
    """Count semantic trajectory identities and optimizer evidence, never infer from exit 0."""
    records = read_jsonl(cell_dir / "artifacts/screening_evidence")
    # Flow schema is normalized here; repeated identities from recovery are not accepted silently.
    identities: set[str] = set()
    counts: Counter[str] = Counter()
    expected_grid = {
        (step, index, sibling, False): train_episode_ids[index]
        for step in range(1, expected_rounds + 1)
        for index in (((step - 1) * 2 + offset) % len(train_episode_ids) for offset in (0, 1))
        for sibling in range(4)
    } if stage != "zero-shot" else {}
    eval_step = 0 if stage == "zero-shot" else expected_rounds
    expected_grid.update({
        (eval_step, index, 0, True): episode_id
        for index, episode_id in enumerate(eval_episode_ids)
    })
    observed_grid = set()
    for record in records:
        if record.get("schema_version") != 1:
            raise RuntimeError("unknown screening trajectory evidence schema")
        identity = record.get("rollout_identity", record.get("identity"))
        if not isinstance(identity, dict):
            raise RuntimeError("missing screening trajectory identity")
        normalized = normalize_rollout_identity(identity)
        grid_key = (
            normalized.step, normalized.sample_index, normalized.rollout_n, normalized.validate,
        )
        if expected_grid.get(grid_key) != record.get("episode_id"):
            raise RuntimeError("trajectory does not match frozen episode/checkpoint grid")
        observed_grid.add(grid_key)
        if record.get("arm") != arm or record.get("base_seed") != seed:
            raise RuntimeError("trajectory arm/seed identity mismatch")
        if record.get("metrics", {}).get("calibration_sha256") != calibration_payload_sha256:
            raise RuntimeError("trajectory calibration identity mismatch")
        key = json.dumps(identity, sort_keys=True)
        if key in identities:
            raise RuntimeError("duplicate screening trajectory evidence")
        identities.add(key)
        complete = record.get("complete", record.get("episode_complete"))
        if complete is not True:
            raise RuntimeError("INCOMPLETE screening trajectory")
        score = record.get("terminal_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise RuntimeError("missing finite common terminal score")
        if record.get("metrics", {}).get("terminal_score") != score:
            raise RuntimeError("common terminal score differs from the ledger metrics")
        counts["eval" if identity["validate"] else "train"] += 1
    expected_train = 0 if stage == "zero-shot" else expected_rounds * 8
    expected_eval = 2 if stage in {"wiring", "pilot"} else 12
    if counts["train"] != expected_train or counts["eval"] != expected_eval:
        raise RuntimeError(f"trajectory count mismatch: {dict(counts)}")
    if observed_grid != set(expected_grid):
        raise RuntimeError("incomplete frozen episode/checkpoint grid")
    updates = summarize_update_evidence(cell_dir / "artifacts/update_probe")
    if stage != "zero-shot":
        diagnostics = read_jsonl(cell_dir / "artifacts/sibling_diagnostics")
        if sorted(r.get("policy_step", -1) for r in diagnostics) != list(
            range(1, expected_rounds + 1)
        ):
            raise RuntimeError("missing or duplicate logical-batch advantage diagnostics")
        for row in diagnostics:
            if (
                row.get("trajectory_count") != 8
                or row.get("schema_version") != 1
                or row.get("phase") != "advantage_pre_update"
                or row.get("arm") != arm
                or row.get("run_seed") != seed
                or row.get("whitening_scope") != "logical_rollout_batch_real_step_rows"
                or row.get("whitening_epsilon") != 1e-8
                or row.get("whitening_correction") != 1
            ):
                raise RuntimeError("screening advantage hook contract mismatch")
        roles = updates["detected_role_counts"]
        if updates["optimizer_step_count"] != 2 * expected_rounds:
            raise RuntimeError("unexpected total optimizer update count")
        if not isinstance(roles, dict) or any(
            roles.get(role, 0) != expected_rounds for role in ("actor", "critic")
        ):
            raise RuntimeError(f"optimizer update mismatch: {roles}")
        final = cell_dir / "checkpoints" / f"global_step_{expected_rounds}"
        validate_checkpoint(final, identity_sha256=execution_identity_sha256)
    elif updates["optimizer_step_count"] != 0 or read_jsonl(
        cell_dir / "artifacts/sibling_diagnostics"
    ):
        raise RuntimeError("zero-shot unexpectedly updated optimizer")
    return {"trajectories": dict(counts), "optimizer_evidence": updates}


def launch_cell(
    *, prepared_dir: Path, settings_path: Path, cell_dir: Path, arm: str, stage: str,
    seed: int, expected_commit: str, dry_run: bool = False, preflight_only: bool = False,
) -> dict[str, Any]:
    from src.data.screening import validate_prepared_screening

    cell_dir = cell_dir.resolve()
    cell_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary: dict[str, Any] = {
        "status": "PREFLIGHT", "started_at": utc_now(), "stage": stage,
        "arm": arm, "seed": seed, "expected_commit": expected_commit,
        "formal_valid_accessed": False, "final_oos_accessed": False,
    }
    atomic_json(cell_dir / "summary.json", summary)
    try:
        settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
        manifest = validate_prepared_screening(prepared_dir, repo_root=REPO_ROOT)
        config = build_cell_config(
            settings=settings, manifest=manifest, prepared_dir=prepared_dir,
            arm=arm, stage=stage, seed=seed, cell_dir=cell_dir,
        )
        agent_path = REPO_ROOT / config["framework"]["agent_r1_path"]
        hashes = collect_run_hashes(
            repo_root=REPO_ROOT, agent_r1_path=agent_path,
            data_version=config["data"]["version"],
            expected_agent_r1_commit=config["framework"]["agent_r1_commit"],
            require_clean_repo=not dry_run,
        )
        if not dry_run:
            # This pre-existing, ignored M1 smoke manifest is not an executable
            # overlay or an input to screening. Preserve it and record its bytes.
            legacy_manifest = REPO_ROOT / "configs/episode_manifest.smoke.reviewed.yaml"
            summary["unused_legacy_artifacts"] = [{
                "path": str(legacy_manifest.relative_to(REPO_ROOT)),
                "sha256": sha256(legacy_manifest.read_bytes()).hexdigest(),
                "used_by_screening": False,
            }] if legacy_manifest.is_file() else []
            summary["identity"] = verify_validation_identity(
                repo_root=REPO_ROOT, agent_r1_path=agent_path,
                expected_repo_commit=expected_commit,
                expected_agent_r1_commit=config["framework"]["agent_r1_commit"],
                required_tracked_paths=(settings_path,),
                allowed_untracked_paths=(legacy_manifest,),
            )
        atomic_json(cell_dir / "resolved_config.json", config)
        atomic_json(cell_dir / "settings.json", settings)
        summary["config_sha256"] = sha256(
            json.dumps(config, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        command = build_cell_command(
            config=config, settings=settings, hashes=hashes, cell_dir=cell_dir,
        )
        options = dict(argument.lstrip("+").split("=", 1) for argument in command[3:])
        summary["execution_identity_sha256"] = options[
            "ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_SCREENING_IDENTITY_SHA256"
        ]
        small = stage in {"wiring", "pilot"}
        train_ids = manifest["datasets"]["pilot_train" if small else "train"]["episode_ids"]
        eval_ids = manifest["datasets"]["pilot_eval" if small else "dev"]["episode_ids"]
        summary.update(
            expected_eval_episode_ids=eval_ids,
            expected_eval_step=(
                0 if stage == "zero-shot" else config["training"]["total_training_steps"]
            ),
            calibration_payload_sha256=manifest["calibration"]["payload_sha256"],
            prepared_manifest_sha256=manifest["hashes"]["payload_sha256"],
        )
        if dry_run:
            atomic_json(cell_dir / "command.json", command)
            summary["status"] = "DRY_RUN"
            return summary
        runtime = verify_gpu_runtime_and_model(config)
        atomic_json(cell_dir / "preflight.json", {"runtime": runtime, "hashes": hashes.as_config()})
        command = _finalize_model_paths(
            command, arm_id=config["arm"]["id"],
            model_snapshot_path=str(runtime["model_snapshot_path"]),
        )
        environment = _training_environment(
            agent_path=agent_path, probe_root=cell_dir / "artifacts/update_probe",
            c_compiler=str(runtime["c_compiler"]),
        )
        environment["PYTHONUNBUFFERED"] = "1"
        # Runtime hooks are enabled only in explicitly configured Ray workers.
        for key in list(environment):
            if key.startswith("FINANCE_SCREENING_"):
                environment.pop(key)
        atomic_json(cell_dir / "command.json", command)
        if preflight_only:
            summary["status"] = "PREFLIGHT_PASSED"
            return summary
        summary["status"] = "RUNNING"
        atomic_json(cell_dir / "summary.json", summary)
        returncode = capture_process(command, environment=environment, cell_dir=cell_dir)
        summary["returncode"] = returncode
        if returncode != 0:
            raise RuntimeError(f"trainer exited {returncode}; inspect {cell_dir / 'trainer.log'}")
        summary["evidence"] = validate_cell_evidence(
            cell_dir, stage=stage, expected_rounds=config["training"]["total_training_steps"],
            arm=arm, seed=seed, train_episode_ids=train_ids, eval_episode_ids=eval_ids,
            execution_identity_sha256=summary["execution_identity_sha256"],
            calibration_payload_sha256=summary["calibration_payload_sha256"],
        )
        summary["status"] = "PASSED"
        return summary
    except BaseException as error:
        summary.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["updated_at"] = utc_now()
        atomic_json(cell_dir / "summary.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=REPO_ROOT / "artifacts/m2_screening")
    parser.add_argument("--settings", type=Path, default=REPO_ROOT / "configs/m2_screening.yaml")
    parser.add_argument("--cell-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.preflight_only:
        parser.error("dry-run and preflight-only are mutually exclusive")
    result = launch_cell(
        prepared_dir=args.prepared_dir.resolve(), settings_path=args.settings.resolve(),
        cell_dir=args.cell_dir, arm=args.arm, stage=args.stage, seed=args.seed,
        expected_commit=args.expected_commit, dry_run=args.dry_run,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

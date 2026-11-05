"""Two-update full-parameter 4B engineering pilot for disk-backed chunked Adam.

Fresh M1 and M2 cells use the approved common SFT, reward and first two
coverage batches. This controller never resumes or retries a failed update.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from src.agents import common_validity_penalty as common
from src.agents import learning_restart as restart
from src.agents import train_coverage as coverage
from src.agents import training_tuning as tuning
from src.agents.launcher import (
    _finalize_model_paths,
    _resolve_exact_model_snapshot,
    _training_environment,
)
from src.agents.metadata import collect_run_hashes
from src.agents.screening_launcher import (
    REPO_ROOT,
    atomic_json,
    build_cell_command,
    build_cell_config,
    capture_process,
    read_jsonl,
    set_overrides,
    validate_cell_evidence,
    verify_gpu_runtime_and_model,
)

PROFILE = "common_sft_low_memory_pilot_v1"
PENALTY = 0.05
PLAN = "docs/2026-09-30_M2_LOW_MEMORY_PILOT_PLAN.md"
TRAIN_NUMBERS = [1, 65, 9, 73]
TRAIN_IDS = [f"formal-train-{i:04d}" for i in TRAIN_NUMBERS]
EVAL_IDS = restart.EVAL_IDS
EVAL_STEPS = [2]
DEFAULT = REPO_ROOT / "runs/m2-low-memory-pilot-20260930"
INPUTS = (
    "settings.json",
    "model_identity.json",
    "input_preflight.json",
    "prepared/manifest.json",
    "selection.json",
    "source_identity.json",
    "plan.md",
    "gpu_handoff_config.json",
)
CellSpec = tuning.CellSpec
MIN_FREE_GPU_MIB = 60 * 1024
WAIT_SECONDS = 12 * 3600


@dataclass(frozen=True)
class ExperimentProfile:
    """A reviewed fixed budget, passed explicitly through the shared controller."""

    name: str
    plan: str
    train_numbers: tuple[int, ...]
    evaluation_steps: tuple[int, ...]
    rounds: int
    include_sft: bool = False
    retire_completed_moments: bool = False
    seed: int = 42
    eval_numbers: tuple[int, ...] = (113, 117, 121, 125)
    actor_learning_rate: float = 2e-7
    cell_specs: tuple[CellSpec, ...] = ()
    cell_actor_learning_rates: tuple[tuple[str, float], ...] = ()
    archive_completed_actors: bool = False
    evaluation_incomplete_mode: str | None = None

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("experiment seed must be an integer in [0, 2**32)")
        if (not self.eval_numbers or len(set(self.eval_numbers)) != len(self.eval_numbers)
                or any(type(n) is not int or n < 1 for n in self.eval_numbers)):
            raise ValueError("evaluation window indices must be unique positive integers")
        rates = [self.actor_learning_rate, *[lr for _, lr in self.cell_actor_learning_rates]]
        if any(type(lr) not in (float, int) or not math.isfinite(lr) or lr <= 0 for lr in rates):
            raise ValueError("actor learning rates must be finite positive numbers")
        if self.cell_specs:
            ids = [s.cell_id for s in self.cell_specs]
            if (len(set(ids)) != len(ids) or any(Path(i).name != i for i in ids)
                    or sum(s.zero for s in self.cell_specs) != int(self.include_sft)
                    or any(s.rounds != self.rounds for s in self.cell_specs if not s.zero)):
                raise ValueError("custom cells must have unique identities and the fixed budget")
            overrides = [name for name, _ in self.cell_actor_learning_rates]
            if len(set(overrides)) != len(overrides) or not set(overrides) <= set(ids):
                raise ValueError("learning-rate overrides must reference unique fixed cells")
        if self.archive_completed_actors and not self.retire_completed_moments:
            raise ValueError("actor archival requires successful-cell moment retirement")
        if self.evaluation_incomplete_mode not in (None, "retain_prefix_v1"):
            raise ValueError("unknown incomplete evaluation profile")
        if self.evaluation_incomplete_mode and not self.cell_specs:
            raise ValueError("incomplete evaluation isolation requires an explicit fixed grid")

    @property
    def train_ids(self):
        return [f"formal-train-{number:04d}" for number in self.train_numbers]

    @property
    def eval_ids(self):
        return [f"formal-train-{number:04d}" for number in self.eval_numbers]

    def actor_lr(self, cell_id):
        return dict(self.cell_actor_learning_rates).get(cell_id, self.actor_learning_rate)

    @property
    def planned_trajectories(self):
        return sum(
            len(self.eval_ids) * 5 if s.zero else
            s.rounds * 8 + len(self.evaluation_steps) * len(self.eval_ids) * 5
            for s in cells(self)
        )


ONE_ROUND_DISK_PROFILE = "one_round_fresh_adam_v1"
RECOVERY_OBSERVATION_PROFILE = "history_features_recovery_v2"


@dataclass(frozen=True)
class RecoveryExperimentProfile(ExperimentProfile):
    """Opt-in acceptance contract; legacy profile serialization stays unchanged."""

    observation_profile: str = RECOVERY_OBSERVATION_PROFILE
    test_freq: int = 1
    disk_budget_profile: str = ONE_ROUND_DISK_PROFILE
    acceptance_gate_sha256: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if (self.rounds != 1 or self.evaluation_steps != (1,) or self.test_freq != 1
                or self.observation_profile != RECOVERY_OBSERVATION_PROFILE
                or self.disk_budget_profile != ONE_ROUND_DISK_PROFILE
                or not self.retire_completed_moments or self.archive_completed_actors):
            raise ValueError("recovery acceptance requires the fresh one-round lifecycle")
        if (not isinstance(self.acceptance_gate_sha256, str)
                or len(self.acceptance_gate_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.acceptance_gate_sha256)):
            raise ValueError("recovery acceptance must bind its predeclared endpoint gate SHA256")


@dataclass(frozen=True)
class ActionPoolingExperimentProfile(ExperimentProfile):
    """Four-round pooling ablation with the accepted common recovery observation.

    This uses the existing 90-GiB multi-round moment budget. The fresh one-round
    budget cannot be selected through this profile.
    """

    observation_profile: str = RECOVERY_OBSERVATION_PROFILE
    test_freq: int = 2

    def __post_init__(self):
        super().__post_init__()
        if (self.rounds != 4 or self.evaluation_steps != (2, 4) or self.test_freq != 2
                or self.observation_profile != RECOVERY_OBSERVATION_PROFILE
                or not self.retire_completed_moments or not self.archive_completed_actors
                or not self.include_sft or self.evaluation_incomplete_mode != "retain_prefix_v1"
                or len(self.cell_specs) != 5):
            raise ValueError("pooling comparison requires the fixed four-round lifecycle")


def recovery_observation_enabled(profile):
    return isinstance(profile, (RecoveryExperimentProfile, ActionPoolingExperimentProfile))


def observation_profile_for(profile):
    return getattr(profile, "observation_profile", "history_features_v1")


def disk_requirement_bytes(profile, remaining_exports):
    """Budget all remaining HF actors and reserve, using the approved lifecycle."""
    if type(remaining_exports) is not int or remaining_exports < 0:
        raise ValueError("remaining export count must be a nonnegative integer")
    moments = 30 if isinstance(profile, RecoveryExperimentProfile) else 90
    return (moments + 9 * remaining_exports + 8) * 1024**3


def verify_one_round_model_budget(model_path):
    """Read BF16 headers without loading a device; reject a larger architecture.

    A fresh actor starts with zero moments. The first actor->critic exchange writes
    an empty snapshot; the first critic->actor exchange writes 8*Ncritic bytes.
    Actor update, final evaluation and actor-only export perform no more exchange.
    Round two would need replacement coexistence and must use the legacy gate.
    """
    import struct

    model_path = Path(model_path)
    config = restart.read(model_path / "config.json")
    if (config.get("model_type") != "qwen3" or config.get("hidden_size") != 2560
            or config.get("tie_word_embeddings") is not True):
        raise RuntimeError("one-round disk budget requires the reviewed tied Qwen3-4B")
    tensors = {}
    for shard in sorted(model_path.glob("*.safetensors")):
        with shard.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise RuntimeError("invalid shared SFT safetensors header")
            size = struct.unpack("<Q", prefix)[0]
            if not 0 < size <= 8 * 1024**2:
                raise RuntimeError("invalid shared SFT safetensors header")
            header = json.loads(stream.read(size))
        for name, item in header.items():
            if name == "__metadata__":
                continue
            shape = item.get("shape")
            offsets = item.get("data_offsets")
            if (name in tensors or item.get("dtype") != "BF16"
                    or not isinstance(shape, list)
                    or any(type(n) is not int or n <= 0 for n in shape)
                    or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(n) is not int for n in offsets)
                    or offsets[1] - offsets[0] != 2 * math.prod(shape)):
                raise RuntimeError("shared SFT tensor differs from BF16 parameter budget")
            tensors[name] = shape
    embedding = tensors.get("model.embed_tokens.weight")
    head = tensors.pop("lm_head.weight", None)
    if embedding != [151936, 2560] or head not in (None, embedding):
        raise RuntimeError("shared SFT embedding or tied head budget changed")
    actor_parameters = sum(math.prod(shape) for shape in tensors.values())
    if actor_parameters != 4022468096:
        raise RuntimeError("shared SFT parameter count exceeds the reviewed one-round budget")
    critic_parameters = actor_parameters + config["hidden_size"] + 1
    moment_bytes = 8 * critic_parameters
    # Export includes tied lm_head as a separate HF tensor. Metadata is covered
    # by the conservative 9 GiB allowance, not another FP32/checkpoint payload.
    export_tensor_bytes = 2 * (actor_parameters + math.prod(embedding))
    if moment_bytes > 30 * 1024**3 or export_tensor_bytes > 9 * 1024**3:
        raise RuntimeError("shared SFT exceeds one-round moment or actor export allowance")
    return dict(
        profile=ONE_ROUND_DISK_PROFILE, actor_parameters=actor_parameters,
        critic_parameters=critic_parameters, maximum_moment_bytes=moment_bytes,
        actor_export_tensor_bytes=export_tensor_bytes,
        moment_allowance_bytes=30 * 1024**3, actor_allowance_bytes=9 * 1024**3,
        reserve_bytes=8 * 1024**3, optimizer_updates_per_role=1,
        role_exchanges=2, optimizer_included=False, actor_only_export=True,
        automatic_retry=False, resume_supported=False,
    )


PILOT = ExperimentProfile(PROFILE, PLAN, tuple(TRAIN_NUMBERS), tuple(EVAL_STEPS), 2)


def settings(model, *, profile=PILOT):
    if model.get("source") != "local_export" or model.get("revision") is not None:
        raise ValueError("low-memory pilot comparison requires the shared pinned local SFT")
    result = restart.settings()
    result.update(
        artifact_profile="actor_bf16_hf_v1",
        action_protocol=tuning.PROTOCOL,
        critic_initialization="cpu_seed_v1",
        seeds=[profile.seed],
        m2_residual_eta=0.0,
        common_validity_penalty=dict(
            profile="environment_rejection_v1",
            coefficient=PENALTY,
            applies_to="training_only",
            financial_terminal_score_unchanged=True,
        ),
        plan=profile.plan,
        arms=["M1", "M2"],
        training_coverage=dict(
            profile=profile.name,
            evaluation_steps=list(profile.evaluation_steps),
            actor_learning_rate=profile.actor_learning_rate,
            critic_learning_rate=1e-6,
            training_episode_ids=profile.train_ids,
            evaluation_episode_ids=profile.eval_ids,
            evaluation_lanes={"greedy": [0], "sampled": [1, 2, 3, 4]},
            training_trajectories_per_arm=profile.rounds * 8,
            planned_trajectories=profile.planned_trajectories,
            automatic_retry=False,
            resume_supported=False,
            adaptive_gate=False,
        ),
    )
    if profile.retire_completed_moments:
        result["training_coverage"]["moment_retention"] = "retire_completed_cell_raw_v1"
    if profile.cell_specs:
        result["training_coverage"]["fixed_cells"] = [asdict(s) for s in cells(profile)]
        result["training_coverage"]["cell_actor_learning_rates"] = dict(
            profile.cell_actor_learning_rates
        )
    if profile.archive_completed_actors:
        result["training_coverage"]["actor_retention"] = "verified_lossless_tensor_xor_v1"
    if profile.evaluation_incomplete_mode:
        result["training_coverage"]["evaluation_incomplete_mode"] = (
            profile.evaluation_incomplete_mode
        )
    if isinstance(profile, RecoveryExperimentProfile):
        result["observation_profile"] = profile.observation_profile
        result["training_coverage"].update(
            purpose="stage1_recovery_sanity_not_four_round_performance_v1",
            disk_budget_profile=profile.disk_budget_profile,
            acceptance_gate_sha256=profile.acceptance_gate_sha256,
            consumed_training_episode_ids=profile.train_ids[:2],
            prepared_training_episode_ids=profile.train_ids,
        )
    elif isinstance(profile, ActionPoolingExperimentProfile):
        result["arms"] = ["M1", "M2-plain", "M2"]
        result["observation_profile"] = profile.observation_profile
        result["training_coverage"].update(
            purpose="stage2_four_round_action_pooling_comparison_v1",
            disk_budget_profile="multi_round_three_generation_adam_v1",
            consumed_training_episode_ids=profile.train_ids,
        )
    result["models"]["screening"] = copy.deepcopy(model)
    result["training"].update(
        episodes=len(profile.train_numbers),
        total_epochs=1,
        total_training_steps=profile.rounds,
        learning_rate=profile.actor_learning_rate,
        critic_learning_rate=1e-6,
        test_freq=getattr(profile, "test_freq", 2),
        save_freq=profile.rounds,
        max_checkpoints=1,
        max_infrastructure_resumes=0,
    )
    result["evaluation"].update(
        rollout_n=5,
        do_sample=True,
        temperature=1.0,
        primary="same_worker_separate_sampled_and_greedy",
    )
    result["resources"].update(
        profile="chunked_adam_disk_v1",
        chunk_bytes=64 * 1024**2,
        gpu_reserve_bytes=2 * 1024**3,
        host_reserve_bytes=2 * 1024**3,
        disk_reserve_bytes=8 * 1024**3,
    )
    return result


def read_settings(base, *, profile=PILOT):
    value = restart.read(base / "settings.json")
    if not tuning._same(value, settings(value["models"]["screening"], profile=profile)):
        raise RuntimeError("low-memory pilot settings differ from the fixed plan")
    return value


def cells(profile=PILOT):
    if profile.cell_specs:
        return profile.cell_specs
    baseline = (CellSpec("sft0", "M1", 0.0, 1e-6, 0),) if profile.include_sft else ()
    return baseline + (
        CellSpec("m1", "M1", 0.0, 1e-6, profile.rounds),
        CellSpec("m2_eta025", "M2", 0.25, 1e-6, profile.rounds),
    )


def cell_settings(run_settings, cell_id, *, profile=PILOT):
    spec = next(spec for spec in cells(profile) if spec.cell_id == cell_id)
    value = copy.deepcopy(run_settings)
    value["training_coverage"]["cell"] = asdict(spec)
    value["m2_residual_eta"] = spec.eta
    value["training"].update(
        learning_rate=profile.actor_lr(cell_id), critic_learning_rate=spec.critic_lr,
    )
    value["training_coverage"]["actor_learning_rate"] = profile.actor_lr(cell_id)
    if spec.zero:
        value["training"].update(total_training_steps=1, test_freq=1, save_freq=1)
    return value


def select_episodes(formal, old, *, profile=PILOT):
    """Use the fixed prefix of the preserved date-only coverage selection."""
    full, source = coverage.select_episodes(formal, old)
    count = len(profile.train_numbers)
    selected = full[:count]
    if [row["episode_id"] for row in selected] != profile.train_ids:
        raise RuntimeError("coverage windows differ from the fixed plan")
    selection = copy.deepcopy(source)
    selection.update(
        criterion=(f"first {count} windows of the preserved date-only coverage selection"),
        coverage_reference_profile=coverage.PROFILE,
        coverage_full_training_episode_ids=list(coverage.TRAIN_IDS),
        training_episode_ids=profile.train_ids,
        unique_training_sessions=len(
            {str(day)[:10] for row in selected for day in row["episode_dates"]}
        ),
        windows=source["windows"][:count],
    )
    return selected, selection


def select_evaluation_episodes(formal, old, *, profile=PILOT):
    """Expand only into the previously reserved original-Train screen-dev role."""
    by_id = {row["episode_id"]: row for row in formal["splits"]["train"]}
    reserved = {row["episode_id"] for row in old["selected_episodes"]["dev"]}
    if not set(profile.eval_ids) <= reserved:
        raise RuntimeError("expanded evaluation must remain in reserved Train screen-dev")
    selected = [by_id[episode_id] for episode_id in profile.eval_ids]
    seen = set()
    for row in selected:
        days = {str(day)[:10] for day in row["episode_dates"]}
        if len(days) != 21 or seen & days:
            raise RuntimeError("expanded evaluation windows must have disjoint 21-session dates")
        seen.update(days)
    criterion = (
        "fixed known-rejection and normal regression Train windows; not an untouched holdout"
        if isinstance(profile, RecoveryExperimentProfile) else
        "fixed chronological original-Train screen-dev windows; no outcome selection"
    )
    return selected, dict(
        criterion=criterion,
        episode_ids=profile.eval_ids,
        unique_evaluation_sessions=len(seen),
        first_session=min(seen), last_session=max(seen),
        old_full_train_calibration_included_dev=True,
        is_untouched_holdout=False,
    )


def prepare(base, *, model_path, merge_receipt_sha256, reservation_directory=None,
            coordinated_reservation_directory=None, coordinator_token=None, profile=PILOT):
    import pandas as pd

    from src.agents.config import load_experiment_config
    from src.data.screening import _dataset_rows, _environment_common

    if base.exists():
        raise RuntimeError("low-memory pilot run requires a new directory")
    if reservation_directory is not None and coordinated_reservation_directory is not None:
        raise ValueError("one-shot and persistent GPU handoff cannot both be enabled")
    model = dict(
        source="local_export",
        name=str(model_path.resolve()),
        merge_receipt_sha256=merge_receipt_sha256,
    )
    name, identity, path = _resolve_exact_model_snapshot({"model": model})
    model["name"] = name
    if isinstance(profile, RecoveryExperimentProfile):
        verify_one_round_model_budget(path)
    original = REPO_ROOT / "artifacts/m2_screening"
    old = restart.read(original / "manifest.json")
    formal_path = Path(old["original_manifest"]["path"])
    if restart.digest(formal_path) != old["original_manifest"]["sha256"]:
        raise RuntimeError("reviewed formal manifest changed")
    calibration = original / old["calibration"]["path"]
    if restart.digest(calibration) != old["calibration"]["sha256"]:
        raise RuntimeError("existing calibration changed")
    formal = yaml.safe_load(formal_path.read_text())
    selected, selection = select_episodes(formal, old, profile=profile)
    config = load_experiment_config(REPO_ROOT / "configs/base.yaml")
    environment = _environment_common(
        config=config, source=old["source"], calibration=restart.read(calibration)
    )
    if recovery_observation_enabled(profile):
        environment["observation_profile"] = profile.observation_profile
    training = _dataset_rows(
        role="train",
        episodes=selected,
        indices=[n - 1 for n in profile.train_numbers],
        environment_common=environment,
    )
    for row in training:
        row["extra_info"]["screening_role"] = "low_memory_pilot_train"
    old_run = REPO_ROOT / "runs/m2-training-tuning-20260914"
    old_dev = old_run / "prepared/dev.parquet"
    old_manifest = restart.read(old_run / "prepared/manifest.json")
    if restart.digest(old_dev) != old_manifest["datasets"]["dev"]["sha256"]:
        raise RuntimeError("fixed four Train evaluation windows changed")
    evaluation = None
    if profile.eval_ids != EVAL_IDS:
        selected_eval, evaluation_selection = select_evaluation_episodes(
            formal, old, profile=profile,
        )
        evaluation = _dataset_rows(
            role="dev", episodes=selected_eval,
            indices=[n - 1 for n in profile.eval_numbers], environment_common=environment,
        )
        train_days = {str(day)[:10] for row in selected for day in row["episode_dates"]}
        eval_days = {str(day)[:10] for row in selected_eval for day in row["episode_dates"]}
        if train_days & eval_days:
            raise RuntimeError("expanded Train evaluation dates overlap training")
        selection["evaluation_selection"] = evaluation_selection
    base.mkdir(parents=True, exist_ok=False)
    handoff = {"profile": "disabled"}
    if reservation_directory is not None:
        from src.agents.gpu_handoff import config_for_holder

        handoff = config_for_holder(reservation_directory)
    atomic_json(base / "gpu_handoff_config.json", handoff)
    if coordinated_reservation_directory is not None:
        from uuid import uuid4

        from src.agents.gpu_memory_coordinator import config_for_holder, freeze_config

        coordinated = config_for_holder(
            coordinated_reservation_directory,
            lease_token=coordinator_token or uuid4().hex,
            evidence_directory=base / "artifacts/gpu_memory_coordinator",
        )
        freeze_config(base / "gpu_coordinator_config.json", coordinated)
    target = base / "prepared"
    target.mkdir()
    pd.DataFrame(training).to_parquet(target / "train.parquet", index=False)
    if evaluation is None:
        shutil.copy2(old_dev, target / "dev.parquet")
    else:
        pd.DataFrame(evaluation).to_parquet(target / "dev.parquet", index=False)
    shutil.copy2(calibration, target / "reward_calibration.json")
    datasets = {
        n: dict(
            path=n + ".parquet",
            episode_ids=ids,
            row_count=len(ids),
            sha256=restart.digest(target / (n + ".parquet")),
        )
        for n, ids in (("train", profile.train_ids), ("dev", profile.eval_ids))
    }
    manifest = dict(
        datasets=datasets,
        calibration=old["calibration"],
        source_manifest_sha256=restart.digest(formal_path),
        eval_is_reserved_train_subset=True,
        train_eval_dates_disjoint=True,
        created_at_SGT=restart.now(),
        selection=selection,
    )
    manifest["hashes"] = {
        "payload_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    }
    atomic_json(target / "manifest.json", manifest)
    selection.update(
        actor_learning_rate=profile.actor_learning_rate,
        critic_learning_rate=1e-6,
        actor_lr_and_date_coverage_changed_together=False,
        **{("same_as_coverage_first_round" if profile.rounds == 1 else
            "same_as_coverage_first_two_rounds" if profile.rounds == 2 else
            "same_as_coverage_first_four_rounds"): True},
        common_invalid_action_penalty=PENALTY,
        critic_lr_basis=(
            "completed same-batch critic-fit comparison; unchanged from previous main run"
        ),
        source_sha256={
            str(p): restart.digest(p)
            for p in (
                formal_path,
                original / "manifest.json",
                calibration,
                old_dev,
                old_run / "selection.json",
            )
        },
    )
    atomic_json(base / "selection.json", selection)
    atomic_json(base / "settings.json", settings(model, profile=profile))
    atomic_json(
        base / "model_identity.json",
        dict(
            model=name,
            model_revision=identity,
            model_snapshot_path=str(path),
            model_source="local_export",
            merge_receipt_sha256=merge_receipt_sha256,
        ),
    )


async def input_preflight(base, *, profile=PILOT):
    import pandas as pd

    agent_path = str(REPO_ROOT / "third_party/agent-r1")
    if agent_path not in sys.path:
        sys.path.insert(0, agent_path)
    from agent_r1.env.base import Action
    from transformers import AutoTokenizer

    from src.agents.screening_prompt import apply_screening_action_protocol
    from src.env.trading import TradingEnv

    config = read_settings(base, profile=profile)
    _, identity, snapshot = _resolve_exact_model_snapshot({"model": config["models"]["screening"]})
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    records, session_sets = [], {}
    for dataset, expected in (("train", profile.train_ids), ("dev", profile.eval_ids)):
        rows = pd.read_parquet(base / "prepared" / f"{dataset}.parquet").to_dict("records")
        if [r["extra_info"]["episode_id"] for r in rows] != expected or [
            r["extra_info"]["index"] for r in rows
        ] != list(range(len(expected))):
            raise RuntimeError("prepared episode or sample-index order changed")
        session_sets[dataset] = set()
        for row in rows:
            if row["extra_info"]["source_split"] != "train":
                raise RuntimeError("low-memory pilot trial may only use original Train rows")
            kwargs = json.loads(row["env_kwargs"])
            kwargs.update(
                observation_profile=observation_profile_for(profile),
                reward_mode="finance_trajectory"
            )
            if len(kwargs["episode_dates"]) != 21:
                raise RuntimeError("low-memory pilot trial requires twenty-one-session windows")
            session_sets[dataset].update(kwargs["episode_dates"])
            env = TradingEnv(**kwargs)
            obs = env.reset(raw_prompt=list(row["prompt"]))
            rejection_tokens = None
            if recovery_observation_enabled(profile):
                tool = {"name": "sell", "arguments": {
                    "symbol": env.symbols[0], "target_weight": 0.1,
                }}
                rejected, _, done, info = await env.step(Action(
                    text="<tool_call>" + json.dumps(tool) + "</tool_call>",
                ))
                if not info.get("invalid_action") or done or env.current_session_index != 0:
                    raise RuntimeError("CPU rejection feedback replay failed")
                messages = apply_screening_action_protocol(rejected.messages, tuning.PROTOCOL)
                rejection_tokens = len(tokenizer.apply_chat_template(
                    messages, tools=env.tool_schemas, add_generation_prompt=True,
                    enable_thinking=False, tokenize=True,
                ))
                if rejection_tokens > 8192:
                    raise RuntimeError("recovery feedback prompt exceeds common budget")
                obs = env.reset(raw_prompt=list(row["prompt"]))
            counts = []
            for session in range(21):
                messages = apply_screening_action_protocol(obs.messages, tuning.PROTOCOL)
                tokens = tokenizer.apply_chat_template(
                    messages,
                    tools=env.tool_schemas,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=True,
                )
                counts.append(len(tokens))
                if len(tokens) > 8192:
                    raise RuntimeError("low-memory pilot prompt exceeds common budget")
                if session < 20:
                    obs, _, done, info = await env.step(
                        Action(text='<tool_call>{"name":"hold","arguments":{}}</tool_call>')
                    )
                    if info.get("invalid_action") or done != (session == 19):
                        raise RuntimeError("CPU complete-window replay failed")
            records.append(
                dict(
                    episode=row["extra_info"]["episode_id"],
                    complete=env.episode_complete,
                    max_prompt_tokens=max(counts),
                    **({"rejection_prompt_tokens": rejection_tokens}
                       if rejection_tokens is not None else {}),
                )
            )
    if session_sets["train"] & session_sets["dev"]:
        raise RuntimeError("training and evaluation sessions overlap")
    atomic_json(
        base / "input_preflight.json",
        dict(
            status="PASSED",
            at_SGT=restart.now(),
            windows=records,
            model_revision=identity,
            tokenizer_path=str(snapshot),
            action_protocol=tuning.PROTOCOL,
            enable_thinking=False,
            **({"observation_profile": profile.observation_profile}
               if recovery_observation_enabled(profile) else {}),
        ),
    )


def source_files():
    return tuning._source_files()


def input_files(base):
    """Seal an explicitly enabled coordinator without changing old input inventories."""
    optional = ("gpu_coordinator_config.json",) if (
        base / "gpu_coordinator_config.json"
    ).exists() else ()
    return (*INPUTS, *optional)


def seal(base, *, profile=PILOT):
    if (base / "run_contract.json").exists() or (base / "state.json").exists():
        raise RuntimeError("low-memory pilot trial already sealed or started")
    read_settings(base, profile=profile)
    pre = restart.read(base / "input_preflight.json")
    if (
        pre.get("status") != "PASSED"
        or [r.get("episode") for r in pre.get("windows", [])]
        != profile.train_ids + profile.eval_ids
        or any(r.get("complete") is not True for r in pre["windows"])
    ):
        raise RuntimeError("all fixed complete CPU windows must pass")
    if recovery_observation_enabled(profile) and (
        pre.get("observation_profile") != profile.observation_profile
        or any(type(r.get("rejection_prompt_tokens")) is not int
               or not 0 < r["rejection_prompt_tokens"] <= 8192 for r in pre["windows"])
    ):
        raise RuntimeError("recovery observation and feedback preflight must match the contract")
    shutil.copy2(REPO_ROOT / profile.plan, base / "plan.md")
    sources = {}
    for relative, path in source_files().items():
        target = base / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[relative] = restart.digest(path)
    atomic_json(base / "source_identity.json", sources)
    atomic_json(
        base / "run_contract.json",
        dict(
            profile=profile.name,
            sealed_at_SGT=restart.now(),
            inputs={p: restart.digest(base / p) for p in input_files(base)},
        ),
    )
    check_contract(base, profile=profile)


def check_contract(base, *, profile=PILOT):
    seal = restart.read(base / "run_contract.json")
    if (seal.get("profile") != profile.name
            or set(seal.get("inputs", {})) != set(input_files(base))):
        raise RuntimeError("low-memory pilot identity is incomplete")
    for p, digest in seal["inputs"].items():
        if restart.digest(base / p) != digest:
            raise RuntimeError("sealed low-memory pilot input changed: " + p)
    sources = restart.read(base / "source_identity.json")
    if set(sources) != set(source_files()):
        raise RuntimeError("low-memory pilot source inventory changed")
    for p, path in source_files().items():
        if restart.digest(path) != sources[p]:
            raise RuntimeError("low-memory pilot source changed: " + p)
    for p, digest in restart.read(base / "selection.json")["source_sha256"].items():
        if restart.digest(Path(p)) != digest:
            raise RuntimeError("low-memory pilot selection source changed: " + p)
    manifest = restart.read(base / "prepared/manifest.json")
    for name, ids in (("train", profile.train_ids), ("dev", profile.eval_ids)):
        row = manifest["datasets"][name]
        if row["episode_ids"] != ids or row["row_count"] != len(ids):
            raise RuntimeError("low-memory pilot window budget changed")
    for row in [*manifest["datasets"].values(), manifest["calibration"]]:
        if restart.digest(base / "prepared" / row["path"]) != row["sha256"]:
            raise RuntimeError("low-memory pilot prepared data changed")
    cfg = read_settings(base, profile=profile)
    _, identity, path = _resolve_exact_model_snapshot({"model": cfg["models"]["screening"]})
    recorded = restart.read(base / "model_identity.json")
    pre = restart.read(base / "input_preflight.json")
    if (
        recorded["model_revision"] != identity
        or recorded["model_snapshot_path"] != str(path)
        or pre["model_revision"] != identity
        or pre["action_protocol"] != tuning.PROTOCOL
        or pre["enable_thinking"] is not False
    ):
        raise RuntimeError("shared SFT or prompt identity changed")
    if (recovery_observation_enabled(profile)
            and pre.get("observation_profile") != profile.observation_profile):
        raise RuntimeError("recovery preflight observation profile changed")
    if isinstance(profile, RecoveryExperimentProfile):
        verify_one_round_model_budget(path)


def make_config(base, manifest, cell_id, *, profile=PILOT):
    cfg = cell_settings(read_settings(base, profile=profile), cell_id, profile=profile)
    spec = cfg["training_coverage"]["cell"]
    result = build_cell_config(
        settings=cfg,
        manifest=manifest,
        prepared_dir=base / "prepared",
        arm=spec["arm"],
        stage="zero-shot" if spec["rounds"] == 0 else "screening",
        seed=profile.seed,
        cell_dir=base / cell_id,
    )
    result["training"].update(cfg["training"], critic_initialization="cpu_seed_v1")
    result["run"].update(name=f"low-memory-pilot-{cell_id}-s{profile.seed}", kind=profile.name)
    result["screening"].update(
        coverage_cell_id=cell_id,
        invalid_action_penalty=PENALTY,
        common_validity_penalty=copy.deepcopy(cfg["common_validity_penalty"]),
    )
    if profile.evaluation_incomplete_mode:
        result["screening"]["evaluation_incomplete_mode"] = profile.evaluation_incomplete_mode
    if handoff_config(base) is not None:
        result["screening"]["gpu_handoff_config_sha256"] = restart.digest(
            base / "gpu_handoff_config.json"
        )
    if memory_coordinator_config(base) is not None:
        result["screening"]["gpu_coordinator_config_sha256"] = restart.digest(
            base / "gpu_coordinator_config.json"
        )
    return result


def handoff_config(base):
    path = base / "gpu_handoff_config.json"
    if not path.exists():
        # Old fixtures/config construction may omit the optional resource hook.
        # A real prepared run must include the file in its sealed INPUTS.
        return None
    value = restart.read(path)
    if value == {"profile": "disabled"}:
        return None
    from src.agents.gpu_handoff import load_config

    return load_config(path, restart.digest(path))


def memory_coordinator_config(base):
    path = base / "gpu_coordinator_config.json"
    if not path.exists():
        return None
    if handoff_config(base) is not None:
        raise RuntimeError("one-shot and persistent GPU handoff conflict")
    from src.agents.gpu_memory_coordinator import load_config

    return load_config(path, restart.digest(path))


def build_command(*, config, run_settings, hashes, cell_dir, profile=PILOT):
    base_settings = settings(run_settings["models"]["screening"], profile=profile)
    cell_id = run_settings["training_coverage"]["cell"]["cell_id"]
    spec = next(s for s in cells(profile) if s.cell_id == cell_id)
    if not tuning._same(run_settings, cell_settings(base_settings, spec.cell_id, profile=profile)):
        raise RuntimeError("cell condition differs from fixed low-memory pilot settings")
    expected = {**run_settings["training"], "critic_initialization": "cpu_seed_v1"}
    if any(not tuning._same(config["training"].get(k), v) for k, v in expected.items()):
        raise RuntimeError("resolved training budget/LR differs")
    if (
        config["screening"]["arm"] != spec.arm
        or config["screening"]["residual_eta"] != spec.eta
        or config["model"] != run_settings["models"]["screening"]
        or config["run"]["seed"] != profile.seed
        or config["screening"]["stage"] != ("zero-shot" if spec.zero else "screening")
        or config["screening"]["artifact_profile"] != "actor_bf16_hf_v1"
        or config["screening"]["coverage_cell_id"] != spec.cell_id
        or config["screening"].get("invalid_action_penalty") != PENALTY
        or config["screening"].get("common_validity_penalty")
        != run_settings["common_validity_penalty"]
    ):
        raise RuntimeError("resolved arm, eta or shared model differs")
    command = build_cell_command(
        config=config, settings=run_settings, hashes=hashes, cell_dir=cell_dir
    )
    if profile.cell_specs:
        command = set_overrides(command, {"data.val_max_samples": len(profile.eval_ids)})
    prefix = "+ray_kwargs.ray_init.runtime_env.env_vars."
    command = set_overrides(
        command,
        {
            prefix + "FINANCE_COMMON_INVALID_PENALTY": json.dumps(str(PENALTY)),
            prefix + "FINANCE_SCREENING_CRITIC_INIT": "cpu_seed_v1",
            prefix + "FINANCE_SCREENING_CRITIC_INIT_EVIDENCE_DIR": str(
                cell_dir / "artifacts/critic_initialization"
            ),
            prefix + "FINANCE_CRITIC_FIT_PROBE_DIR": str(cell_dir / "artifacts/critic_fit"),
            prefix + "FINANCE_ACTOR_UPDATE_PROBE_DIR": str(cell_dir / "artifacts/actor_update"),
            prefix + "FINANCE_PAIRED_VALIDATION_PROFILE": "greedy1_sample4_v1",
            prefix + "FINANCE_PAIRED_VALIDATION_EVIDENCE_DIR": str(
                cell_dir / "artifacts/paired_validation"
            ),
            "critic.model.tokenizer_path": config["model"]["name"],
            "+actor_rollout_ref.ref.model.path": config["model"]["name"],
            "data.validation_shuffle": False,
            "actor_rollout_ref.rollout.val_kwargs.n": 5,
            "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
            "actor_rollout_ref.rollout.val_kwargs.temperature": 1.0,
        },
    )
    if profile.evaluation_incomplete_mode:
        if config["screening"].get("evaluation_incomplete_mode") != (
            profile.evaluation_incomplete_mode
        ):
            raise RuntimeError("resolved incomplete evaluation profile differs")
        command = set_overrides(command, {
            prefix + "FINANCE_GRID_EVALUATION_INCOMPLETE": profile.evaluation_incomplete_mode,
        })
    if handoff_config(cell_dir.parent) is not None:
        path = cell_dir.parent / "gpu_handoff_config.json"
        digest = restart.digest(path)
        if config["screening"].get("gpu_handoff_config_sha256") != digest:
            raise RuntimeError("handoff config differs from execution identity")
        command = set_overrides(command, {
            prefix + "FINANCE_GPU_HANDOFF_CONFIG": str(path.resolve()),
            prefix + "FINANCE_GPU_HANDOFF_CONFIG_SHA256": json.dumps(digest),
            prefix + "FINANCE_GPU_HANDOFF_EVIDENCE_DIR": str(
                (cell_dir / "artifacts/gpu_handoff").resolve()
            ),
        })
    if memory_coordinator_config(cell_dir.parent) is not None:
        path = cell_dir.parent / "gpu_coordinator_config.json"
        digest = restart.digest(path)
        if config["screening"].get("gpu_coordinator_config_sha256") != digest:
            raise RuntimeError("persistent GPU coordinator differs from execution identity")
        command = set_overrides(command, {
            prefix + "FINANCE_GPU_COORDINATOR_CONFIG": str(path.resolve()),
            prefix + "FINANCE_GPU_COORDINATOR_CONFIG_SHA256": json.dumps(digest),
            prefix + "VLLM_LOGGING_LEVEL": "INFO",
        })
    return command


def validate_extra_evidence(cell, spec, identity, *, profile=PILOT):
    # Each implementation validates its real runtime schema and fixed budget.
    from src.agents.actor_update_probe import validate_run_evidence as validate_actor
    from src.agents.paired_validation import validate_run_evidence as validate_paired

    actor = validate_actor(
        cell / "artifacts/actor_update",
        expected_updates=spec.rounds,
        arm=spec.arm,
        execution_identity=identity,
    )
    paired = validate_paired(
        cell / "artifacts/paired_validation",
        expected_steps=[0] if spec.zero else list(profile.evaluation_steps),
        expected_episode_ids=profile.eval_ids,
        arm=spec.arm,
        execution_identity=identity,
        **({"allow_incomplete": True} if profile.evaluation_incomplete_mode else {}),
    )
    penalty = common.validate_common_penalty_evidence(
        cell, spec, expected_seed=profile.seed,
        **({"allow_incomplete_evaluation": True} if profile.evaluation_incomplete_mode else {}),
    )
    if not spec.zero and profile.cell_specs:
        rows = read_jsonl(cell / "artifacts/actor_update")
        if len(rows) != spec.rounds or any(
            not row.get("actor_learning_rates")
            or any(rate != profile.actor_lr(spec.cell_id) for rate in row["actor_learning_rates"])
            for row in rows
        ):
            raise RuntimeError("actual actor optimizer learning rate differs from fixed cell")
    return dict(actor_update=actor, paired_validation=paired, common_penalty_evidence=penalty)


def validate_placement(cell, spec, identity):
    rows = read_jsonl(cell / "artifacts/placement")
    if not rows or any(
        row.get("profile") != "chunked_adam_disk_v1"
        or row.get("identity_sha256") != identity
        or row.get("failed") is True
        or str(row.get("phase", "")).endswith(":error")
        for row in rows
    ):
        raise RuntimeError("chunked placement identity or boundary failure")
    if spec.zero and (
        any(str(r.get("phase", "")).split(":", 1)[0]
            in {"update_actor", "update_critic", "save_checkpoint"} for r in rows)
        or any((cell / "checkpoints").rglob("*"))
    ):
        raise RuntimeError("SFT0 unexpectedly trained or exported a checkpoint")
    events = [r["detail"] for r in rows if r.get("event") == "exchange"]
    if any(r.get("event") == "failed" for r in events):
        raise RuntimeError("chunked placement recorded a failed operation")
    result = {"status": "PASSED", "updates": {}, "exchanges": []}
    for role in ("critic", "actor"):
        done = [r for r in events if r.get("event") == "optimizer_complete"
                and r.get("role") == role]
        updates = [r["detail"] for r in events if r.get("event") == "optimizer_chunked"
                   and r.get("role") == role]
        if [r.get("role_step") for r in done] != list(range(1, spec.rounds + 1)) or (
            len(updates) != spec.rounds or any(
                r.get("event") != "chunked_adamw_step_complete"
                or r.get("chunks", 0) < 1
                or not 0 < r.get("maximum_temporary_bytes", 0) <= 64 * 1024**2
                or r.get("moment_transfer_bytes", 0) <= 0
                for r in updates
            )
        ):
            raise RuntimeError("chunked Adam did not complete the fixed real CUDA updates")
        result["updates"][role] = updates
    pending = None
    for row in events:
        if row.get("event") == "exchange_start":
            if pending is not None:
                raise RuntimeError("overlapping optimizer state exchanges")
            pending = row
        elif row.get("event") == "exchange_complete":
            if pending is None:
                raise RuntimeError("optimizer state exchange is missing its start")
            active = row["active_role"]
            inactive = "critic" if active == "actor" else "actor"
            if (row["moment_cpu_bytes_by_role"][inactive] != 0
                    or row["disk_status"][inactive] != "offloaded"
                    or row["disk_status"][active] != "resident"):
                raise RuntimeError("inactive optimizer moments remained in CPU memory")
            result["exchanges"].append(dict(
                active_role=active,
                elapsed_seconds=(row["time_ns"] - pending["time_ns"]) / 1e9,
                moment_cpu_bytes_by_role=row["moment_cpu_bytes_by_role"],
            ))
            pending = None
    if pending is not None or len(result["exchanges"]) != 2 * spec.rounds:
        raise RuntimeError("incomplete actor/critic disk exchanges")
    for field in ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"):
        values = [r[field] for r in rows if field in r]
        if not values:
            raise RuntimeError("worker CUDA peak evidence missing")
        result[field] = max(values)
    result["peak_container_memory_bytes_sampled"] = max(
        int(r.get("memory.current", 0)) for r in rows
    )
    return result


def validate_handoff(cell, identity):
    config = handoff_config(cell.parent)
    if config is None:
        return {"status": "DISABLED"}
    paths = list((cell / "artifacts/gpu_handoff").glob("receipt-*.json"))
    if len(paths) != 1:
        raise RuntimeError("one shared-worker GPU handoff receipt is required")
    receipt = restart.read(paths[0])
    if (receipt.get("status") != "PASSED" or receipt.get("identity") != identity
            or receipt.get("holder_config") != config or receipt.get("first_role") != "critic"
            or [row.get("role") for row in receipt.get("role_checks", [])]
            != ["reference", "actor"]
            or receipt.get("worker_memory", {}).get("driver_free_bytes", 0)
            < MIN_FREE_GPU_MIB * 1024**2):
        raise RuntimeError("GPU handoff initialization evidence differs from the fixed profile")
    states = [receipt["after"], *[row["holder_status"] for row in receipt["role_checks"]]]
    if any(state.get("paused") is not True or any(
        state.get(key) != 0 for key in (
            "payload_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes"
        )
    ) for state in states):
        raise RuntimeError("GPU holder was not inactive throughout model initialization")
    return dict(status="PASSED", receipt=str(paths[0]), worker_pid=receipt["worker_pid"],
                free_after_release_bytes=receipt["worker_memory"]["driver_free_bytes"])


def evaluation_readout(cell, *, policy_step=None, profile=PILOT):
    """Descriptive only; the caller has already validated the complete fixed grid."""
    rows = read_jsonl(cell / "artifacts/screening_evidence")
    result = {}
    for lane, siblings in (("greedy", {0}), ("sampled", {1, 2, 3, 4})):
        selected = []
        for row in rows:
            identity = row.get("rollout_identity", row.get("identity"))
            if (identity["validate"] and identity["rollout_n"] in siblings
                    and (policy_step is None or identity["step"] == policy_step)):
                selected.append(row)
        if len(selected) != len(profile.eval_ids) * len(siblings):
            raise RuntimeError("evaluation lane size differs")
        attempted = selected
        if profile.evaluation_incomplete_mode:
            selected = [row for row in attempted
                        if row.get("complete") is True and row.get("status") == "COMPLETE"]
            failed = [row for row in attempted
                      if row.get("complete") is not True or row.get("status") != "COMPLETE"]
            if any(row.get("status") != "INCOMPLETE" or row.get("complete") is not False
                   or row.get("evaluation_failure_profile") != profile.evaluation_incomplete_mode
                   or row.get("failure_kind") != "trajectory_max_steps"
                   or row.get("metrics") is not None or row.get("terminal_score") is not None
                   for row in failed):
                raise RuntimeError("incomplete evaluation is not a registered unscored prefix")
        metrics = {}
        fields = ("raw_net_return", "transaction_cost", "terminal_score")
        if profile.include_sft:
            fields += ("max_drawdown", "agent_volatility", "benchmark_excess", "turnover",
                       "maximum_position_weight")
        for field in fields:
            values = [r["metrics"][field] for r in selected]
            if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
                raise RuntimeError("nonfinite evaluation outcome")
            metrics["mean_" + field] = math.fsum(values) / len(values) if values else None
        steps = [step for r in selected for step in r["steps"]]
        result[lane] = dict(
            trajectories=len(selected), decisions=len(steps),
            invalid_decisions=sum(s.get("invalid_action") is True for s in steps),
            **metrics,
        )
        if profile.evaluation_incomplete_mode:
            result[lane].update(
                planned_trajectories=len(profile.eval_ids) * len(siblings),
                attempted_trajectories=len(attempted), complete_trajectories=len(selected),
                incomplete_trajectories=len(attempted) - len(selected),
                financial_status="INCOMPLETE" if len(selected) != len(attempted) else "COMPLETE",
                financial_scope="complete_trajectories_only",
            )
    return result


def refresh_status(base, *, profile=PILOT):
    state = restart.read(base / "state.json") if (base / "state.json").exists() else {}
    queue = restart.read(base / "gpu_queue_state.json") if (
        base / "gpu_queue_state.json"
    ).exists() else {}
    status = state.get("status", "PREPARED")
    current = state.get("current_cell")
    if status == "RUNNING" and queue.get("cell") == current and (
        queue.get("status") == "WAITING_FOR_GPU"
    ):
        status = "WAITING_FOR_GPU"
    title = ("共同恢复 SFT / M1 / M2 单轮工程验收" if
             isinstance(profile, RecoveryExperimentProfile) else
             "4B SFT / M1 / M2 参数网格" if profile.cell_specs else
             "4B SFT / M1 / M2 四轮对照" if profile.include_sft else "4B 低显存两轮工程验收")
    evaluation_n = len(profile.eval_ids) * 5
    budget = (f"固定预算：共 {profile.planned_trajectories} 条；每训练臂 "
              f"{profile.rounds * 8} 条训练 + "
              f"{evaluation_n * len(profile.evaluation_steps)} 条评估；"
              f"SFT0 {evaluation_n} 条评估。" if profile.include_sft else
              f"固定预算：共 {profile.planned_trajectories} 条；M1/M2 各 "
              f"{profile.rounds * 8} 条训练 + {evaluation_n} 条同 worker 评估。")
    lines = ["# " + title, "", f"当前进展（新加坡时间）：{restart.now()}",
             "", f"状态：**{status}**；当前臂：{current or '尚未开始'}。",
             f"本批 seed：{profile.seed}。",
             "", budget,
             ("4B 全参数；新共同恢复 SFT；单轮 stage1 sanity；非四轮性能证据。" if
              isinstance(profile, RecoveryExperimentProfile) else
              "4B 全参数；共同 SFT v3；non-thinking；CPU 活跃 Adam moments + 磁盘非活跃 moments。"),
             "模型导出不含优化器；本轮不支持断点续训或自动重跑。", ""]
    if queue:
        free = queue.get("free_MiB")
        lines += [f"GPU 准入读数：{free / 1024:.2f} GiB 空闲（门槛 60 GiB）。" if free
                  is not None else "GPU 空闲读数暂缺。", ""]
        if queue.get("holder_payload_bytes") is not None:
            lines += [f"占位实际持有 {queue['holder_payload_bytes'] / 1024**3:.2f} GiB；"
                      f"空闲加可释放占位 {queue['available_bytes'] / 1024**3:.2f} GiB。",
                      "只在 critic 首次加载前释放；训练期间及两臂之间不自动回补。", ""]
    for spec in cells(profile):
        cell = base / spec.cell_id
        summary = restart.read(cell / "summary.json") if (
            cell / "summary.json"
        ).exists() else {}
        observed = 0
        for path in (cell / "artifacts/screening_evidence").glob("*.jsonl"):
            with path.open() as stream:
                observed += sum(bool(line.strip()) for line in stream)
        planned = evaluation_n if spec.zero else (
            spec.rounds * 8 + evaluation_n * len(profile.evaluation_steps)
        )
        lines += [f"- {spec.cell_id}：{summary.get('status', 'PENDING')}；"
                  f"已落盘轨迹 {observed}/{planned}。"]
        if summary.get("error"):
            lines += [f"  错误：{summary['error']}"]
        if summary.get("placement"):
            p = summary["placement"]
            lines += [f"  训练 worker 的 PyTorch 峰值分配："
                      f"{p['gpu_peak_allocated_bytes'] / 1024**3:.2f} GiB；"
                      f"整臂耗时 {summary['elapsed_seconds'] / 60:.1f} 分钟。"]
        if summary.get("evaluation"):
            for lane, row in summary["evaluation"].items():
                lines += [f"  {lane}：平均净收益 {100 * row['mean_raw_net_return']:.4f}%；"
                          f"非法请求 {row['invalid_decisions']}/{row['decisions']}。"]
    if state.get("error"):
        lines += ["", "停止原因：" + state["error"]]
    scope = ("本轮比较共同 SFT 起点、M1 和 M2；单 seed Train 读数不替代正式验证。"
             if profile.include_sft else
             "本轮衡量资源路径能否跑通；两轮 Train 读数不能证明 M2 优于 M1。")
    lines += ["", scope,
              "阶段显存包含本进程指标；nvidia-smi 的整卡读数还包含其他用户进程。", ""]
    path = base / "LIVE_STATUS.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(path)
    if state.get("status") in {"COMPLETE", "FAILED"}:
        shutil.copy2(path, base / "FINAL_RESULTS.md")


def monitor_status(base, stop, profile=PILOT):
    while not stop.is_set():
        try:
            refresh_status(base, profile=profile)
        except Exception as error:
            atomic_json(base / "monitor_error.json", dict(at_SGT=restart.now(), error=str(error)))
        stop.wait(30)


def capture_cell_process(command, *, environment, cell_dir, profile=PILOT):
    if isinstance(profile, ActionPoolingExperimentProfile):
        from src.agents.action_pooling_gpu_monitor import capture_process as guarded_capture

        return guarded_capture(command, environment=environment, cell_dir=cell_dir)
    return capture_process(command, environment=environment, cell_dir=cell_dir)


def run_cell(base, cell_id, *, profile=PILOT):
    started = time.monotonic()
    check_contract(base, profile=profile)
    phase = read_settings(base, profile=profile)
    spec = next(s for s in cells(profile) if s.cell_id == cell_id)
    cfg = cell_settings(phase, cell_id, profile=profile)
    manifest = restart.read(base / "prepared/manifest.json")
    cell = base / cell_id
    cell.mkdir(exist_ok=False)
    config = make_config(base, manifest, cell_id, profile=profile)
    agent = REPO_ROOT / config["framework"]["agent_r1_path"]
    hashes = collect_run_hashes(
        repo_root=REPO_ROOT,
        agent_r1_path=agent,
        data_version=config["data"]["version"],
        expected_agent_r1_commit=config["framework"]["agent_r1_commit"],
        require_clean_repo=False,
    )
    atomic_json(cell / "resolved_config.json", config)
    atomic_json(cell / "settings.json", cfg)
    command = build_command(config=config, run_settings=cfg, hashes=hashes, cell_dir=cell,
                            profile=profile)
    runtime_path = base / "runtime_preflight.json"
    if not runtime_path.exists():
        atomic_json(runtime_path, verify_gpu_runtime_and_model(config))
    runtime = restart.read(runtime_path)
    recorded = restart.read(base / "model_identity.json")
    if any(runtime[k] != recorded[k] for k in ("model_revision", "model_snapshot_path")):
        raise RuntimeError("runtime preflight does not belong to shared SFT")
    command = _finalize_model_paths(
        command, arm_id=config["arm"]["id"], model_snapshot_path=runtime["model_snapshot_path"]
    )
    env = _training_environment(
        agent_path=agent,
        probe_root=cell / "artifacts/update_probe",
        c_compiler=runtime["c_compiler"],
    )
    env["PYTHONUNBUFFERED"] = "1"
    for key in list(env):
        if key.startswith(
            (
                "FINANCE_SCREENING_",
                "FINANCE_LEARNING_",
                "FINANCE_CRITIC_",
                "FINANCE_ACTOR_",
                "FINANCE_PAIRED_",
                "FINANCE_COMMON_",
                "FINANCE_CHUNKED_",
                "FINANCE_GPU_HANDOFF_",
                "FINANCE_GPU_COORDINATOR_",
                "FINANCE_GRID_",
            )
        ):
            env.pop(key)
    # Apply the exact same setting to the launcher and its Ray workers.  Never
    # inherit an unrelated experiment's penalty from the supervising shell.
    env["FINANCE_COMMON_INVALID_PENALTY"] = str(PENALTY)
    if profile.evaluation_incomplete_mode:
        env["FINANCE_GRID_EVALUATION_INCOMPLETE"] = profile.evaluation_incomplete_mode
    if memory_coordinator_config(base) is not None:
        env["FINANCE_GPU_COORDINATOR_CONFIG"] = str(
            (base / "gpu_coordinator_config.json").resolve()
        )
        env["FINANCE_GPU_COORDINATOR_CONFIG_SHA256"] = restart.digest(
            base / "gpu_coordinator_config.json"
        )
    atomic_json(cell / "command.json", command)
    opts = dict(a.lstrip("+").split("=", 1) for a in command[3:])
    identity = opts["ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_SCREENING_IDENTITY_SHA256"]
    evsteps = [0] if spec.zero else list(profile.evaluation_steps)
    summary = dict(
        status="RUNNING",
        cell=asdict(spec),
        started_at_SGT=restart.now(),
        requested_rounds=spec.rounds,
        seed=profile.seed,
        model_revision=recorded["model_revision"],
        artifact_profile=cfg["artifact_profile"],
        optimizer_included=False,
        resume_supported=False,
        evaluation_steps=evsteps,
        execution_identity_sha256=identity,
        common_validity_penalty=copy.deepcopy(cfg["common_validity_penalty"]),
    )
    atomic_json(cell / "summary.json", summary)
    try:
        # Finish hashes, runtime/compiler checks and command construction while
        # the separate holder still owns its allocations. The worker performs
        # the actual release and a second admission immediately before loading.
        wait_gpu(base, cell_id)
        code = capture_cell_process(command, environment=env, cell_dir=cell, profile=profile)
        summary["returncode"] = code
        if code:
            raise RuntimeError(f"{cell_id} trainer exited {code}; evidence retained")
        summary["evidence"] = validate_cell_evidence(
            cell,
            stage=config["screening"]["stage"],
            expected_rounds=max(1, spec.rounds),
            arm=spec.arm,
            seed=profile.seed,
            train_episode_ids=profile.train_ids,
            eval_episode_ids=profile.eval_ids,
            execution_identity_sha256=identity,
            calibration_payload_sha256=manifest["calibration"]["payload_sha256"],
            expected_eval_steps=evsteps,
            expected_eta=spec.eta,
            artifact_profile=cfg["artifact_profile"],
            eval_rollout_n=5,
            **({"allow_incomplete_evaluation": True}
               if profile.evaluation_incomplete_mode else {}),
        )
        summary["critic_fit"] = tuning.validate_fit_evidence(
            cell, spec=spec, execution_identity=identity
        )
        summary.update(validate_extra_evidence(cell, spec, identity, profile=profile))
        summary["placement"] = validate_placement(cell, spec, identity)
        summary["gpu_handoff"] = validate_handoff(cell, identity)
        summary["evaluation_by_step"] = {
            str(step): evaluation_readout(cell, policy_step=step, profile=profile)
            for step in evsteps
        }
        summary["evaluation"] = summary["evaluation_by_step"][str(evsteps[-1])]
        if profile.evaluation_incomplete_mode:
            failed_evaluations = sum(
                lane["incomplete_trajectories"]
                for step in summary["evaluation_by_step"].values() for lane in step.values()
            )
            summary.update(
                evaluation_incomplete_mode=profile.evaluation_incomplete_mode,
                incomplete_evaluations=failed_evaluations,
                financial_status="INCOMPLETE" if failed_evaluations else "COMPLETE",
            )
        head_path = cell / "artifacts/critic_initialization/head-rank0.json"
        # Zero-shot proves the shared actor's behavior, not a critic update. The
        # current upstream worker initializes its critic even for val_only, but
        # that implementation detail is not a required SFT baseline artifact.
        if not spec.zero or head_path.exists():
            head = restart.read(head_path)
            if (
                head.get("seed") != profile.seed
                or head.get("mode") != "cpu_seed_v1"
                or head.get("arm") != spec.arm
                or head.get("execution_identity_sha256") != identity
                or head.get("model_path") != str(Path(runtime["model_snapshot_path"]).resolve())
                or head.get("phase") != "critic_head_after_hf_load_before_fsdp"
            ):
                raise RuntimeError("actual initial critic head identity differs")
            summary["critic_initial_head_sha256"] = head["head_sha256"]
        if profile.include_sft and not spec.zero:
            baseline = restart.read(base / "sft0/summary.json")
            if (baseline.get("status") != "COMPLETE" or (
                baseline.get("critic_initial_head_sha256") is not None
                and baseline["critic_initial_head_sha256"]
                != summary["critic_initial_head_sha256"]
            )):
                raise RuntimeError("initial critic head differs from the shared SFT baseline")
        if spec.arm == "M2" and not profile.cell_specs:
            baseline = restart.read(base / "m1/summary.json")
            if (
                baseline["status"] != "COMPLETE"
                or summary["critic_initial_head_sha256"]
                != baseline["critic_initial_head_sha256"]
            ):
                raise RuntimeError("initial critic head differs from the common M1 seed")
        check_contract(base, profile=profile)
        summary.update(actual_rounds=spec.rounds, status="COMPLETE")
        return summary
    except BaseException as error:
        summary.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["finished_at_SGT"] = restart.now()
        atomic_json(cell / "summary.json", summary)


def gpu_free_mib():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True, timeout=15,
    ).strip().splitlines()
    if len(output) != 1 or not output[0].strip().isdigit():
        raise RuntimeError("expected one GPU with an integer memory.free reading")
    return int(output[0])


def wait_gpu(base, cell_id, timeout=WAIT_SECONDS):
    handoff = handoff_config(base)
    coordinated = memory_coordinator_config(base)
    claimed = False
    started = time.monotonic()
    record = dict(status="WAITING_FOR_GPU", cell=cell_id, started_at_SGT=restart.now(),
                  minimum_free_MiB=(max(coordinated["phase_peak_bytes"].values()) // 1024**2
                                    if coordinated else MIN_FREE_GPU_MIB),
                  maximum_wait_seconds=timeout,
                  stops_other_processes=False, automatic_retry=False)
    while True:
        record.update(at_SGT=restart.now(), waited_seconds=time.monotonic() - started)
        try:
            if coordinated is not None:
                from src.agents.gpu_memory_coordinator import admission_snapshot

                resources = admission_snapshot(coordinated)
                record.update(resources)
                free = resources["driver_free_bytes"] // 1024**2
                available_mib = resources["available_bytes"] // 1024**2
            elif handoff is None:
                free = gpu_free_mib()
                available_mib = free
            else:
                from src.agents.gpu_handoff import available_bytes

                resources = available_bytes(handoff)
                record.update(resources)
                free = resources["driver_free_bytes"] // 1024**2
                available_mib = resources["available_bytes"] // 1024**2
            record.update(free_MiB=free, last_telemetry_error=None)
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            free = -1
            available_mib = -1
            record.update(free_MiB=None, last_telemetry_error=str(error))
            if handoff is not None or coordinated is not None:
                # A dead/changed holder cannot become valid through polling.
                record["status"] = "HANDOFF_UNAVAILABLE"
                atomic_json(base / "gpu_queue_state.json", record)
                raise
        if time.monotonic() - started >= timeout:
            record["status"] = "WAIT_TIMEOUT"
            atomic_json(base / "gpu_queue_state.json", record)
            raise RuntimeError(
                f"{record['minimum_free_MiB'] / 1024:g} GiB GPU capacity "
                "did not arrive within queue limit"
            )
        if available_mib >= record["minimum_free_MiB"]:
            if coordinated is not None and not claimed:
                from src.agents.gpu_memory_coordinator import (
                    admission_snapshot,
                    controller_claim,
                )

                # Idle growth continues until enough retained + free capacity
                # exists. Claim only then, and confirm again after pausing.
                controller_claim(coordinated)
                claimed = True
                resources = admission_snapshot(coordinated)
                record.update(resources)
                available_mib = resources["available_bytes"] // 1024**2
                record["free_MiB"] = resources["driver_free_bytes"] // 1024**2
                if available_mib < record["minimum_free_MiB"]:
                    atomic_json(base / "gpu_queue_state.json", record)
                    time.sleep(min(45, max(0, timeout - (time.monotonic() - started))))
                    continue
            record["status"] = "ADMITTED"
            atomic_json(base / "gpu_queue_state.json", record)
            return
        atomic_json(base / "gpu_queue_state.json", record)
        time.sleep(min(45, max(0, timeout - (time.monotonic() - started))))


def stop_reservation(base):
    """Release this run's holder on exit; never manipulate training/foreign PIDs."""
    from src.agents.gpu_handoff import reservation_status
    from src.agents.gpu_reservation_service import request

    receipt = dict(at_SGT=restart.now())
    try:
        config = handoff_config(base)
        if config is None:
            return
        receipt["before"] = reservation_status(config, minimum_lease_seconds=0)
        receipt["after"] = request(config["directory"], {"op": "stop"})
        receipt["status"] = "STOP_REQUESTED"
    except Exception as error:
        receipt.update(status="STOP_UNCONFIRMED", error=f"{type(error).__name__}: {error}")
    atomic_json(base / "reservation_cleanup.json", receipt)



def wait_moment_worker_exit(pid, *, timeout=30):
    """Wait only for this owned worker's async teardown; never kill or rerun."""
    deadline = time.monotonic() + timeout
    process = Path("/proc") / str(pid)
    while process.exists():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("moment worker has not exited before scratch retirement")
        time.sleep(min(1, remaining))


def retire_completed_moments(base, spec, summary, *, profile=PILOT):
    """Retire only verified raw payloads produced by this successful new cell.

    This explicit no-resume profile retains ownership manifests and a complete
    before-deletion receipt. Failed cells and the original pilot retain scratch.
    """
    if not profile.retire_completed_moments or spec.zero:
        return {"status": "RETAINED"}
    base = base.resolve()
    cell = base / spec.cell_id
    if (
        summary.get("status") != "COMPLETE"
        or summary.get("returncode") != 0
        or summary.get("actual_rounds") != spec.rounds
        or summary.get("cell") != asdict(spec)
        or restart.read(cell / "summary.json") != summary
    ):
        raise RuntimeError("only the current fully completed cell may retire moments")
    check_contract(base, profile=profile)
    from src.agents.actor_export import validate_export_artifact

    identity = summary["execution_identity_sha256"]
    validate_export_artifact(
        cell / "checkpoints" / f"global_step_{spec.rounds}", identity_sha256=identity,
        verify_hashes=True,
    )
    root = cell / "artifacts/placement/moment_scratch"
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("current-cell moment scratch is missing or linked")
    ancestor = base
    for component in root.relative_to(base).parts:
        ancestor = ancestor / component
        if ancestor.is_symlink():
            raise RuntimeError("moment scratch ancestors must not be symbolic links")
    stores = list(root.glob("*/manifest.json"))  # Layout is PID/role/manifest.json.
    if stores:
        raise RuntimeError("unexpected moment-store layout")
    stores = sorted(root.glob("*/*/manifest.json"))
    if len(stores) != 2 or {p.parent.name for p in stores} != {"actor", "critic"}:
        raise RuntimeError("exactly two owned moment manifests are required")
    payloads, manifests = [], []
    expected_raw = set()
    for path in stores:
        role_dir = path.parent
        pid_dir = role_dir.parent
        if (not pid_dir.name.isdecimal() or path.is_symlink()
                or role_dir.is_symlink() or pid_dir.is_symlink()):
            raise RuntimeError("moment worker directory identity differs")
        manifest = restart.read(path)
        owner_path = role_dir / "owner.json"
        if owner_path.is_symlink():
            raise RuntimeError("moment owner must not be a symbolic link")
        owner = restart.read(owner_path)
        role = role_dir.name
        generation = manifest.get("generation", "")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("identity") != identity
            or owner.get("identity") != identity
            or manifest.get("owner") != owner.get("owner")
            or manifest.get("role") != role or owner.get("role") != role
            or not generation.startswith("generation-")
            or not generation.removeprefix("generation-").isdecimal()
        ):
            raise RuntimeError("moment retirement ownership or generation differs")
        wait_moment_worker_exit(int(pid_dir.name))
        generation_dir = role_dir / generation
        if not generation_dir.is_dir() or generation_dir.is_symlink():
            raise RuntimeError("moment generation is missing or linked")
        role_payload_bytes = 0
        for parameter in manifest["parameters"]:
            for chunks in parameter["moments"].values():
                for chunk in chunks:
                    filename = chunk["file"]
                    if Path(filename).name != filename or not filename.endswith(".raw"):
                        raise RuntimeError("moment chunk name leaves its generation")
                    payload = generation_dir / filename
                    if (payload in expected_raw or payload.is_symlink()
                            or not payload.is_file()
                            or payload.stat().st_nlink != 1
                            or payload.stat().st_size != chunk["bytes"]
                            or restart.digest(payload) != chunk["sha256"]):
                        raise RuntimeError("moment retirement chunk inventory or hash differs")
                    expected_raw.add(payload)
                    role_payload_bytes += chunk["bytes"]
                    stat = payload.stat()
                    payloads.append(dict(
                        path=str(payload.relative_to(cell)), bytes=chunk["bytes"],
                        sha256=chunk["sha256"], device=stat.st_dev, inode=stat.st_ino,
                    ))
        if role_payload_bytes != manifest["payload_bytes"]:
            raise RuntimeError("moment manifest payload total differs")
        expected_role_raw = {p for p in expected_raw if p.parent == generation_dir}
        if set(generation_dir.iterdir()) != expected_role_raw:
            raise RuntimeError("foreign files are present in a moment generation")
        manifests.append(dict(path=str(path.relative_to(cell)), sha256=restart.digest(path),
                              owner=owner["owner"], role=role, generation=generation))
    actual_raw = set(root.rglob("*.raw"))
    if actual_raw != expected_raw:
        raise RuntimeError("unmanifested moment payloads must be preserved")
    receipt_path = cell / "moment_pruning_receipt.json"
    if receipt_path.exists():
        raise RuntimeError("moment retirement has already been attempted; no retry")
    receipt = dict(
        status="PREPARED", at_SGT=restart.now(), execution_identity_sha256=identity,
        cell=spec.cell_id, purpose="temporary moments; no resume supported",
        kept_manifests=manifests, files=payloads,
        planned_bytes=sum(row["bytes"] for row in payloads),
        deleted_files=0, deleted_bytes=0,
    )
    atomic_json(receipt_path, receipt)
    try:
        for row in payloads:
            payload = cell / row["path"]
            stat = payload.stat()
            if (payload.is_symlink() or (stat.st_dev, stat.st_ino, stat.st_size)
                    != (row["device"], row["inode"], row["bytes"])):
                raise RuntimeError("moment chunk changed after the retirement receipt")
            payload.unlink()
            receipt["deleted_files"] += 1
            receipt["deleted_bytes"] += row["bytes"]
        receipt.update(status="COMPLETE", finished_at_SGT=restart.now())
    except BaseException as error:
        receipt.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(receipt_path, receipt)
    return dict(status="COMPLETE", receipt=str(receipt_path),
                deleted_files=receipt["deleted_files"], deleted_bytes=receipt["deleted_bytes"])


def run(base, *, profile=PILOT, post_cell=None):
    if (base / "state.json").exists():
        raise RuntimeError("new state required; no implicit retry")
    check_contract(base, profile=profile)
    state = dict(
        status="RUNNING",
        started_at_SGT=restart.now(),
        cells=[],
        current_cell=None,
        profile=profile.name,
        seed=profile.seed,
        planned_trajectories=profile.planned_trajectories,
        automatic_retry=False,
        resume_supported=False,
        train_episode_ids=profile.train_ids,
        evaluation_episode_ids=profile.eval_ids,
        evaluation_steps=list(profile.evaluation_steps),
    )
    atomic_json(base / "state.json", state)
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_status, args=(base, stop, profile), daemon=True)
    monitor.start()
    try:
        for index, spec in enumerate(cells(profile)):
            exports = sum(not s.zero for s in cells(profile)[index:])
            # Two retained role snapshots plus one replacement generation (~90 GiB),
            # remaining actor exports and reserve. The pilot retains scratch;
            # the comparison retires verified successful-cell raw payloads.
            if profile.archive_completed_actors:
                if post_cell is None:
                    raise RuntimeError("lossless actor archival callback is required")
                exports = int(exports > 0)
            required = disk_requirement_bytes(profile, exports)
            free = shutil.disk_usage(base).free
            if free < required:
                raise RuntimeError(
                    "insufficient space for disk moment generations, exports and reserve"
                )
            state.update(current_cell=spec.cell_id, free_disk_bytes=free)
            atomic_json(base / "state.json", state)
            completed = run_cell(base, spec.cell_id, profile=profile)
            state["cells"].append(completed)
            atomic_json(base / "state.json", state)
            coordinated = memory_coordinator_config(base)
            if coordinated is not None:
                from src.agents.gpu_memory_coordinator import controller_idle

                try:
                    idle = controller_idle(coordinated)
                    idle_record = dict(status="REFILLING", at_SGT=restart.now(),
                                       payload_bytes=idle["payload_bytes"])
                except (OSError, RuntimeError) as error:
                    # A surviving Ray child or uncertain residency must keep
                    # the holder paused; it never invalidates completed RL.
                    idle_record = dict(status="DEFERRED", at_SGT=restart.now(),
                                       reason=f"{type(error).__name__}: {error}")
                atomic_json(base / "gpu_coordinator_idle.json", idle_record)
            if profile.retire_completed_moments and not spec.zero:
                completed["moment_pruning"] = retire_completed_moments(
                    base, spec, completed, profile=profile,
                )
                atomic_json(base / spec.cell_id / "summary.json", completed)
                atomic_json(base / "state.json", state)
            if post_cell is not None:
                post_cell(base, spec, completed)
                atomic_json(base / spec.cell_id / "summary.json", completed)
                atomic_json(base / "state.json", state)
        state["status"] = "COMPLETE"
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["finished_at_SGT"] = restart.now()
        atomic_json(base / "state.json", state)
        stop.set()
        monitor.join(timeout=20)
        stop_reservation(base)
        refresh_status(base, profile=profile)


def current_reservation_directory():
    """Select the bound project holder for a fresh CLI prepare, never an old run."""
    pointer = REPO_ROOT / "runs/GPU_RESERVATION_CURRENT.json"
    if not pointer.exists():
        return None
    from src.agents.gpu_memory_coordinator import PROFILE as COORDINATOR_PROFILE
    from src.agents.gpu_reservation_service import process_identity

    value = restart.read(pointer)
    directory = Path(value["directory"])
    if (value.get("profile") != COORDINATOR_PROFILE or not directory.is_absolute()
            or directory.resolve() != directory):
        raise RuntimeError("current persistent GPU holder pointer is invalid")
    owner_path = directory / "owner.json"
    owner = restart.read(owner_path)
    actual = process_identity(owner["pid"])
    if (restart.digest(owner_path) != value.get("owner_sha256")
            or owner.get("persistent") is not True
            or owner["pid"] != value.get("holder_pid")
            or actual["start_time_ticks"] != value.get("holder_start_time_ticks")):
        raise RuntimeError("current persistent GPU holder identity changed")
    return directory


def main(*, profile=PILOT, default_directory=DEFAULT):
    description = (__doc__ if not profile.include_sft else
                   "Fixed SFT0 + four-round full-parameter 4B M1/M2 comparison; "
                   "policy 0/2/4 evaluation, disk-backed chunked Adam, no retries or resumes.")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("command", choices=("prepare", "preflight", "seal", "check", "run"))
    parser.add_argument("--directory", type=Path, default=default_directory)
    parser.add_argument("--reservation-directory", type=Path)
    parser.add_argument("--coordinated-reservation-directory", type=Path)
    parser.add_argument(
        "--model", type=Path, default=REPO_ROOT / "runs/m2-coldstart-v3-20260914/merged_model"
    )
    parser.add_argument(
        "--merge-receipt-sha256",
        default="cf3b8e5a4e2e45708943f4c71c822d41d60cfae13170e760cb5979e1df2356db",
    )
    args = parser.parse_args()
    base = args.directory.resolve()
    if args.command == "prepare":
        coordinated = args.coordinated_reservation_directory
        if coordinated is None and args.reservation_directory is None:
            coordinated = current_reservation_directory()
        prepare(base, model_path=args.model, merge_receipt_sha256=args.merge_receipt_sha256,
                reservation_directory=args.reservation_directory,
                coordinated_reservation_directory=coordinated,
                profile=profile)
    elif args.command == "preflight":
        asyncio.run(input_preflight(base, profile=profile))
    elif args.command == "seal":
        seal(base, profile=profile)
    elif args.command == "check":
        check_contract(base, profile=profile)
    else:
        run(base, profile=profile)
    print(json.dumps(dict(command=args.command, status="PASS", at_SGT=restart.now())))


if __name__ == "__main__":
    main()

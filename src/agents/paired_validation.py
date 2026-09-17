"""Opt-in greedy plus sampled validation lanes on the existing Ray rollout worker.

One native trainer validation repeats each window five times. Lane zero is
greedy; lanes one through four sample the unchanged current profile. Training
dispatch is untouched. The original seed wrapper and actual policy-step identity
remain in place; no actor, critic, trainer loss, or checkpoint logic is replaced.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from functools import wraps
from hashlib import sha256
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PROFILE = "greedy1_sample4_v1"
PROFILE_ENV = "FINANCE_PAIRED_VALIDATION_PROFILE"
DIRECTORY_ENV = "FINANCE_PAIRED_VALIDATION_EVIDENCE_DIR"
TARGET_MODULE = "agent_r1.agent_flow.agent_flow"
_MARKER = "_finance_paired_validation_directory"


def _get(config: Any, key: str, default: Any = None) -> Any:
    return config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)


def lane_settings(rollout_n: int) -> dict[str, Any]:
    if type(rollout_n) is not int or not 0 <= rollout_n < 5:
        raise ValueError("paired validation requires exactly lanes 0..4")
    return {
        "mode": "greedy" if rollout_n == 0 else "sampled",
        "sampled_sibling": None if rollout_n == 0 else rollout_n - 1,
        "sampling_params": {
            "temperature": 0.0 if rollout_n == 0 else 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
    }


def _validate_worker_config(worker: Any) -> None:
    rollout = worker.config.actor_rollout_ref.rollout
    validation = rollout.val_kwargs
    expected = {"n": 5, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "do_sample": True}
    if any(_get(validation, key) != value for key, value in expected.items()):
        raise RuntimeError("worker validation configuration differs from five-lane contract")
    if _get(_get(rollout, "agent"), "num_workers") != 1:
        raise RuntimeError("five-lane identity requires the fixed single agent-flow worker")
    if _get(rollout, "n") != 4:
        raise RuntimeError("paired validation must preserve four training siblings")


def _write(directory: Path, record: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(
        directory / f"paired-validation-{os.getpid()}.jsonl",
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        if os.write(descriptor, payload) != len(payload):
            raise RuntimeError("paired validation evidence write was incomplete")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def patch_worker_class(worker_class: type[Any], directory: Path) -> bool:
    from src.agents.rollout_seed import normalize_rollout_identity

    directory = Path(directory).expanduser()
    if not directory.is_absolute():
        raise ValueError("paired validation evidence directory must be absolute")
    directory = directory.resolve()
    original = worker_class._run_agent_flow
    installed = getattr(original, _MARKER, None)
    if installed is not None:
        if installed != str(directory):
            raise RuntimeError("paired validation cannot change its evidence directory")
        return False
    if not getattr(original, "_finance_rollout_seed", False):
        raise RuntimeError("paired validation must wrap the existing deterministic seed dispatch")

    @wraps(original)
    async def dispatch(
        instance: Any,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        _ray_trace_ctx: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if trajectory.get("validate") is False:
            # Do not copy, normalize, change configuration, or emit extra training evidence.
            return await original(
                instance,
                sampling_params,
                trajectory,
                agent_name=agent_name,
                trace=trace,
                _ray_trace_ctx=_ray_trace_ctx,
                **kwargs,
            )
        identity = normalize_rollout_identity(trajectory)
        if not identity.validate or agent_name != "trading":
            raise RuntimeError("paired validation is limited to actual trading validation rows")
        _validate_worker_config(instance)
        lane = lane_settings(identity.rollout_n)
        if sampling_params.get("temperature") != 1.0 or sampling_params.get("top_p") != 1.0:
            raise RuntimeError(
                "upstream validation sampling defaults differ from the fixed profile"
            )
        forwarded = {**sampling_params, **lane["sampling_params"]}
        execution = os.environ.get("FINANCE_SCREENING_IDENTITY_SHA256", "")
        arm = os.environ.get("FINANCE_SCREENING_ARM", "")
        if not re.fullmatch(r"[0-9a-f]{64}", execution) or arm not in {"M1", "M2"}:
            raise RuntimeError("paired validation requires the actual screening execution identity")
        extra = kwargs.get("extra_info", {})
        if isinstance(extra, str):
            extra = json.loads(extra)
        episode = kwargs.get("episode_id") or extra.get("episode_id")
        if not isinstance(episode, str) or not episode:
            raise RuntimeError("paired validation requires its original episode identity")
        record = {
            "schema_version": 1,
            "profile": PROFILE,
            "status": "RUNNING",
            "started_at_SGT": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
            "pid": os.getpid(),
            "arm": arm,
            "execution_identity_sha256": execution,
            "identity": identity.as_dict(),
            "policy_step": identity.step,
            "episode_id": episode,
            "group_uid": kwargs.get("uid"),
            "validation_rollout_n": 5,
            "mode": lane["mode"],
            "sampled_sibling": lane["sampled_sibling"],
            "upstream_sampling_params": dict(sampling_params),
            "forwarded_sampling_params": dict(forwarded),
            "scope": "agent-flow dispatch before the existing seed and response-cap wrappers",
            "seed_identity_preserved": True,
            "training_dispatch_changed": False,
        }
        try:
            result = await original(
                instance,
                forwarded,
                trajectory,
                agent_name=agent_name,
                trace=trace,
                _ray_trace_ctx=_ray_trace_ctx,
                **kwargs,
            )
            record["status"] = "COMPLETE"
            return result
        except BaseException as error:
            record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["finished_at_SGT"] = datetime.now(ZoneInfo("Asia/Singapore")).isoformat()
            _write(directory, record)

    setattr(dispatch, _MARKER, str(directory))
    worker_class._run_agent_flow = dispatch
    return True


def validate_run_evidence(
    directory: str | Path,
    *,
    expected_steps: list[int],
    expected_episode_ids: list[str],
    arm: str,
    execution_identity: str,
) -> dict[str, Any]:
    """Require every registered lane with its actual unchanged seed identity and parameters."""
    from src.agents.rollout_seed import normalize_rollout_identity

    directory = Path(directory)
    if (
        not expected_steps
        or any(type(step) is not int or step < 0 for step in expected_steps)
        or len(set(expected_steps)) != len(expected_steps)
        or len(expected_episode_ids) != 4
        or len(set(expected_episode_ids)) != 4
        or arm not in {"M1", "M2"}
        or not re.fullmatch(r"[0-9a-f]{64}", execution_identity)
    ):
        raise ValueError("invalid expected paired validation grid")
    expected = {
        (step, index, lane): episode
        for step in expected_steps
        for index, episode in enumerate(expected_episode_ids)
        for lane in range(5)
    }
    observed, sources = set(), {}
    for path in sorted(directory.glob("*.jsonl")):
        payload = path.read_bytes()
        sources[path.name] = sha256(payload).hexdigest()
        if not payload.endswith(b"\n"):
            raise RuntimeError("paired validation evidence has an uncommitted final record")
        for line in payload.splitlines():
            row = json.loads(line)
            identity = normalize_rollout_identity(row.get("identity")).as_dict()
            key = (identity["step"], identity["sample_index"], identity["rollout_n"])
            if key not in expected or key in observed or identity["validate"] is not True:
                raise RuntimeError(
                    "paired validation has missing, repeated or unexpected identities"
                )
            lane = lane_settings(identity["rollout_n"])
            upstream, forwarded = (
                row.get("upstream_sampling_params", {}),
                row.get("forwarded_sampling_params", {}),
            )
            if (
                row.get("schema_version") != 1
                or row.get("profile") != PROFILE
                or row.get("status") != "COMPLETE"
                or row.get("arm") != arm
                or row.get("execution_identity_sha256") != execution_identity
                or row.get("policy_step") != identity["step"]
                or row.get("episode_id") != expected[key]
                or row.get("validation_rollout_n") != 5
                or row.get("mode") != lane["mode"]
                or row.get("sampled_sibling") != lane["sampled_sibling"]
                or row.get("seed_identity_preserved") is not True
                or row.get("training_dispatch_changed") is not False
                or upstream.get("temperature") != 1.0
                or upstream.get("top_p") != 1.0
                or forwarded != {**upstream, **lane["sampling_params"]}
                or not isinstance(row.get("group_uid"), str)
                or not row["group_uid"]
                or type(row.get("pid")) is not int
                or row["pid"] <= 0
            ):
                raise RuntimeError("actual paired validation lane, parameters or execution differs")
            observed.add(key)
    if observed != set(expected):
        raise RuntimeError("paired validation is missing one or more fixed validation lanes")
    return {
        "status": "PASS",
        "profile": PROFILE,
        "trajectories": len(observed),
        "greedy_trajectories": len(expected_steps) * 4,
        "sampled_trajectories": len(expected_steps) * 16,
        "policy_steps": list(expected_steps),
        "source_sha256": sources,
        "seed_identity_contains_actual_policy_step": True,
    }


def _patch_module(module: Any, directory: Path) -> None:
    base_class = getattr(module, "AgentFlowWorkerBase", None)
    metadata = getattr(getattr(module, "AgentFlowWorker", None), "__ray_metadata__", None)
    worker_class = getattr(metadata, "modified_class", None)
    if not isinstance(base_class, type) or not isinstance(worker_class, type):
        raise RuntimeError("pinned Agent-R1 Ray dispatch metadata is unavailable")
    if not issubclass(worker_class, base_class):
        raise RuntimeError("Ray agent-flow worker no longer inherits the pinned base class")
    patch_worker_class(worker_class, directory)


def install_from_environment() -> bool:
    """Chain onto the existing seed import callback; no competing module finder."""
    profile = os.environ.get(PROFILE_ENV)
    if not profile:
        return False
    if profile != PROFILE:
        raise ValueError("unknown paired validation profile")
    value = os.environ.get(DIRECTORY_ENV, "")
    directory = Path(value).expanduser()
    if not value or not directory.is_absolute():
        raise ValueError("paired validation requires an absolute evidence directory")
    directory = directory.resolve()
    from src.agents import rollout_seed

    if rollout_seed._configured_evidence_root is None:
        raise RuntimeError("install deterministic rollout seeds before paired validation")
    original = rollout_seed._patch_loaded_agent_flow_module
    installed = getattr(original, _MARKER, None)
    if installed is not None and installed != str(directory):
        raise RuntimeError("paired validation import hook cannot change its evidence directory")
    if installed is None:

        @wraps(original)
        def patch_module(module: Any) -> None:
            original(module)
            _patch_module(module, directory)

        setattr(patch_module, _MARKER, str(directory))
        rollout_seed._patch_loaded_agent_flow_module = patch_module
    loaded = sys.modules.get(TARGET_MODULE)
    if loaded is not None:
        _patch_module(loaded, directory)
    return True

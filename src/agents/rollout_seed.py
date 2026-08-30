"""Deterministic per-request seeds for concurrent Agent-R1 rollouts."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import inspect
import json
import operator
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from functools import wraps
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Any

ROLLOUT_SEED_DIR_ENV = "FINANCE_ROLLOUT_SEED_DIR"
ROLLOUT_IDENTITY_KWARG = "_finance_rollout_identity_v1"
ROLLOUT_SEED_SCHEME = "finance-rollout-seed-v1"
ROLLOUT_SEED_EVIDENCE_SCHEMA_VERSION = 1

_IDENTITY_FIELDS = frozenset({"step", "sample_index", "rollout_n", "validate"})
_EVIDENCE_FIELDS = frozenset(
    {
        "base_seed",
        "env_step",
        "pid",
        "prompt_token_count",
        "request_seed",
        "rollout_n",
        "sample_index",
        "schema_version",
        "scheme",
        "step",
        "validate",
    }
)
_AGENT_FLOW_MODULE = "agent_r1.agent_flow.agent_flow"
_configured_evidence_root: Path | None = None


@dataclass(frozen=True)
class RolloutIdentity:
    """Stable upstream identity for one sampled trajectory."""

    step: int
    sample_index: int
    rollout_n: int
    validate: bool

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "step": self.step,
            "sample_index": self.sample_index,
            "rollout_n": self.rollout_n,
            "validate": self.validate,
        }


def _integer(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    normalized = int(normalized)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def normalize_rollout_identity(value: Any) -> RolloutIdentity:
    """Validate the exact identity supplied by pinned Agent-R1."""

    if isinstance(value, RolloutIdentity):
        value = value.as_dict()
    if not isinstance(value, Mapping):
        raise TypeError("rollout identity must be a mapping")
    fields = set(value)
    if fields != _IDENTITY_FIELDS:
        missing = sorted(_IDENTITY_FIELDS - fields)
        unexpected = sorted(fields - _IDENTITY_FIELDS)
        raise ValueError(
            f"rollout identity fields changed (missing={missing}, unexpected={unexpected})"
        )
    validate = value["validate"]
    if not isinstance(validate, bool):
        raise TypeError("rollout identity validate must be bool")
    return RolloutIdentity(
        step=_integer(value["step"], name="rollout identity step", minimum=0),
        sample_index=_integer(
            value["sample_index"], name="rollout identity sample_index", minimum=0
        ),
        rollout_n=_integer(value["rollout_n"], name="rollout identity rollout_n", minimum=0),
        validate=validate,
    )


def derive_request_seed(
    base_seed: Any,
    trajectory: RolloutIdentity | Mapping[str, Any],
    env_step: Any,
) -> int:
    """Derive a schedule-independent vLLM request seed from frozen identity."""

    normalized_base_seed = _integer(base_seed, name="base seed", minimum=0)
    identity = normalize_rollout_identity(trajectory)
    normalized_env_step = _integer(env_step, name="environment step", minimum=0)
    payload = {
        "base_seed": normalized_base_seed,
        "env_step": normalized_env_step,
        "rollout_n": identity.rollout_n,
        "sample_index": identity.sample_index,
        "scheme": ROLLOUT_SEED_SCHEME,
        "step": identity.step,
        "validate": identity.validate,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return int.from_bytes(sha256(encoded).digest()[:8], byteorder="big") & 0x7FFFFFFFFFFFFFFF


class DeterministicSeedServerManager:
    """Per-flow proxy that copies sampling parameters and adds request seeds."""

    def __init__(
        self,
        server_manager: Any,
        *,
        base_seed: Any,
        trajectory: RolloutIdentity | Mapping[str, Any],
        evidence_root: str | Path,
    ) -> None:
        self._server_manager = server_manager
        self._base_seed = _integer(base_seed, name="base seed", minimum=0)
        self._identity = normalize_rollout_identity(trajectory)
        root = Path(evidence_root).expanduser()
        if not root.is_absolute():
            raise ValueError("rollout seed evidence directory must be absolute")
        self._evidence_root = root.resolve()
        self._evidence_root.mkdir(parents=True, exist_ok=True)
        self._evidence_path = self._evidence_root / f"request-seeds-{os.getpid()}.jsonl"
        self._request_seeds: list[int] = []
        self._in_flight = False

    @property
    def request_seeds(self) -> tuple[int, ...]:
        return tuple(self._request_seeds)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._server_manager, name)

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
    ) -> Any:
        if not isinstance(sampling_params, Mapping):
            raise TypeError("sampling_params must be a mapping")
        if "seed" in sampling_params:
            raise RuntimeError(
                "upstream sampling_params already contains seed; refusing ambiguous seed ownership"
            )
        if self._in_flight:
            raise RuntimeError("concurrent generate calls on one seeded flow are not deterministic")
        self._in_flight = True
        try:
            env_step = len(self._request_seeds)
            request_seed = derive_request_seed(self._base_seed, self._identity, env_step)
            seeded_params = dict(sampling_params)
            seeded_params["seed"] = request_seed
            output = await self._server_manager.generate(
                request_id,
                prompt_ids=prompt_ids,
                sampling_params=seeded_params,
                image_data=image_data,
                video_data=video_data,
            )
            record = {
                "base_seed": self._base_seed,
                "env_step": env_step,
                "pid": os.getpid(),
                "prompt_token_count": len(prompt_ids),
                "request_seed": request_seed,
                "rollout_n": self._identity.rollout_n,
                "sample_index": self._identity.sample_index,
                "schema_version": ROLLOUT_SEED_EVIDENCE_SCHEMA_VERSION,
                "scheme": ROLLOUT_SEED_SCHEME,
                "step": self._identity.step,
                "validate": self._identity.validate,
            }
            with self._evidence_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            self._request_seeds.append(request_seed)
            return output
        finally:
            self._in_flight = False


def build_seeded_server_manager(
    server_manager: Any,
    *,
    base_seed: Any,
    trajectory: RolloutIdentity | Mapping[str, Any],
) -> DeterministicSeedServerManager:
    """Build a per-flow proxy only after the launcher-controlled hook is active."""

    if _configured_evidence_root is None:
        raise RuntimeError("deterministic rollout seed hook is not installed")
    return DeterministicSeedServerManager(
        server_manager,
        base_seed=base_seed,
        trajectory=trajectory,
        evidence_root=_configured_evidence_root,
    )


def patch_agent_flow_worker_class(agent_flow_worker_class: type[Any]) -> bool:
    """Pass upstream trajectory identity into the configured finance flow."""

    original = agent_flow_worker_class._run_agent_flow
    if getattr(original, "_finance_rollout_seed", False):
        return False
    parameters = inspect.signature(original, follow_wrapped=False).parameters
    expected_without_ray_trace = (
        ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("sampling_params", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("trajectory", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("agent_name", inspect.Parameter.KEYWORD_ONLY),
        ("trace", inspect.Parameter.KEYWORD_ONLY),
        ("kwargs", inspect.Parameter.VAR_KEYWORD),
    )
    expected_with_ray_trace = expected_without_ray_trace[:-1] + (
        ("_ray_trace_ctx", inspect.Parameter.KEYWORD_ONLY),
        expected_without_ray_trace[-1],
    )
    actual = tuple((name, parameter.kind) for name, parameter in parameters.items())
    if (
        actual not in {expected_without_ray_trace, expected_with_ray_trace}
        or parameters["trace"].default is not True
    ):
        raise RuntimeError(
            f"Agent-R1 _run_agent_flow signature changed: {inspect.signature(original)}"
        )
    ray_tracing_wrapper = bool(getattr(original, "__ray_tracing_wrapped__", False))
    if ray_tracing_wrapper and actual != expected_with_ray_trace:
        raise RuntimeError("Ray tracing wrapper is missing its required _ray_trace_ctx parameter")

    @wraps(original)
    async def seeded_run(
        instance: Any,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        _ray_trace_ctx: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if agent_name != "trading":
            raise RuntimeError(
                f"deterministic rollout seeding requires agent_name='trading', got {agent_name!r}"
            )
        if ROLLOUT_IDENTITY_KWARG in kwargs:
            raise RuntimeError(f"reserved rollout identity key collision: {ROLLOUT_IDENTITY_KWARG}")
        identity = normalize_rollout_identity(trajectory)
        if "index" not in kwargs:
            raise RuntimeError("rollout dataset index is required for deterministic seeding")
        dataset_index = _integer(
            kwargs["index"], name="rollout dataset index", minimum=0
        )
        if dataset_index != identity.sample_index:
            raise RuntimeError(
                f"rollout dataset index {dataset_index} does not match trajectory "
                f"sample_index {identity.sample_index}"
            )
        forwarded = dict(kwargs)
        forwarded[ROLLOUT_IDENTITY_KWARG] = identity.as_dict()
        if ray_tracing_wrapper:
            return await original(
                instance,
                sampling_params,
                trajectory,
                agent_name=agent_name,
                trace=trace,
                _ray_trace_ctx=_ray_trace_ctx,
                **forwarded,
            )
        if _ray_trace_ctx is not None:
            raise RuntimeError("unexpected Ray trace context on an unwrapped agent flow method")
        return await original(
            instance,
            sampling_params,
            trajectory,
            agent_name=agent_name,
            trace=trace,
            **forwarded,
        )

    seeded_run._finance_rollout_seed = True  # type: ignore[attr-defined]
    agent_flow_worker_class._run_agent_flow = seeded_run
    return True


def _patch_loaded_agent_flow_module(module: ModuleType) -> None:
    base_class = getattr(module, "AgentFlowWorkerBase", None)
    remote_actor = getattr(module, "AgentFlowWorker", None)
    metadata = getattr(remote_actor, "__ray_metadata__", None)
    modified_class = getattr(metadata, "modified_class", None)
    if not isinstance(base_class, type) or not isinstance(modified_class, type):
        raise RuntimeError("pinned Agent-R1 Ray worker classes are unavailable")
    if not issubclass(modified_class, base_class):
        raise RuntimeError("Agent-R1 Ray modified worker no longer inherits AgentFlowWorkerBase")

    # Ray copies and trace-wraps inherited methods when @ray.remote executes.
    # Patch that concrete dispatch class as well as the base class used directly.
    patch_agent_flow_worker_class(modified_class)
    patch_agent_flow_worker_class(base_class)
    for cls in (modified_class, base_class):
        worker_class: Any = cls
        if not getattr(worker_class._run_agent_flow, "_finance_rollout_seed", False):
            raise RuntimeError("deterministic rollout seed patch did not attach to worker class")


class _AgentFlowPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader, finder: _AgentFlowPatchFinder) -> None:
        self._wrapped = wrapped
        self._finder = finder

    def create_module(self, spec: Any) -> ModuleType | None:
        create_module = getattr(self._wrapped, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module: ModuleType) -> None:
        exec_module = getattr(self._wrapped, "exec_module", None)
        if exec_module is None:
            raise ImportError(f"loader for {_AGENT_FLOW_MODULE} cannot execute modules")
        exec_module(module)
        _patch_loaded_agent_flow_module(module)
        if self._finder in sys.meta_path:
            sys.meta_path.remove(self._finder)


class _AgentFlowPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> Any:
        if fullname != _AGENT_FLOW_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None or not isinstance(spec.loader, importlib.abc.Loader):
            raise ImportError(f"cannot install deterministic seed hook for {fullname}")
        spec.loader = _AgentFlowPatchLoader(spec.loader, self)
        return spec


def install_from_environment() -> None:
    """Install deterministic rollout seeding in launcher-controlled workers."""

    output_value = os.environ.get(ROLLOUT_SEED_DIR_ENV)
    if not output_value:
        return
    output = Path(output_value).expanduser()
    if not output.is_absolute():
        raise ValueError("rollout seed evidence directory must be absolute")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    global _configured_evidence_root
    if _configured_evidence_root is not None and _configured_evidence_root != output:
        raise RuntimeError("rollout seed hook cannot change evidence directory after installation")

    _configured_evidence_root = output
    loaded_module = sys.modules.get(_AGENT_FLOW_MODULE)
    if loaded_module is not None:
        _patch_loaded_agent_flow_module(loaded_module)
        return
    if not any(isinstance(finder, _AgentFlowPatchFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _AgentFlowPatchFinder())


def summarize_rollout_seed_evidence(
    evidence_root: str | Path,
    *,
    expected_base_seed: Any | None = None,
) -> dict[str, object]:
    """Validate and summarize per-request deterministic seed evidence."""

    root = Path(evidence_root)
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.jsonl")) if root.is_dir() else []:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            document = json.loads(line)
            if not isinstance(document, dict):
                raise RuntimeError(f"invalid rollout seed evidence in {path}")
            fields = set(document)
            if fields != _EVIDENCE_FIELDS:
                missing = sorted(_EVIDENCE_FIELDS - fields)
                unexpected = sorted(fields - _EVIDENCE_FIELDS)
                raise RuntimeError(
                    f"rollout seed evidence fields changed in {path} "
                    f"(missing={missing}, unexpected={unexpected})"
                )
            schema_version = _integer(
                document["schema_version"],
                name="rollout evidence schema_version",
                minimum=1,
            )
            if schema_version != ROLLOUT_SEED_EVIDENCE_SCHEMA_VERSION:
                raise RuntimeError(f"unexpected rollout seed evidence schema in {path}")
            identity = normalize_rollout_identity(
                {field: document.get(field) for field in _IDENTITY_FIELDS}
            )
            expected_seed = derive_request_seed(
                document.get("base_seed"), identity, document.get("env_step")
            )
            if document.get("scheme") != ROLLOUT_SEED_SCHEME:
                raise RuntimeError(f"unexpected rollout seed scheme in {path}")
            request_seed = _integer(
                document.get("request_seed"),
                name="rollout evidence request_seed",
                minimum=0,
            )
            if request_seed != expected_seed:
                raise RuntimeError(f"invalid derived rollout seed in {path}")
            _integer(document.get("pid"), name="rollout evidence pid", minimum=1)
            _integer(
                document.get("prompt_token_count"),
                name="rollout evidence prompt_token_count",
                minimum=0,
            )
            records.append(document)

    seed_identities: dict[int, tuple[int, int, int, bool, int]] = {}
    request_identities: set[tuple[int, int, int, bool, int]] = set()
    base_seeds: set[int] = set()
    for record in records:
        request_seed = int(record["request_seed"])
        request_identity = (
            int(record["step"]),
            int(record["sample_index"]),
            int(record["rollout_n"]),
            bool(record["validate"]),
            int(record["env_step"]),
        )
        if request_identity in request_identities:
            raise RuntimeError(f"duplicate rollout seed evidence for {request_identity}")
        request_identities.add(request_identity)
        base_seeds.add(int(record["base_seed"]))
        previous = seed_identities.setdefault(request_seed, request_identity)
        if previous != request_identity:
            raise RuntimeError(
                f"rollout seed collision: seed {request_seed} maps to "
                f"{previous} and {request_identity}"
            )

    if len(base_seeds) > 1:
        raise RuntimeError(f"multiple rollout base seeds were recorded: {sorted(base_seeds)}")
    if expected_base_seed is not None:
        normalized_expected_seed = _integer(
            expected_base_seed, name="expected base seed", minimum=0
        )
        if base_seeds and base_seeds != {normalized_expected_seed}:
            raise RuntimeError(
                f"rollout base seed mismatch: expected {normalized_expected_seed}, "
                f"recorded {sorted(base_seeds)}"
            )

    trajectories = {
        (
            int(record["step"]),
            int(record["sample_index"]),
            int(record["rollout_n"]),
            bool(record["validate"]),
        )
        for record in records
    }
    canonical_records = [
        {key: value for key, value in record.items() if key != "pid"}
        for record in sorted(
            records,
            key=lambda record: (
                int(record["step"]),
                int(record["sample_index"]),
                int(record["rollout_n"]),
                bool(record["validate"]),
                int(record["env_step"]),
            ),
        )
    ]
    canonical_sha256 = sha256(
        json.dumps(canonical_records, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return {
        "schema_version": ROLLOUT_SEED_EVIDENCE_SCHEMA_VERSION,
        "scheme": ROLLOUT_SEED_SCHEME,
        "request_count": len(records),
        "trajectory_count": len(trajectories),
        "training_request_count": sum(record["validate"] is False for record in records),
        "validation_request_count": sum(record["validate"] is True for record in records),
        "unique_seed_count": len(seed_identities),
        "base_seed": next(iter(base_seeds)) if base_seeds else None,
        "canonical_sha256": canonical_sha256,
        "worker_files": len(list(root.glob("*.jsonl"))) if root.is_dir() else 0,
    }

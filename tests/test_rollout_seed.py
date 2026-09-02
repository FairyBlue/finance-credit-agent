from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.agents.rollout_seed import (
    ROLLOUT_IDENTITY_KWARG,
    ROLLOUT_SEED_EVIDENCE_SCHEMA_VERSION,
    ROLLOUT_SEED_SCHEME,
    DeterministicSeedServerManager,
    derive_request_seed,
    normalize_rollout_identity,
    patch_agent_flow_worker_class,
    summarize_rollout_seed_evidence,
)


def _identity(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "step": 1,
        "sample_index": 3,
        "rollout_n": 0,
        "validate": False,
    }
    value.update(overrides)
    return value


def test_request_seed_is_stable_and_every_identity_dimension_matters() -> None:
    baseline = derive_request_seed(42, _identity(), 0)

    variants = {
        derive_request_seed(43, _identity(), 0),
        derive_request_seed(42, _identity(step=2), 0),
        derive_request_seed(42, _identity(sample_index=4), 0),
        derive_request_seed(42, _identity(rollout_n=1), 0),
        derive_request_seed(42, _identity(validate=True), 0),
        derive_request_seed(42, _identity(), 1),
    }

    assert derive_request_seed(42, _identity(), 0) == baseline
    assert baseline == 8143404199048609985
    assert len(variants) == 6
    assert baseline not in variants
    assert 0 <= baseline <= 0x7FFFFFFFFFFFFFFF


def test_rollout_identity_accepts_numpy_integers_without_accepting_loose_types() -> None:
    normalized = normalize_rollout_identity(
        _identity(step=np.int64(1), sample_index=np.int32(3), rollout_n=np.int64(0))
    )

    assert normalized.as_dict() == _identity()
    for invalid in (True, 1.0, "1", None):
        with pytest.raises((TypeError, ValueError)):
            normalize_rollout_identity(_identity(sample_index=invalid))
    with pytest.raises(TypeError, match="validate must be bool"):
        normalize_rollout_identity(_identity(validate=np.bool_(False)))
    with pytest.raises(ValueError, match="fields changed"):
        normalize_rollout_identity({**_identity(), "new_upstream_field": 1})
    with pytest.raises(ValueError, match="must be at least 0"):
        normalize_rollout_identity(_identity(step=-1))


class _ServerManagerFixture:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
    ) -> str:
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": prompt_ids,
                "sampling_params": sampling_params,
                "image_data": image_data,
                "video_data": video_data,
            }
        )
        return "generated"


def test_proxy_copies_params_passes_request_through_and_records_seed(tmp_path: Path) -> None:
    upstream = _ServerManagerFixture()
    proxy = DeterministicSeedServerManager(
        upstream,
        base_seed=42,
        trajectory=_identity(),
        evidence_root=tmp_path,
    )
    params = {"temperature": 0.7, "top_p": 0.9}

    result = asyncio.run(
        proxy.generate(
            "uuid-passthrough",
            prompt_ids=[1, 2, 3],
            sampling_params=params,
            image_data=["image"],
            video_data=["video"],
        )
    )

    assert result == "generated"
    assert params == {"temperature": 0.7, "top_p": 0.9}
    assert upstream.calls == [
        {
            "request_id": "uuid-passthrough",
            "prompt_ids": [1, 2, 3],
            "sampling_params": {
                "temperature": 0.7,
                "top_p": 0.9,
                "seed": derive_request_seed(42, _identity(), 0),
            },
            "image_data": ["image"],
            "video_data": ["video"],
        }
    ]
    assert proxy.request_seeds == (derive_request_seed(42, _identity(), 0),)
    evidence = summarize_rollout_seed_evidence(tmp_path)
    canonical_sha256 = evidence.pop("canonical_sha256")
    trajectory_identity_sha256 = evidence.pop("trajectory_identity_sha256")
    assert len(str(canonical_sha256)) == 64
    assert len(str(trajectory_identity_sha256)) == 64
    assert evidence == {
        "schema_version": ROLLOUT_SEED_EVIDENCE_SCHEMA_VERSION,
        "scheme": ROLLOUT_SEED_SCHEME,
        "request_count": 1,
        "trajectory_count": 1,
        "training_request_count": 1,
        "validation_request_count": 0,
        "training_trajectory_count": 1,
        "validation_trajectory_count": 0,
        "unique_seed_count": 1,
        "base_seed": 42,
        "worker_files": 1,
    }


def test_each_proxy_has_an_independent_environment_step_counter(tmp_path: Path) -> None:
    upstream = _ServerManagerFixture()
    first = DeterministicSeedServerManager(
        upstream, base_seed=42, trajectory=_identity(), evidence_root=tmp_path
    )
    second = DeterministicSeedServerManager(
        upstream, base_seed=42, trajectory=_identity(), evidence_root=tmp_path
    )

    async def run_requests() -> None:
        await first.generate("first-0", prompt_ids=[], sampling_params={})
        await first.generate("first-1", prompt_ids=[], sampling_params={})
        await second.generate("second-0", prompt_ids=[], sampling_params={})

    asyncio.run(run_requests())

    assert first.request_seeds == (
        derive_request_seed(42, _identity(), 0),
        derive_request_seed(42, _identity(), 1),
    )
    assert second.request_seeds == (derive_request_seed(42, _identity(), 0),)


def test_proxy_rejects_an_upstream_owned_seed(tmp_path: Path) -> None:
    proxy = DeterministicSeedServerManager(
        _ServerManagerFixture(),
        base_seed=42,
        trajectory=_identity(),
        evidence_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="ambiguous seed ownership"):
        asyncio.run(proxy.generate("request", prompt_ids=[], sampling_params={"seed": None}))


def test_proxy_rejects_concurrent_calls_before_sampling(tmp_path: Path) -> None:
    class BlockingServerManager:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0

        async def generate(self, *_args: Any, **_kwargs: Any) -> str:
            self.calls += 1
            self.started.set()
            await self.release.wait()
            return "generated"

    async def exercise() -> None:
        upstream = BlockingServerManager()
        proxy = DeterministicSeedServerManager(
            upstream, base_seed=42, trajectory=_identity(), evidence_root=tmp_path
        )
        first = asyncio.create_task(
            proxy.generate("first", prompt_ids=[], sampling_params={})
        )
        await upstream.started.wait()
        with pytest.raises(RuntimeError, match="concurrent generate calls"):
            await proxy.generate("second", prompt_ids=[], sampling_params={})
        upstream.release.set()
        assert await first == "generated"
        assert upstream.calls == 1

    asyncio.run(exercise())


def test_worker_patch_forwards_identity_without_mutating_inputs() -> None:
    class WorkerFixture:
        async def _run_agent_flow(
            self,
            sampling_params: dict[str, Any],
            trajectory: dict[str, Any],
            *,
            agent_name: str,
            trace: bool = True,
            **kwargs: Any,
        ) -> dict[str, Any]:
            return {
                "sampling_params": sampling_params,
                "trajectory": trajectory,
                "agent_name": agent_name,
                "trace": trace,
                "kwargs": kwargs,
            }

    original_signature = inspect.signature(WorkerFixture._run_agent_flow)
    params = {"temperature": 1.0}
    trajectory = _identity()

    assert patch_agent_flow_worker_class(WorkerFixture) is True
    result = asyncio.run(
        WorkerFixture()._run_agent_flow(
            params,
            trajectory,
            agent_name="trading",
            trace=False,
            index=3,
            dataset_value="kept",
        )
    )

    assert params == {"temperature": 1.0}
    assert trajectory == _identity()
    assert result["kwargs"] == {
        "index": 3,
        "dataset_value": "kept",
        ROLLOUT_IDENTITY_KWARG: _identity(),
    }
    assert inspect.signature(WorkerFixture._run_agent_flow) == original_signature
    assert patch_agent_flow_worker_class(WorkerFixture) is False


def test_worker_patch_fails_closed_on_collision_nontrading_and_signature_drift() -> None:
    class WorkerFixture:
        async def _run_agent_flow(
            self,
            sampling_params: dict[str, Any],
            trajectory: dict[str, Any],
            *,
            agent_name: str,
            trace: bool = True,
            **kwargs: Any,
        ) -> None:
            return None

    patch_agent_flow_worker_class(WorkerFixture)
    with pytest.raises(RuntimeError, match="requires agent_name='trading'"):
        asyncio.run(
            WorkerFixture()._run_agent_flow({}, _identity(), agent_name="other")
        )
    with pytest.raises(RuntimeError, match="reserved rollout identity key collision"):
        collision_kwargs: dict[str, Any] = {ROLLOUT_IDENTITY_KWARG: _identity()}
        asyncio.run(
            WorkerFixture()._run_agent_flow(
                {},
                _identity(),
                agent_name="trading",
                index=3,
                **collision_kwargs,
            )
        )
    with pytest.raises(RuntimeError, match="dataset index is required"):
        asyncio.run(
            WorkerFixture()._run_agent_flow({}, _identity(), agent_name="trading")
        )
    with pytest.raises(RuntimeError, match="does not match trajectory sample_index"):
        asyncio.run(
            WorkerFixture()._run_agent_flow(
                {}, _identity(), agent_name="trading", index=4
            )
        )

    class DriftedWorkerFixture:
        async def _run_agent_flow(self, sampling_params: dict[str, Any]) -> None:
            return None

    with pytest.raises(RuntimeError, match="signature changed"):
        patch_agent_flow_worker_class(DriftedWorkerFixture)


def test_worker_patch_preserves_the_pinned_ray_trace_context() -> None:
    class RayWrappedWorkerFixture:
        async def _run_agent_flow(
            self,
            sampling_params: dict[str, Any],
            trajectory: dict[str, Any],
            *,
            agent_name: str,
            trace: bool = True,
            _ray_trace_ctx: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            return {"ray_trace_ctx": _ray_trace_ctx, "kwargs": kwargs}

    RayWrappedWorkerFixture._run_agent_flow.__ray_tracing_wrapped__ = True  # type: ignore[attr-defined]
    patch_agent_flow_worker_class(RayWrappedWorkerFixture)

    result = asyncio.run(
        RayWrappedWorkerFixture()._run_agent_flow(
            {},
            _identity(),
            agent_name="trading",
            index=3,
            _ray_trace_ctx={"traceparent": "fixture"},
        )
    )

    assert result["ray_trace_ctx"] == {"traceparent": "fixture"}
    assert result["kwargs"][ROLLOUT_IDENTITY_KWARG] == _identity()


def test_evidence_summary_rejects_tampered_seed(tmp_path: Path) -> None:
    path = tmp_path / "tampered.jsonl"
    path.write_text(
        json.dumps(
            {
                "base_seed": 42,
                "env_step": 0,
                "pid": 7,
                "prompt_token_count": 3,
                "request_seed": 123,
                "rollout_n": 0,
                "sample_index": 3,
                "schema_version": ROLLOUT_SEED_EVIDENCE_SCHEMA_VERSION,
                "scheme": ROLLOUT_SEED_SCHEME,
                "step": 1,
                "validate": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="invalid derived rollout seed"):
        summarize_rollout_seed_evidence(tmp_path)


def test_evidence_summary_rejects_duplicate_identity_and_wrong_base_seed(
    tmp_path: Path,
) -> None:
    upstream = _ServerManagerFixture()
    first = DeterministicSeedServerManager(
        upstream, base_seed=42, trajectory=_identity(), evidence_root=tmp_path
    )
    second = DeterministicSeedServerManager(
        upstream, base_seed=42, trajectory=_identity(), evidence_root=tmp_path
    )
    asyncio.run(first.generate("first", prompt_ids=[], sampling_params={}))
    asyncio.run(second.generate("duplicate", prompt_ids=[], sampling_params={}))

    with pytest.raises(RuntimeError, match="duplicate rollout seed evidence"):
        summarize_rollout_seed_evidence(tmp_path)

    for path in tmp_path.glob("*.jsonl"):
        path.unlink()
    third = DeterministicSeedServerManager(
        upstream, base_seed=42, trajectory=_identity(), evidence_root=tmp_path
    )
    asyncio.run(third.generate("third", prompt_ids=[], sampling_params={}))
    with pytest.raises(RuntimeError, match="rollout base seed mismatch"):
        summarize_rollout_seed_evidence(tmp_path, expected_base_seed=43)

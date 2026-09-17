"""Specialization and failure-boundary tests using existing CPU-only executor fakes.

The fake helper is imported into its own module and rebound only with pytest's
restoring monkeypatch. Neither a live diagnostic module nor a prior test module
is patched. Actual inference, filesystem durability, and market sources are not
retested here; this suite checks the new population, adapter and negative gates.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Dataclass annotation processing needs a module during fake import only.
    assert name not in sys.modules
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[name]
    return module


S = load("semantics_probe_under_test", ROOT / "scripts/diagnostics/m2_semantics_probe.py")
T = load("semantics_private_cpu_fakes", ROOT / "tests/test_m2_decision_probe.py")


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(T, "P", S.E)
    monkeypatch.setattr(T, "CODES", {condition: i + 1 for i, condition in enumerate(S.CONDITIONS)})

    def write_new(path: Path, value: Any) -> None:
        with path.open("x") as handle:
            json.dump(value, handle)

    # Keep the production evidence shapes and exclusive writes; avoid thousands
    # of physical fsyncs in a pure CPU population-control test.
    monkeypatch.setattr(S.H, "write_new", write_new)
    return T


def fixture_manifest(fakes: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    manifest = fakes.fixture_manifest()
    manifest.update(diagnostic_id="m2_semantics_v1", executor_binding=S.executor_binding())
    old = {
        "models": {
            "wiring": {
                "initial_prompts": {
                    "memory_only": copy.deepcopy(
                        manifest["models"]["wiring"]["initial_prompts"]["memory_only"]
                    )
                }
            }
        }
    }
    monkeypatch.setattr(S, "baseline", lambda: copy.deepcopy(old))
    return manifest


def test_isolation_preserves_second_original_module_and_exact_budget() -> None:
    original = S.isolated_executor()
    assert len(original.grid_plan()) == 96
    assert original.MODEL_CONDITIONS["wiring"] == ("raw", "history_only", "memory_only", "both")
    assert original.HERE == ROOT / "runs/m2-decision-diagnostic-20260909/probe-v1"
    assert list(original.H.MODELS) == ["screening", "wiring"]
    assert original is not S.E and original.H is not S.H
    assert len(S.grid_plan()) == 64
    assert [p["grid_index"] for p in S.grid_plan()] == list(range(64))
    assert sum(p["request_limit"] for p in S.grid_plan()) == 4096
    assert all(p["start_session"] == 0 and p["model_key"] == "wiring" for p in S.grid_plan())
    assert (
        S.E.MAX_REQUESTS == 4096 and S.E.budget_contract()["max_response_tokens_total"] == 16777216
    )
    for window in (0, 1):
        cohort = [p for p in S.grid_plan() if p["cohort_index"] == window]
        assert len(cohort) == 32 and {p["window_index"] for p in cohort} == {window}
        assert [p["condition"] for p in cohort[::8]] == list(S.CONDITIONS)
        assert len({S.E.request_seed(p, 0) for p in cohort}) == 8
    assert len({S.E.request_seed(p, 0) for p in S.grid_plan()}) == 16
    assert hashlib.sha256(S.EXECUTOR_SCRIPT.read_bytes()).hexdigest() == S.EXECUTOR_SHA
    assert S.E.execute_grid.__code__.co_filename == str(S.EXECUTOR_SCRIPT)
    assert S.E.run.__code__.co_filename == str(S.EXECUTOR_SCRIPT)
    assert S.E.HERE / "wiring/result.json" == S.HERE / "wiring/result.json"


def test_binding_drift_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(S.E, "MAX_REQUESTS", 8192)
    with pytest.raises(RuntimeError, match="binding drift"):
        S.verify_binding()


@pytest.mark.parametrize(
    "mutation",
    [
        "grid",
        "order",
        "budget",
        "binding",
        "baseline",
        "missing_factor",
        "thinking",
        "resource",
        "model",
    ],
)
def test_manifest_rejects_scientific_or_adapter_drift(
    fakes: Any, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    manifest = fixture_manifest(fakes, monkeypatch)
    S.validate_manifest(manifest)
    model = manifest["models"]["wiring"]
    if mutation == "grid":
        manifest["grid"][0]["identity"]["rollout_n"] = 8
    if mutation == "order":
        manifest["grid"][0], manifest["grid"][1] = manifest["grid"][1], manifest["grid"][0]
    if mutation == "budget":
        manifest["budget"]["max_requests_total"] = 8192
    if mutation == "binding":
        manifest["executor_binding"]["overrides"]["MAX_REQUESTS"] = 8192
    if mutation == "baseline":
        model["initial_prompts"]["memory_only"][0]["prompt_ids"].append(42)
    if mutation == "missing_factor":
        model["initial_prompts"]["availability"] = copy.deepcopy(
            model["initial_prompts"]["memory_only"]
        )
    if mutation == "thinking":
        model["initial_prompts"]["time_semantics"][0]["template_kwargs"]["enable_thinking"] = True
    if mutation == "resource":
        model["engine_kwargs"]["max_num_seqs"] = 8
    if mutation == "model":
        model["name"] = "Qwen/Qwen3-4B"
    with pytest.raises(RuntimeError):
        S.validate_manifest(manifest)


def test_fixed64_complete_uses_only_new_cohorts_and_semantic_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fakes: Any
) -> None:
    original_environment = fakes.Environment

    def semantic_environment(*args: Any, **kwargs: Any) -> Any:
        env = original_environment(*args, **kwargs)
        env.feature_evidence = {"kind": "semantic_display", "transport_field": "feature_evidence"}
        return env

    monkeypatch.setattr(fakes, "Environment", semantic_environment)
    engine, state, coroutine = fakes.execute(tmp_path, monkeypatch, "normal", "wiring")
    asyncio.run(coroutine)
    assert state["status"] == "OBSERVED" and state["complete_count"] == 64
    assert state["requests_submitted"] == state["actions_applied"] == 1280
    assert state["prefix_actions_applied"] == 0 and engine.batch_sizes == [32] * 40
    assert [r["trajectory_id"] for r in state["grid"]] == [
        p["trajectory_id"] for p in S.grid_plan()
    ]
    assert all(
        r["prefix_steps_count"] == 0 and r["terminal_metrics"]["session_count"] == 21
        for r in state["grid"]
    )
    first = S.H.read(tmp_path / "cohort-0-round-000.submitted.json")["requests"]
    assert [r["grid_index"] for r in first] == list(range(32))
    assert first[0]["seed_env_step"] == first[0]["environment_trace_index"] == 0
    assert first[0]["feature_evidence"]["kind"] == "semantic_display"
    second = S.H.read(tmp_path / "cohort-1-round-000.submitted.json")["requests"]
    assert [r["grid_index"] for r in second] == list(range(32, 64))


@pytest.mark.parametrize("mode", ["response_cap", "prompt_cap", "horizon"])
def test_negative_trajectory_has_no_retry_or_false_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fakes: Any, mode: str
) -> None:
    engine, state, coroutine = fakes.execute(tmp_path, monkeypatch, mode, "wiring")
    asyncio.run(coroutine)
    row = state["grid"][0]
    assert state["status"] == "OBSERVED" and state["complete_count"] == 63
    assert row["status"] == ("HORIZON_REACHED" if mode == "horizon" else "INCOMPLETE")
    assert row["requests_submitted"] == (64 if mode == "horizon" else 1)
    assert "terminal_metrics" not in row and row["complete"] is False
    assert all(r["status"] == "COMPLETE" for r in state["grid"][1:])
    assert max(engine.batch_sizes) == 32 and len(state["grid"]) == 64
    if mode == "response_cap":
        assert row["actions_applied"] == 0
        assert not (tmp_path / "request-000000.applied.json").exists()


@pytest.mark.parametrize(
    "mode,returned,mapped,decoded,applied",
    [
        ("partial_return", 3, 0, 0, 0),
        ("decode_failure", 32, 32, 3, 0),
        ("env_failure", 32, 32, 32, 3),
        ("engine_failure", 32, 32, 32, 32),
    ],
)
def test_runtime_failure_preserves_returned_prefix_and_unstarted_second_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fakes: Any,
    mode: str,
    returned: int,
    mapped: int,
    decoded: int,
    applied: int,
) -> None:
    _, state, coroutine = fakes.execute(tmp_path, monkeypatch, mode, "wiring")
    with pytest.raises((RuntimeError, ValueError)):
        asyncio.run(coroutine)
    assert state["status"] == "STOPPED"
    assert state["responses_returned"] == returned and state["responses_mapped"] == mapped
    assert state["responses_decoded"] == decoded and state["actions_applied"] == applied
    assert state["returned_but_unmapped"] == returned - mapped
    assert all(r["status"] == "ABORTED_RUNTIME" for r in state["grid"][:32])
    assert all(r["status"] == "UNSTARTED" for r in state["grid"][32:])
    assert not list(tmp_path.glob("cohort-1-*"))

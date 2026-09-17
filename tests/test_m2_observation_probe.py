"""Fixed-population and failure semantics for the complete Train observation diagnostic."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.agents.rollout_seed import derive_request_seed

_PATH = Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_observation_probe.py"
_SPEC = importlib.util.spec_from_file_location("observation_probe_tests", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


@dataclass
class Step:
    step_index: int
    action: str
    decision_session_index: int
    result_session_index: int
    invalid_action: bool
    done: bool


class Tokenizer:
    def apply_chat_template(self, messages: list[Any], **_: Any) -> list[int]:
        state = json.loads(messages[-1]["content"])
        if state.get("overflow"):
            return [10] * (probe.MAX_PROMPT + 1)
        return [
            100 + state["session"],
            200 + state["calls"],
            int(state["profile"] == "history_memory_v1"),
        ]

    def decode(self, tokens: list[int], **_: Any) -> str:
        return {11: "hold", 88: "compute", 99: "invalid"}[tokens[0]]


class Environment:
    def __init__(
        self, *, overflow: bool = False, stuck: bool = False, profile: str = "legacy"
    ) -> None:
        self.overflow = overflow
        self.stuck = stuck
        self.profile = profile
        self.tool_schemas = [{"type": "function", "function": {"name": "hold"}}]
        self.credit_step_trace: list[Step] = []
        self.interaction_trace: list[dict[str, Any]] = []
        self.current_session_index = 0
        self.episode_session_count = 21
        self.episode_complete = False
        self.ledger = SimpleNamespace(fills=[])
        self.terminal_metric_calls = 0

    def observation(self) -> Any:
        return SimpleNamespace(
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "session": self.current_session_index,
                            "calls": len(self.credit_step_trace),
                            "overflow": self.overflow and bool(self.credit_step_trace),
                            "stuck": self.stuck,
                            "profile": self.profile,
                        }
                    ),
                }
            ],
            text=None,
            token_ids=None,
        )

    def reset(self, **_: Any) -> Any:
        return self.observation()

    def screening_terminal_metrics(self) -> dict[str, Any]:
        assert self.episode_complete and self.current_session_index == 20
        self.terminal_metric_calls += 1
        return {
            "scorer_id": "benchmark_scale_v2",
            "raw_net_return": 0.0,
            "terminal_score": 0.0,
            "session_count": 21,
            "step_count": len(self.credit_step_trace),
            "invocations": self.terminal_metric_calls,
        }

    async def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        before = self.current_session_index
        if action.text == "hold":
            self.current_session_index += 1
        self.episode_complete = self.current_session_index == 20
        self.credit_step_trace.append(
            Step(
                len(self.credit_step_trace),
                action.text,
                before,
                self.current_session_index,
                action.text == "invalid",
                self.episode_complete,
            )
        )
        self.interaction_trace.append(
            {
                "response": {
                    "text": action.text,
                    "sha256": hashlib.sha256(action.text.encode()).hexdigest(),
                },
                "info": {"invalid_action": action.text == "invalid"},
            }
        )
        return self.observation(), 0.0, self.episode_complete, self.interaction_trace[-1]["info"]


class Engine:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls = 0
        self.next_id = 0
        self.batch_sizes: list[int] = []
        self.stuck_seeds = {probe.request_seed(probe.grid_plan()[0], step) for step in range(256)}

    def get_tokenizer(self) -> Tokenizer:
        return Tokenizer()

    def generate(self, prompts: list[Any], params: list[Any], **_: Any) -> list[Any]:
        self.calls += 1
        self.batch_sizes.append(len(prompts))
        assert len(prompts) == len(params) and all(p.temperature == 1 for p in params)
        if self.mode == "engine_failure" and self.calls == 2:
            raise RuntimeError("synthetic engine failure")
        outputs = []
        for prompt, params_for_request in zip(prompts, params, strict=True):
            index = self.next_id
            self.next_id += 1
            tokens, finish = [11, 151645], "stop"
            if (
                self.mode == "step_cap"
                and self.calls <= 256
                and params_for_request.seed in self.stuck_seeds
            ):
                tokens = [88, 151645]
            if self.mode == "response_cap" and index == 0:
                tokens, finish = [11] * 4096, "length"
            if self.mode == "invalid" and index == 1:
                tokens = [99, 151645]
            outputs.append(
                SimpleNamespace(
                    request_id=str(index),
                    prompt_token_ids=prompt["prompt_token_ids"],
                    outputs=[
                        SimpleNamespace(
                            index=0,
                            token_ids=tokens,
                            finish_reason=finish,
                            stop_reason=None,
                            text="native output",
                        )
                    ],
                )
            )
        if self.mode == "mapping_failure":
            outputs[-1].request_id = "9999"
        return outputs


def fixture_manifest(mode: str = "normal") -> dict[str, Any]:
    tokenizer = Tokenizer()
    models = {}
    for key, (name, revision) in probe.H.MODELS.items():
        prompts: dict[str, list[dict[str, Any]]] = {}
        for profile in probe.PROFILES:
            prompts[profile] = []
            for _index in range(2):
                env = Environment(profile=profile)
                prompts[profile].append(probe.render(tokenizer, env.reset(), env.tool_schemas))
        models[key] = {
            "name": name,
            "revision": revision,
            "snapshot": str(probe.H.snapshot(key)),
            "engine_kwargs": probe.engine_kwargs(probe.H.snapshot(key)),
            "initial_prompts": prompts,
        }
    return {
        "status": "PREPARED",
        "schema_version": 1,
        "diagnostic_id": "m2_observation_trajectory_v1",
        "budget": probe.budget_contract(),
        "grid": probe.grid_plan(),
        "common_objective_text": None,
        "action_protocol": "single_tool_v1",
        "windows": [{"raw_prompt": []}, {"raw_prompt": []}],
        "models": models,
    }


def execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str = "normal"
) -> tuple[Any, Any, Any]:
    manifest = fixture_manifest(mode)
    state = probe.initial_state("screening", "synthetic-manifest-sha")
    engine = Engine(mode)
    made: list[Environment] = []

    def make_env(index: int, profile: str) -> Environment:
        env = Environment(
            overflow=mode == "prompt_cap" and len(made) == 0,
            stuck=False,
            profile=profile,
        )
        made.append(env)
        return env

    monkeypatch.setattr(
        probe.H, "effective_params", lambda llm, params: {"sampling_params": vars(params)}
    )
    coroutine = probe.execute_grid(
        engine,
        SimpleNamespace,
        manifest,
        "screening",
        tmp_path,
        state,
        make_env,
        lambda text, ids: SimpleNamespace(text=text, token_ids=ids),
        lambda: None,
    )
    return engine, state, coroutine


def test_exact_fixed_32_grid_and_paired_production_seed_budget() -> None:
    grid = probe.grid_plan()
    expected = [
        (m, p, w, s)
        for m in ("screening", "wiring")
        for p in probe.PROFILES
        for w in range(2)
        for s in range(4)
    ]
    assert [
        (p["model_key"], p["profile"], p["window_index"], p["rollout_id"]) for p in grid
    ] == expected
    assert [p["grid_index"] for p in grid] == list(range(32))
    assert len({p["trajectory_id"] for p in grid}) == 32
    assert all(p["initial_status"] == "UNSTARTED" for p in grid)
    for env_step in (0, 19, 255):
        grouped: dict[Any, list[int]] = {}
        for row in grid:
            seed = probe.request_seed(row, env_step)
            assert seed == derive_request_seed(42, row["identity"], env_step)
            grouped.setdefault((row["window_index"], row["rollout_id"]), []).append(seed)
        assert all(len(set(seeds)) == 1 and len(seeds) == 4 for seeds in grouped.values())
    assert probe.budget_contract()["max_requests_total"] == 32 * 256 == 8192
    assert probe.budget_contract()["max_response_tokens_total"] == 33554432
    with pytest.raises(RuntimeError, match="outside budget"):
        probe.request_seed(grid[0], 256)


@pytest.mark.parametrize(
    "mutation", ["population", "cap", "objective", "profile", "seed", "template"]
)
def test_manifest_rejects_unregistered_population_and_budget(mutation: str) -> None:
    manifest = fixture_manifest()
    probe.validate_manifest(manifest)
    if mutation == "population":
        manifest["grid"].pop()
    elif mutation == "cap":
        manifest["budget"]["max_requests_per_trajectory"] = 257
    elif mutation == "objective":
        manifest["common_objective_text"] = probe.H.OBJECTIVE
    elif mutation == "profile":
        manifest["models"]["screening"]["initial_prompts"]["extra"] = []
    elif mutation == "seed":
        manifest["grid"][0]["identity"]["step"] = 0
    else:
        manifest["models"]["screening"]["initial_prompts"]["legacy"][0]["template_kwargs"][
            "enable_thinking"
        ] = True
    with pytest.raises(RuntimeError):
        probe.validate_manifest(manifest)


def test_complete_grid_requires_20_advances_without_refill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, state, coroutine = execute(tmp_path, monkeypatch)
    result = asyncio.run(coroutine)
    assert result is state and state["status"] == "OBSERVED"
    assert state["complete_count"] == 16 and state["incomplete_count"] == 0
    assert (
        state["requests_submitted"]
        == state["responses_returned"]
        == state["actions_applied"]
        == 320
    )
    assert engine.calls == 40 and engine.batch_sizes == [8] * 40
    assert all(
        len(row["steps"]) == 20 and row["current_session_index"] == 20 for row in state["grid"]
    )
    assert all(
        row["terminal_metrics"]["invocations"] == 1
        and row["terminal_metrics"]["step_count"] == len(row["steps"])
        for row in state["grid"]
    )
    assert all(
        probe.H.digest(tmp_path / name) == sha
        for name, sha in state["evidence_files_sha256"].items()
    )


@pytest.mark.parametrize("mode", ["response_cap", "prompt_cap"])
def test_individual_incomplete_continues_remaining_fixed_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    engine, state, coroutine = execute(tmp_path, monkeypatch, mode)
    asyncio.run(coroutine)
    assert (
        state["status"] == "OBSERVED"
        and state["complete_count"] == 15
        and state["incomplete_count"] == 1
    )
    assert state["grid"][0]["status"] == "INCOMPLETE"
    assert "terminal_metrics" not in state["grid"][0]
    assert all(row["status"] == "COMPLETE" for row in state["grid"][1:])
    assert state["requests_submitted"] == state["responses_returned"] == 301
    assert max(engine.batch_sizes) == 8 and 7 in engine.batch_sizes
    assert len(state["grid"]) == 16
    if mode == "response_cap":
        assert state["actions_applied"] == 300 and not state["grid"][0]["steps"]
    else:
        failure = probe.H.read(tmp_path / "trajectory-00.prompt_failure.json")
        assert failure["prompt_token_count"] == 8193 and len(state["grid"][0]["steps"]) == 1


def test_invalid_feedback_is_consumed_before_later_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, state, coroutine = execute(tmp_path, monkeypatch, "invalid")
    asyncio.run(coroutine)
    assert state["complete_count"] == 16 and state["requests_submitted"] == 321
    row = state["grid"][1]
    assert row["invalid_count"] == 1 and row["steps"][0]["invalid_action"] is True
    assert row["steps"][0]["result_session_index"] == 0 and len(row["steps"]) == 21
    assert engine.batch_sizes[20] == 1


def test_256_request_limit_stops_nonadvancing_trajectories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _engine, state, coroutine = execute(tmp_path, monkeypatch, "step_cap")
    asyncio.run(coroutine)
    assert state["complete_count"] == 15 and state["incomplete_count"] == 1
    row = state["grid"][0]
    assert row["status"] == "INCOMPLETE" and row["failure_kind"] == "STEP_CAP"
    assert row["requests_submitted"] == len(row["steps"]) == 256
    assert state["requests_submitted"] == 256 + 15 * 20


@pytest.mark.parametrize("mode", ["engine_failure", "mapping_failure"])
def test_runtime_failure_stops_model_and_preserves_unstarted_condition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    engine, state, coroutine = execute(tmp_path, monkeypatch, mode)
    with pytest.raises(RuntimeError):
        asyncio.run(coroutine)
    assert state["status"] == "STOPPED" and all(
        row["status"] == "UNSTARTED" for row in state["grid"][8:]
    )
    assert all(row["status"] == "ABORTED_RUNTIME" for row in state["grid"][:8])
    assert not list(tmp_path.glob("condition-history_memory_v1-*.json"))
    if mode == "engine_failure":
        assert engine.calls == 2 and state["actions_applied"] == 8
        assert (tmp_path / "condition-legacy-round-001.submitted.json").exists()
        assert not (tmp_path / "condition-legacy-round-001.raw.json").exists()
    else:
        assert engine.calls == 1 and state["actions_applied"] == 0
        assert len(probe.H.read(tmp_path / "condition-legacy-round-000.raw.json")["outputs"]) == 8


@pytest.mark.parametrize(
    "mutation", ["early_done", "missing_advance", "invalid_advance", "gap", "postterminal"]
)
def test_trace_validation_refuses_false_terminal(mutation: str) -> None:
    trace = [
        {
            "step_index": i,
            "action": "hold",
            "decision_session_index": i,
            "result_session_index": i + 1,
            "invalid_action": False,
            "done": i == 19,
        }
        for i in range(20)
    ]
    probe.check_trace(trace, done=True, session=20, sessions=21)
    bad = copy.deepcopy(trace)
    if mutation == "early_done":
        bad[0]["done"] = True
    elif mutation == "missing_advance":
        bad.pop()
    elif mutation == "invalid_advance":
        bad[4]["invalid_action"] = True
    elif mutation == "gap":
        bad[4]["decision_session_index"] = 3
    else:
        bad.append({**bad[-1], "step_index": 20})
    with pytest.raises(RuntimeError):
        probe.check_trace(bad, done=True, session=20, sessions=21)

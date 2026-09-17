"""CPU-only fixed decision grids, separate prefixes and irreversible terminal states."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.agents.rollout_seed import derive_request_seed
from src.agents.screening_prompt import apply_screening_action_protocol

_PATH = Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_decision_probe.py"
_SPEC = importlib.util.spec_from_file_location("decision_probe_tests", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
P = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(P)
CODES = {"raw": 1, "features": 2, "history_only": 3, "memory_only": 4, "both": 5}


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
        if state["overflow"]:
            return [10] * 8193
        return [
            state["session"] + 10,
            state["calls"] + 200,
            CODES[state["condition"]],
            state["window"],
        ]

    def decode(self, tokens: list[int], **_: Any) -> str:
        if tokens[0] == 77:
            raise ValueError("synthetic decode error")
        return {11: "hold", 88: "compute", 99: "invalid"}[tokens[0]]


class Environment:
    def __init__(
        self, window: int, condition: str, start: int, *, fail: bool = False, overflow: bool = False
    ) -> None:
        self.window = window
        self.decision_condition = condition
        self.diagnostic_start_session = start
        self.fail = fail
        self.overflow = overflow
        self.in_prefix = True
        self.current_session_index = 0
        self.episode_complete = False
        self.episode_session_count = 21
        self.credit_step_trace: list[Step] = []
        self.interaction_trace: list[dict[str, Any]] = []
        self.tool_schemas = [{"type": "function", "function": {"name": "hold"}}]
        self.ledger = SimpleNamespace(positions={}, fills=[])
        self.feature_evidence = None
        self.format_wrapper = SimpleNamespace(
            parse_response=lambda text: (None, [SimpleNamespace(name=text, arguments={})])
        )

    @property
    def diagnostic_pit_counts(self) -> dict[str, int]:
        return {f"SYM{i}": self.current_session_index + self.window * 4 + 1 for i in range(20)}

    def observation(self) -> Any:
        return SimpleNamespace(
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "condition": self.decision_condition,
                            "window": self.window,
                            "session": self.current_session_index,
                            "calls": len(self.credit_step_trace),
                            "overflow": self.overflow
                            and len(self.credit_step_trace) > self.diagnostic_start_session,
                        }
                    ),
                }
            ],
            text=None,
            token_ids=None,
        )

    async def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        if self.fail and not self.in_prefix:
            raise RuntimeError("synthetic environment failure")
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
        info: dict[str, Any] = {"invalid_action": action.text == "invalid", "fills": []}
        if action.text == "invalid":
            info["error"] = "synthetic invalid tool arguments"
        self.interaction_trace.append(
            {
                "response": {
                    "text": action.text,
                    "sha256": hashlib.sha256(action.text.encode()).hexdigest(),
                },
                "info": info,
            }
        )
        return self.observation(), 0.0, self.episode_complete, info

    def screening_terminal_metrics(self) -> dict[str, Any]:
        assert self.episode_complete and self.current_session_index == 20
        return {
            "session_count": 21,
            "step_count": len(self.credit_step_trace),
            "raw_net_return": 0.0,
            "terminal_score": 0.0,
        }


class Conditions:
    @staticmethod
    async def reset_with_prefix(env: Environment) -> tuple[Any, dict[str, Any]]:
        for _ in range(env.diagnostic_start_session):
            await env.step(SimpleNamespace(text="hold"))
        env.in_prefix = False
        return env.observation(), {
            "is_model_request": False,
            "model_requests": 0,
            "start_session": env.diagnostic_start_session,
            "prefix_action_count": env.diagnostic_start_session,
            "credit_step_trace": [asdict(s) for s in env.credit_step_trace],
            "interaction_trace": env.interaction_trace,
        }

    @staticmethod
    def render_prompt(tokenizer: Tokenizer, observation: Any, tools: Any) -> dict[str, Any]:
        messages = apply_screening_action_protocol(observation.messages, "single_tool_v1")
        ids = tokenizer.apply_chat_template(messages)
        return {
            "messages": messages,
            "tools": copy.deepcopy(tools),
            "prompt_ids": ids,
            "prompt_sha256": P.H.token_sha(ids),
            "prompt_token_count": len(ids),
            "template_kwargs": {"enable_thinking": False},
            "action_protocol": "single_tool_v1",
        }


class Engine:
    def __init__(self, mode: str, model: str) -> None:
        self.mode = mode
        self.calls = 0
        self.next_id = 0
        self.batch_sizes: list[int] = []
        plan = next(p for p in P.grid_plan() if p["model_key"] == model)
        self.seed_to_step = {
            P.request_seed(plan, step): step for step in range(plan["request_limit"])
        }

    def get_tokenizer(self) -> Tokenizer:
        return Tokenizer()

    def generate(self, prompts: list[Any], params: list[Any], **_: Any) -> list[Any]:
        self.calls += 1
        self.batch_sizes.append(len(prompts))
        assert len(prompts) == len(params) and len(prompts) <= 32
        if self.mode == "engine_failure" and self.calls == 2:
            raise RuntimeError("synthetic engine failure")
        outputs = []
        for prompt, param in zip(prompts, params, strict=True):
            assert param.temperature == 1 and param.n == 1 and param.max_tokens == 4096
            index = self.next_id
            self.next_id += 1
            tokens, finish = [11, 151645], "stop"
            raw_first = prompt["prompt_token_ids"][2:] == [1, 0] and param.seed in self.seed_to_step
            if raw_first and self.mode == "horizon":
                tokens = [88, 151645]
            if (
                raw_first
                and self.mode == "last_request_terminal"
                and self.seed_to_step[param.seed] < 112
            ):
                tokens = [88, 151645]
            if self.mode == "response_cap" and index == 0:
                tokens, finish = [11] * 4096, "length"
            if self.mode == "invalid" and index == 1:
                tokens = [99, 151645]
            if self.mode == "decode_failure" and index == 3:
                tokens = [77, 151645]
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
                            text="native text",
                        )
                    ],
                )
            )
        if self.mode == "partial_return":
            return outputs[:3]
        return outputs


def fixture_manifest() -> dict[str, Any]:
    models = {}
    for model, (name, revision) in P.H.MODELS.items():
        initial: dict[str, list[dict[str, Any]]] = {}
        for condition in P.MODEL_CONDITIONS[model]:
            initial[condition] = []
            for window in range(2):
                env = Environment(window, condition, P.START_SESSIONS[model])
                observation, _ = asyncio.run(Conditions.reset_with_prefix(env))
                initial[condition].append(
                    Conditions.render_prompt(Tokenizer(), observation, env.tool_schemas)
                )
        models[model] = {
            "name": name,
            "revision": revision,
            "snapshot": str(P.H.snapshot(model)),
            "engine_kwargs": P.engine_kwargs(P.H.snapshot(model)),
            "initial_prompts": initial,
        }
    return {
        "status": "PREPARED",
        "schema_version": 1,
        "diagnostic_id": "m2_decision_v1",
        "grid": P.grid_plan(),
        "budget": P.budget_contract(),
        "models": models,
        "windows": [{}, {}],
    }


def execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, model: str = "screening"
) -> tuple[Any, Any, Any]:
    state = P.initial_state(model, "synthetic_manifest")
    manifest = fixture_manifest()
    engine = Engine(mode, model)
    created: list[Environment] = []

    def make_env(window: int, condition: str, start: int) -> Environment:
        env = Environment(
            window,
            condition,
            start,
            fail=mode == "env_failure" and len(created) == 3,
            overflow=mode == "prompt_cap" and len(created) == 0,
        )
        created.append(env)
        return env

    monkeypatch.setattr(
        P.H, "effective_params", lambda llm, params: {"sampling_params": vars(params)}
    )
    coroutine = P.execute_grid(
        engine,
        SimpleNamespace,
        manifest,
        model,
        tmp_path,
        state,
        Conditions,
        make_env,
        lambda text, ids: SimpleNamespace(text=text, token_ids=ids),
        lambda: None,
    )
    return engine, state, coroutine


def test_exact_population_cohorts_prefix_horizons_and_seeds() -> None:
    grid = P.grid_plan()
    assert len(grid) == len({p["trajectory_id"] for p in grid}) == 96
    assert [p["grid_index"] for p in grid] == list(range(96))
    assert all(p["initial_status"] == "UNSTARTED" for p in grid)
    assert sum(p["request_limit"] for p in grid) == 8192
    assert P.budget_contract()["max_response_tokens_total"] == 33554432
    assert len([p for p in grid if p["model_key"] == "screening"]) == 32
    for model, count in (("screening", 1), ("wiring", 2)):
        for cohort in range(count):
            cohort_rows = [
                p for p in grid if p["model_key"] == model and p["cohort_index"] == cohort
            ]
            assert len(cohort_rows) == 32
            assert all(
                p["start_session"] == (4 if model == "screening" else 0) for p in cohort_rows
            )
            if model == "wiring":
                assert {p["window_index"] for p in cohort_rows} == {cohort}
    grouped: dict[Any, set[int]] = {}
    for plan in grid:
        assert P.request_seed(plan, 0) == derive_request_seed(42, plan["identity"], 0)
        grouped.setdefault((plan["window_index"], plan["rollout_id"]), set()).add(
            P.request_seed(plan, 0)
        )
        with pytest.raises(RuntimeError, match="horizon"):
            P.request_seed(plan, plan["request_limit"])
    assert len(grouped) == 16 and all(len(seeds) == 1 for seeds in grouped.values())
    assert len({next(iter(seeds)) for seeds in grouped.values()}) == 16


def test_gpu_contract_is_new_and_does_not_mutate_frozen_helper() -> None:
    path = P.H.snapshot("screening")
    kwargs = P.engine_kwargs(path)
    assert kwargs["gpu_memory_utilization"] == 0.5 and kwargs["max_num_seqs"] == 32
    assert kwargs["max_model_len"] == 12288 and kwargs["max_num_batched_tokens"] == 16384
    assert kwargs["compilation_config"]["cudagraph_capture_sizes"] == [1, 2, 4, 8, 16, 32]
    assert P.H.engine_kwargs(path)["max_num_seqs"] == 8


@pytest.mark.parametrize("mutation", ["grid", "budget", "prefix", "memory", "engine", "thinking"])
def test_manifest_refuses_unregistered_changes(mutation: str) -> None:
    manifest = fixture_manifest()
    P.validate_manifest(manifest)
    if mutation == "grid":
        manifest["grid"].pop()
    elif mutation == "budget":
        manifest["budget"]["max_requests_total"] = 8193
    elif mutation == "prefix":
        manifest["grid"][0]["start_session"] = 0
    elif mutation == "memory":
        manifest["models"]["wiring"]["initial_prompts"]["memory_only"] = copy.deepcopy(
            manifest["models"]["wiring"]["initial_prompts"]["raw"]
        )
    elif mutation == "engine":
        manifest["models"]["screening"]["engine_kwargs"]["gpu_memory_utilization"] = 0.9
    else:
        manifest["models"]["screening"]["initial_prompts"]["features"][0]["template_kwargs"][
            "enable_thinking"
        ] = True
    with pytest.raises(RuntimeError):
        P.validate_manifest(manifest)


def test_full_4b_prefix_is_not_counted_as_model_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, state, coroutine = execute(tmp_path, monkeypatch, "normal")
    asyncio.run(coroutine)
    assert state["status"] == "OBSERVED" and state["complete_count"] == 32
    assert state["prefix_actions_applied"] == 128
    assert (
        state["requests_submitted"] == state["responses_mapped"] == state["actions_applied"] == 512
    )
    assert engine.calls == 16 and engine.batch_sizes == [32] * 16
    assert all(
        row["prefix_steps_count"] == 4
        and row["action_counts"] == {"hold": 16}
        and len(row["steps"]) == 20
        for row in state["grid"]
    )
    assert all(
        group["action_counts"] == {"hold": 128} and group["prefix_steps_count"] == 32
        for group in state["summary_by_condition_window"]
    )
    first = P.H.read(tmp_path / "cohort-0-round-000.submitted.json")["requests"][0]
    assert first["seed_env_step"] == first["trajectory_request_index"] == 0
    assert first["environment_trace_index"] == first["environment_session_index"] == 4
    assert set(first["diagnostic_pit_counts"].values()) == {5}


@pytest.mark.parametrize("mode", ["response_cap", "prompt_cap"])
def test_cap_permanently_exits_without_resubmission_or_refill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    engine, state, coroutine = execute(tmp_path, monkeypatch, mode)
    asyncio.run(coroutine)
    row = state["grid"][0]
    assert state["complete_count"] == 31 and state["incomplete_count"] == 1
    assert row["status"] == "INCOMPLETE" and row["requests_submitted"] == 1
    assert row["actions_applied"] == (0 if mode == "response_cap" else 1)
    assert state["requests_submitted"] == 497 and engine.batch_sizes == [32] + [31] * 15
    submitted = [
        request
        for path in tmp_path.glob("*.submitted.json")
        for request in P.H.read(path)["requests"]
    ]
    assert len([r for r in submitted if r["grid_index"] == 0]) == 1
    assert "terminal_metrics" not in row


@pytest.mark.parametrize("model", ["screening", "wiring"])
def test_horizon_is_not_complete_and_has_no_terminal_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    _engine, state, coroutine = execute(tmp_path, monkeypatch, "horizon", model)
    asyncio.run(coroutine)
    row = state["grid"][0]
    limit = 128 if model == "screening" else 64
    assert row["status"] == "HORIZON_REACHED" and row["complete"] is False
    assert row["requests_submitted"] == row["actions_applied"] == limit
    assert "terminal_metrics" not in row and "failure_kind" not in row
    assert row["nonadvancing_repeated_tool_calls"] == limit - 1
    assert state["complete_count"] == len(state["grid"]) - 1 and state["horizon_count"] == 1


def test_terminal_on_last_registered_request_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _engine, state, coroutine = execute(tmp_path, monkeypatch, "last_request_terminal")
    asyncio.run(coroutine)
    row = state["grid"][0]
    assert row["status"] == "COMPLETE" and row["requests_submitted"] == 128
    assert len(row["steps"]) == 132 and row["current_session_index"] == 20
    assert row["terminal_metrics"]["step_count"] == 132


@pytest.mark.parametrize(
    "mode,returned,mapped,decoded,applied",
    [
        ("partial_return", 3, 0, 0, 0),
        ("decode_failure", 32, 32, 3, 0),
        ("env_failure", 32, 32, 32, 3),
        ("engine_failure", 32, 32, 32, 32),
    ],
)
def test_stopped_returned_and_applied_prefixes_are_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    returned: int,
    mapped: int,
    decoded: int,
    applied: int,
) -> None:
    _engine, state, coroutine = execute(tmp_path, monkeypatch, mode, "wiring")
    with pytest.raises((RuntimeError, ValueError)):
        asyncio.run(coroutine)
    assert state["status"] == "STOPPED"
    assert state["responses_returned"] == returned and state["responses_mapped"] == mapped
    assert state["responses_decoded"] == decoded and state["actions_applied"] == applied
    assert state["returned_but_unmapped"] == returned - mapped
    assert sum(row["responses_mapped"] for row in state["grid"]) == mapped
    assert sum(row["actions_applied"] for row in state["grid"]) == applied
    assert all(row["status"] == "UNSTARTED" for row in state["grid"][32:])
    assert not list(tmp_path.glob("cohort-1-*.json"))
    if mode == "decode_failure":
        assert len(P.H.read(tmp_path / "cohort-0-round-000.audited_prefix.json")["rows"]) == 3
    if mode == "env_failure":
        assert len(list(tmp_path.glob("*.applied.json"))) == 3


def test_invalid_error_summary_excludes_scripted_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _engine, state, coroutine = execute(tmp_path, monkeypatch, "invalid")
    asyncio.run(coroutine)
    row = state["grid"][1]
    assert row["action_counts"] == {"invalid": 1, "hold": 16}
    assert row["invalid_errors"] == {"synthetic invalid tool arguments": 1}
    assert row["prefix_steps_count"] == 4 and row["invalid_count"] == 1
    assert state["complete_count"] == 32 and state["requests_submitted"] == 513

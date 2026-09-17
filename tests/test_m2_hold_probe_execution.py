"""The fixed first-step grid continues negatives but never refills runtime failures."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.agents.screening_prompt import apply_screening_action_protocol

_path = Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_hold_probe.py"
_spec = importlib.util.spec_from_file_location("hold_execution_probe", _path)
assert _spec is not None and _spec.loader is not None
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


@dataclass
class Step:
    action: str = "hold"
    invalid_action: bool = False


class Tokenizer:
    def apply_chat_template(self, *_: Any, **__: Any) -> list[int]:
        return [10]

    def decode(self, tokens: list[int], **_: Any) -> str:
        return "invalid" if tokens[0] == 99 else "<tool_call>hold</tool_call>"


class Engine:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls = 0

    def get_tokenizer(self) -> Tokenizer:
        return Tokenizer()

    def generate(self, prompts: list[Any], params: list[Any], **_: Any) -> list[Any]:
        start = self.calls * 8
        self.calls += 1
        if self.mode == "engine_failure" and self.calls == 2:
            raise RuntimeError("engine failed after submitting the second batch")
        assert len(prompts) == len(params) == 8
        outputs = []
        for offset, prompt in enumerate(prompts):
            index = start + offset
            tokens = [11, 151645]
            finish = "stop"
            if self.mode == "negative" and index == 0:
                tokens, finish = [11] * 4096, "length"
            if self.mode == "negative" and index == 1:
                tokens = [99, 151645]
            output = SimpleNamespace(
                request_id=str(index),
                prompt_token_ids=prompt["prompt_token_ids"],
                outputs=[
                    SimpleNamespace(
                        index=0,
                        token_ids=tokens,
                        finish_reason=finish,
                        stop_reason=None,
                        text="raw engine text",
                    )
                ],
            )
            outputs.append(output)
        if self.mode == "wrong_mapping":
            outputs[-1].request_id = "999"
        return outputs


def fixture_manifest() -> dict[str, Any]:
    initial = apply_screening_action_protocol(
        [{"role": "user", "content": "state"}], "single_tool_v1"
    )
    prompt = {
        "messages": initial,
        "tools": [],
        "prompt_ids": [10],
        "template_kwargs": {"enable_thinking": False},
    }
    return {
        "request_plan": probe.build_request_plan(),
        "models": {
            "screening": {
                "prompts": {
                    c: [copy.deepcopy(prompt), copy.deepcopy(prompt)] for c in probe.CONDITIONS
                }
            }
        },
        "windows": [
            {
                "raw_prompt": [{"role": "user", "content": "state"}],
                "initial_messages": copy.deepcopy(initial),
            }
            for _ in range(2)
        ],
    }


def execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> tuple[Any, Any, Any]:
    calls: list[str] = []

    class Environment:
        def __init__(self) -> None:
            self.interaction_trace: list[Any] = []
            self.credit_step_trace: list[Any] = []
            self.ledger = SimpleNamespace(fills=[])
            self.current_session_index = 0
            self.episode_complete = False

        def reset(self, *, raw_prompt: list[Any]) -> Any:
            return SimpleNamespace(messages=raw_prompt)

        async def step(self, action: Any) -> None:
            calls.append(action.text)
            self.credit_step_trace = [Step(invalid_action=action.text == "invalid")]
            self.current_session_index = 0 if action.text == "invalid" else 1
            self.interaction_trace = [
                {
                    "response": {"sha256": hashlib.sha256(action.text.encode()).hexdigest()},
                    "info": {"invalid_action": action.text == "invalid"},
                }
            ]

    monkeypatch.setattr(
        probe, "effective_params", lambda llm, param: {"sampling_params": vars(param)}
    )
    engine = Engine(mode)
    coroutine = probe.execute_batches(
        engine,
        SimpleNamespace,
        fixture_manifest(),
        "screening",
        tmp_path,
        lambda index: Environment(),
        lambda text, ids: SimpleNamespace(text=text, token_ids=ids),
        lambda: None,
    )
    return engine, calls, coroutine


def test_length_and_invalid_continue_without_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, actions, coroutine = execute(tmp_path, monkeypatch, "negative")
    rows = asyncio.run(coroutine)
    assert engine.calls == 3 and len(rows) == 24 and len(actions) == 23
    assert rows[0]["complete"] is False and rows[0]["environment_applied"] is False
    assert rows[1]["complete"] is True and rows[1]["invalid_action"] is True
    assert rows[1]["environment_interaction_trace"][0]["info"]["invalid_action"] is True
    assert [row["model_request_index"] for row in rows] == list(range(24))
    assert len(list(tmp_path.glob("condition-*.raw.json"))) == 3
    assert len(list(tmp_path.glob("request-*.json"))) == 24


def test_bad_mapping_preserves_entire_returned_batch_before_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, actions, coroutine = execute(tmp_path, monkeypatch, "wrong_mapping")
    with pytest.raises(probe.DiagnosticStop, match="ID/order"):
        asyncio.run(coroutine)
    assert engine.calls == 1 and actions == []
    assert len(probe.read(tmp_path / "condition-A.raw.json")["outputs"]) == 8
    assert not list(tmp_path.glob("request-*.json"))
    assert not (tmp_path / "condition-B.submitted.json").exists()


def test_engine_failure_preserves_completed_batch_and_unreturned_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, actions, coroutine = execute(tmp_path, monkeypatch, "engine_failure")
    with pytest.raises(RuntimeError, match="engine failed"):
        asyncio.run(coroutine)
    assert engine.calls == 2 and len(actions) == 8
    assert len(list(tmp_path.glob("request-*.json"))) == 8
    assert len(probe.read(tmp_path / "condition-B.submitted.json")["requests"]) == 8
    assert not (tmp_path / "condition-B.raw.json").exists()
    assert not (tmp_path / "condition-C.submitted.json").exists()


def test_write_new_never_overwrites_negative_evidence(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    probe.write_new(path, {"status": "STOPPED"})
    with pytest.raises(FileExistsError):
        probe.write_new(path, {"status": "OBSERVED"})
    assert probe.read(path) == {"status": "STOPPED"}


def test_decode_failure_keeps_all_raw_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, actions, coroutine = execute(tmp_path, monkeypatch, "normal")

    def fail_decode(*args: Any, **kwargs: Any) -> str:
        raise ValueError("decode failed")

    monkeypatch.setattr(Tokenizer, "decode", fail_decode)
    with pytest.raises(ValueError, match="decode failed"):
        asyncio.run(coroutine)
    assert engine.calls == 1 and actions == []
    raw = probe.read(tmp_path / "condition-A.raw.json")["outputs"]
    assert len(raw) == 8
    assert all(row["outputs"][0]["token_ids"] == [11, 151645] for row in raw)
    assert not list(tmp_path.glob("request-*.json"))

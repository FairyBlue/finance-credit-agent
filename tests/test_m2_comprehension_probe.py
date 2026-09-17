"""CPU-only checks of fixed comprehension identities, strict grading and failure retention."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.agents.rollout_seed import derive_request_seed

PATH = Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_comprehension_probe.py"
SPEC = importlib.util.spec_from_file_location("comprehension_probe_tests", PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def material(window: int = 0) -> dict[str, Any]:
    return {
        "window_index": window,
        "episode_id": probe.H.EPISODES[window],
        "quoted_serial_trading_rules": "return exactly one <tool_call>",
        "quoted_tool_schemas": [{"type": "function", "function": {"name": "buy"}}],
        "feature_state": {
            "session_index": 4,
            "positions": {},
            "cash": 1000000,
            "decision_features": {
                "columns": ["symbol", "lookback_used", *probe.METRICS],
                "rows": [
                    [symbol, 5, 0.123456789 + i, 0.002345678, 0.012345678]
                    for i, symbol in enumerate(probe.FIRST_SYMBOLS)
                ],
            },
        },
        "interface_reference": probe.INTERFACE_REFERENCE,
        "metric_reference": probe.METRIC_REFERENCE,
    }


@pytest.fixture
def expected() -> list[dict[str, Any]]:
    return [probe.expected_answer(q, material()) for q in probe.QUESTION_IDS]


def test_fixed_quiz_grid_has_no_training_rollout_alias() -> None:
    grid = probe.grid_plan()
    assert len(grid) == 64 and [p["request_index"] for p in grid] == list(range(64))
    assert {(p["window_index"], p["question_index"], p["sibling_id"]) for p in grid} == {
        (w, q, s) for w in range(2) for q in range(4) for s in range(8)
    }
    for p in grid:
        identity = {
            "step": 1,
            "sample_index": 4 * p["window_index"] + p["question_index"],
            "rollout_n": p["sibling_id"],
            "validate": False,
        }
        assert p["seed_identity"] == identity
        assert p["seed"] == derive_request_seed(42, identity, 0)
        assert p["identity_scope"] == "one_shot_comprehension_not_training_rollout"


@pytest.mark.parametrize("index", range(4))
def test_declared_facts_are_exactly_scored_without_free_reasoning(
    index: int, expected: list[Any]
) -> None:
    q = probe.QUESTION_IDS[index]
    grade = probe.grade_answer(q, json.dumps(expected[index]), expected[index], complete=True)
    assert grade["all_correct"] and grade["schema_valid"] and grade["json_valid"]
    assert grade["planned_facts"] == [8, 9, 5, 6][index]
    assert grade["correct_facts"] == grade["planned_facts"]


@pytest.mark.parametrize(
    "text",
    [
        '{"task_code": "portfolio_management", "task_code": "portfolio_management"}',
        "```json\n{}\n```",
        "<tool_call>{}</tool_call>",
        "<think></think>{}",
        "{} explanation",
        '{"x": NaN}',
        '{"x": Infinity}',
        '{"x": 1e400}',
        '{"x": -1e400}',
        "[" * 1500 + "0" + "]" * 1500,
        "[" * 40 + "0" + "]" * 40,
    ],
)
def test_malformed_answers_are_retained_as_errors_not_repaired(
    text: str, expected: list[Any]
) -> None:
    grade = probe.grade_answer(probe.QUESTION_IDS[0], text, expected[0], complete=True)
    assert not grade["json_valid"] and not grade["all_correct"]
    assert grade["correct_facts"] == 0 and grade["planned_facts"] == 8
    json.dumps(grade, allow_nan=False)


def test_types_enum_and_extra_keys_preserve_field_errors(expected: list[Any]) -> None:
    answer = {
        **expected[0],
        "tool_calls_per_trading_turn": True,
        "task_code": "manage portfolio",
        "reason": "unrequested explanation",
    }
    grade = probe.grade_answer(
        probe.QUESTION_IDS[0], json.dumps(answer), expected[0], complete=True
    )
    assert grade["json_valid"] and not grade["schema_valid"] and not grade["all_correct"]
    fields = {f["field"]: f for f in grade["facts"]}
    assert not fields["tool_calls_per_trading_turn"]["type_valid"]
    assert fields["task_code"]["type_valid"] and not fields["task_code"]["correct"]
    assert grade["correct_facts"] == 6


def test_fixed_value_order_and_absolute_precision_are_enforced(expected: list[Any]) -> None:
    q, truth = probe.QUESTION_IDS[1], expected[1]
    near = copy.deepcopy(truth)
    near["rows"][0]["price_change"] += 0.9e-8
    assert probe.grade_answer(q, json.dumps(near), truth, complete=True)["all_correct"]
    near["rows"][0]["price_change"] += 0.2e-8
    assert not probe.grade_answer(q, json.dumps(near), truth, complete=True)["all_correct"]
    swapped = {"rows": list(reversed(truth["rows"]))}
    grade = probe.grade_answer(q, json.dumps(swapped), truth, complete=True)
    assert not grade["schema_valid"] and grade["correct_facts"] == 3
    string_number = copy.deepcopy(truth)
    string_number["rows"][0]["volatility"] = "0.012345678"
    assert (
        probe.grade_answer(q, json.dumps(string_number), truth, complete=True)["correct_facts"] == 8
    )


def test_huge_integer_is_a_serializable_field_error(expected: list[Any]) -> None:
    answer = copy.deepcopy(expected[1])
    answer["rows"][0]["volatility"] = 10**1000
    grade = probe.grade_answer(
        probe.QUESTION_IDS[1], json.dumps(answer), expected[1], complete=True
    )
    assert not grade["schema_valid"] and grade["correct_facts"] == 8
    assert not probe.finite_number(10**1000)
    json.dumps(grade, allow_nan=False)


def test_truncated_correct_json_does_not_receive_credit(expected: list[Any]) -> None:
    grade = probe.grade_answer(
        probe.QUESTION_IDS[3], json.dumps(expected[3]), expected[3], complete=False
    )
    assert grade["correct_facts"] == 0 and grade["planned_facts"] == 6
    assert not grade["json_valid"] and not grade["all_correct"]


class Tokenizer:
    def __init__(self) -> None:
        self.answers: dict[int, str] = {}
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        body = json.loads(messages[1]["content"])
        index = 4 * body["quoted_material"]["window_index"] + probe.QUESTION_IDS.index(
            body["question"]["question_id"]
        )
        return [100 + index, 200]

    def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
        text = self.answers[tokens[0] - 10000]
        if len(tokens) > 1 and tokens[-1] == 151645 and not skip_special_tokens:
            return text + "<|im_end|>"
        return text


def test_quiz_has_only_new_system_and_no_callable_tools() -> None:
    tokenizer = Tokenizer()
    source = material()
    original = copy.deepcopy(source)
    prompt = probe.render_prompt(tokenizer, source, probe.question_definitions()[0])
    assert source == original
    assert len(prompt["messages"]) == 2 and prompt["messages"][0]["content"] == probe.SYSTEM
    assert '"quoted_serial_trading_rules"' in prompt["messages"][1]["content"]
    assert prompt["tools_attached"] is False and "tools" not in tokenizer.calls[0]["kwargs"]
    assert tokenizer.calls[0]["kwargs"]["enable_thinking"] is False


class LLM:
    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        fail_batch: int | None = None,
        wrong_mapping: bool = False,
        cap_first: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.batches = 0
        self.next_index = 0
        self.fail_batch = fail_batch
        self.wrong_mapping = wrong_mapping
        self.cap_first = cap_first

    def get_tokenizer(self) -> Tokenizer:
        return self.tokenizer

    def generate(
        self, prompts: list[dict[str, Any]], params: list[Any], **kwargs: Any
    ) -> list[Any]:
        if self.batches == self.fail_batch:
            raise RuntimeError("injected engine failure")
        self.batches += 1
        outputs = []
        for prompt in prompts:
            index = self.next_index
            self.next_index += 1
            capped = self.cap_first and index == 0
            tokens = [10000 + index] * 4096 if capped else [10000 + index, 151645]
            outputs.append(
                SimpleNamespace(
                    request_id=str(index + int(self.wrong_mapping)),
                    prompt_token_ids=prompt["prompt_token_ids"],
                    outputs=[
                        SimpleNamespace(
                            index=0,
                            token_ids=tokens,
                            text=self.tokenizer.answers[index],
                            finish_reason="length" if capped else "stop",
                            stop_reason=None,
                        )
                    ],
                )
            )
        return outputs


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    tokenizer = Tokenizer()
    prompts = []
    answers = []
    for w in range(2):
        for definition in probe.question_definitions():
            prompts.append(probe.render_prompt(tokenizer, material(w), definition))
            answers.append(probe.expected_answer(definition["question_id"], material(w)))
    manifest = {"grid": probe.grid_plan(), "prompts": prompts, "answers": answers}
    for row in manifest["grid"]:
        tokenizer.answers[row["request_index"]] = json.dumps(answers[row["prompt_index"]])
    monkeypatch.setattr(
        probe.H, "effective_params", lambda llm, param: {"sampling_params": vars(param)}
    )
    return manifest, tokenizer


def test_fixed_batches_keep_bad_json_wrong_facts_and_caps_without_retries(
    runtime: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    manifest, tokenizer = runtime
    tokenizer.answers[1] = '{"x":1e400}'
    tokenizer.answers[2] = "```json\n{}\n```"
    tokenizer.answers[3] = json.dumps({**manifest["answers"][0], "shorting_allowed": True})
    state = probe.initial_state(manifest, "a" * 64)
    llm = LLM(tokenizer, cap_first=True)
    probe.execute_batches(llm, SimpleNamespace, manifest, tmp_path, state, lambda: None)
    assert llm.batches == 2 and llm.next_index == 64
    assert state["status"] == "OBSERVED" and state["model_actions_applied"] == 0
    assert len(state["grid"]) == 64 and state["summary"]["planned_facts"] == 448
    assert [r["status"] for r in state["grid"][:4]] == [
        "RESPONSE_INCOMPLETE",
        "JSON_INVALID",
        "JSON_INVALID",
        "GRADED",
    ]
    assert state["summary"]["all_correct"] == 60
    assert state["responses_returned"] == state["responses_graded"] == 64
    record = json.loads((tmp_path / "request-004.json").read_text())
    assert record["response"]["raw_text"].endswith("<|im_end|>")
    assert record["response"]["json_text_after_final_EOS_only"] == tokenizer.answers[4]


def test_mapping_failure_keeps_raw_batch_and_all_unstarted_identities(
    runtime: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    manifest, tokenizer = runtime
    state = probe.initial_state(manifest, "a" * 64)
    with pytest.raises(ValueError, match="mapping drift"):
        probe.execute_batches(
            LLM(tokenizer, wrong_mapping=True),
            SimpleNamespace,
            manifest,
            tmp_path,
            state,
            lambda: None,
        )
    assert (tmp_path / "window-0.raw.json").exists()
    assert state["responses_returned"] == 32 and state["responses_mapped"] == 0
    assert len(state["grid"]) == 64 and sum(r["status"] == "UNSTARTED" for r in state["grid"]) == 32
    assert state["summary"]["planned_facts"] == 448


def test_later_engine_failure_never_discards_earlier_answers(
    runtime: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    manifest, tokenizer = runtime
    state = probe.initial_state(manifest, "a" * 64)
    with pytest.raises(RuntimeError, match="engine failure"):
        probe.execute_batches(
            LLM(tokenizer, fail_batch=1), SimpleNamespace, manifest, tmp_path, state, lambda: None
        )
    assert state["responses_graded"] == 32 and state["requests_submitted"] == 64
    assert sum(r["status"] == "SUBMITTED" for r in state["grid"]) == 32
    assert len(list(tmp_path.glob("request-*.json"))) == 32
    assert state["summary"]["planned_facts"] == 448

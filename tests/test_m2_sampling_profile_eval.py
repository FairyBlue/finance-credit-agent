"""Fixed paired sampling profiles and native incomplete outcomes, without a GPU."""

import asyncio
import copy
import importlib.util
import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "sampling_profile_eval",
    Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_sampling_profile_eval.py",
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def contexts():
    return {
        "short": [],
        "teacher_labels_included": False,
        "full": [
            {
                "id": episode,
                "group": "training_windows" if index < 8 else "evaluation_windows",
                "prefix": [],
                "state_kind": "cash",
                "expected_start_session_index": 0,
                "raw_prompt": [{"role": "user", "content": "Manage the portfolio."}],
                "env_kwargs": {
                    "episode_dates": list(range(21)),
                    "observation_profile": "history_features_v1",
                    "reward_mode": "finance_trajectory",
                    "initial_cash": 1000000,
                    "screening_calibration_sha256": "a" * 64,
                },
            }
            for index, episode in enumerate(entry.TRAIN_IDS + entry.EVAL_IDS)
        ],
    }


class NativeParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.stop = []
        self._all_stop_token_ids = set()
        self.__struct_fields__ = tuple(kwargs) + ("stop", "_all_stop_token_ids")


def test_native_enum_fields_are_explicit_and_json_serializable():
    class OutputKind(Enum):
        CUMULATIVE = 0

    evidence = entry._native_json(OutputKind.CUMULATIVE)
    assert evidence == {"enum": "OutputKind", "name": "CUMULATIVE", "value": 0}
    assert json.loads(json.dumps(evidence)) == evidence


def test_fixed_96_grid_interleaves_profiles_with_identical_pair_seeds():
    rows = entry.grid(contexts())
    assert len(rows) == len({row["id"] for row in rows}) == 96
    assert Counter(row["profile"] for row in rows) == {"current": 48, "qwen_recommended": 48}
    for start in range(0, 96, 32):
        assert len({row["context_id"] for row in rows[start : start + 32]}) == 4
    for current, recommended in zip(rows[::2], rows[1::2], strict=True):
        assert current["profile"] == "current"
        assert recommended["profile"] == "qwen_recommended"
        assert current["identity"] == recommended["identity"]
        assert current["identity"]["validate"] is False
        assert current["sampling"] == recommended["sampling"] == "sample"
        first = entry.sampling_parameters("current", current["identity"], 7)
        second = entry.sampling_parameters("qwen_recommended", recommended["identity"], 7)
        assert first["seed"] == second["seed"]
        assert {key: first[key] for key in entry.PROFILES["current"]} == entry.PROFILES["current"]
        assert {key: second[key] for key in entry.PROFILES["qwen_recommended"]} == entry.PROFILES[
            "qwen_recommended"
        ]
        for params in (first, second):
            assert params["max_tokens"] == 4096 and params["n"] == 1
            assert params["repetition_penalty"] == 1
            assert params["presence_penalty"] == params["frequency_penalty"] == 0


@pytest.mark.parametrize(
    "damage", ["order", "prefix", "group", "horizon", "teacher", "observation"]
)
def test_unregistered_context_changes_rejected(damage):
    value = contexts()
    if damage == "order":
        value["full"].reverse()
    elif damage == "prefix":
        value["full"][0]["prefix"] = ["hold"]
    elif damage == "group":
        value["full"][0]["group"] = "evaluation_windows"
    elif damage == "horizon":
        value["full"][0]["env_kwargs"]["episode_dates"].pop()
    elif damage == "teacher":
        value["teacher_labels_included"] = True
    else:
        value["full"][0]["env_kwargs"]["observation_profile"] = "other"
    with pytest.raises(ValueError):
        entry.grid(value)


def test_actual_native_fields_recorded_and_parameter_drift_rejected():
    row = entry.grid(contexts())[0]
    params = entry.sampling_parameters("current", row["identity"], 0)
    _, evidence, representation = entry.native_parameters(NativeParams, params)
    assert {key: evidence[key] for key in params} == params
    assert evidence["stop"] == evidence["_all_stop_token_ids"] == []
    assert representation

    class Changed(NativeParams):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.temperature = 0.0

    with pytest.raises(RuntimeError, match="altered"):
        entry.native_parameters(Changed, params)
    with pytest.raises(ValueError):
        entry.sampling_parameters("another_profile", row["identity"], 0)


@dataclass
class Trace:
    step_index: int
    action: str
    invalid_action: bool
    done: bool
    result_session_index: int


@pytest.mark.parametrize("failure", [None, "REQUEST_CAP", "RESPONSE_INCOMPLETE", "PROMPT_CAP"])
def test_runtime_keeps_all_96_lanes_and_never_fabricates_incomplete_outcomes(
    tmp_path,
    monkeypatch,
    failure,
):
    from src.agents import coldstart_data
    from src.env import trading

    class FakeEnv:
        created = 0

        def __init__(self, **kwargs):
            self.serial = FakeEnv.created
            FakeEnv.created += 1
            self.current_session_index = 0
            self.episode_complete = False
            self.ledger = SimpleNamespace(fills=[])
            self.credit_step_trace = []
            self.interaction_trace = []

        def observation(self):
            return {
                "session_index": self.current_session_index,
                "positions": {},
                "cash": 1000000,
                "portfolio_equity": 1000000,
            }

        def reset(self, **kwargs):
            return self.observation()

        async def step(self, action):
            invalid = action.text == "invalid"
            if not invalid:
                self.current_session_index += 1
            self.episode_complete = self.current_session_index == 20
            self.credit_step_trace.append(
                Trace(
                    len(self.credit_step_trace),
                    "invalid" if invalid else "hold",
                    invalid,
                    self.episode_complete,
                    self.current_session_index,
                )
            )
            return (
                self.observation(),
                0,
                self.episode_complete,
                {
                    "equity": 1000000,
                    "invalid_action": invalid,
                },
            )

        def screening_terminal_metrics(self):
            return {"session_count": 21, "terminal_score": -1, "calibration_sha256": "a" * 64}

    class Tokenizer:
        def encode(self, prompt, **kwargs):
            serial, session = [int(value) for value in prompt.split(":")]
            if failure == "PROMPT_CAP" and serial == 0:
                return [0] * 8193
            return [serial, session]

        def decode(self, tokens, **kwargs):
            return "invalid" if tokens[0] == 1 else "hold"

    class LLM:
        def generate(self, prompts, params, **kwargs):
            result = []
            for prompt in prompts:
                first_lane = prompt["prompt_token_ids"][0] == 0
                tokens = [1] if failure == "REQUEST_CAP" and first_lane else [2]
                reason = "stop"
                if failure == "RESPONSE_INCOMPLETE" and first_lane:
                    tokens, reason = [2] * 4096, "length"
                result.append(
                    SimpleNamespace(
                        prompt_token_ids=prompt["prompt_token_ids"],
                        outputs=[SimpleNamespace(token_ids=tokens, finish_reason=reason)],
                    )
                )
            return result

    monkeypatch.setattr(trading, "TradingEnv", FakeEnv)
    monkeypatch.setattr(coldstart_data, "parse_state", lambda obs: obs)
    monkeypatch.setattr(
        coldstart_data,
        "render_prompt",
        lambda tok, env, obs, **kw: f"{env.serial}:{env.current_session_index}",
    )
    result = asyncio.run(entry.evaluate(LLM(), Tokenizer(), NativeParams, contexts(), tmp_path))
    assert result["status"] == "OBSERVED" and FakeEnv.created == 96
    assert json.loads((tmp_path / "state.json").read_text()) == result
    assert json.loads((tmp_path / "result.json").read_text()) == result
    assert sum(row["status"] == "COMPLETE" for row in result["grid"]) == 96 - bool(failure)
    first = result["grid"][0]
    if failure:
        assert first["status"] == "INCOMPLETE" and first["failure"] == failure
        assert not any(
            key in first for key in ("takeover_net_return", "terminal_metrics", "final_state")
        )
        assert (
            first["requests"]
            == {"REQUEST_CAP": 128, "RESPONSE_INCOMPLETE": 1, "PROMPT_CAP": 0}[failure]
        )
        assert len(first["steps_including_prefix"]) == (128 if failure == "REQUEST_CAP" else 0)
    for record in map(json.loads, (tmp_path / "requests.jsonl").read_text().splitlines()):
        assert record["profile"] in entry.PROFILES
        for key, value in record["sampling_params"].items():
            assert record["native_sampling_params"][key] == value


def test_terminal_outcome_rejects_illegal_or_nonterminal_transition():
    row = entry.grid(contexts())[0]
    row["takeover_equity"] = 1000000
    env = SimpleNamespace(
        episode_complete=False,
        current_session_index=19,
        screening_terminal_metrics=lambda: {
            "session_count": 21,
            "terminal_score": 1,
            "calibration_sha256": "a" * 64,
        },
    )
    before = copy.deepcopy(row)
    with pytest.raises(RuntimeError, match="terminal state"):
        entry.finish_trajectory(row, env, {"session_index": 19}, {"equity": 1000000})
    assert row == before


def test_engine_resources_match_existing_standalone_defaults():
    engine = entry.engine_settings(Path("/fixed/model"))
    assert engine["dtype"] == "bfloat16"
    assert engine["max_num_seqs"] == 32 and engine["max_model_len"] == 12288
    assert engine["gpu_memory_utilization"] == 0.5
    assert engine["seed"] == 42
    assert engine["compilation_config"]["cudagraph_capture_sizes"] == [1, 2, 4, 8, 16, 32]

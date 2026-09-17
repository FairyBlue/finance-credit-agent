"""Real-ledger initialization and complete runner contracts, without GPU."""

import asyncio
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "practice_test", ROOT / "scripts/diagnostics/m2_train_practice.py"
)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
S.H.add_import_paths()
C = S.E.module_from_path("practice_condition_test", S.CONDITION)


@pytest.fixture(scope="module")
def rows():
    return S.H.read_train_rows()


@pytest.mark.parametrize("start", ["cash", "invested"])
def test_same_initial_portfolio_across_prompt_conditions(rows, start):
    states = []
    for kind in ["baseline", "guide", "examples"]:
        cond = kind + "_" + start
        env = C.create_env(rows, 0, cond, 4)
        _, rec = asyncio.run(C.reset_with_prefix(env))
        S.validate_prefix(env, rec, {"condition": cond})
        states.append(rec["state_at_model_start"])
    assert states[0] == states[1] == states[2]
    assert set(states[0]["positions"]) == (set(C.SEED_SYMBOLS) if start == "invested" else set())
    assert rows[0]["prompt"][0]["role"] == "user"


class Tokenizer:
    def __init__(self):
        self.ids = {}

    def apply_chat_template(self, messages, **kwargs):
        text = json.dumps(messages, sort_keys=True)
        if text not in self.ids:
            self.ids[text] = len(self.ids) + 1000
        return [self.ids[text]]

    def decode(self, ids, skip_special_tokens):
        return "".join(chr(i) for i in ids if i != 151645)


@pytest.mark.parametrize("start", ["cash", "invested"])
def test_actual_executor_full_episode_buy_sell_and_persist(rows, start, tmp_path, monkeypatch):
    from agent_r1.env.base import Action

    condition = "guide_" + start
    plan = next(
        copy.deepcopy(p)
        for p in S.grid_plan()
        if p["condition"] == condition and p["window_index"] == 0 and p["rollout_id"] == 0
    )
    monkeypatch.setattr(S.E, "grid_plan", lambda: [plan])
    monkeypatch.setattr(S.H, "effective_params", lambda llm, p: {"fixture": True})
    tokenizer = Tokenizer()
    env = C.create_env(rows, 0, condition, 4)
    obs, _ = asyncio.run(C.reset_with_prefix(env))
    prompt = C.render_prompt(tokenizer, obs, env.tool_schemas)
    manifest = {"models": {"screening": {"initial_prompts": {condition: [prompt]}}}}

    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class LLM:
        def __init__(self):
            self.count = 0

        def get_tokenizer(self):
            return tokenizer

        def generate(self, inputs, params, use_tqdm):
            assert len(inputs) == 1
            if self.count == 0:
                tool = (
                    {"name": "buy", "arguments": {"symbol": "MSFT", "target_weight": 0.1}}
                    if start == "cash"
                    else {"name": "sell", "arguments": {"symbol": "AAPL", "target_weight": 0.1}}
                )
            elif self.count == 1:
                tool = (
                    {"name": "sell", "arguments": {"symbol": "MSFT", "target_weight": 0.0}}
                    if start == "cash"
                    else {"name": "buy", "arguments": {"symbol": "MSFT", "target_weight": 0.1}}
                )
            else:
                tool = {"name": "hold", "arguments": {}}
            text = "<tool_call>" + json.dumps(tool) + "</tool_call>"
            response = SimpleNamespace(
                index=0,
                token_ids=[ord(c) for c in text] + [151645],
                finish_reason="stop",
                stop_reason=None,
                text=text,
            )
            output = SimpleNamespace(
                request_id=str(self.count),
                prompt_token_ids=inputs[0]["prompt_token_ids"],
                outputs=[response],
            )
            self.count += 1
            return [output]

    state = S.E.initial_state("screening", "fixture")
    result = asyncio.run(
        S.E.execute_grid(
            LLM(),
            Params,
            manifest,
            "screening",
            tmp_path,
            state,
            C,
            lambda wi, c, start_session: C.create_env(rows, wi, c, start_session),
            lambda text, ids: Action(text=text, token_ids=ids),
            lambda: None,
        )
    )
    row = result["grid"][0]
    assert row["status"] == "COMPLETE" and row["requests_submitted"] == 16
    assert (
        row["fill_count"] == 2 and row["action_counts"]["buy"] == row["action_counts"]["sell"] == 1
    )
    assert row["invalid_count"] == 0 and len(row["fills"]) == (5 if start == "invested" else 2)
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["complete_count"] == 1 and saved["actions_applied"] == 16

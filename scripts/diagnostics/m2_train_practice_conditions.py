"""Train-only portfolio decision guidance and in-context examples."""

# ruff: noqa: E501 - preserve readable paragraph literals in prompt text
from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "practice_features", ROOT / "scripts/diagnostics/m2_decision_conditions.py"
)
assert spec and spec.loader
LEGACY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LEGACY)
render_prompt = LEGACY.render_prompt
CONDITIONS = tuple(
    f"{p}_{s}" for p in ("baseline", "guide", "examples") for s in ("cash", "invested")
)
SEED_SYMBOLS = ("AAPL", "AMZN", "JPM")
GUIDE = """Portfolio decision guide:
Your job is to manage the actual portfolio, choosing one executable action each turn.
The current market, existing positions and decision_features are usable evidence. You do not need a guarantee about future prices to make a measured allocation decision. Missing news or a longer history alone does not make the supplied five-close table unusable.
price_change is total close-to-close return over the displayed window; mean_return and volatility describe daily simple returns, not forecasts. All returns and weights are decimals: 0.01 means 1%.
Compare available stocks using recent return direction, consistency and volatility, then review your current holdings and cash. Positive recent returns can support a small exploratory allocation, while high volatility reduces the size justified by that signal. These are imperfect signals, not a requirement to buy.
When a candidate is attractive relative to alternatives and cash, consider a modest buy. A typical initial position is 5-15% of portfolio value; avoid concentrating more than 30% in one stock. When a held stock weakens, has an oversized weight, or a better use of capital is available, consider reducing or exiting it. Do not sell a stock you do not own.
Use hold when keeping the current portfolio is the best decision after considering costs; holding cash is also allowed when signals are weak or conflicting. Use abstain for an actual inability to evaluate the visible state, not as the default response to ordinary market uncertainty.
Avoid repeated buying and selling for tiny daily changes. Transaction costs matter. target_weight is the desired FINAL fraction of the whole portfolio, not a change in weight or a fraction of cash: buy must increase current weight, sell must reduce it. Review the new portfolio after every result. Output exactly one allowed tool call, without a separate explanation. Do not fabricate tool results."""
EXAMPLES = """Illustrative fictional decisions, not recommendations for the actual stocks. X and Y below are placeholders; choose only a symbol in the real visible universe. Do not copy an example unless the actual situation supports that decision.
Example 1: All cash. X has five-close price_change +0.04, mean_return +0.0098, volatility 0.006; alternatives are weaker. A small exposure is reasonable despite uncertainty: <tool_call>{"name":"buy","arguments":{"symbol":"X","target_weight":0.1}}</tool_call>
Example 2: X is already 20% of equity. Its recent direction has weakened (price_change -0.05, mean_return -0.012, volatility 0.025). Reducing exposure is reasonable: <tool_call>{"name":"sell","arguments":{"symbol":"X","target_weight":0.1}}</tool_call>
Example 3: X is already 10%, remains moderately positive, and no alternative is clearly better. Small changes do not justify another fee: <tool_call>{"name":"hold","arguments":{}}</tool_call>
Example 4: All cash, all candidates show weak or conflicting signals with high volatility. Preserve cash: <tool_call>{"name":"hold","arguments":{}}</tool_call>
These examples illustrate action selection and sizing, not optimal investment rules. Apply the current data; do not follow an alternating buy/sell pattern."""


def addition(condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError("Unknown practice condition")
    kind = condition.split("_")[0]
    return "" if kind == "baseline" else GUIDE + ("\n\n" + EXAMPLES if kind == "examples" else "")


def create_env(
    rows: list[dict[str, Any]], index: int, condition: str, start_session: int = 4
) -> Any:
    extra = addition(condition)
    copied = deepcopy(rows)
    if extra:
        copied[index]["prompt"].insert(0, {"role": "system", "content": extra})
    env = LEGACY.create_env(copied, index, "features", start_session)
    env.practice_condition = condition
    return env


async def reset_with_prefix(env: Any):
    from agent_r1.env.base import Action

    obs = env.reset()
    initial = deepcopy(obs.messages)
    invested = env.practice_condition.endswith("_invested")
    for i in range(4):
        tool = (
            {"name": "buy", "arguments": {"symbol": SEED_SYMBOLS[i], "target_weight": 0.2}}
            if invested and i < 3
            else {"name": "hold", "arguments": {}}
        )
        obs, _, done, info = await env.step(
            Action(text="<tool_call>" + json.dumps(tool) + "</tool_call>")
        )
        if done or info.get("invalid_action"):
            raise RuntimeError("Initial portfolio construction failed")
    state = json.loads(env._state_text())
    return obs, {
        "kind": "scripted_initial_portfolio_not_model_actions",
        "is_model_request": False,
        "model_requests": 0,
        "start_session": 4,
        "prefix_action_count": 4,
        "condition": env.practice_condition,
        "initial_observation": initial,
        "credit_step_trace": [asdict(t) for t in env.credit_step_trace],
        "interaction_trace": env.interaction_trace,
        "cash": env.ledger.cash,
        "positions": dict(env.ledger.positions),
        "fill_count": len(env.ledger.fills),
        "equity_at_model_start": state["portfolio_equity"],
        "state_at_model_start": state,
        "feature_evidence": env.feature_evidence,
    }

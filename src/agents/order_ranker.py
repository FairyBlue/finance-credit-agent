"""Visible-state order menus and a frozen, CPU-only learned scoring interface.

This module does not create an environment, inspect a future bar, or update an
LLM. The same candidate constructor is used by the menu control and the scorer.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

PROFILE = "visible_order_ranker_mlp_v1"
SYMBOLS = (
    "GS",
    "MSFT",
    "HD",
    "V",
    "SHW",
    "CAT",
    "MCD",
    "UNH",
    "AXP",
    "AMGN",
    "TRV",
    "CRM",
    "JPM",
    "IBM",
    "HON",
    "BA",
    "AMZN",
    "AAPL",
    "PG",
    "JNJ",
)
FAMILIES = ("buy", "sell", "hold", "abstain")
TARGET_GRID = (0.0, 0.1, 0.2)
DIRECTION_MARGIN = 1e-6
FEATURE_NAMES = (
    "session_fraction",
    "remaining_fraction",
    "cash_fraction",
    "position_fraction",
    "total_position_weight",
    "max_position_weight",
    "weight_concentration",
    "current_weight",
    "target_weight",
    "weight_delta",
    "absolute_weight_delta",
    "cash_after_delta_approx",
    "symbol_intraday_return",
    "symbol_intraday_range",
    "symbol_high_to_close",
    "symbol_low_to_close",
    "symbol_log_volume",
    "symbol_history_price_change",
    "symbol_history_mean_return",
    "symbol_history_volatility",
    "symbol_history_fraction",
    "symbol_history_present",
    "market_intraday_mean",
    "market_intraday_std",
    "market_history_mean_return",
    "market_history_return_std",
    "symbol_relative_history_return",
    "delta_times_history_return",
    "absolute_delta_times_volatility",
    "is_original_proposal",
    *(f"symbol_{symbol}" for symbol in SYMBOLS),
    *(f"family_{family}" for family in FAMILIES),
)


def _finite(value, default=0.0):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = float(value)
        if math.isfinite(value):
            return value
    return default


def stable_seed(seed: int, key: str) -> int:
    payload = json.dumps([PROFILE, seed, key], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def make_order(family: str, symbol=None, target=None, *, response_text=None):
    arguments = (
        {}
        if family in {"hold", "abstain"}
        else {"symbol": symbol.strip().upper(), "target_weight": float(target)}
    )
    call = dict(name=family, arguments=arguments)
    action_key = json.dumps(call, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return dict(
        action_key=action_key,
        family=family,
        symbol=arguments.get("symbol"),
        target_weight=arguments.get("target_weight"),
        response_text=response_text or "<tool_call>" + action_key + "</tool_call>",
    )


def parse_proposal(text: str, visible: dict) -> tuple[dict | None, str]:
    """Conservative checks using only visible weights and strict tool arguments."""
    calls = re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL)
    if len(calls) != 1:
        return None, "not_one_tool_call"
    try:
        raw = json.loads(calls[0])
        if not isinstance(raw, dict) or set(raw) != {"name", "arguments"}:
            return None, "call_fields"
        family, arguments = raw["name"], raw["arguments"]
        if family not in FAMILIES or not isinstance(arguments, dict):
            return None, "unsupported_tool"
        if family in {"hold", "abstain"}:
            if arguments:
                return None, "unexpected_arguments"
            return make_order(family, response_text=text), "visible_legal"
        if set(arguments) != {"symbol", "target_weight"}:
            return None, "target_fields"
        symbol, target = arguments["symbol"], arguments["target_weight"]
        if not isinstance(symbol, str):
            return None, "symbol_type"
        symbol = symbol.strip().upper()
        if symbol not in visible["market"] or symbol not in SYMBOLS:
            return None, "symbol_outside_universe"
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            return None, "target_type"
        target = float(target)
        if not math.isfinite(target) or not 0 <= target <= 1:
            return None, "target_range"
        current = _finite(visible.get("positions", {}).get(symbol, {}).get("weight"))
        if family == "buy" and target <= current + DIRECTION_MARGIN:
            return None, "visible_buy_direction"
        if family == "sell" and target >= current - DIRECTION_MARGIN:
            return None, "visible_sell_direction"
        return make_order(family, symbol, target, response_text=text), "visible_legal"
    except (ValueError, TypeError, KeyError, AttributeError):
        return None, "unparseable_call"


def candidate_orders(visible: dict, proposed_response_text: str, *, seed: int, key: str):
    """Original + hold + seeded family diversity, never alphabetical preference."""
    symbols = [symbol for symbol in SYMBOLS if symbol in visible["market"]]
    if len(symbols) != len(SYMBOLS):
        raise ValueError("visible frozen twenty-symbol universe required")
    proposal, reason = parse_proposal(proposed_response_text, visible)
    rng = random.Random(stable_seed(seed, key))
    rng.shuffle(symbols)
    menu = []
    positions = visible.get("positions", {})
    for symbol in symbols:
        current = _finite(positions.get(symbol, {}).get("weight"))
        targets = list(TARGET_GRID)
        rng.shuffle(targets)
        for target in targets:
            if target > current + DIRECTION_MARGIN:
                menu.append(make_order("buy", symbol, target))
            elif target < current - DIRECTION_MARGIN:
                menu.append(make_order("sell", symbol, target))
    result = []

    def add(row):
        if row is not None and row["action_key"] not in {r["action_key"] for r in result}:
            result.append(dict(row))

    add(proposal)
    add(make_order("hold"))
    families = ["buy", "sell"]
    rng.shuffle(families)
    for family in families:
        add(
            next(
                (
                    row
                    for row in menu
                    if row["family"] == family
                    and row["action_key"] not in {r["action_key"] for r in result}
                ),
                None,
            )
        )
        if len(result) == 4:
            break
    for row in menu:
        if len(result) == 4:
            break
        add(row)
    if len(result) != 4:
        raise ValueError("four distinct visible orders unavailable")
    for index, row in enumerate(result):
        row.update(candidate_id=index, is_proposal=proposal is not None and index == 0)
        row["features"] = order_features(visible, row)
    return result, proposal is not None, reason


def _history_rows(visible):
    table = visible.get("decision_features", {})
    columns = table.get("columns", [])
    return {
        row[columns.index("symbol")]: dict(zip(columns, row, strict=True))
        for row in table.get("rows", [])
        if "symbol" in columns and len(row) == len(columns)
    }


def order_features(visible: dict, candidate: dict) -> list[float]:
    positions, market = visible.get("positions", {}), visible["market"]
    weights = [_finite(row.get("weight")) for row in positions.values()]
    cash_fraction = _finite(visible.get("cash")) / max(
        _finite(visible.get("portfolio_equity")), 1e-8
    )
    history = _history_rows(visible)
    intraday = [
        _finite(row.get("close")) / max(_finite(row.get("open")), 1e-8) - 1
        for row in market.values()
    ]
    history_returns = [_finite(row.get("mean_return")) for row in history.values()]
    symbol = candidate.get("symbol")
    own = market.get(symbol, {})
    close = max(_finite(own.get("close")), 1e-8)
    current = _finite(positions.get(symbol, {}).get("weight"))
    target = _finite(candidate.get("target_weight"), current)
    delta = target - current if symbol is not None else 0.0
    h = history.get(symbol, {})
    own_return = _finite(h.get("mean_return"))
    own_volatility = _finite(h.get("volatility"))
    market_return = float(np.mean(history_returns)) if history_returns else 0.0
    values = [
        _finite(visible.get("session_index")) / 20,
        _finite(visible.get("sessions_remaining")) / 20,
        cash_fraction,
        len(positions) / 20,
        sum(weights),
        max(weights, default=0),
        sum(weight * weight for weight in weights),
        current,
        target,
        delta,
        abs(delta),
        cash_fraction - delta,
        close / max(_finite(own.get("open")), 1e-8) - 1 if own else 0,
        (_finite(own.get("high")) - _finite(own.get("low"))) / close if own else 0,
        _finite(own.get("high")) / close - 1 if own else 0,
        _finite(own.get("low")) / close - 1 if own else 0,
        math.log1p(max(_finite(own.get("volume")), 0)),
        _finite(h.get("price_change")),
        own_return,
        own_volatility,
        _finite(h.get("lookback_used")) / 5,
        float(h.get("mean_return") is not None),
        float(np.mean(intraday)),
        float(np.std(intraday)),
        market_return,
        float(np.std(history_returns)) if history_returns else 0,
        own_return - market_return if symbol is not None else 0,
        delta * own_return,
        abs(delta) * own_volatility,
        float(candidate.get("is_proposal", False)),
        *(float(symbol == item) for item in SYMBOLS),
        *(float(candidate["family"] == family) for family in FAMILIES),
    ]
    if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite or inconsistent visible order features")
    return values


@lru_cache(maxsize=4)
def load_ranker(path: str):
    location = Path(path).resolve()
    if location.is_dir():
        location = location / "ranker.json"
    payload = json.loads(location.read_text())
    if payload["profile"] != PROFILE or payload["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("ranker feature contract mismatch")
    if payload.get("training_split") != "train":
        raise ValueError("only a Train-fitted ranker is allowed")
    return payload


def predict_features(features, payload):
    array = np.asarray(features, dtype=np.float64)
    mean = np.asarray(payload["feature_mean"])
    std = np.asarray(payload["feature_std"])
    array = np.clip((array - mean) / std, -10, 10)
    for index, layer in enumerate(payload["layers"]):
        array = array @ np.asarray(layer["weight"]).T + np.asarray(layer["bias"])
        if index < len(payload["layers"]) - 1:
            array = np.maximum(array, 0)
    result = np.asarray(array).reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite learned candidate scores")
    return result.tolist()


def select_order(
    visible: dict,
    proposed_response_text: str,
    *,
    mode: str = "menu",
    seed: int = 43,
    key: str,
    ranker_path=None,
) -> dict:
    candidates, proposal_legal, reason = candidate_orders(
        visible, proposed_response_text, seed=seed, key=key
    )
    scores = None
    if mode == "menu":
        chosen = (
            0
            if proposal_legal
            else random.Random(stable_seed(seed, key + "/fallback")).randrange(4)
        )
    elif mode == "ranker":
        if ranker_path is None:
            raise ValueError("learned order selection requires ranker_path")
        payload = load_ranker(str(Path(ranker_path).resolve()))
        scores = predict_features([row["features"] for row in candidates], payload)
        chosen = max(range(len(candidates)), key=lambda index: (scores[index], -index))
    else:
        raise ValueError("decision mode must be menu or ranker")
    selected = candidates[chosen]
    return dict(
        response_text=selected["response_text"],
        economic_order_key=selected["action_key"],
        selected_candidate_id=chosen,
        proposal_legal=proposal_legal,
        proposal_legality_reason=reason,
        selection_mode=mode,
        candidates=candidates,
        scores=scores,
        uses_future_information=False,
        external_action_selection=True,
    )

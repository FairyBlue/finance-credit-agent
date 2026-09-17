"""Neutral historical feature display, sharing exact arithmetic with compute.

The caller supplies only point-in-time visible close records in adapter order.
Rendering this table neither invokes policy tools nor changes trading state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

FEATURE_LOOKBACK = 5
FEATURE_METRICS = ("price_change", "mean_return", "volatility")
FEATURE_COLUMNS = ("symbol", "lookback_used", *FEATURE_METRICS)


def compute_close_metrics(closes: Sequence[float]) -> dict[str, float]:
    """The production compute formulas, with no annualization or rounding."""
    if len(closes) < 2:
        raise ValueError("close metrics require at least two visible close prices")
    array = np.asarray(closes, dtype=np.float64)
    returns = array[1:] / array[:-1] - 1.0
    return {
        "price_change": float(array[-1] / array[0] - 1.0),
        "mean_return": float(returns.mean()),
        "volatility": float(returns.std(ddof=0)),
    }


def render_decision_features(
    visible_closes: Mapping[str, Sequence[float]], *, as_of: str,
) -> dict[str, Any]:
    """Display every symbol, explicitly marking insufficient history as null.

    Inputs may include pre-episode records but must already be filtered by the
    production adapter's availability cutoff. Mapping order is universe order.
    """
    rows: list[list[Any]] = []
    for symbol, history in visible_closes.items():
        selected = history[-FEATURE_LOOKBACK:]
        values = compute_close_metrics(selected) if len(selected) >= 2 else None
        rows.append([
            symbol, len(selected),
            *[values[metric] if values is not None else None for metric in FEATURE_METRICS],
        ])
    return {
        "source": "TradingEnv._compute",
        "as_of": as_of,
        "lookback_requested": FEATURE_LOOKBACK,
        "columns": list(FEATURE_COLUMNS),
        "rows": rows,
        "definitions": {
            "history": (
                "Last up to 5 market close records available at as_of, in availability order. "
                "Available pre-episode records are included. lookback_used counts close records; "
                "adjacent returns number lookback_used minus 1."
            ),
            "price_change": "Last close / first close - 1.",
            "mean_return": "Arithmetic mean of adjacent close-to-close simple returns.",
            "volatility": "Population standard deviation (ddof=0) of those returns.",
            "units": (
                "Returns and volatility are fractions, not percentages, and not annualized. "
                "These describe visible history, not future returns."
            ),
            "missing": "Fewer than 2 visible close records: all 3 metrics are null, not zero.",
        },
    }

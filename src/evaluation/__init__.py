"""Shared financial/agent metrics, baseline replay, and standard plots."""

from .baselines import (
    ReplayResult,
    all_cash_targets,
    buy_and_hold_equal_weight_targets,
    replay_target_weights,
)
from .metrics import (
    AgentMetrics,
    FinancialMetrics,
    agent_metrics_by_regime,
    compute_agent_metrics,
    compute_financial_metrics,
    metrics_by_regime,
)
from .plots import plot_equity_vs_buy_and_hold, plot_reward_vs_step, plot_tool_call_success

__all__ = [
    "AgentMetrics",
    "FinancialMetrics",
    "ReplayResult",
    "agent_metrics_by_regime",
    "all_cash_targets",
    "buy_and_hold_equal_weight_targets",
    "compute_agent_metrics",
    "compute_financial_metrics",
    "metrics_by_regime",
    "plot_equity_vs_buy_and_hold",
    "plot_reward_vs_step",
    "plot_tool_call_success",
    "replay_target_weights",
]

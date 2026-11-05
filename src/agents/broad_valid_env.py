"""Explicit history-Valid adapter for the authorized broad experiment only.

The original TradingEnv screening entry point remains Train-only. This adapter
validates the separate Valid boundary and then reuses its ledger and frozen
benchmark-scaled terminal scorer; no action executes during base construction.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from src.agents.broad_prepare import ROOT, TEMPLATE, load_config
from src.env.trading import TradingEnv


@lru_cache(maxsize=1)
def frozen_settings():
    import pyarrow.parquet as pq

    row = pq.read_table(TEMPLATE).slice(0, 1).to_pylist()[0]
    return json.loads(row["env_kwargs"])


class BroadValidTradingEnv(TradingEnv):
    """Same trading/scoring implementation with an explicit, bounded Valid scope."""

    def __init__(self, *, experiment_config, episode_id, **kwargs):
        config = load_config(experiment_config)
        if episode_id not in config["valid_episode_ids"]:
            raise ValueError("broad Valid adapter requires a declared Valid episode")
        dates = [str(day)[:10] for day in kwargs.get("episode_dates", [])]
        if len(dates) != 21 or not all("2024-11-01" <= day <= "2025-02-28" for day in dates):
            raise ValueError("broad Valid dates must remain in historical Valid, never Final OOS")
        if kwargs.get("screening") is not True or kwargs.get("reward_mode") != "finance_trajectory":
            raise ValueError("explicit frozen terminal-Y Valid profile required")
        if kwargs.get("data_start") != "2023-01-03" or kwargs.get("data_end") != "2025-02-28":
            raise ValueError("broad Valid storage must be the bounded historical snapshot")
        history = (ROOT / "data/history/storage").resolve()
        if any(
            Path(kwargs.get(key, "")).resolve() != history
            for key in ("manifest_root", "storage_root")
        ):
            raise ValueError("broad Valid may only load the history snapshot")
        if (
            kwargs.get("input_sessions_per_symbol") != 541
            or kwargs.get("input_file_count") != 10820
        ):
            raise ValueError("broad Valid must use the frozen541-session history source")
        expected = frozen_settings()
        for key in (
            "finance_scorer_id",
            "finance_volatility_floor",
            "screening_calibration_sha256",
            "risk_lambda",
            "turnover_limit",
            "position_limit",
            "observation_profile",
            "drawdown_coefficient",
            "turnover_coefficient",
            "initial_cash",
            "commission_bps",
            "slippage_bps",
            "symbols",
        ):
            if kwargs.get(key) != expected[key]:
                raise ValueError(f"broad Valid frozen Train scoring/observation differs: {key}")
        floor = kwargs["finance_volatility_floor"]
        if not isinstance(floor, (int, float)) or not math.isfinite(floor) or floor <= 0:
            raise ValueError("positive frozen volatility floor required")
        base_kwargs = dict(kwargs)
        # Base initialization performs calendar, storage, ledger and tool checks.
        # Select its generic terminal mode transiently so the Train-only screening
        # entry point is never widened. No reset/step occurs before restoring Y.
        base_kwargs.update(screening=False, reward_mode="outcome_only")
        super().__init__(**base_kwargs)
        self._screening = True
        self._reward_mode = "finance_trajectory"
        self._broad_valid_profile = config["profile"]


def make_valid_environment(kwargs, *, experiment_config, episode_id):
    return BroadValidTradingEnv(
        experiment_config=experiment_config, episode_id=episode_id, **kwargs
    )

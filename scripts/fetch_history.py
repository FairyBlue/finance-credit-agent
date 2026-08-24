#!/usr/bin/env python3
"""Freeze the approved train+valid daily history into the offline storage layout."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.config import load_experiment_config  # noqa: E402
from src.data.history import (  # noqa: E402
    HistoryWindow,
    StooqDailyBarProvider,
    YFinanceDailyBarProvider,
    materialize_daily_history,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--provider", choices=("yfinance", "stooq"))
    parser.add_argument("--output")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    config = load_experiment_config(config_path)
    data = config["data"]
    history = data["sources"]["history"]
    windows = data["windows"]
    provider_name = args.provider or str(history["provider"])
    provider = (
        YFinanceDailyBarProvider(
            corporate_actions_through=date.fromisoformat(
                str(history["corporate_actions_through"])
            )
        )
        if provider_name == "yfinance"
        else StooqDailyBarProvider()
    )
    output = Path(args.output) if args.output else Path(str(history["storage_root"]))
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    manifest = materialize_daily_history(
        provider=provider,
        symbols=data["universe"]["symbols"],
        window=HistoryWindow(
            start=date.fromisoformat(str(windows["train"]["start"])),
            end=date.fromisoformat(str(windows["valid"]["end"])),
        ),
        output_root=output,
    )
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

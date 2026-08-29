#!/usr/bin/env python3
"""Run the G1 real-tokenizer prompt-budget gate without starting training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.config import load_experiment_config  # noqa: E402
from src.agents.context_budget import verify_smoke_context_budget  # noqa: E402
from src.agents.launcher import verify_gpu_runtime_and_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_1p7b.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_experiment_config(config_path)
    if config.get("run", {}).get("smoke") is not True:
        raise RuntimeError("standalone context-budget verification only accepts smoke configs")

    runtime = verify_gpu_runtime_and_model(config)
    audit = verify_smoke_context_budget(config, runtime=runtime, repo_root=PROJECT_ROOT)
    print(json.dumps({"status": "CONTEXT_BUDGET_PASSED", **audit}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

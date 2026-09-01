#!/usr/bin/env python3
"""Run the preregistered history-only T2 v2 leakage probe."""

from __future__ import annotations

import argparse
import sys
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.config import load_experiment_config  # noqa: E402
from src.evaluation.t2_v2 import (  # noqa: E402
    run_configured_t2_v2_probe,
    t2_v2_artifact_bytes,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    config_path = Path(arguments.config).expanduser()
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    configured_run = run_configured_t2_v2_probe(
        load_experiment_config(config_path), repo_root=REPO_ROOT
    )
    output_path = Path(arguments.output).expanduser()
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = t2_v2_artifact_bytes(configured_run)
    output_path.write_bytes(raw_bytes)
    print(f"T2 v2 artifact: {output_path}")
    print(f"T2 v2 artifact SHA-256: {sha256(raw_bytes).hexdigest()}")
    print("T2_V2_PASSED" if configured_run.passed else "T2_V2_FAILED")
    return 0 if configured_run.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

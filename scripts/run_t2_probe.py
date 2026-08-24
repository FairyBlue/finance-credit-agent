#!/usr/bin/env python3
"""Run the frozen T2 shifted-label probe against the real two-source timeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.config import load_experiment_config  # noqa: E402
from src.agents.launcher import (  # noqa: E402
    verify_configured_overlap,
    verify_configured_source_manifests,
)
from src.evaluation import run_configured_temporal_leakage_probe  # noqa: E402


def _resolved(path: object) -> Path:
    candidate = Path(str(path))
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_experiment_config(_resolved(args.config))
    verify_configured_source_manifests(config, repo_root=PROJECT_ROOT)
    verify_configured_overlap(config, repo_root=PROJECT_ROOT)
    configured = run_configured_temporal_leakage_probe(
        config=config,
        repo_root=PROJECT_ROOT,
    )
    report = {
        "data_version": config["data"]["version"],
        "feature_names": configured.dataset.feature_names,
        "first_feature_session": str(configured.dataset.sessions[0]),
        "last_feature_session": str(configured.dataset.sessions[-1]),
        "criteria": configured.criteria,
        "passed": configured.passed,
        "result": asdict(configured.result),
    }
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = _resolved(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if configured.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

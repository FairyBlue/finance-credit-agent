#!/usr/bin/env python3
"""CLI for building audited Agent-R1 episode parquet files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.episodes import prepare_episode_datasets  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    outputs = prepare_episode_datasets(
        config_path=PROJECT_ROOT / args.config,
        manifest_path=args.manifest,
        repo_root=PROJECT_ROOT,
    )
    for split, output in outputs.items():
        print(f"{split}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

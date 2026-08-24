#!/usr/bin/env python3
"""Freeze reviewed episode benchmark NAV through the shared accounting engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.episodes import freeze_episode_benchmarks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = freeze_episode_benchmarks(
        config_path=PROJECT_ROOT / args.config,
        draft_manifest_path=args.draft,
        output_path=args.output,
        repo_root=PROJECT_ROOT,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

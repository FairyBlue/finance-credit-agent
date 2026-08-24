#!/usr/bin/env python3
"""Disabled legacy StockBench OOS T2 entrypoint retained for audit history."""

from __future__ import annotations

import argparse

LEGACY_T2_DISABLED_MESSAGE = (
    "legacy StockBench OOS T2 is permanently disabled after its single approved 2026-08-24 run; "
    "implement T2 v2 on history train/valid instead"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output")
    parser.parse_args()
    parser.error(LEGACY_T2_DISABLED_MESSAGE)


if __name__ == "__main__":
    raise SystemExit(main())

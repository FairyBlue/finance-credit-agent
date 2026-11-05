#!/usr/bin/env python3
"""Prepare frozen Train panel and inspect historical learning signals on CPU."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.financial_learning_data import main  # noqa: E402

if __name__ == "__main__":
    main()

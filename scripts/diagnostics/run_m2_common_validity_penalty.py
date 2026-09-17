#!/usr/bin/env python3
"""CLI wrapper for the bounded common-rejection-penalty training controller."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.common_validity_penalty import main  # noqa: E402

if __name__ == "__main__":
    main()

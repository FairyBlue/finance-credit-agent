#!/usr/bin/env python3
"""Build or verify isolated M2 screening datasets and the new CPU reward calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.screening import prepare_screening_datasets, validate_prepared_screening  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--manifest", default="configs/episode_manifest.reviewed.yaml")
    parser.add_argument("--output-dir", default="artifacts/m2_screening")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify all source/parameter/row identities without fitting again",
    )
    args = parser.parse_args()
    result = (
        validate_prepared_screening(args.output_dir, PROJECT_ROOT)
        if args.check
        else prepare_screening_datasets(
            config_path=args.config,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            repo_root=PROJECT_ROOT,
        )
    )
    print(
        json.dumps(
            {
                "output_dir": str((PROJECT_ROOT / args.output_dir).resolve()),
                "manifest_sha256": result["hashes"]["payload_sha256"],
                "calibration": result["calibration"],
                "datasets": result["datasets"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

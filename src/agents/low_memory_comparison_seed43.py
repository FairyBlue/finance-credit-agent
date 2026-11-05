"""The approved seed43 replication of the fixed four-round comparison."""

from dataclasses import replace

from src.agents import low_memory_pilot as runner
from src.agents.low_memory_comparison import PROFILE as ORIGINAL_PROFILE

PROFILE = replace(
    ORIGINAL_PROFILE,
    name="common_sft_low_memory_comparison_seed43_v1",
    plan="docs/2026-10-05_M2_SEED43_REPLICATION_PLAN.md",
    seed=43,
)
DEFAULT = runner.REPO_ROOT / "runs/m2-sft-four-round-comparison-seed43-20261005"


def main():
    runner.main(profile=PROFILE, default_directory=DEFAULT)


if __name__ == "__main__":
    main()

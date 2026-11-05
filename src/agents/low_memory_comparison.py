"""Fixed SFT0 + four-round M1/M2 comparison using the accepted low-memory path.

This experiment owns its own sealed identity and budget. The shared controller
receives an immutable profile; the original two-round pilot remains unchanged.
"""

from src.agents import low_memory_pilot as runner

PROFILE = runner.ExperimentProfile(
    name="common_sft_low_memory_comparison_v1",
    plan="docs/2026-10-04_M2_SFT_FOUR_ROUND_COMPARISON_PLAN.md",
    train_numbers=(1, 65, 9, 73, 17, 149, 25, 157),
    evaluation_steps=(2, 4),
    rounds=4,
    include_sft=True,
    retire_completed_moments=True,
)
NAME = PROFILE.name
DEFAULT = runner.REPO_ROOT / "runs/m2-sft-four-round-comparison-20261004"


def main():
    runner.main(profile=PROFILE, default_directory=DEFAULT)


if __name__ == "__main__":
    main()

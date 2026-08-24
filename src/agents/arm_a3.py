"""A3: StepPO with finance-native trajectory signal and pluggable step credit."""

from .launcher import REPO_ROOT, cli

if __name__ == "__main__":
    raise SystemExit(
        cli(expected_arm="A3", default_config=str(REPO_ROOT / "configs" / "arm_a3.yaml"))
    )

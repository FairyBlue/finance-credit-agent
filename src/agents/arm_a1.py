"""A1: outcome-only GRPO."""

from .launcher import REPO_ROOT, cli

if __name__ == "__main__":
    raise SystemExit(
        cli(expected_arm="A1", default_config=str(REPO_ROOT / "configs" / "arm_a1.yaml"))
    )

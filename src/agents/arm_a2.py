"""A2: Agent-R1 StepPO (`adv_estimator=gae`) with generic step rewards."""

from .launcher import REPO_ROOT, cli

if __name__ == "__main__":
    raise SystemExit(
        cli(expected_arm="A2", default_config=str(REPO_ROOT / "configs" / "arm_a2.yaml"))
    )

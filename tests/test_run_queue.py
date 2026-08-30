"""G1 queue remains disabled until stage-aware process-group handling is reviewed."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "scripts/run_queue.sh"


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("configs/smoke_1p7b.yaml",),
        ("--formal", "configs/arm_a1.yaml"),
    ],
)
def test_queue_is_direct_only_and_never_starts_uv(
    tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "uv-started"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        f"#!/usr/bin/env bash\ntouch {marker!s}\nexit 0\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"

    completed = subprocess.run(
        [str(QUEUE), *arguments],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "disabled fail-closed" in completed.stderr
    assert not marker.exists()

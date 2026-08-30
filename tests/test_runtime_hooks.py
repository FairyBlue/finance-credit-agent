"""Interpreter-startup tests for required G1 runtime instrumentation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

RUNTIME_HOOKS = Path(__file__).resolve().parents[1] / "runtime_hooks"


def _write_fake_hooks(tmp_path: Path, *, update_body: str, compat_body: str) -> None:
    agents = tmp_path / "src" / "agents"
    agents.mkdir(parents=True)
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (agents / "__init__.py").write_text("", encoding="utf-8")
    (agents / "update_probe.py").write_text(update_body, encoding="utf-8")
    (agents / "vllm_compat.py").write_text(compat_body, encoding="utf-8")


def _start_python(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(RUNTIME_HOOKS), str(tmp_path)))
    environment.pop("FINANCE_UPDATE_PROBE_DIR", None)
    environment.pop("FINANCE_VLLM_DRAIN_COMPAT", None)
    return subprocess.run(
        [sys.executable, "-c", 'print("BODY_RAN")'],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_sitecustomize_installs_both_hooks_before_user_code(tmp_path: Path) -> None:
    _write_fake_hooks(
        tmp_path,
        update_body='def install_from_environment():\n    print("UPDATE_INSTALLED")\n',
        compat_body='def install_from_environment():\n    print("COMPAT_INSTALLED")\n',
    )

    completed = _start_python(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["UPDATE_INSTALLED", "COMPAT_INSTALLED", "BODY_RAN"]


@pytest.mark.parametrize("failing_hook", ["update", "compat"])
def test_sitecustomize_hook_failure_stops_interpreter(
    tmp_path: Path,
    failing_hook: str,
) -> None:
    success = 'def install_from_environment():\n    print("HOOK_INSTALLED")\n'
    failure = (
        'def install_from_environment():\n    raise RuntimeError("instrumentation exploded")\n'
    )
    _write_fake_hooks(
        tmp_path,
        update_body=failure if failing_hook == "update" else success,
        compat_body=failure if failing_hook == "compat" else success,
    )

    completed = _start_python(tmp_path)

    assert completed.returncode != 0
    assert "BODY_RAN" not in completed.stdout
    assert "finance runtime hook installation failed: instrumentation exploded" in completed.stderr

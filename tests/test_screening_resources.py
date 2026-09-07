"""CPU-only tests for observational resource phase wrappers."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from src.agents import screening_resources
from src.agents.screening_resources import RESOURCE_PHASE_METHODS, install_resource_phase_hooks


def _trainer_class() -> type[Any]:
    class Base:
        calls: list[tuple[Any, Any]]
        failure: BaseException | None
        callback: Any
        result: Any

        def _compute_old_log_prob(self, batch: Any, *, marker: Any = None) -> Any:
            self.calls.append((batch, marker))
            if self.failure is not None:
                raise self.failure
            if self.callback is not None:
                self.callback()
            return self.result

        _compute_ref_log_prob = _compute_old_log_prob
        _compute_values = _compute_old_log_prob
        _update_critic = _compute_old_log_prob
        _update_actor = _compute_old_log_prob

    class Trainer(Base):
        def __init__(self) -> None:
            self.calls: list[tuple[Any, Any]] = []
            self.failure: BaseException | None = None
            self.callback: Any = None
            self.result = object()
            self.global_steps = 7

    return Trainer


@pytest.fixture
def cgroup(tmp_path: Path) -> Path:
    directory = tmp_path / "cgroup"
    directory.mkdir()
    (directory / "memory.current").write_text("1024\n")
    (directory / "memory.events").write_text("low 0\nhigh 12\nmax 0\noom 0\noom_kill 0\n")
    (directory / "memory.stat").write_text(
        "anon 100\nfile 200\nfile_dirty 50\npgscan_direct 40\n"
        "pgsteal_direct 30\nunknown_field 999\n"
    )
    return directory


def _records(diagnostics: Path) -> list[dict[str, Any]]:
    directory = diagnostics.parent / "resource_phases"
    return [
        json.loads(line)
        for path in sorted(directory.glob("phase-*.jsonl"))
        for line in path.read_text().splitlines()
    ]


@pytest.mark.parametrize("method", RESOURCE_PHASE_METHODS)
def test_each_phase_preserves_original_calls_and_logs_boundary_snapshots(
    tmp_path: Path,
    cgroup: Path,
    method: str,
) -> None:
    trainer_class = _trainer_class()
    inherited = getattr(trainer_class.__base__, method)
    diagnostics = tmp_path / "run/artifacts/sibling_diagnostics"
    assert install_resource_phase_hooks(
        trainer_class, diagnostics_dir=diagnostics, cgroup_root=cgroup, enabled=True
    )
    trainer = trainer_class()
    batch, marker = object(), object()
    trainer.callback = lambda: (cgroup / "memory.current").write_text("2048\n")
    assert getattr(trainer, method)(batch, marker=marker) is trainer.result
    assert trainer.calls == [(batch, marker)]
    assert getattr(trainer_class.__base__, method) is inherited
    assert inspect.signature(getattr(trainer_class, method)) == inspect.signature(inherited)
    rows = _records(diagnostics)
    assert [row["event"] for row in rows] == ["enter", "success"]
    assert [row["memory.current"] for row in rows] == [1024, 2048]
    assert all(row["phase"] == method.removeprefix("_") and row["policy_step"] == 7 for row in rows)
    assert all(row["memory.events"]["high"] == 12 for row in rows)
    assert rows[0]["memory.stat"] == {
        "anon": 100,
        "file": 200,
        "file_dirty": 50,
        "pgscan_direct": 40,
        "pgsteal_direct": 30,
    }
    assert rows[0]["time"] <= rows[1]["time"]
    assert rows[0]["monotonic_ns"] <= rows[1]["monotonic_ns"]


@pytest.mark.parametrize("error", [RuntimeError("original failure"), KeyboardInterrupt()])
def test_original_exception_identity_is_preserved(
    tmp_path: Path,
    cgroup: Path,
    error: BaseException,
) -> None:
    trainer_class = _trainer_class()
    diagnostics = tmp_path / "diagnostics"
    install_resource_phase_hooks(
        trainer_class, diagnostics_dir=diagnostics, cgroup_root=cgroup, enabled=True
    )
    trainer = trainer_class()
    trainer.failure = error
    with pytest.raises(type(error)) as raised:
        trainer._update_actor(object())
    assert raised.value is error
    rows = _records(diagnostics)
    assert [row["event"] for row in rows] == ["enter", "error"]
    assert rows[-1]["error_type"] == type(error).__name__
    assert len(trainer.calls) == 1


def test_write_failures_cannot_change_return_or_replace_training_exception(
    tmp_path: Path,
    cgroup: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("filesystem full")

    monkeypatch.setattr(screening_resources, "_append_record", fail)
    trainer_class = _trainer_class()
    install_resource_phase_hooks(
        trainer_class, diagnostics_dir=tmp_path / "diagnostics", cgroup_root=cgroup, enabled=True
    )
    trainer = trainer_class()
    assert trainer._compute_values(object()) is trainer.result
    error = ValueError("original error")
    trainer.failure = error
    with pytest.raises(ValueError) as raised:
        trainer._update_critic(object())
    assert raised.value is error
    assert len(trainer.calls) == 2


def test_missing_cgroup_fields_remain_nonfatal_and_are_explicit(
    tmp_path: Path,
    cgroup: Path,
) -> None:
    (cgroup / "memory.events").unlink()
    (cgroup / "memory.stat").write_text("malformed\n")
    trainer_class = _trainer_class()
    diagnostics = tmp_path / "diagnostics"
    install_resource_phase_hooks(
        trainer_class, diagnostics_dir=diagnostics, cgroup_root=cgroup, enabled=True
    )
    trainer = trainer_class()
    assert trainer._compute_values(object()) is trainer.result
    rows = _records(diagnostics)
    assert rows[0]["memory.current"] == 1024
    assert rows[0]["memory.events"] == rows[0]["memory.stat"] == {}
    assert rows[0]["read_errors"] == {
        "memory.events": "FileNotFoundError",
        "memory.stat": "ValueError",
    }


def test_disabled_install_is_noop_and_repeat_install_does_not_nest(
    tmp_path: Path,
    cgroup: Path,
) -> None:
    trainer_class = _trainer_class()
    originals = {name: getattr(trainer_class, name) for name in RESOURCE_PHASE_METHODS}
    diagnostics = tmp_path / "diagnostics"
    assert not install_resource_phase_hooks(trainer_class, diagnostics_dir=diagnostics)
    assert all(getattr(trainer_class, name) is method for name, method in originals.items())
    assert not (tmp_path / "resource_phases").exists()
    assert install_resource_phase_hooks(
        trainer_class, diagnostics_dir=diagnostics, cgroup_root=cgroup, enabled=True
    )
    installed = {name: getattr(trainer_class, name) for name in RESOURCE_PHASE_METHODS}
    assert not install_resource_phase_hooks(
        trainer_class, diagnostics_dir=diagnostics, cgroup_root=cgroup, enabled=True
    )
    assert all(getattr(trainer_class, name) is method for name, method in installed.items())
    trainer_class()._compute_old_log_prob(object())
    assert len(_records(diagnostics)) == 2


def test_different_upstream_class_is_left_unmodified(tmp_path: Path) -> None:
    class Unsupported:
        def _compute_values(self, batch: Any) -> Any:
            return batch

    original = Unsupported._compute_values
    assert not install_resource_phase_hooks(Unsupported, diagnostics_dir=tmp_path, enabled=True)
    assert Unsupported._compute_values is original

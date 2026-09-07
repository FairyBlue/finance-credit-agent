"""Best-effort, opt-in driver phase resource logging for M2 screening.

Only the Agent-R1 subclass is wrapped. The wrappers do not inspect batches,
change return values, synchronize devices, or touch model/RNG state.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, cast

RESOURCE_PHASE_METHODS = (
    "_compute_old_log_prob",
    "_compute_ref_log_prob",
    "_compute_values",
    "_update_critic",
    "_update_actor",
)
MEMORY_STAT_FIELDS = frozenset(
    {
        "anon",
        "file",
        "kernel",
        "kernel_stack",
        "pagetables",
        "sock",
        "shmem",
        "file_mapped",
        "file_dirty",
        "file_writeback",
        "swapcached",
        "inactive_anon",
        "active_anon",
        "inactive_file",
        "active_file",
        "pgscan",
        "pgsteal",
        "pgscan_direct",
        "pgsteal_direct",
        "workingset_refault_file",
    }
)
_CLASS_MARKER = "_finance_screening_resource_hooks"
_METHOD_MARKER = "_finance_screening_resource_phase"


def _read_counters(path: Path) -> dict[str, int]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, value = line.split()
        result[name] = int(value)
    return result


def _append_record(directory: Path, record: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"phase-{record['pid']}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def record_resource_phase(
    *,
    diagnostics_dir: Path,
    phase: str,
    event: str,
    trainer: Any,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    error: BaseException | None = None,
) -> None:
    """Emit one phase boundary; unavailable counters and diagnostic I/O are non-fatal."""
    try:
        step = getattr(trainer, "global_steps", None)
        record: dict[str, Any] = {
            "schema_version": 1,
            "phase": phase,
            "event": event,
            "time": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "policy_step": step if type(step) is int else None,
            "cgroup_root": str(cgroup_root),
            "memory.current": None,
            "memory.events": {},
            "memory.stat": {},
            "read_errors": {},
        }
        for filename in ("memory.current", "memory.events", "memory.stat"):
            try:
                if filename == "memory.current":
                    record[filename] = int((cgroup_root / filename).read_text().strip())
                else:
                    counters = _read_counters(cgroup_root / filename)
                    record[filename] = (
                        {key: value for key, value in counters.items() if key in MEMORY_STAT_FIELDS}
                        if filename == "memory.stat"
                        else counters
                    )
            except (OSError, ValueError) as failure:
                record["read_errors"][filename] = type(failure).__name__
        if error is not None:
            record["error_type"] = type(error).__name__
        _append_record(Path(diagnostics_dir).parent / "resource_phases", record)
    except Exception:
        # This observational log is deliberately not a training/evidence gate.
        # A full filesystem or unavailable cgroup must not replace a model result/error.
        return


def _wrap_phase(
    original: Callable[..., Any],
    *,
    phase: str,
    diagnostics_dir: Path,
    cgroup_root: Path,
) -> Callable[..., Any]:
    @wraps(original)
    def wrapped(trainer: Any, *args: Any, **kwargs: Any) -> Any:
        record_resource_phase(
            diagnostics_dir=diagnostics_dir,
            phase=phase,
            event="enter",
            trainer=trainer,
            cgroup_root=cgroup_root,
        )
        try:
            result = original(trainer, *args, **kwargs)
        except BaseException as error:
            record_resource_phase(
                diagnostics_dir=diagnostics_dir,
                phase=phase,
                event="error",
                trainer=trainer,
                cgroup_root=cgroup_root,
                error=error,
            )
            raise
        record_resource_phase(
            diagnostics_dir=diagnostics_dir,
            phase=phase,
            event="success",
            trainer=trainer,
            cgroup_root=cgroup_root,
        )
        return result

    setattr(wrapped, _METHOD_MARKER, True)
    return wrapped


def install_resource_phase_hooks(
    trainer_class: type[Any],
    *,
    diagnostics_dir: Path,
    enabled: bool = False,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> bool:
    """Install only for an explicit screening caller, without nesting repeated wrappers."""
    if not enabled or trainer_class.__dict__.get(_CLASS_MARKER, False):
        return False
    # Resolve every method before installing; a different upstream class remains unchanged.
    originals = {name: getattr(trainer_class, name, None) for name in RESOURCE_PHASE_METHODS}
    if any(not callable(method) for method in originals.values()):
        return False
    installed = False
    for name, original in originals.items():
        if getattr(original, _METHOD_MARKER, False):
            continue
        setattr(
            trainer_class,
            name,
            _wrap_phase(
                cast(Callable[..., Any], original),
                phase=name.removeprefix("_"),
                diagnostics_dir=Path(diagnostics_dir),
                cgroup_root=Path(cgroup_root),
            ),
        )
        installed = True
    setattr(trainer_class, _CLASS_MARKER, True)
    return installed

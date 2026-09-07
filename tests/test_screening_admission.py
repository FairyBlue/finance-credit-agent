"""Host capacity admission uses charged bytes and preserves the hard-limit reserve."""

from pathlib import Path

import pytest

from src.agents.screening_exchange import ExchangeError
from src.agents.screening_placement import admit_host_transfer

GIB = 1024**3


def accounting(root: Path, *, current: int, maximum: int | str, high: int) -> None:
    for name, value in (
        ("memory.current", current),
        ("memory.max", maximum),
        ("memory.high", high),
    ):
        (root / name).write_text(str(value))


def test_observed_v6_transfer_is_admitted_without_crediting_page_cache(tmp_path: Path):
    # Real v6 rejection: 55.41 GiB charged, next moment ~0.376 GiB, 64 GiB hard cap.
    accounting(tmp_path, current=59494535168, maximum=64 * GIB, high=61847527424)
    admit_host_transfer(tmp_path, 403723264, reserve=2 * GIB)
    # The remaining 1.504 GiB also fits under the hard limit, with the same reserve.
    admit_host_transfer(tmp_path, 1614893056, reserve=2 * GIB)


@pytest.mark.parametrize("high", (1, 61847527424, 64 * GIB))
def test_soft_threshold_does_not_change_hard_capacity_admission(tmp_path: Path, high: int):
    accounting(tmp_path, current=59 * GIB, maximum=64 * GIB, high=high)
    admit_host_transfer(tmp_path, GIB, reserve=2 * GIB)


def test_reserve_boundary_and_no_speculative_cache_reclaim(tmp_path: Path):
    accounting(tmp_path, current=61 * GIB, maximum=64 * GIB, high=61847527424)
    admit_host_transfer(tmp_path, GIB, reserve=2 * GIB)
    (tmp_path / "memory.current").write_text(str(61 * GIB + 1))
    # Even an implausibly large clean-cache counter must not increase the budget.
    (tmp_path / "memory.stat").write_text(f"file {128 * GIB}\nfile_dirty 0\n")
    with pytest.raises(MemoryError, match="hard_limit=.*transfer=.*reserve="):
        admit_host_transfer(tmp_path, GIB, reserve=2 * GIB)


@pytest.mark.parametrize(("current", "maximum"), ((1, "max"), (-1, 64 * GIB), (1, 0)))
def test_missing_finite_capacity_or_invalid_accounting_fails_closed(
    tmp_path: Path,
    current: int,
    maximum: int | str,
):
    accounting(tmp_path, current=current, maximum=maximum, high=61847527424)
    with pytest.raises(ExchangeError):
        admit_host_transfer(tmp_path, GIB, reserve=2 * GIB)

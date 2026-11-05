"""Filesystem-accounting waits use real retirement receipts and a fake clock."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents import actor_export
from src.agents import low_memory_pilot as runner
from src.agents import order_recovery_learning as entry

GIB = 1024**3


class Clock:
    def __init__(self):
        self.now = 0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def completed_previous_seed(base, monkeypatch):
    """Create and retire tiny genuine raw payloads through the production API."""
    monkeypatch.setattr(runner, "check_contract", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_export, "validate_export_artifact", lambda *args, **kwargs: None)
    child, profile = base / "seed42", entry.PROFILES[42]
    summaries = []
    for index, spec in enumerate(runner.cells(profile)):
        cell = child / spec.cell_id
        identity = str(index) * 64
        summary = dict(status="COMPLETE", returncode=0, actual_rounds=spec.rounds,
                       cell=asdict(spec), seed=42, execution_identity_sha256=identity)
        runner.atomic_json(cell / "summary.json", summary)
        if not spec.zero:
            scratch = cell / "artifacts/placement/moment_scratch" / str(999999990 + index)
            for role in ("actor", "critic"):
                directory = scratch / role
                generation = directory / "generation-000001"
                generation.mkdir(parents=True)
                chunks = []
                if role == "critic":
                    payload = generation / "p000000-exp_avg-000000.raw"
                    payload.write_bytes(b"real CPU moments")
                    chunks.append(dict(file=payload.name, bytes=payload.stat().st_size,
                                       sha256=runner.restart.digest(payload)))
                owner = dict(identity=identity, owner=f"owner-{index}-{role}", role=role)
                runner.atomic_json(directory / "owner.json", owner)
                runner.atomic_json(directory / "manifest.json", dict(
                    schema_version=1, **owner, generation=generation.name,
                    payload_bytes=sum(chunk["bytes"] for chunk in chunks),
                    parameters=[dict(moments={"exp_avg": chunks})],
                ))
            summary["moment_pruning"] = runner.retire_completed_moments(
                child, spec, summary, profile=profile,
            )
            runner.atomic_json(cell / "summary.json", summary)
        summaries.append(summary)
    runner.atomic_json(child / "state.json", dict(
        status="COMPLETE", profile=profile.name, seed=42, cells=summaries,
    ))
    return child


@pytest.fixture
def retired(tmp_path, monkeypatch):
    completed_previous_seed(tmp_path, monkeypatch)
    old = tmp_path / "old-run/moment.raw"
    old.parent.mkdir()
    old.write_bytes(b"historical scratch must remain")
    return tmp_path


def wait(base, values, clock):
    samples = iter(values)
    last = values[-1]
    return entry.wait_seed_disk_reclaim(
        base, 42, 43, clock=clock, sleep=clock.sleep,
        disk_usage=lambda path: SimpleNamespace(free=next(samples, last) * GIB),
    )


def test_immediate_actual_capacity_needs_no_sleep(retired):
    clock = Clock()
    result = wait(retired, [56], clock)
    assert result["status"] == "READY" and result["required_free_bytes"] == 56 * GIB
    assert result["samples"] == [dict(elapsed_seconds=0, free_bytes=56 * GIB)]
    assert clock.sleeps == []
    assert (retired / "old-run/moment.raw").read_bytes() == b"historical scratch must remain"


def test_delayed_capacity_requires_owned_receipts_and_real_df(retired):
    clock = Clock()
    result = wait(retired, [53, 53, 56], clock)
    assert result["status"] == "READY" and clock.sleeps == [5, 5]
    assert [row["free_bytes"] for row in result["samples"]] == [53 * GIB, 53 * GIB, 56 * GIB]
    evidence = result["verified_retirement_evidence_sha256"]
    assert len([path for path in evidence if path.endswith("moment_pruning_receipt.json")]) == 2
    assert len([path for path in evidence if path.endswith("owner.json")]) == 4
    assert all(Path(path).is_relative_to(retired / "seed42") for path in evidence)
    assert runner.restart.read(retired / "seed43_disk_reclaim_receipt.json") == result
    assert (retired / "old-run/moment.raw").exists()


def test_timeout_stops_at_300_seconds_without_synthetic_free_credit(retired):
    clock = Clock()
    with pytest.raises(RuntimeError, match="unchanged next-seed gate"):
        wait(retired, [53], clock)
    result = runner.restart.read(retired / "seed43_disk_reclaim_receipt.json")
    assert result["status"] == "TIMEOUT" and clock.now == 300
    assert clock.sleeps == [5] * 60
    assert len(result["samples"]) == 61
    assert all(row["free_bytes"] == 53 * GIB for row in result["samples"])
    assert (retired / "old-run/moment.raw").exists()


def test_retirement_validation_time_counts_against_the_wait_deadline(retired, monkeypatch):
    clock, original = Clock(), entry._verified_retirements

    def verify(base, seed):
        result = original(base, seed)
        clock.now = 299
        return result

    monkeypatch.setattr(entry, "_verified_retirements", verify)
    result = wait(retired, [53, 56], clock)
    assert result["status"] == "READY" and clock.sleeps == [1]
    assert result["samples"][-1]["elapsed_seconds"] == 300


@pytest.mark.parametrize("mutation", [
    "missing", "foreign", "identity", "manifest", "owner", "raw", "counts", "state",
])
def test_missing_foreign_or_tampered_retirement_never_authorizes_wait(retired, mutation):
    cell = retired / "seed42/m1"
    receipt_path = cell / "moment_pruning_receipt.json"
    receipt = runner.restart.read(receipt_path)
    first = receipt["kept_manifests"][0]
    if mutation == "missing":
        receipt_path.unlink()
    elif mutation == "foreign":
        summary_path = cell / "summary.json"
        summary = runner.restart.read(summary_path)
        summary["moment_pruning"]["receipt"] = str(retired / "old-run/receipt.json")
        runner.atomic_json(summary_path, summary)
        state_path = retired / "seed42/state.json"
        state = runner.restart.read(state_path)
        state["cells"][1] = summary
        runner.atomic_json(state_path, state)
    elif mutation == "identity":
        receipt["execution_identity_sha256"] = "f" * 64
        runner.atomic_json(receipt_path, receipt)
    elif mutation == "manifest":
        (cell / first["path"]).write_text("changed manifest bytes")
    elif mutation == "owner":
        path = (cell / first["path"]).with_name("owner.json")
        owner = runner.restart.read(path)
        owner["owner"] = "foreign owner"
        runner.atomic_json(path, owner)
    elif mutation == "raw":
        (cell / receipt["files"][0]["path"]).write_bytes(b"undeleted moments")
    elif mutation == "counts":
        receipt["deleted_bytes"] += 1
        runner.atomic_json(receipt_path, receipt)
    else:
        path = retired / "seed42/state.json"
        state = runner.restart.read(path)
        state["seed"] = 43
        runner.atomic_json(path, state)
    clock = Clock()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        wait(retired, [53, 56], clock)
    assert clock.sleeps == []
    assert runner.restart.read(retired / "seed43_disk_reclaim_receipt.json")["status"] == "FAILED"


def test_retirement_evidence_changed_during_wait_is_rejected(retired):
    clock = Clock()

    def sleep(seconds):
        clock.sleep(seconds)
        path = next((retired / "seed42/m1").rglob("owner.json"))
        path.write_text("tampered after verification")

    with pytest.raises(RuntimeError, match="changed while waiting"):
        entry.wait_seed_disk_reclaim(
            retired, 42, 43, clock=clock, sleep=sleep,
            disk_usage=lambda path: SimpleNamespace(free=53 * GIB),
        )
    assert clock.now == 5


def test_wait_is_not_available_for_first_seed_or_unrelated_batch(retired):
    for previous, following in ((None, 42), (43, 42), (42, 99)):
        with pytest.raises(RuntimeError, match="this batch's seed boundary"):
            entry.wait_seed_disk_reclaim(retired, previous, following)
    assert not (retired / "seed43_disk_reclaim_receipt.json").exists()


@pytest.mark.parametrize("succeeds", [True, False])
def test_driver_waits_before_first_next_seed_execution_and_never_retries(
    tmp_path, monkeypatch, succeeds,
):
    calls, clock = [], Clock()
    monkeypatch.setattr(entry, "check_contract", lambda path: None)
    monkeypatch.setattr(runner, "memory_coordinator_config", lambda path: {"holder": "frozen"})
    monkeypatch.setattr(entry.time, "monotonic", clock)
    monkeypatch.setattr(entry.time, "sleep", clock.sleep)
    free = iter([100 * GIB, 53 * GIB])
    monkeypatch.setattr(entry.shutil, "disk_usage", lambda path:
                        SimpleNamespace(free=next(free, (56 if succeeds else 53) * GIB)))

    def execute(child, *, profile):
        calls.append(profile.seed)
        if profile.seed == 42:
            completed_previous_seed(tmp_path, monkeypatch)
        else:
            assert clock.now == 5
            runner.atomic_json(child / "state.json", {"status": "COMPLETE"})

    monkeypatch.setattr(runner, "run", execute)
    monkeypatch.setattr(entry, "export_model_spec", lambda base: base / "models.json")
    monkeypatch.setattr(entry, "_readout", lambda *args, **kwargs:
                        {"incomplete_evaluations": 0, "financial_status": "COMPLETE"})
    if succeeds:
        entry.run(tmp_path)
        assert calls == [42, 43]
    else:
        with pytest.raises(RuntimeError, match="unchanged next-seed gate"):
            entry.run(tmp_path)
        assert calls == [42] and clock.now == 300
        assert runner.restart.read(tmp_path / "state.json")["status"] == "FAILED"
        with pytest.raises(RuntimeError, match="no retry or resume"):
            entry.run(tmp_path)
        assert calls == [42]

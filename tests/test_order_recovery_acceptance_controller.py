"""CPU fail-stop tests for the real holder controller's acceptance worker boundary."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from src.agents import order_recovery_acceptance_controller as controller
from src.agents import order_recovery_probe as probe


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    directory, holder = tmp_path / "acceptance", tmp_path / "holder"
    directory.mkdir()
    source = tmp_path / "frozen-source.py"
    source.write_text("stable runtime\n")
    keys = ("sft", "seed42_m1", "seed42_m2", "seed43_m1", "seed43_m2")
    models = {}
    for key in keys:
        model = tmp_path / key
        model.mkdir()
        config = model / "config.json"
        config.write_text('{}\n')
        models[key] = dict(model_path=str(model), files={"config.json": {
            "bytes": config.stat().st_size, "sha256": probe.sha(config),
        }})
    manifest = {"mode": "all", "models": models, "source_sha256": probe.sha(source)}
    probe.save(directory / "manifest.json", manifest)
    digest = probe.sha(directory / "manifest.json")
    probe.save(directory / "state.json", {"status": "PREPARED", "manifest_sha256": digest})
    events, workers = [], []

    def verify(actual, expected, **kwargs):
        events.append("verify")
        if probe.sha(actual / "manifest.json") != expected:
            raise ValueError("Acceptance manifest digest differs")
        value = probe.read(actual / "manifest.json")
        if probe.sha(source) != value["source_sha256"]:
            raise ValueError("Frozen acceptance source changed")
        return value

    monkeypatch.setattr(controller.acceptance, "verify", verify)
    monkeypatch.setattr(controller.acceptance, "keys_for", lambda mode: keys)

    def holder_config(*args, **kwargs):
        events.append("holder_config")
        return {"fixture": True}

    monkeypatch.setattr(controller, "config_for_holder", holder_config)
    monkeypatch.setattr(controller, "freeze_config", lambda *args: events.append("freeze"))
    monkeypatch.setattr(controller, "release_for_inference",
                        lambda config: events.append("release") or {"released": True})
    monkeypatch.setattr(controller, "controller_idle", lambda config: events.append("idle"))

    def run_monitored(command, **kwargs):
        key = command[command.index("--model-key") + 1]
        workers.append(key)
        events.append("worker")
        assert command[command.index("--expected-manifest-sha256") + 1] == digest
        observed = dict(status="OBSERVED", model_key=key, profile=controller.acceptance.PROFILE)
        (directory / "workers" / key).mkdir(parents=True)
        for name in ("state.json", "result.json"):
            probe.save(directory / "workers" / key / name, observed)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(controller, "run_monitored", run_monitored)

    def readout(*args):
        observed = {key: "OBSERVED" if key in workers else "UNOBSERVED" for key in keys}
        return dict(status="OBSERVED" if len(workers) == len(keys) else "INCOMPLETE",
                    short_workers=observed, full_workers=observed,
                    behavioral_gate="PASS", acceptance="PENDING_AUDIT")

    monkeypatch.setattr(controller.acceptance, "readout", readout)
    return SimpleNamespace(directory=directory, holder=holder, source=source, keys=keys,
                           events=events, workers=workers, run_monitored=run_monitored,
                           readout=readout)


@pytest.mark.parametrize("corruption", ["digest", "source", "prepared", "existing_state"])
def test_invalid_prepared_source_or_existing_state_never_releases_gpu(prepared, corruption):
    case = prepared
    if corruption == "digest":
        (case.directory / "manifest.json").write_text('{}\n')
    elif corruption == "source":
        case.source.write_text("changed runtime\n")
    elif corruption == "prepared":
        probe.save(case.directory / "state.json",
                   {"status": "OBSERVED", "manifest_sha256": "a" * 64})
    else:
        probe.save(case.directory / "controller_state.json", {"status": "FAILED", "old": True})
    with pytest.raises((RuntimeError, ValueError)):
        controller.run(case.directory, case.holder)
    assert "release" not in case.events and "holder_config" not in case.events
    assert case.workers == []
    if corruption == "existing_state":
        assert probe.read(case.directory / "controller_state.json")["old"] is True


def test_worker_nonzero_stops_remaining_models(prepared, monkeypatch):
    case = prepared

    def failed(command, **kwargs):
        result = case.run_monitored(command, **kwargs)
        result.returncode = 1
        return result

    monkeypatch.setattr(controller, "run_monitored", failed)
    with pytest.raises(RuntimeError, match="worker failed"):
        controller.run(case.directory, case.holder)
    assert case.workers == ["sft"]
    state = probe.read(case.directory / "controller_state.json")
    assert state["status"] == state["models"][0]["status"] == "FAILED"


def test_timeout_marks_current_slot_failed_and_stops_remaining_models(prepared, monkeypatch):
    case = prepared

    def timeout(command, **kwargs):
        case.run_monitored(command, **kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(controller, "run_monitored", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        controller.run(case.directory, case.holder)
    assert case.workers == ["sft"]
    state = probe.read(case.directory / "controller_state.json")
    assert state["status"] == state["models"][0]["status"] == "FAILED"
    assert "TimeoutExpired" in state["models"][0]["error"]


def test_runtime_change_after_first_worker_is_rejected_before_next_release(prepared, monkeypatch):
    case = prepared

    def changed(command, **kwargs):
        result = case.run_monitored(command, **kwargs)
        case.source.write_text("runtime changed during first worker\n")
        return result

    monkeypatch.setattr(controller, "run_monitored", changed)
    with pytest.raises(ValueError, match="Frozen acceptance source changed"):
        controller.run(case.directory, case.holder)
    assert case.workers == ["sft"]
    assert case.events.count("release") == 1


def test_return_zero_without_observed_population_stops_before_other_models(prepared, monkeypatch):
    case = prepared

    def incomplete(*args):
        value = case.readout(*args)
        value["short_workers"]["sft"] = "UNOBSERVED"
        value["status"] = "INCOMPLETE"
        return value

    monkeypatch.setattr(controller.acceptance, "readout", incomplete)
    with pytest.raises(RuntimeError, match="observed|OBSERVED|complete|population"):
        controller.run(case.directory, case.holder)
    assert case.workers == ["sft"]
    assert case.events.count("release") == 1
    assert probe.read(case.directory / "controller_state.json")["status"] == "FAILED"


def test_behavior_failure_keeps_all_models_and_does_not_claim_audited_acceptance(
    prepared, monkeypatch,
):
    case = prepared

    def behavior_failed(*args):
        value = case.readout(*args)
        value.update(behavioral_gate="FAIL", acceptance="FAIL")
        return value

    monkeypatch.setattr(controller.acceptance, "readout", behavior_failed)
    state = controller.run(case.directory, case.holder)
    assert case.workers == list(case.keys)
    assert case.events.count("release") == len(case.keys)
    assert state["status"] == "COMPLETE"
    assert state["behavioral_acceptance"] == state["acceptance"] == "FAIL"
    assert state["raw_audit_required"] is True


def test_success_requires_full_population_and_waits_for_offline_audit(prepared):
    case = prepared
    state = controller.run(case.directory, case.holder)
    assert case.workers == list(case.keys)
    assert state["status"] == "COMPLETE"
    assert state["behavioral_acceptance"] == "PASS"
    assert state["acceptance"] == "PENDING_AUDIT" and state["raw_audit_required"] is True


def test_final_readout_must_still_have_all_observed_workers(prepared, monkeypatch):
    case, readouts = prepared, []

    def disappeared(*args):
        value = case.readout(*args)
        readouts.append(len(case.workers))
        if len(readouts) == len(case.keys) + 1:
            value["status"] = "INCOMPLETE"
        return value

    monkeypatch.setattr(controller.acceptance, "readout", disappeared)
    with pytest.raises(RuntimeError, match="observed before completion"):
        controller.run(case.directory, case.holder)
    assert case.workers == list(case.keys)
    assert probe.read(case.directory / "controller_state.json")["status"] == "FAILED"


@pytest.mark.parametrize("which", ["state", "result"])
def test_combined_worker_evidence_must_match_before_next_model(prepared, monkeypatch, which):
    case = prepared

    def corrupted(command, **kwargs):
        result = case.run_monitored(command, **kwargs)
        path = case.directory / "workers/sft" / f"{which}.json"
        observed = probe.read(path)
        observed["status"] = "FAILED"
        probe.save(path, observed)
        return result

    monkeypatch.setattr(controller, "run_monitored", corrupted)
    with pytest.raises(RuntimeError, match="complete observed evidence"):
        controller.run(case.directory, case.holder)
    assert case.workers == ["sft"] and case.events.count("release") == 1
    state = probe.read(case.directory / "controller_state.json")
    assert state["status"] == state["models"][0]["status"] == "FAILED"

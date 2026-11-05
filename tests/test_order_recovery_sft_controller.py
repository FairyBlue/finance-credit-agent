"""Actual production controller with CPU child artifacts and real validators."""

import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
from test_order_recovery_sft_validation import evidence as evidence_fixture

from src.agents import order_recovery_sft_controller as c


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture
def evidence(tmp_path):
    return evidence_fixture.__wrapped__(tmp_path)


@pytest.fixture
def integrated(tmp_path, monkeypatch, evidence):
    root = tmp_path / "controller"
    run = root / "run"
    run.mkdir(parents=True)
    base = root / "base"
    base.mkdir()
    data = run / "train.jsonl"
    data.write_text("{}\n" * 384)
    write(base / "merge_receipt.json", {"status": "COMPLETE"})
    plan = root / "docs/2026-10-11_M2_ORDER_RECOVERY_LEARNING_PLAN.md"
    plan.parent.mkdir()
    plan.write_text("fixed CPU mock plan")
    implementation = root / "src/agents/coldstart_sft.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("# cpu mock")
    bank = {
        "training_dataset_sha256": c.digest(data),
        "cases": [{}] * 24,
        "source_split": "train",
        "teacher_labels_force_actions": False,
        "train_decision_dates": ["2023-01-01"],
        "recovery_training_source_dates": ["2023-01-01"],
        "acceptance_source_dates": ["2023-06-01"],
        "known_failure_source_dates": ["2023-11-01"],
    }
    write(run / "acceptance_bank.json", bank)
    manifest = {
        "status": "COMPLETE",
        "rows": 384,
        "recovery_rows": 192,
        "unchanged_replay_rows": 192,
        "acceptance_is_training": False,
        "training_observation_profile": "history_features_recovery_v2",
        "train_jsonl_sha256": c.digest(data),
        "acceptance_bank_sha256": c.digest(run / "acceptance_bank.json"),
        "training_acceptance_known_failure_dates_disjoint": True,
        "input_sha256": {"mock-input": "a"},
        "runtime_sha256": {"mock-generator": "b"},
        "tokenizer_sha256": {"mock-tokenizer": "c"},
        "suggested_fixed_training_budget": {
            "epochs": 3,
            "micro_batch": 1,
            "gradient_accumulation": 16,
            "optimizer_updates": 72,
            "learning_rate": 5e-5,
        },
    }
    write(run / "dataset_manifest.json", manifest)
    write(
        run / "dataset_source_receipt.json",
        {
            k: manifest[k]
            for k in (
                "input_sha256",
                "runtime_sha256",
                "tokenizer_sha256",
                "train_jsonl_sha256",
                "acceptance_bank_sha256",
            )
        },
    )
    for name, value in [("ROOT", root), ("RUN", run), ("BASE", base), ("DATA", data)]:
        monkeypatch.setattr(c, name, value)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: object())),
    )
    monkeypatch.setattr(c, "validate_model", lambda *a: None)
    monkeypatch.setattr(c, "model_contracts", lambda *a: {"sft": {}})
    examples = [dict(e, input_ids=[1, 2, 3, 4, 5]) for e in evidence.preflight["examples_manifest"]]
    monkeypatch.setattr(c, "read_examples", lambda *a: examples)
    monkeypatch.setattr(c.acceptance, "runtime_sources", lambda: {})
    config_kwargs = []

    def config(*args, **kwargs):
        config_kwargs.append(kwargs)
        return {}

    monkeypatch.setattr(c, "config_for_holder", config)
    monkeypatch.setattr(c, "freeze_config", lambda *a: None)
    calls = []

    def release(config):
        calls.append("admission")
        return {"driver_total_bytes": 100 * 1024**3, "driver_free_bytes": 92 * 1024**3}

    monkeypatch.setattr(c, "release_for_inference", release)
    monkeypatch.setattr(c, "controller_idle", lambda *a: calls.append("idle") or {})
    options = {"missing_steps": False, "corrupt_adapter": False, "bad_merge": False}

    def train(command, **kwargs):
        calls.append("train")
        assert "src.agents.order_recovery_sft_worker" in command
        assert "--allocator-receipt" in command
        training = run / "training"
        shutil.copytree(evidence.training, training)
        settings = dict(
            evidence.settings,
            data=str(data),
            model=str(base),
            output=str(training),
            data_sha256=c.digest(data),
            implementation_sha256=c.digest(implementation),
        )
        write(training / "settings.json", settings)
        write(
            training / "summary.json",
            dict(evidence.summary, data_sha256=c.digest(data), adapter=str(training / "adapter")),
        )
        write(
            run / "training_allocator_receipt.json",
            dict(
                evidence.allocator,
                training_output=str(training.resolve()),
                device_total_bytes=100 * 1024**3,
                requested_fraction=0.6,
                actual_fraction=0.6,
            ),
        )
        if options["missing_steps"]:
            (training / "steps.jsonl").unlink()
        if options["corrupt_adapter"]:
            (training / "adapter/adapter_model.safetensors").write_bytes(b"changed after summary")
        return subprocess.CompletedProcess(command, 0)

    def merge(command, **kwargs):
        calls.append("merge")
        model = run / "merged_model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "tokenizer_config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"mock actual export")
        receipt = {
            "status": "FAILED" if options["bad_merge"] else "COMPLETE",
            "device": "cpu",
            "model": str(base),
            "adapter": str(run / "training/adapter"),
            "files": c.acceptance.hf_inventory(model, excluded=set()),
        }
        write(model / "merge_receipt.json", receipt)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(c, "run_monitored", train)
    monkeypatch.setattr(c.subprocess, "run", merge)
    return SimpleNamespace(run=run, calls=calls, options=options, config_kwargs=config_kwargs)


def test_actual_helpers_allow_complete_cpu_mock(integrated):
    h = integrated
    c.main()
    assert h.calls == ["admission", "train", "idle", "merge"]
    assert set(h.config_kwargs[0]["phase_peak_bytes"].values()) == {60 * 1024**3}
    assert json.loads((h.run / "controller_state.json").read_text())["status"] == "COMPLETE"
    lineage = json.loads((h.run / "lineage_receipt.json").read_text())
    assert str(h.run / "training/steps.jsonl") in lineage["training_evidence_sha256"]


@pytest.mark.parametrize("fault", ["missing_steps", "corrupt_adapter"])
def test_invalid_training_cannot_call_merge(integrated, fault):
    h = integrated
    h.options[fault] = True
    with pytest.raises((ValueError, FileNotFoundError)):
        c.main()
    assert "merge" not in h.calls and not (h.run / "merged_model").exists()
    assert not (h.run / "lineage_receipt.json").exists()
    assert json.loads((h.run / "controller_state.json").read_text())["status"] == "FAILED"


def test_invalid_merge_receipt_cannot_produce_lineage(integrated):
    h = integrated
    h.options["bad_merge"] = True
    with pytest.raises(ValueError, match="CPU merge receipt"):
        c.main()
    assert h.calls[-1] == "merge" and not (h.run / "lineage_receipt.json").exists()
    assert json.loads((h.run / "controller_state.json").read_text())["status"] == "FAILED"

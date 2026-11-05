"""Bounded CPU contract tests; every rejected case would otherwise permit merge."""

import json
from types import SimpleNamespace

import pytest

from src.agents import order_recovery_sft_validation as v


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def inventory(model, *, excluded):
    return {
        str(p.relative_to(model)): {"bytes": p.stat().st_size, "sha256": v.digest(p)}
        for p in sorted(model.iterdir())
        if p.is_file() and p not in excluded
    }


@pytest.fixture
def evidence(tmp_path):
    training = tmp_path / "training"
    adapter = training / "adapter"
    adapter.mkdir(parents=True)
    implementation = tmp_path / "implementation.py"
    implementation.write_text("# cpu mock")
    (adapter / "adapter_model.safetensors").write_bytes(b"mock adapter")
    (training / "optimizer_rng.pt").write_bytes(b"mock optimizer")
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n" * 384)
    examples = [
        {
            "id": f"e{i:03}",
            "kind": "api" if i < 192 else "decision",
            "prompt_tokens": 2,
            "target_tokens": 3,
            "input_ids": [1, 2, 3, 4, 5],
        }
        for i in range(384)
    ]
    contract = {
        "data": str(data),
        "model": str(tmp_path / "base"),
        "data_sha256": v.digest(data),
        "epochs": 3,
        "micro_batch": 1,
        "gradient_accumulation": 16,
        "learning_rate": 5e-5,
        "max_length": 8192,
        "seed": 42,
        "optimizer_steps": 72,
    }
    settings = {
        "command": "train",
        "data": contract["data"],
        "model": contract["model"],
        "output": str(training),
        "seed": 42,
        "epochs": 3,
        "micro_batch": 1,
        "gradient_accumulation": 16,
        "lr": 5e-5,
        "max_length": 8192,
        "device": "cuda",
        "data_sha256": contract["data_sha256"],
        "implementation_sha256": v.digest(implementation),
        "schedule": "constant",
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "gradient_checkpointing": True,
        "attention": "sdpa",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "lora_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "versions": {"torch": "2.7.0+cu128"},
    }
    summary = {
        "status": "COMPLETE",
        "optimizer_steps": 72,
        "expected_optimizer_steps": 72,
        "probe_changed_steps": 72,
        "epochs": 3,
        "examples_per_epoch": 384,
        "example_presentations": 1152,
        "data_sha256": contract["data_sha256"],
        "adapter": str(adapter),
        "trainable_sha256_before": "a" * 64,
        "trainable_sha256_after": "b" * 64,
        "trainable_parameters": 32,
        "elapsed_training_seconds": 73.0,
        "artifacts": {
            "adapter/adapter_model.safetensors": v.digest(adapter / "adapter_model.safetensors"),
            "optimizer_rng.pt": v.digest(training / "optimizer_rng.pt"),
        },
    }
    preflight = {
        "status": "PASSED",
        "examples": 384,
        "kinds": {"api": 192, "decision": 192},
        "max_tokens": 5,
        "min_target_tokens": 3,
        "total_target_tokens": 1152,
        "examples_manifest": [
            {k: e[k] for k in ("id", "kind", "prompt_tokens", "target_tokens")} for e in examples
        ],
    }
    for name, d in [
        ("settings.json", settings),
        ("summary.json", summary),
        ("data_preflight.json", preflight),
    ]:
        write(training / name, d)
    rows = []
    for i in range(72):
        ids = [e["id"] for e in examples[(i % 24) * 16 : (i % 24 + 1) * 16]]
        rows.append(
            {
                "step": i + 1,
                "epoch": i // 24 + 1,
                "example_ids": ids,
                "examples": 16,
                "supervised_tokens": 48,
                "lr": 5e-5,
                "loss": 0.2,
                "grad_norm_before_clip": 0.1,
                "elapsed_seconds": i + 1.0,
                "probe_max_abs_delta": 1e-6,
                "adapter_update_detected": True,
                "probe_parameter": "model.q_proj.lora_B.default.weight",
                "cuda_peak_allocated_bytes": 30 * 1024**3,
                "cuda_peak_reserved_bytes": 40 * 1024**3,
            }
        )

    def write_steps():
        (training / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    write_steps()
    allocator_path = tmp_path / "training_allocator_receipt.json"
    allocator = {
        "status": "CONFIGURED",
        "allocator_config": "backend:native,expandable_segments:True",
        "allocator_backend": "native",
        "allocator_cap_bytes": 60 * 1024**3,
        "device_total_bytes": 140 * 1024**3,
        "training_output": str(training),
        "torch_version": "2.7.0+cu128",
        "cuda_device": 0,
        "requested_fraction": 60 / 140,
        "actual_fraction": 60 / 140,
    }
    write(allocator_path, allocator)

    def validate():
        return v.validate_training_evidence(
            training,
            examples,
            contract,
            implementation=implementation,
            allocator_receipt=allocator_path,
        )

    return SimpleNamespace(
        root=tmp_path,
        training=training,
        rows=rows,
        write_steps=write_steps,
        validate=validate,
        summary=summary,
        settings=settings,
        preflight=preflight,
        allocator=allocator,
        allocator_path=allocator_path,
    )


def test_exact_384_three_epochs_72_updates(evidence):
    result = evidence.validate()
    assert result["optimizer_steps"] == 72 and result["example_presentations"] == 1152
    assert len(result["evidence_sha256"]) == 5


@pytest.mark.parametrize(
    "fault", ["missing", "coverage", "nan_loss", "zero_delta", "lr", "tokens", "order"]
)
def test_bad_update_evidence_rejected_before_merge(evidence, fault):
    if fault == "missing":
        evidence.rows.pop()
    elif fault == "coverage":
        evidence.rows[1]["example_ids"] = evidence.rows[0]["example_ids"][:]
    elif fault == "nan_loss":
        evidence.rows[0]["loss"] = float("nan")
    elif fault == "zero_delta":
        evidence.rows[0]["probe_max_abs_delta"] = 0.0
    elif fault == "lr":
        evidence.rows[0]["lr"] = 1e-4
    elif fault == "tokens":
        evidence.rows[0]["supervised_tokens"] = 47
    elif fault == "order":
        evidence.rows[0]["step"] = 2
    evidence.write_steps()
    with pytest.raises(ValueError):
        evidence.validate()


def test_missing_steps_file_rejected(evidence):
    (evidence.training / "steps.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        evidence.validate()


@pytest.mark.parametrize("fault", ["settings", "preflight", "artifact", "allocator", "cap"])
def test_inconsistent_receipts_rejected(evidence, fault):
    if fault == "settings":
        evidence.settings["seed"] = 43
        write(evidence.training / "settings.json", evidence.settings)
    elif fault == "preflight":
        evidence.preflight["examples_manifest"][0]["target_tokens"] = 4
        write(evidence.training / "data_preflight.json", evidence.preflight)
    elif fault == "artifact":
        (evidence.training / "adapter/adapter_model.safetensors").write_bytes(b"corrupted")
    elif fault == "allocator":
        evidence.allocator["actual_fraction"] = 1.0
        write(evidence.allocator_path, evidence.allocator)
    elif fault == "cap":
        evidence.rows[0]["cuda_peak_reserved_bytes"] = 61 * 1024**3
        evidence.write_steps()
    with pytest.raises(ValueError):
        evidence.validate()


@pytest.mark.parametrize("fault", [None, "status", "device", "base", "adapter", "inventory"])
def test_merge_receipt_matches_real_files_and_cpu_lineage(tmp_path, fault):
    model = tmp_path / "model"
    model.mkdir()
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    (model / "model.safetensors").write_bytes(b"mock model")
    receipt = {
        "status": "COMPLETE",
        "device": "cpu",
        "model": str(base),
        "adapter": str(adapter),
        "files": inventory(model, excluded=set()),
    }
    if fault == "status":
        receipt["status"] = "FAILED"
    elif fault == "device":
        receipt["device"] = "cuda"
    elif fault == "base":
        receipt["model"] = "wrong-base"
    elif fault == "adapter":
        receipt["adapter"] = "wrong-adapter"
    elif fault == "inventory":
        receipt["files"] = {}
    write(model / "merge_receipt.json", receipt)
    if fault:
        with pytest.raises(ValueError):
            v.validate_merge_evidence(model, base=base, adapter=adapter, inventory=inventory)
    else:
        assert (
            v.validate_merge_evidence(model, base=base, adapter=adapter, inventory=inventory)[
                "status"
            ]
            == "PASSED"
        )


def test_no_overwrite_guard_allows_data_files_but_rejects_training_artifact(tmp_path):
    write(tmp_path / "dataset_manifest.json", {"data": True})
    v.refuse_execution_overwrite(tmp_path)
    write(tmp_path / "training_contract.json", {"original": True})
    with pytest.raises(ValueError, match="training_contract"):
        v.refuse_execution_overwrite(tmp_path)


def test_dataset_source_receipt_is_bound_and_dates_are_independent(tmp_path):
    (tmp_path / "train.jsonl").write_text("{}\n")
    bank = {
        "training_dataset_sha256": v.digest(tmp_path / "train.jsonl"),
        "source_split": "train",
        "teacher_labels_force_actions": False,
        "train_decision_dates": ["2023-01-01"],
        "recovery_training_source_dates": ["2023-01-01", "2023-01-02"],
        "acceptance_source_dates": ["2023-06-01"],
        "known_failure_source_dates": ["2023-11-01"],
    }
    write(tmp_path / "acceptance_bank.json", bank)
    dataset = {
        "input_sha256": {"old-data": "a"},
        "runtime_sha256": {"generator": "b"},
        "tokenizer_sha256": {"tokenizer": "c"},
        "train_jsonl_sha256": bank["training_dataset_sha256"],
        "acceptance_bank_sha256": v.digest(tmp_path / "acceptance_bank.json"),
    }
    write(tmp_path / "dataset_source_receipt.json", dataset)
    assert v.validate_dataset_receipt(tmp_path, dataset, bank) == dataset
    wrong = dict(dataset, train_jsonl_sha256="wrong")
    write(tmp_path / "dataset_source_receipt.json", wrong)
    with pytest.raises(ValueError, match="receipt mismatch"):
        v.validate_dataset_receipt(tmp_path, dataset, bank)
    write(tmp_path / "dataset_source_receipt.json", dataset)
    bank["acceptance_source_dates"] = ["2023-01-02"]
    with pytest.raises(ValueError, match="leak"):
        v.validate_dataset_receipt(tmp_path, dataset, bank)

"""CPU-only checks for the shared recovery SFT's real training/export evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def finite(value, *, minimum=None, strictly_positive=False):
    require(
        type(value) in (int, float) and math.isfinite(value), "Nonfinite or nonnumeric evidence"
    )
    if minimum is not None:
        require(value >= minimum, "Evidence is below its allowed minimum")
    if strictly_positive:
        require(value > 0, "Expected a positive numeric value")
    return value


def refuse_execution_overwrite(run: Path) -> None:
    for name in (
        "controller_state.json",
        "training_contract.json",
        "runtime_sources.zip",
        "training",
        "merged_model",
        "training.log",
        "merge.log",
        "gpu-training.jsonl",
        "training_allocator_receipt.json",
        "lineage_receipt.json",
        "model_spec_sft.json",
        "training_evidence_readout.json",
        "merge_evidence_readout.json",
        "gpu_handoff",
    ):
        require(not (run / name).exists(), f"Fresh execution required; existing {name}")


def validate_dataset_receipt(run: Path, dataset: dict, bank: dict) -> dict:
    receipt = read(run / "dataset_source_receipt.json")
    for key in (
        "input_sha256",
        "runtime_sha256",
        "tokenizer_sha256",
        "train_jsonl_sha256",
        "acceptance_bank_sha256",
    ):
        require(
            receipt.get(key) == dataset.get(key) and key in receipt,
            f"Dataset/source receipt mismatch: {key}",
        )
    require(
        receipt["train_jsonl_sha256"] == digest(run / "train.jsonl"),
        "Receipt dataset hash mismatch",
    )
    require(
        receipt["acceptance_bank_sha256"] == digest(run / "acceptance_bank.json"),
        "Receipt acceptance bank hash mismatch",
    )
    require(
        bank["training_dataset_sha256"] == receipt["train_jsonl_sha256"],
        "Bank dataset hash mismatch",
    )
    training = set(bank["train_decision_dates"])
    recovery = set(bank["recovery_training_source_dates"])
    acceptance = set(bank["acceptance_source_dates"])
    known = set(bank["known_failure_source_dates"])
    require(
        bool(training and recovery and acceptance and known), "Missing date separation evidence"
    )
    require(
        not ((training | recovery) & (acceptance | known)),
        "Training dates leak into acceptance/dev",
    )
    require(not acceptance & known, "New acceptance dates overlap the known dev cases")
    require(
        bank.get("source_split") == "train" and bank.get("teacher_labels_force_actions") is False,
        "Acceptance must use independent Train cases with no forced labels",
    )
    return receipt


def validate_training_evidence(
    training: Path,
    examples: list[dict],
    contract: dict,
    *,
    implementation: Path,
    allocator_receipt: Path,
) -> dict:
    summary, settings, preflight = (
        read(training / name) for name in ("summary.json", "settings.json", "data_preflight.json")
    )
    epochs = contract["epochs"]
    batch = contract["micro_batch"] * contract["gradient_accumulation"]
    steps_per_epoch = math.ceil(len(examples) / batch)
    expected_steps = epochs * steps_per_epoch
    require(expected_steps == contract["optimizer_steps"], "Contract update count is inconsistent")
    checks = {
        "command": "train",
        "data": contract["data"],
        "model": contract["model"],
        "output": str(training),
        "seed": contract["seed"],
        "epochs": epochs,
        "micro_batch": contract["micro_batch"],
        "gradient_accumulation": contract["gradient_accumulation"],
        "lr": contract["learning_rate"],
        "max_length": contract["max_length"],
        "device": "cuda",
        "data_sha256": contract["data_sha256"],
        "implementation_sha256": digest(implementation),
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
    }
    for key, value in checks.items():
        require(settings.get(key) == value, f"Actual SFT settings mismatch: {key}")
    require(summary.get("status") == "COMPLETE", "SFT summary is incomplete")
    for key, value in {
        "optimizer_steps": expected_steps,
        "expected_optimizer_steps": expected_steps,
        "probe_changed_steps": expected_steps,
        "epochs": epochs,
        "examples_per_epoch": len(examples),
        "example_presentations": epochs * len(examples),
        "data_sha256": contract["data_sha256"],
        "adapter": str(training / "adapter"),
    }.items():
        require(summary.get(key) == value, f"SFT summary mismatch: {key}")
    before, after = (
        summary.get("trainable_sha256_before", ""),
        summary.get("trainable_sha256_after", ""),
    )
    require(
        bool(re.fullmatch("[0-9a-f]{64}", before) and re.fullmatch("[0-9a-f]{64}", after))
        and before != after,
        "Trainable adapter hashes do not prove an actual update",
    )
    finite(summary["elapsed_training_seconds"], strictly_positive=True)
    require(
        type(summary["trainable_parameters"]) is int and summary["trainable_parameters"] > 0,
        "Missing trainable parameter count",
    )
    expected_manifest = [
        {key: e[key] for key in ("id", "kind", "prompt_tokens", "target_tokens")} for e in examples
    ]
    require(
        preflight.get("status") == "PASSED"
        and preflight.get("examples") == len(examples)
        and preflight.get("examples_manifest") == expected_manifest
        and preflight.get("kinds") == dict(Counter(e["kind"] for e in examples))
        and preflight.get("max_tokens") == max(len(e["input_ids"]) for e in examples)
        and preflight.get("min_target_tokens") == min(e["target_tokens"] for e in examples)
        and preflight.get("total_target_tokens") == sum(e["target_tokens"] for e in examples),
        "SFT data preflight differs from the frozen tokenized dataset",
    )
    by_id = {e["id"]: e for e in examples}
    require(len(by_id) == len(examples), "Duplicate example ids in controller data")
    rows = [json.loads(line) for line in (training / "steps.jsonl").read_text().splitlines()]
    require(len(rows) == expected_steps, "Missing or extra optimizer-step evidence")
    coverage = {epoch: Counter() for epoch in range(1, epochs + 1)}
    last_elapsed = -1.0
    for index, row in enumerate(rows, 1):
        epoch = (index - 1) // steps_per_epoch + 1
        require(
            type(row.get("step")) is int
            and row["step"] == index
            and type(row.get("epoch")) is int
            and row["epoch"] == epoch,
            "Optimizer steps/epochs are out of order",
        )
        ids = row["example_ids"]
        require(
            isinstance(ids, list) and len(ids) == len(set(ids)) and all(i in by_id for i in ids),
            "Invalid step example identities",
        )
        size = min(batch, len(examples) - ((index - 1) % steps_per_epoch) * batch)
        require(
            row["examples"] == size == len(ids), "Step micro/accumulated example count mismatch"
        )
        require(
            row["supervised_tokens"] == sum(by_id[i]["target_tokens"] for i in ids),
            "Step supervised-token accounting differs",
        )
        require(
            finite(row["lr"], strictly_positive=True) == contract["learning_rate"],
            "Step LR differs",
        )
        finite(row["loss"])
        finite(row["grad_norm_before_clip"], minimum=0)
        elapsed = finite(row["elapsed_seconds"], minimum=0)
        require(elapsed >= last_elapsed, "Step elapsed time runs backwards")
        last_elapsed = elapsed
        finite(row["probe_max_abs_delta"], strictly_positive=True)
        require(
            row["adapter_update_detected"] is True and "lora_B" in row["probe_parameter"],
            "Step lacks an actual adapter update",
        )
        allocated = finite(row["cuda_peak_allocated_bytes"], strictly_positive=True)
        reserved = finite(row["cuda_peak_reserved_bytes"], strictly_positive=True)
        require(reserved >= allocated, "Allocator peak evidence is inconsistent")
        coverage[epoch].update(ids)
    for epoch, counts in coverage.items():
        require(
            counts == Counter({i: 1 for i in by_id}),
            f"Epoch {epoch} does not cover each example once",
        )
    adapter_files = list((training / "adapter").iterdir())
    require(
        adapter_files and all(p.is_file() and not p.is_symlink() for p in adapter_files),
        "Unexpected/nonregular adapter export",
    )
    expected_artifacts = {str(p.relative_to(training)): digest(p) for p in adapter_files}
    expected_artifacts["optimizer_rng.pt"] = digest(training / "optimizer_rng.pt")
    require(summary.get("artifacts") == expected_artifacts, "SFT saved artifact hashes differ")
    allocator = read(allocator_receipt)
    total = allocator["device_total_bytes"]
    cap = 60 * 1024**3
    require(
        allocator.get("status") == "CONFIGURED"
        and allocator.get("allocator_config") == "backend:native,expandable_segments:True"
        and allocator.get("allocator_backend") == "native"
        and allocator.get("allocator_cap_bytes") == cap
        and total >= cap
        and allocator.get("training_output") == str(training.resolve())
        and allocator.get("torch_version") == settings["versions"]["torch"]
        and allocator.get("cuda_device") == 0,
        "Frozen training allocator receipt differs",
    )
    for key in ("requested_fraction", "actual_fraction"):
        require(
            math.isclose(
                finite(allocator[key], strictly_positive=True),
                cap / total,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "Allocator fraction differs from cap",
        )
    require(
        all(row["cuda_peak_reserved_bytes"] <= cap for row in rows),
        "Allocator peak exceeds its cap",
    )
    paths = [
        training / name
        for name in ("summary.json", "settings.json", "data_preflight.json", "steps.jsonl")
    ]
    paths.append(allocator_receipt)
    return {
        "status": "PASSED",
        "optimizer_steps": expected_steps,
        "epochs": epochs,
        "unique_examples_per_epoch": len(examples),
        "example_presentations": epochs * len(examples),
        "all_steps_actual_update": True,
        "all_artifact_hashes_match": True,
        "evidence_sha256": {str(p): digest(p) for p in paths},
        "summary": summary,
    }


def validate_merge_evidence(model: Path, *, base: Path, adapter: Path, inventory) -> dict:
    receipt_path = model / "merge_receipt.json"
    receipt = read(receipt_path)
    require(
        receipt.get("status") == "COMPLETE"
        and receipt.get("device") == "cpu"
        and receipt.get("model") == str(base)
        and receipt.get("adapter") == str(adapter),
        "CPU merge receipt status/base/adapter/device mismatch",
    )
    files = inventory(model, excluded={receipt_path})
    require(
        receipt.get("files") == files, "CPU merge receipt inventory differs from actual HF export"
    )
    return {
        "status": "PASSED",
        "model_files": len(files),
        "receipt_sha256": digest(receipt_path),
        "files": files,
    }

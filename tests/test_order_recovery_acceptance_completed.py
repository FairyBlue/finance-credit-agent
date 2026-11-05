"""Historical completion validation uses real manifests, hashes and ZIP entries."""

from __future__ import annotations

import copy
import stat
import warnings
import zipfile
from pathlib import Path

import pytest

from src.agents import order_recovery_acceptance as acceptance
from src.agents import order_recovery_episode as episode
from src.agents import order_recovery_probe as probe


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    probe.save(path, value)


@pytest.fixture
def completed(tmp_path):
    directory = tmp_path / "completed"
    directory.mkdir()
    runtime_file = tmp_path / "runtime" / "engine.py"
    runtime_file.parent.mkdir()
    runtime_file.write_text("FROZEN_ENGINE = 1\n")
    audit_helper = tmp_path / "audit_helper.py"
    audit_helper.write_text("INDEPENDENT_AUDITOR = 1\n")
    evidence = tmp_path / "training.jsonl"
    evidence.write_text('{"update": 72}\n')
    model = tmp_path / "model"
    model.mkdir()
    for name, content in (
        ("config.json", b"{}"),
        ("tokenizer_config.json", b"{}"),
        ("model.safetensors", b"actual CPU fixture export"),
    ):
        (model / name).write_bytes(content)
    lineage = tmp_path / "lineage.json"
    write(lineage, {"status": "COMPLETE"})
    files = acceptance.hf_inventory(model, excluded={lineage})
    models = {
        "sft": {
            "model_path": str(model),
            "lineage_receipt": str(lineage),
            "files": files,
            "fingerprint": acceptance.fingerprint(files),
            "evidence": {str(evidence): probe.sha(evidence)},
            "identities": [{"id": f"seed{seed}_sft", "base_seed": seed} for seed in (42, 43)],
        }
    }
    known = {
        "cases": [
            {
                "id": case_id,
                "source_identity": {
                    "step": step,
                    "sample_index": sample,
                    "rollout_n": 0,
                    "validate": True,
                },
                "source_base_seed": 42,
                "source": {"episode_id": case_id},
                "prefix": ["legal-buy", "rejected-sell"],
            }
            for case_id, (step, sample) in zip(
                acceptance.KNOWN_IDS,
                ((4, 0), (0, 2), (2, 6), (0, 9), (4, 10), (4, 11), (0, 1), (0, 3)),
                strict=True,
            )
        ]
    }
    fresh = {
        "training_dataset_sha256": probe.sha(evidence),
        "cases": [
            {
                "id": f"fresh-{index:02}",
                "source_identity": {
                    "step": 0,
                    "sample_index": index,
                    "rollout_n": 0,
                    "validate": True,
                },
                "source_base_seed": 44,
                "source": {"episode_id": f"heldout-{index // 8}"},
                "prefix": ["legal-buy", "rejected-sell"],
            }
            for index in range(24)
        ],
    }
    known_path, bank_path = tmp_path / "known.json", tmp_path / "bank.json"
    write(known_path, known)
    write(bank_path, fresh)
    start, assigned = acceptance.request_seed_identities(known, fresh)
    short = copy.deepcopy(known["cases"] + fresh["cases"])
    for case in short[8:]:
        case["request_seed_identity"] = assigned[case["id"]]
    full = copy.deepcopy(known["cases"])
    runtime = {str(runtime_file): probe.sha(runtime_file)}
    archive = directory / "runtime_sources.zip"
    with zipfile.ZipFile(archive, "x") as output:
        output.write(runtime_file, arcname=str(runtime_file)[1:])
    contract = {
        "profile": acceptance.PROFILE,
        "status": "PREPARED",
        "mode": "sft",
        "budget": acceptance.budget("sft"),
        "action_protocol": acceptance.PROTOCOL,
        "observation_profile": acceptance.OBSERVATION_PROFILE,
        "enable_thinking": False,
        "endpoint_gate": acceptance.ENDPOINT_GATE,
        "endpoint_gate_sha256": acceptance.fingerprint(acceptance.ENDPOINT_GATE),
        "short_cases": short,
        "full_cases": full,
        "models": models,
        "evidence_sha256": {
            str(path): probe.sha(path) for path in (evidence, lineage, known_path, bank_path)
        },
        "runtime_sha256": runtime,
        "runtime_snapshot_sha256": probe.sha(archive),
        "request_seed_namespace": {
            "scheme": acceptance.REQUEST_SEED_NAMESPACE,
            "known_manifest": str(known_path),
            "known_manifest_sha256": probe.sha(known_path),
            "acceptance_bank": str(bank_path),
            "acceptance_bank_sha256": probe.sha(bank_path),
            "sample_index_start": start,
            "seed_collision_check": acceptance.check_request_seed_collisions(short, full),
        },
    }
    phases = {
        "short": {
            "profile": acceptance.SHORT_PROFILE,
            "models": models,
            "cases": short,
        },
        "full": {"profile": episode.PROFILE, "models": models, "cases": full},
    }
    contract["phase_manifest_sha256"] = {}
    sources = {**contract["evidence_sha256"], **runtime, str(audit_helper): probe.sha(audit_helper)}
    for phase, document in phases.items():
        path = directory / phase / "manifest.json"
        write(path, document)
        contract["phase_manifest_sha256"][phase] = probe.sha(path)
        sources[str(path)] = probe.sha(path)
        for name in (
            "state.json",
            "result.json",
            "config.json",
            "requests.jsonl",
            "raw.jsonl",
            "submissions.jsonl",
        ):
            path = directory / phase / "workers" / "sft" / name
            write(path, {"status": "OBSERVED", "fixture": name})
            sources[str(path)] = probe.sha(path)
    for name in ("state.json", "result.json"):
        path = directory / "workers" / "sft" / name
        write(path, {"status": "OBSERVED"})
        sources[str(path)] = probe.sha(path)
    write(directory / "manifest.json", contract)
    digest = probe.sha(directory / "manifest.json")
    sources[str(directory / "manifest.json")] = digest
    sources[str(archive)] = probe.sha(archive)
    write(
        directory / "raw_audit.json",
        {
            "profile": acceptance.RAW_AUDIT_PROFILE,
            "status": "PASS",
            "mode": "sft",
            "audit_scope": "complete_population",
            "manifest_sha256": digest,
            "endpoint_gate_sha256": acceptance.fingerprint(acceptance.ENDPOINT_GATE),
            "short_trajectories": 320,
            "full_trajectories": 80,
            "workers": {"sft": {}},
            "acceptance_pass": True,
            "source_sha256": sources,
        },
    )
    raw_digest = probe.sha(directory / "raw_audit.json")
    write(
        directory / "state.json",
        {
            "status": "PREPARED",
            "profile": acceptance.PROFILE,
            "manifest_sha256": digest,
        },
    )
    write(
        directory / "controller_state.json",
        {
            "status": "COMPLETE",
            "manifest_sha256": digest,
            "models": [{"model_key": "sft", "status": "COMPLETE"}],
        },
    )
    write(
        directory / "completion_runtime_receipt.json",
        {
            "status": "PASS",
            "controller_status": "COMPLETE",
            "acceptance": "PASS",
            "manifest_sha256": digest,
            "raw_audit_sha256": raw_digest,
            "runtime_files_verified": len(runtime),
            "runtime_archive_sha256": probe.sha(archive),
        },
    )
    write(
        directory / "readout.json",
        {
            "profile": acceptance.PROFILE,
            "mode": "sft",
            "status": "OBSERVED",
            "acceptance": "PASS",
            "behavioral_gate": "PASS",
            "manifest_sha256": digest,
            "endpoint_gate_sha256": acceptance.fingerprint(acceptance.ENDPOINT_GATE),
            "raw_audit_required": True,
            "raw_audit_status": "PASS",
            "raw_audit_sha256": raw_digest,
            "short_workers": {"sft": "OBSERVED"},
            "full_workers": {"sft": "OBSERVED"},
            "short_grid": [{"status": "RECOVERED"}] * 320,
            "full_grid": [{"status": "COMPLETE"}] * 80,
        },
    )
    return directory, runtime_file, evidence, audit_helper


def mutate(path: Path, **fields) -> None:
    document = probe.read(path)
    document.update(fields)
    write(path, document)


def rebind_manifest(directory: Path, contract: dict) -> None:
    """Rebind the envelope to expose deeper validation, rather than just outer hashes."""
    write(directory / "manifest.json", contract)
    digest = probe.sha(directory / "manifest.json")
    audit = probe.read(directory / "raw_audit.json")
    audit["manifest_sha256"] = digest
    audit["source_sha256"][str(directory / "manifest.json")] = digest
    audit["source_sha256"][str(directory / "runtime_sources.zip")] = contract[
        "runtime_snapshot_sha256"
    ]
    write(directory / "raw_audit.json", audit)
    raw_digest = probe.sha(directory / "raw_audit.json")
    for name in ("controller_state.json", "readout.json", "completion_runtime_receipt.json"):
        update = {"manifest_sha256": digest}
        if name != "controller_state.json":
            update["raw_audit_sha256"] = raw_digest
        if name == "completion_runtime_receipt.json":
            update["runtime_archive_sha256"] = contract["runtime_snapshot_sha256"]
        mutate(directory / name, **update)


def test_completed_runtime_drift_passes_only_historical_mode(completed):
    directory, runtime, _, _ = completed
    contract = acceptance.verify_completed(directory)
    runtime.write_text("FROZEN_ENGINE = 2\n")
    with pytest.raises(ValueError, match="Frozen acceptance source"):
        acceptance.verify(directory)
    with pytest.raises(ValueError, match="Independently audited source"):
        acceptance.raw_audit_gate(directory, contract)
    assert acceptance.verify_completed(directory) == contract
    assert acceptance.raw_audit_gate(directory, contract, completed_runtime=True) == {
        "status": "PASS",
        "acceptance_pass": True,
        "sha256": probe.sha(directory / "raw_audit.json"),
    }
    # The archived runtime may be absent locally; immutable evidence may not.
    runtime.unlink()
    assert acceptance.verify_completed(directory) == contract


@pytest.mark.parametrize(
    "filename,field,value",
    [
        ("controller_state.json", "status", "RUNNING"),
        ("controller_state.json", "manifest_sha256", "0" * 64),
        ("controller_state.json", "models", [{"model_key": "sft", "status": "RUNNING"}]),
        ("readout.json", "status", "INCOMPLETE"),
        ("readout.json", "acceptance", "PENDING_AUDIT"),
        ("readout.json", "raw_audit_sha256", "0" * 64),
        ("readout.json", "endpoint_gate_sha256", "0" * 64),
        ("readout.json", "short_grid", []),
        ("completion_runtime_receipt.json", "status", "FAIL"),
        ("completion_runtime_receipt.json", "manifest_sha256", "0" * 64),
        ("completion_runtime_receipt.json", "raw_audit_sha256", "0" * 64),
        ("completion_runtime_receipt.json", "runtime_archive_sha256", "0" * 64),
        ("completion_runtime_receipt.json", "runtime_files_verified", 0),
    ],
)
def test_completion_chain_tamper_is_rejected(completed, filename, field, value):
    directory, *_ = completed
    mutate(directory / filename, **{field: value})
    with pytest.raises(ValueError, match="Historical acceptance"):
        acceptance.verify_completed(directory, check_models=False)


@pytest.mark.parametrize("kind", ["evidence", "auditor", "raw", "model", "phase"])
def test_nonruntime_artifacts_remain_live(completed, kind):
    directory, _, evidence, auditor = completed
    target = {
        "evidence": evidence,
        "auditor": auditor,
        "raw": directory / "short/workers/sft/requests.jsonl",
        "model": evidence.parent / "model/model.safetensors",
        "phase": directory / "short/manifest.json",
    }[kind]
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(ValueError):
        acceptance.verify_completed(directory)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "extra", "traversal", "symlink", "bytes"])
def test_archive_members_are_checked_after_outer_hash_rebinding(completed, kind):
    directory, runtime, *_ = completed
    contract = probe.read(directory / "manifest.json")
    archive_path = directory / "runtime_sources.zip"
    name = str(runtime)[1:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive_path, "w") as output:
            if kind not in {"missing", "symlink"}:
                output.writestr(name, b"changed" if kind == "bytes" else runtime.read_bytes())
            if kind == "duplicate":
                output.writestr(name, runtime.read_bytes())
            if kind in {"extra", "traversal"}:
                output.writestr("unexpected.py" if kind == "extra" else "../outside.py", b"x")
            if kind == "symlink":
                entry = zipfile.ZipInfo(name)
                entry.create_system = 3
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                output.writestr(entry, runtime.read_bytes())
    contract["runtime_snapshot_sha256"] = probe.sha(archive_path)
    rebind_manifest(directory, contract)
    with pytest.raises(ValueError, match="Historical runtime archive"):
        acceptance.verify_completed(directory, check_models=False)


@pytest.mark.parametrize("kind", ["endpoint", "namespace", "model_population", "budget"])
def test_historical_mode_keeps_original_contract_checks(completed, kind):
    directory, *_ = completed
    contract = probe.read(directory / "manifest.json")
    if kind == "endpoint":
        contract["endpoint_gate"]["comparison_tolerance"] = 0.1
    elif kind == "namespace":
        contract["short_cases"][8]["request_seed_identity"]["sample_index"] += 1
    elif kind == "model_population":
        contract["models"]["seed42_m1"] = contract["models"]["sft"]
    else:
        contract["budget"]["short_request_cap"] = 5
    rebind_manifest(directory, contract)
    with pytest.raises(ValueError):
        acceptance.verify_completed(directory, check_models=False)


def test_completed_checks_expected_manifest_and_raw_acceptance(completed):
    directory, *_ = completed
    with pytest.raises(ValueError, match="manifest digest"):
        acceptance.verify_completed(directory, expected_sha256="0" * 64)
    audit = probe.read(directory / "raw_audit.json")
    audit["acceptance_pass"] = False
    write(directory / "raw_audit.json", audit)
    raw_digest = probe.sha(directory / "raw_audit.json")
    for name in ("completion_runtime_receipt.json", "readout.json"):
        mutate(directory / name, raw_audit_sha256=raw_digest)
    with pytest.raises(ValueError, match="raw audit did not pass"):
        acceptance.verify_completed(directory, check_models=False)


def test_runtime_cannot_mask_a_live_evidence_entry(completed):
    directory, runtime, *_ = completed
    contract = probe.read(directory / "manifest.json")
    contract["evidence_sha256"][str(runtime)] = probe.sha(runtime)
    rebind_manifest(directory, contract)
    runtime.write_text("must remain live when also evidence\n")
    with pytest.raises(ValueError, match="Frozen acceptance source"):
        acceptance.verify_completed(directory, check_models=False)


def test_prepare_all_accepts_completed_prior_without_copying_stale_runtime(completed, monkeypatch):
    directory, runtime, *_ = completed
    prior = probe.read(directory / "manifest.json")
    runtime.write_text("FROZEN_ENGINE = 2\n")
    models = copy.deepcopy(prior["models"])
    for key in acceptance.MODEL_KEYS[1:]:
        models[key] = {**models["sft"], "fingerprint": key}

    async def bank(*_args):
        return copy.deepcopy(prior["short_cases"]), copy.deepcopy(prior["full_cases"])

    monkeypatch.setattr(acceptance, "validate_bank", bank)
    monkeypatch.setattr(acceptance, "model_contracts", lambda *_args: models)
    monkeypatch.setattr(acceptance, "runtime_sources", lambda: {str(runtime): probe.sha(runtime)})
    specification = directory.parent / "models.json"
    write(specification, {})
    namespace = prior["request_seed_namespace"]
    destination = directory.parent / "new-all"
    acceptance.prepare(
        destination,
        Path(namespace["known_manifest"]),
        Path(namespace["acceptance_bank"]),
        specification,
        "all",
        sft_gate_readout=directory / "readout.json",
    )
    prepared = acceptance.verify(destination, check_models=False)
    assert str(runtime) not in prepared["evidence_sha256"]
    assert prepared["runtime_sha256"][str(runtime)] == probe.sha(runtime)
    for name in ("controller_state.json", "completion_runtime_receipt.json", "runtime_sources.zip"):
        path = directory / name
        assert prepared["evidence_sha256"][str(path)] == probe.sha(path)

"""Independent audit tamper tests and the complete 1,200-trajectory population."""

from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

import pytest
from test_order_recovery_acceptance import (
    Engine,
    Tokenizer,
    acceptance,
    auditor,
    create_model,
    prepared_fixture,
)
from test_order_recovery_acceptance import (
    environment_bank as _environment_bank_fixture,
)

from src.agents import order_recovery_probe as probe
from src.agents.coldstart_data import tool_call


@pytest.fixture
def environment_bank(tmp_path):
    return _environment_bank_fixture.__wrapped__(tmp_path)


class RejectionTokenizer(Tokenizer):
    def decode(self, tokens, **kwargs):
        if tokens and tokens[0] == 0:
            return tool_call("buy", {"symbol": "AAA", "target_weight": 0.2})
        return super().decode(tokens, **kwargs)


class RejectOnceEngine(Engine):
    def generate(self, prompts, params, **kwargs):
        first = self.count == 0
        outputs = super().generate(prompts, params, **kwargs)
        if first:
            outputs[0].outputs[0].token_ids = [0]
        return outputs


def execute(directory, contract, key, replay, reset, *, reject_once=True, oversized=False):
    return asyncio.run(
        acceptance.execute(
            RejectOnceEngine() if reject_once else Engine(),
            RejectionTokenizer(oversized=oversized),
            SimpleNamespace,
            contract,
            key,
            directory,
            replay=replay,
            reset=reset,
        )
    )


def documents(path):
    return probe.read(path) if path.suffix == ".json" else acceptance.json_lines(path)


def write(path, data):
    if path.suffix == ".json":
        probe.save(path, data)
    else:
        path.write_text("".join(probe.canonical(row) + "\n" for row in data))


def test_independent_audit_rejects_tampering(tmp_path, monkeypatch, environment_bank):
    directory, contract, replay, reset = prepared_fixture(tmp_path, monkeypatch, environment_bank)
    execute(directory, contract, "sft", replay, reset)
    acceptance.readout(directory)
    original = auditor.audit(directory)
    assert original["status"] == "PASS" and original["acceptance_pass"] is True
    assert original["workers"]["sft"]["short"]["rejections_with_unchanged_ledger"] == 1
    rejected = []

    def reject(label, filenames, mutate):
        paths = [directory / name for name in filenames]
        originals = [path.read_bytes() for path in paths]
        content = [copy.deepcopy(documents(path)) for path in paths]
        mutate(content)
        for path, data in zip(paths, content, strict=True):
            write(path, data)
        try:
            with pytest.raises((ValueError, KeyError, FileNotFoundError)):
                auditor.audit(directory)
            rejected.append(label)
        finally:
            for path, data in zip(paths, originals, strict=True):
                path.write_bytes(data)

    short = "short/workers/sft/"
    full = "full/workers/sft/"
    reject(
        "invalid ledger changed",
        [short + "requests.jsonl"],
        lambda data: next(
            record for record in data[0] if record.get("feedback", {}).get("invalid_action")
        )["ledger_after"].update(cash=0),
    )
    reject(
        "raw response token mapping",
        [short + "raw.jsonl"],
        lambda data: data[0][0]["outputs"][0]["outputs"][0]["token_ids"].append(99),
    )

    def wrong_seed(data):
        data[0][0]["sampling_params"]["seed"] = 0
        data[1][0]["requests"][0]["sampling_params"]["seed"] = 0

    reject(
        "derived request seed", [short + "requests.jsonl", short + "submissions.jsonl"], wrong_seed
    )

    def wrong_finish(data):
        data[0][0]["finish_reason"] = "length"
        data[1][0]["outputs"][0]["outputs"][0]["finish_reason"] = "length"

    reject("truncation applied", [short + "requests.jsonl", short + "raw.jsonl"], wrong_finish)
    reject(
        "more than one session",
        [short + "requests.jsonl"],
        lambda data: next(
            record for record in data[0] if record.get("ledger_after", {}).get("session_index") == 2
        )["ledger_after"].update(session_index=3),
    )
    reject(
        "phase parent manifest",
        [full + "config.json"],
        lambda data: data[0].update(acceptance_manifest_sha256="0" * 64),
    )

    def wrong_terminal(data):
        for document in data:
            document["grid"][0]["terminal_metrics"]["raw_net_return"] += 1

    reject(
        "terminal financial outcome", [full + "state.json", full + "result.json"], wrong_terminal
    )

    def missing_cell(data):
        for document in data:
            document["grid"].pop()

    reject("missing trajectory cell", [short + "state.json", short + "result.json"], missing_cell)
    reject(
        "engine configuration",
        [short + "config.json"],
        lambda data: data[0]["engine"].update(gpu_memory_utilization=0.9),
    )
    reject(
        "parent state/result", ["workers/sft/state.json"], lambda data: data[0].update(phase="full")
    )
    assert len(rejected) == 10
    assert auditor.audit(directory)["acceptance_pass"] is True
    # A partial raw audit is valid evidence but can never authorize all-mode.
    partial = auditor.audit(directory, ("sft",))
    assert partial["audit_scope"] == "read_only_partial" and partial["acceptance_pass"] is False
    probe.save(directory / "raw_audit.json", partial)
    with pytest.raises(ValueError, match="Independent raw audit population"):
        acceptance.raw_audit_gate(directory, contract)


def test_prompt_caps_keep_every_cell_and_no_requests(tmp_path, monkeypatch, environment_bank):
    directory, contract, replay, reset = prepared_fixture(tmp_path, monkeypatch, environment_bank)
    execute(directory, contract, "sft", replay, reset, reject_once=False, oversized=True)
    readout = acceptance.readout(directory)
    assert readout["acceptance"] == "FAIL"
    assert all(
        row["requests"] == 0 and row["failure"] == "PROMPT_CAP"
        for row in readout["short_grid"] + readout["full_grid"]
    )
    audited = auditor.audit(directory)
    assert audited["status"] == "PASS" and audited["requests"] == 0
    assert audited["trajectories"] == 400 and audited["acceptance_pass"] is False


def test_fresh_all_mode_requires_raw_pass_and_audits_1200(tmp_path, monkeypatch, environment_bank):
    directory, contract, replay, reset = prepared_fixture(tmp_path, monkeypatch, environment_bank)
    execute(directory, contract, "sft", replay, reset)
    readout = acceptance.readout(directory)
    assert readout["acceptance"] == "PENDING_AUDIT"
    all_spec = {
        "sft": {
            "model_path": contract["models"]["sft"]["model_path"],
            "lineage_receipt": contract["models"]["sft"]["lineage_receipt"],
        }
    }
    for key in acceptance.MODEL_KEYS[1:]:
        all_spec[key] = create_model(
            tmp_path, key=key, sft_fingerprint=contract["models"]["sft"]["fingerprint"]
        )
    spec = tmp_path / "all_models.json"
    probe.save(spec, all_spec)
    new = tmp_path / "all_acceptance"
    with pytest.raises(ValueError, match="acceptance gate has not passed"):
        acceptance.prepare(
            new,
            tmp_path / "known.json",
            tmp_path / "bank.json",
            spec,
            "all",
            sft_gate_readout=directory / "readout.json",
        )
    assert not new.exists()
    probe.save(directory / "raw_audit.json", auditor.audit(directory))
    assert acceptance.readout(directory)["acceptance"] == "PASS"
    # A historical gate also requires the controller's final state and its
    # completion receipt; a PASS readout alone cannot activate archive mode.
    digest = probe.sha(directory / "manifest.json")
    probe.save(
        directory / "controller_state.json",
        {
            "status": "COMPLETE",
            "manifest_sha256": digest,
            "models": [{"model_key": "sft", "status": "COMPLETE"}],
        },
    )
    probe.save(
        directory / "completion_runtime_receipt.json",
        {
            "status": "PASS",
            "controller_status": "COMPLETE",
            "acceptance": "PASS",
            "manifest_sha256": digest,
            "raw_audit_sha256": probe.sha(directory / "raw_audit.json"),
            "runtime_files_verified": len(contract["runtime_sha256"]),
            "runtime_archive_sha256": contract["runtime_snapshot_sha256"],
        },
    )
    prepared = acceptance.prepare(
        new,
        tmp_path / "known.json",
        tmp_path / "bank.json",
        spec,
        "all",
        sft_gate_readout=directory / "readout.json",
    )
    assert prepared["budget"]["planned_trajectories"] == 1200
    all_contract = acceptance.verify(new)
    incomplete = acceptance.readout(new)
    assert len(incomplete["short_grid"]) == 960 and len(incomplete["full_grid"]) == 240
    for key in acceptance.MODEL_KEYS:
        execute(new, all_contract, key, replay, reset)
    before_audit = acceptance.readout(new)
    assert before_audit["acceptance"] == "PENDING_AUDIT"
    result = auditor.audit(new)
    assert result["short_trajectories"] == 960 and result["full_trajectories"] == 240
    assert result["trajectories"] == 1200 and result["acceptance_pass"] is True
    assert set(result["workers"]) == set(acceptance.MODEL_KEYS)
    probe.save(new / "raw_audit.json", result)
    final = acceptance.readout(new)
    assert final["acceptance"] == "PASS" and final["repeated_sft_acceptance"] is True
    assert final["endpoint_adherence"]["endpoint_adherent"] == 720
    assert len(final["full_financial_activity"]) == 240

"""Synthetic integration fixtures; these are not the real pilot's observations."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/diagnostics/m2_same_prefix_gate_probe.py"


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location("same_prefix_gate_probe_fixture", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def write_rows(path, values):
    with path.open("w") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def isolated_run(output):
    # pytest's global conftest imports src.env. A fresh CPU process exercises the
    # actual production backend guard rather than weakening or mocking it.
    code = (
        "import importlib.util,sys;from pathlib import Path;"
        f"s=importlib.util.spec_from_file_location('probe',{str(PROBE)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "m.verify_source=lambda path,expected=None:"
        "{'source_artifact_sha256':{},'inherited_source_count':0,'inherited_source_bytes':0};"
        "m.run(Path(sys.argv[1]))"
    )
    return subprocess.run(
        [sys.executable, "-c", code, str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def synthetic_source(source):
    """32 explicit groups: 29 full, two k=3, one k=2, no observed pilot q."""
    source.mkdir()
    inputs, generated, evaluated, group_flags = [], [], [], []
    for group in range(32):
        anchor = f"synthetic-{group:02d}"
        if group < 7:
            orders = [["buy", "A", 0.1], ["buy", "A", 0.2], ["buy", "B", 0.1], ["buy", "B", 0.2]]
            effects, family_gate, order_gate = (3.0, 1.0, -1.0, -3.0), False, True
        elif group < 11:
            orders = [["buy", "A", 0.1], ["buy", "A", 0.2], ["sell", "B", 0.0], ["hold"]]
            effects, family_gate, order_gate = (3.0, 1.0, -1.0, 0.0), True, True
        else:
            orders = [["hold"]] * 4
            effects, family_gate, order_gate = (0.0,) * 4, False, False
        k = 4 if group < 29 else 3 if group < 31 else 2
        group_flags.append(
            {
                "anchor_id": anchor,
                "full_n4_support": k == 4,
                "n4_family_gate": family_gate if k == 4 else None,
                "n4_order_gate": order_gate if k == 4 else None,
            }
        )
        context = {
            "source_seed": 42,
            "source_sibling_id": 0,
            "source_episode_id": f"synthetic-window-{group // 4}",
            "source_policy_step": 1 + group // 8,
            "decision_session_index": (0, 5, 10, 15)[group % 4],
        }
        hold = float(100 + group)
        for slot, (order, effect) in enumerate(zip(orders, effects, strict=True)):
            key = f"{anchor}/s{slot}"
            prompt = f"{group + 1:064x}"
            common = {"request_key": key, "anchor_id": anchor, "slot": slot}
            inputs.append(
                {
                    **common,
                    "source_group": context,
                    "request_seed": 1000 + group * 4 + slot,
                    "prompt_sha256": prompt,
                    "prompt_token_ids": [1, 2, 3],
                }
            )
            generated.append(
                {
                    **common,
                    "prompt_sha256": prompt,
                    "returned_prompt_token_ids": [1, 2, 3],
                    "generation_status": "COMPLETE",
                }
            )
            legal = slot < k
            evaluated.append(
                {
                    **common,
                    "generation_status": "COMPLETE",
                    "env_status": "ADVANCING" if legal else "INVALID",
                    "terminal_label_status": "COMPLETE_FIXED_HOLD" if legal else "NA",
                    "financial_state_sha256": f"{group + 100:064x}",
                    "observation_messages_sha256": f"{group + 200:064x}",
                    "hold_Y": hold,
                    "Y": hold + effect if legal else None,
                    "q": effect if legal else None,
                    "economic_order_key": json.dumps(order) if legal else None,
                    "family": order[0] if legal else None,
                    "first_info": {"error": "synthetic rejected order"} if not legal else {},
                }
            )
    write_rows(source / "inputs.jsonl", inputs)
    write_rows(source / "generations.jsonl", generated)
    write_rows(source / "evaluations.jsonl", evaluated)
    write_json(source / "model_preflight_receipt.json", {"model_fingerprint": "f" * 64})
    write_json(
        source / "readout.json",
        {
            "status": "PASS",
            "counts": {
                "full_n4_support_groups": 29,
                "full_n4_family_gate_open_groups": 4,
                "full_n4_order_gate_open_groups": 11,
            },
        },
    )
    with (source / "group_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(group_flags[0]))
        writer.writeheader()
        writer.writerows(group_flags)
    return source


@pytest.fixture
def prepared(tmp_path, monkeypatch, probe):
    source = synthetic_source(tmp_path / "synthetic-source")
    output = tmp_path / "gate"
    fixture, baseline = tmp_path / "unit-fixture.json", tmp_path / "baseline.json"
    write_json(
        fixture,
        {
            "status": "PASS",
            "module_sha256": probe.sha(probe.MODULE),
            "tests_sha256": probe.sha(probe.TEST),
            "runner_sha256": probe.sha(PROBE),
            "runner_tests_sha256": probe.sha(Path(__file__).resolve()),
            "synthetic": True,
        },
    )
    write_json(baseline, {"status": "PASS", "synthetic": True})
    # Only the expensive external historical-tree proof is replaced. prepare,
    # make_inputs, verify, run, the candidate and all output writers execute.
    monkeypatch.setattr(
        probe,
        "verify_source",
        lambda path, expected=None: {
            "source_artifact_sha256": {},
            "inherited_source_count": 0,
            "inherited_source_bytes": 0,
        },
    )
    probe.prepare(output, source, fixture, baseline)
    return output, source


def test_actual_run_preserves_every_partial_slot_as_na_and_isolates_gate(probe, prepared):
    output, _ = prepared
    completed = isolated_run(output)
    assert completed.returncode == 0, completed.stderr
    result = probe.read(output / "readout.json")
    slots = probe.rows(output / "slot_signals.jsonl")
    assert result["counts"]["planned_groups"] == 32
    assert result["counts"]["planned_slots"] == 128
    assert result["counts"]["full_n4_groups"] == 29
    assert result["counts"]["source_legal_slots"] == 124
    assert result["counts"]["operator_supported_slots"] == 116
    assert result["counts"]["operator_NA_slots"] == 12
    assert result["counts"]["k_histogram"] == {"0": 0, "1": 0, "2": 1, "3": 2, "4": 29}
    assert result["counts"]["restored_signal_groups"] == 7
    assert result["counts"]["changed_slots_family_to_order"] == 28
    assert result["counts"]["lost_signal_groups"] == 0
    partial = [row for row in slots if not row["full_n4_support"]]
    assert len(partial) == 12
    assert sum(row["slot_label_supported"] for row in partial) == 8
    for row in partial:
        assert row["operator_status"] == "NA_PARTIAL_GROUP"
        for method in probe.METHODS:
            assert row[f"raw_{method}"] is None
            assert row[f"eta025_{method}"] is None
    first = next(row for row in slots if row["request_key"] == "synthetic-00/s0")
    assert first["raw_family_gate_plain"] == 0.0
    assert first["raw_order_gate_plain"] == 3.0
    assert first["eta025_order_gate_plain"] == 0.75
    methods = {row["method"]: row for row in result["method_summary"]}
    assert methods["zero"]["slots_with_signal"] == 0
    assert methods["family_gate_plain"]["groups_with_signal"] == 4
    assert methods["order_gate_plain"]["groups_with_signal"] == 11
    assert result["checks"]["q_Y_common_H_max_abs_difference"] == 0.0
    assert probe.read(output / "completion.json")["slots_classified"] == 128
    before = probe.sha(output / "completion.json")
    repeated = isolated_run(output)
    assert repeated.returncode != 0 and "FileExistsError" in repeated.stderr
    assert probe.sha(output / "completion.json") == before


@pytest.mark.parametrize(
    "field,value,group",
    (
        ("financial_state_sha256", "e" * 64, 0),
        ("financial_state_sha256", "e" * 64, 29),
        ("observation_messages_sha256", "e" * 64, 0),
    ),
)
def test_shared_prefix_damage_hard_fails_even_for_partial_group(
    probe, prepared, field, value, group
):
    output, source = prepared
    rows = probe.rows(source / "evaluations.jsonl")
    rows[group * 4 + 1][field] = value
    write_rows(source / "evaluations.jsonl", rows)
    # Regenerate/pin the derived input to reach the independent group check.
    write_rows(output / "gate_inputs.jsonl", probe.make_inputs(source))
    contract = probe.read(output / "contract.json")
    contract["inputs_sha256"] = probe.sha(output / "gate_inputs.jsonl")
    write_json(output / "contract.json", contract)
    with pytest.raises(ValueError, match=f"shared {field}"):
        probe.run(output)
    assert not (output / "completion.json").exists()


def test_model_snapshot_cannot_be_changed_in_a_derived_slot(probe, prepared):
    output, _ = prepared
    rows = probe.rows(output / "gate_inputs.jsonl")
    rows[0]["policy_snapshot_sha256"] = "d" * 64
    write_rows(output / "gate_inputs.jsonl", rows)
    contract = probe.read(output / "contract.json")
    contract["inputs_sha256"] = probe.sha(output / "gate_inputs.jsonl")
    write_json(output / "contract.json", contract)
    with pytest.raises(ValueError, match="derived inputs exactly reproduce source"):
        probe.run(output)
    assert not (output / "slot_signals.jsonl").exists()


def test_gate_input_hash_damage_rejected_by_real_verify(probe, prepared):
    output, _ = prepared
    with (output / "gate_inputs.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="sealed gate inputs"):
        probe.verify(output)


def test_declared_source_hash_damage_rejected_by_real_verify(probe, prepared, tmp_path):
    output, _ = prepared
    sentinel = tmp_path / "synthetic-source-identity.py"
    sentinel.write_text("original\n")
    contract = probe.read(output / "contract.json")
    contract["source_sha256"][str(sentinel)] = probe.sha(sentinel)
    write_json(output / "contract.json", contract)
    sentinel.write_text("changed\n")
    with pytest.raises(ValueError, match="sealed new source changed"):
        probe.verify(output)


def test_source_reader_gate_mismatch_is_not_accepted(probe, prepared):
    output, source = prepared
    with (source / "group_metrics.csv").open(newline="") as stream:
        groups = list(csv.DictReader(stream))
    groups[0]["n4_family_gate"] = "True"
    with (source / "group_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    with pytest.raises(ValueError, match="per-anchor source family gate"):
        probe.run(output)


def test_partial_source_gate_cannot_be_falsely_marked_closed(probe, prepared):
    output, source = prepared
    with (source / "group_metrics.csv").open(newline="") as stream:
        groups = list(csv.DictReader(stream))
    groups[29]["n4_family_gate"] = "False"
    with (source / "group_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    with pytest.raises(ValueError, match="partial source gates must be NA"):
        probe.run(output)


def test_same_order_different_fixed_hold_score_is_integrity_failure(probe, prepared):
    output, source = prepared
    values = probe.rows(source / "evaluations.jsonl")
    values[11 * 4]["Y"] += 1.0
    values[11 * 4]["q"] += 1.0
    write_rows(source / "evaluations.jsonl", values)
    write_rows(output / "gate_inputs.jsonl", probe.make_inputs(source))
    contract = probe.read(output / "contract.json")
    contract["inputs_sha256"] = probe.sha(output / "gate_inputs.jsonl")
    write_json(output / "contract.json", contract)
    with pytest.raises(
        ValueError, match="same-state/order deterministic fixed-hold labels identical"
    ):
        probe.run(output)


def test_real_historical_verifier_rejects_an_inherited_byte_change(probe, tmp_path):
    source = tmp_path / "historical-tree"
    source.mkdir()
    inherited = tmp_path / "historical-source.py"
    inherited.write_text("frozen\n")
    write_json(source / "proposal.json", {})
    write_rows(source / "inputs.jsonl", [])
    write_json(
        source / "contract.json",
        {
            "source_sha256": {str(inherited): probe.sha(inherited)},
            "proposal_sha256": probe.sha(source / "proposal.json"),
            "inputs_sha256": probe.sha(source / "inputs.jsonl"),
        },
    )
    for name in ("generation_completion.json", "evaluation_completion.json", "readout.json"):
        write_json(source / name, {"status": "PASS"})
    write_json(source / "final_verification.json", {"status": "PASS", "evidence_sha256": {}})
    expected = probe.verify_source(source)["source_artifact_sha256"]
    assert probe.verify_source(source, expected)["source_artifact_sha256"] == expected
    inherited.write_text("tampered\n")
    with pytest.raises(ValueError, match="inherited source changed"):
        probe.verify_source(source, expected)

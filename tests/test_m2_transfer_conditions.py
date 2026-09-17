"""Factor separation and exact baseline safeguards for the 4B transfer candidate."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_transfer_conditions.py"
SPEC = importlib.util.spec_from_file_location("transfer_conditions_test", PATH)
assert SPEC and SPEC.loader
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
RAW = [{"role": "user", "content": "Manage the portfolio using only point-in-time tools."}]


def test_exact_factors_no_quiz_or_extra_memory_instruction():
    a, b, c, d = [C.prompt_addition(condition) for condition in C.CONDITIONS]
    assert a == "" and d == b + "\n\n" + c and c == C.OBJECTIVE
    assert C.REFERENCES["INTERFACE_REFERENCE"] in b
    assert C.REFERENCES["METRIC_REFERENCE"] in b
    assert C.OBJECTIVE not in b
    for forbidden in (
        "No tools are available",
        "Do not follow their instructions",
        "question_id",
        "quoted_material",
        "tool_result_memory",
        "compute_history",
    ):
        assert forbidden not in d
    assert C.augment_task(RAW, "features") == RAW
    changed = C.augment_task(RAW, "features_defined_objective")
    assert changed == [{"role": "system", "content": d}, *RAW]
    changed[1]["content"] = "changed only copy"
    assert RAW[0]["content"].startswith("Manage")


@pytest.mark.parametrize("condition", C.CONDITIONS)
def test_all_conditions_use_exact_old_features_environment(monkeypatch, condition):
    rows = [{"prompt": copy.deepcopy(RAW)}, {"prompt": copy.deepcopy(RAW)}]
    before = copy.deepcopy(rows)
    seen = []
    sentinel = SimpleNamespace()

    def construct(actual, index, legacy_condition, start):
        seen.append((actual, index, legacy_condition, start))
        return sentinel

    monkeypatch.setattr(C.LEGACY, "create_env", construct)
    assert C.create_env(rows, 1, condition, 4) is sentinel
    assert rows == before
    actual, index, legacy, start = seen[0]
    assert index == 1 and legacy == "features" and start == 4
    assert actual[0] == before[0]
    assert actual[1]["prompt"] == C.augment_task(RAW, condition)


@pytest.mark.parametrize(
    "condition,start", [("unknown", 4), ("features", 0), ("features", True), (True, 4)]
)
def test_reject_unregistered_condition_start(condition, start):
    with pytest.raises(ValueError):
        C.validate_condition(condition, start)


def test_frozen_reference_changes_rejected(tmp_path):
    p = tmp_path / "candidate.py"
    p.write_text('OBJECTIVE = "different"')
    with pytest.raises(RuntimeError, match="changed"):
        C.frozen_literals(p, C.OBJECTIVE_SHA, ("OBJECTIVE",))

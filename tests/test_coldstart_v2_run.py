"""Controller freezes one shared initializer and a bounded sequence without GPU execution."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "coldstart_v2_controller",
    Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_coldstart_v2_run.py",
)
assert SPEC is not None and SPEC.loader is not None
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def test_phase_order_common_model_and_fixed_training_budget(tmp_path):
    model = tmp_path / "shared-model"
    phases = controller.phases(tmp_path, model)
    assert [phase["name"] for phase in phases] == [
        "before_regression", "before_fresh", "sft", "merged_model",
        "after_regression", "after_fresh", "readout",
    ]
    training = phases[2]["argv"]
    assert training[training.index("--model") + 1] == str(model)
    assert training[training.index("--epochs") + 1] == "2"
    assert training[training.index("--lr") + 1] == "0.00005"
    assert training[training.index("--gradient-accumulation") + 1] == "8"
    merge = phases[3]["argv"]
    assert merge[merge.index("--model") + 1] == str(model)
    assert merge[merge.index("--adapter") + 1] == str(tmp_path / "sft/adapter")
    for index, full in ((0, "4"), (1, "1"), (4, "4"), (5, "1")):
        command = phases[index]["argv"]
        assert command[command.index("--expected-full") + 1] == full
        assert command[command.index("--action-protocol") + 1] == controller.PROTOCOL
        selected = model if index < 2 else tmp_path / "merged_model"
        assert command[command.index("--model") + 1] == str(selected)


def test_streamed_common_model_receipt_rejects_same_size_corruption(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"original")
    record = {"bytes": weights.stat().st_size, "sha256": hashlib.sha256(b"original").hexdigest()}
    (tmp_path / "merge_receipt.json").write_text(json.dumps({
        "status": "COMPLETE", "files": {"weights.bin": record},
    }))
    identity = controller.model_identity(tmp_path)
    assert identity["files"]["weights.bin"] == record
    weights.write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="model identity mismatch"):
        controller.model_identity(tmp_path)

"""CPU fakes assert allocator ordering and exclusive output; no CUDA imports."""

import json
import os
import sys
from types import SimpleNamespace

import pytest

from src.agents import order_recovery_sft_worker as w


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    calls = []
    fraction = [None]
    options = {"backend": "native", "ack": True, "total": 140 * 1024**3}

    def load():
        calls.append(("load", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")))
        return torch

    def set_fraction(value, *, device):
        calls.append(("cap", value, device))
        fraction[0] = value

    torch = SimpleNamespace(
        __version__="2.7.0+cu128",
        cuda=SimpleNamespace(
            memory=SimpleNamespace(get_allocator_backend=lambda: options["backend"]),
            get_device_properties=lambda device: SimpleNamespace(total_memory=options["total"]),
            set_per_process_memory_fraction=set_fraction,
            get_per_process_memory_fraction=lambda *, device: (
                fraction[0] if options["ack"] else 1.0
            ),
        ),
    )
    receipt = tmp_path / "training_allocator_receipt.json"
    output = tmp_path / "training"
    monkeypatch.setattr(w, "load_torch", load)

    def delegate(argv):
        assert receipt.exists() and not output.exists()
        calls.append(("delegate", argv))

    monkeypatch.setattr(w, "delegate", delegate)
    argv = [
        "--allocator-receipt",
        str(receipt),
        "--output",
        str(output),
        "--data",
        "train.jsonl",
        "--model",
        "base",
        "--epochs",
        "3",
        "--micro-batch",
        "1",
        "--gradient-accumulation",
        "16",
        "--lr",
        "5e-5",
    ]
    return SimpleNamespace(receipt=receipt, output=output, calls=calls, options=options, argv=argv)


def test_fixed_allocator_precedes_torch_and_delegate(fake, monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    w.main(fake.argv)
    assert [c[0] for c in fake.calls] == ["load", "cap", "delegate"]
    assert fake.calls[0][1] == "backend:native,expandable_segments:True"
    assert fake.calls[1][1] == 60 / 140
    assert fake.calls[2][1][0] == "train" and fake.calls[2][1][-2:] == ["--device", "cuda"]
    receipt = json.loads(fake.receipt.read_text())
    assert receipt["allocator_cap_bytes"] == 60 * 1024**3 and receipt["actual_fraction"] == 60 / 140


@pytest.mark.parametrize("kind", ["receipt", "output"])
def test_no_overwrite_precedes_any_torch_load(fake, kind):
    if kind == "receipt":
        fake.receipt.write_text("original")
    else:
        fake.output.mkdir()
    with pytest.raises(RuntimeError, match="Fresh"):
        w.main(fake.argv)
    assert fake.calls == []
    if kind == "receipt":
        assert fake.receipt.read_text() == "original"


@pytest.mark.parametrize("fault", ["backend", "ack", "total"])
def test_allocator_failure_never_writes_receipt_or_delegates(fake, fault):
    fake.options[fault] = {"backend": "cudaMallocAsync", "ack": False, "total": 40 * 1024**3}[fault]
    with pytest.raises(RuntimeError):
        w.main(fake.argv)
    assert not fake.receipt.exists() and not any(c[0] == "delegate" for c in fake.calls)


def test_preimported_torch_fails_before_configuring(fake, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(RuntimeError, match="before importing"):
        w.main(fake.argv)
    assert fake.calls == [] and not fake.receipt.exists()

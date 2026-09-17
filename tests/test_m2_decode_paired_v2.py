"""Observer non-interference and cached-decode identity checks."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OBSERVER = load("decode_observer_test", "scripts/diagnostics/m2_decode_observer.py")
D = load("decode_paired_test", "scripts/diagnostics/m2_decode_paired_v2.py")


class FakeSampler(torch.nn.Module):
    def forward(self, *, logits, sampling_metadata=None):
        self.returned = SimpleNamespace(sampled_token_ids=logits.argmax(-1)[:, None])
        return self.returned


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_readonly_observer_preserves_logits_output_and_rng(tmp_path, dtype):
    sampler = FakeSampler()
    runner = SimpleNamespace(
        sampler=sampler,
        input_batch=SimpleNamespace(
            req_ids=["0"],
            num_prompt_tokens=[3],
            req_output_token_ids=[[2]],
            num_computed_tokens_cpu=[3],
        ),
        query_start_loc_np=np.array([0, 1]),
        positions_np=np.array([3]),
    )
    worker = SimpleNamespace(model_runner=runner)
    logits = torch.tensor([[1.0, 2.0, 4.0]], dtype=dtype)
    expected = logits.clone()
    rng = torch.random.get_rng_state().clone()
    OBSERVER.install_observer(worker, str(tmp_path / "records.jsonl"))
    result = sampler(logits=logits)
    assert result is sampler.returned
    assert torch.equal(logits, expected)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert result.sampled_token_ids.tolist() == [[2]]
    assert OBSERVER.remove_observer(worker)["sampler_calls"] == 1
    assert not sampler._forward_hooks and not sampler._forward_pre_hooks
    row = json.loads((tmp_path / "records.jsonl").read_text())["rows"][0]
    assert row["logprob"] == float(expected.float().log_softmax(-1)[0, 2])
    assert row["output_prefix"] == [2] and row["last_input_position"] == 3


def records():
    return [
        {
            "rows": [
                {
                    "request_id": "0",
                    "prompt_length": 3,
                    "output_prefix": [],
                    "last_input_position": 1,
                    "computed_before": 0,
                    "scheduled_tokens": 2,
                    "token_id": 5,
                },
                {
                    "request_id": "0",
                    "prompt_length": 3,
                    "output_prefix": [],
                    "last_input_position": 2,
                    "computed_before": 2,
                    "scheduled_tokens": 1,
                    "token_id": 5,
                },
                {
                    "request_id": "0",
                    "prompt_length": 3,
                    "output_prefix": [5],
                    "last_input_position": 3,
                    "computed_before": 3,
                    "scheduled_tokens": 1,
                    "token_id": 7,
                },
            ]
        }
    ]


def test_partial_prefill_is_retained_but_not_output():
    assert len(D.matched_observations(records(), "0", 3, [5, 7])) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("computed_before", 2),
        ("scheduled_tokens", 2),
        ("output_prefix", [6]),
        ("token_id", 8),
        ("last_input_position", 4),
        ("prompt_length", 4),
    ],
)
def test_reject_wrong_decode_identity(field, value):
    r = records()
    r[0]["rows"][-1][field] = value
    with pytest.raises(RuntimeError):
        D.matched_observations(r, "0", 3, [5, 7])


def test_reject_duplicate_observation():
    r = records()
    r[0]["rows"].append(r[0]["rows"][-1].copy())
    with pytest.raises(RuntimeError):
        D.matched_observations(r, "0", 3, [5, 7])


def test_reject_unexplained_extra_observation():
    r = records()
    r[0]["rows"].append({**r[0]["rows"][-1], "output_prefix": [5, 7]})
    with pytest.raises(RuntimeError):
        D.matched_observations(r, "0", 3, [5, 7])


def test_observer_import_survives_external_scripts_namespace(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "scripts", SimpleNamespace(__path__=[]))
    D.P.H.add_import_paths()
    install, remove = D.observer_callbacks()
    assert install.__module__ == "m2_decode_observer"
    assert remove.__module__ == "m2_decode_observer"

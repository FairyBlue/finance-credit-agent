"""Causal token boundaries, fixed work population and unchanged acceptance thresholds."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_logprob_alignment.py"
spec = importlib.util.spec_from_file_location("alignment_tests", path)
assert spec and spec.loader
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_causal_positions_include_first_prediction_and_eos():
    assert M.positions(4, 3) == [3, 4, 5]
    prompt, suffix = [10, 11, 12, 13], [20, 21, 151645]
    full = prompt + suffix[:-1]
    assert len(full) == 6
    assert [full[i] for i in M.positions(4, 3)] == [13, 20, 21]


@pytest.mark.parametrize("p,s", [(0, 1), (1, 0), (True, 2), (3, -1), (3.0, 2)])
def test_invalid_lengths_rejected(p, s):
    with pytest.raises(RuntimeError):
        M.positions(p, s)


def test_fixed_grid_every_token_once_per_path():
    cases = [
        {"case_index": i, "prompt_ids": [10, 11], "suffix_ids": [20] * (30 if i in (2, 3) else 29)}
        for i in range(8)
    ]
    batches = M.request_grid(cases)
    requests = [r for b in batches for r in b]
    assert len(requests) == 250
    assert [r["request_index"] for r in requests] == list(range(250))
    assert len(batches[0]) == 8 and all(len(b) == 1 for b in batches[1:])
    for c in cases:
        for mode in ("full_batch", "full_single", "prefix_single"):
            rows = [r for r in requests if r["case_index"] == c["case_index"] and r["mode"] == mode]
            assert [v for r in rows for v in r["selected_suffix_indices"]] == list(
                range(len(c["suffix_ids"]))
            )
            for r in rows:
                last = max(r["selected_suffix_indices"])
                assert r["input_ids"] == c["prompt_ids"] + c["suffix_ids"][: last + 1]


def test_preserve_token_and_sequence_gates():
    assert M.compare([-1.0, -2.0], [-1.0, -2.0])["status"] == "PASSED"
    assert M.compare([-1.0], [-1.194])["status"] == "FAILED"
    result = M.compare([-1.0] * 20, [-1.03] * 20)
    assert result["status"] == "FAILED" and result["max_token_abs_difference"] < 0.1


@pytest.mark.parametrize(
    "left,right", [([], []), ([-1.0], [-1.0, -2.0]), ([float("nan")], [-1.0]), ([0.2], [-1.0])]
)
def test_invalid_logprob_population_rejected(left, right):
    with pytest.raises(RuntimeError):
        M.compare(left, right)


def test_raw_target_position_and_actual_input_are_checked():
    row = {"input_ids": [10, 11, 20, 21], "selected_suffix_indices": [1]}
    case = {"prompt_ids": [10, 11], "suffix_ids": [20, 21]}
    lp = SimpleNamespace(logprob=-0.4, rank=7)
    response = SimpleNamespace(
        index=0, token_ids=[30], finish_reason="length", stop_reason=None, text="x"
    )
    output = SimpleNamespace(
        request_id="0",
        prompt_token_ids=row["input_ids"],
        prompt_logprobs=[None, {11: lp}, {20: lp}, {21: lp}],
        outputs=[response],
    )
    raw = M.selected_vllm(output, row, case)
    assert raw["selected_raw_logprobs"] == [
        {
            "input_token_position": 3,
            "suffix_token_index": 1,
            "target_token_id": 21,
            "distribution": {"21": {"logprob": -0.4, "rank": 7}},
        }
    ]
    output.prompt_token_ids = [10, 11, 20, 22]
    with pytest.raises(RuntimeError):
        M.selected_vllm(output, row, case)


def test_serial_combination_does_not_mutate_baseline():
    model = {
        "engine_kwargs": {
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "enforce_eager": False,
            "max_num_seqs": 32,
            "compilation_config": 3,
        }
    }
    serial = M.engine_config(model, "vllm_serial")
    assert serial["max_num_seqs"] == 1 and serial["compilation_config"] == 0
    assert serial["enforce_eager"] and not serial["enable_prefix_caching"]
    assert not serial["enable_chunked_prefill"]
    assert M.engine_config(model, "vllm_common") == model["engine_kwargs"]
    assert model["engine_kwargs"]["max_num_seqs"] == 32

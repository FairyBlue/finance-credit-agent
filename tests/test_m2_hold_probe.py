"""CPU-only checks for the fixed first-response diagnostic; no model requests."""

from __future__ import annotations

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.rollout_seed import derive_request_seed

_spec = spec_from_file_location(
    "m2_hold_probe", Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_hold_probe.py",
)
assert _spec is not None and _spec.loader is not None
probe = module_from_spec(_spec)
_spec.loader.exec_module(probe)


def test_exact_48_request_grid_and_model_local_ids():
    plan = probe.build_request_plan()
    expected = [
        (model, condition, window, sibling)
        for model in ("screening", "wiring")
        for condition in ("A", "B", "C")
        for window in (0, 1)
        for sibling in range(4)
    ]
    assert len(plan) == 48
    assert [row["request_index"] for row in plan] == list(range(48))
    assert [row["model_request_index"] for row in plan] == list(range(24)) * 2
    assert [
        (row["model_key"], row["condition"], row["window_index"], row["rollout_id"])
        for row in plan
    ] == expected
    assert len(set(expected)) == 48


def test_seed_is_production_first_step_identity_shared_across_conditions_and_models():
    grouped = {}
    for row in probe.build_request_plan():
        expected = derive_request_seed(
            42,
            {
                "step": 1,
                "sample_index": row["window_index"],
                "rollout_n": row["rollout_id"],
                "validate": False,
            },
            0,
        )
        assert type(row["seed"]) is int and row["seed"] == expected
        grouped.setdefault((row["window_index"], row["rollout_id"]), []).append(row["seed"])
    assert len(grouped) == 8
    assert all(len(seeds) == 6 and len(set(seeds)) == 1 for seeds in grouped.values())
    assert len({seeds[0] for seeds in grouped.values()}) == 8


def test_sampling_kwargs_preserve_registered_settings_and_do_not_mutate_plan():
    plan = probe.build_request_plan()
    before = deepcopy(plan)
    for row in plan:
        params = probe.sampling_kwargs(row)
        assert params == {
            "n": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 4096,
            "ignore_eos": False,
            "seed": row["seed"],
            "repetition_penalty": 1.0,
            "logprobs": None,
        }
        params["max_tokens"] = 999
        assert probe.sampling_kwargs(row)["max_tokens"] == 4096
    assert plan == before
    assert probe.build_request_plan() == before


@pytest.mark.parametrize(
    "tokens,finish,eos,expected",
    [
        ([151645], "stop", {151643, 151645}, True),
        ([15, 151643], "stop", {151643, 151645}, True),
        ([15] * 4094 + [151645], "stop", {151643, 151645}, True),
        ([15] * 4095 + [151645], "stop", {151643, 151645}, False),
        ([15] * 4096 + [151645], "stop", {151643, 151645}, False),
        ([], "stop", {151643, 151645}, False),
        ([15, 16], "stop", {151643, 151645}, False),
        ([151645], "length", {151643, 151645}, False),
        ([151645], "abort", {151643, 151645}, False),
        ([151645], "completed", {151643, 151645}, False),
        ([151645], "unknown", {151643, 151645}, False),
        ([151645], None, {151643, 151645}, False),
        ([151645], "stop", {151643}, False),
        ([151645], "stop", set(), False),
    ],
)
def test_completeness_requires_native_stop_effective_eos_and_strict_original_cap(
    tokens, finish, eos, expected
):
    assert probe.response_complete(tokens, finish, eos) is expected


def _batch(model="screening", condition="A"):
    plans = [
        row
        for row in probe.build_request_plan()
        if row["model_key"] == model and row["condition"] == condition
    ]
    prompts = {0: [151644, 100, 151645], 1: [151644, 200, 151645]}
    outputs = [
        SimpleNamespace(
            request_id=str(row["model_request_index"]),
            prompt_token_ids=list(prompts[row["window_index"]]),
            outputs=[
                SimpleNamespace(
                    index=0,
                    token_ids=[15, 151645],
                    finish_reason="stop",
                    stop_reason=None,
                    text="synthetic response",
                )
            ],
        )
        for row in plans
    ]
    return outputs, plans, prompts


@pytest.mark.parametrize("model", ["screening", "wiring"])
@pytest.mark.parametrize("condition", ["A", "B", "C"])
def test_exact_batch_mapping_accepts_each_model_condition_without_reordering(model, condition):
    outputs, plans, prompts = _batch(model, condition)
    before = deepcopy((outputs, plans, prompts))
    assert probe.validate_batch_mapping(outputs, plans, prompts) is None
    assert (outputs, plans, prompts) == before


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "duplicate_id",
        "reordered",
        "wrong_id",
        "wrong_prompt",
        "other_window_prompt",
        "missing_output",
        "multiple_outputs",
        "wrong_sample_index",
    ],
)
def test_bad_raw_mapping_is_rejected_without_guessing_or_filling(change):
    outputs, plans, prompts = _batch()
    if change == "missing":
        outputs.pop()
    elif change == "extra":
        outputs.append(deepcopy(outputs[-1]))
    elif change == "duplicate_id":
        outputs[1].request_id = outputs[0].request_id
    elif change == "reordered":
        outputs[0], outputs[1] = outputs[1], outputs[0]
    elif change == "wrong_id":
        outputs[0].request_id = "99"
    elif change == "wrong_prompt":
        outputs[0].prompt_token_ids.append(99)
    elif change == "other_window_prompt":
        outputs[0].prompt_token_ids = list(prompts[1])
    elif change == "missing_output":
        outputs[0].outputs = []
    elif change == "multiple_outputs":
        outputs[0].outputs.append(deepcopy(outputs[0].outputs[0]))
    elif change == "wrong_sample_index":
        outputs[0].outputs[0].index = 1
    before = deepcopy((outputs, plans, prompts))
    with pytest.raises(probe.DiagnosticStop):
        probe.validate_batch_mapping(outputs, plans, prompts)
    assert (outputs, plans, prompts) == before


def test_fresh_second_model_cannot_use_global_request_ids_as_engine_ids():
    outputs, plans, prompts = _batch("wiring")
    for output, row in zip(outputs, plans, strict=True):
        output.request_id = str(row["request_index"])
    with pytest.raises(probe.DiagnosticStop):
        probe.validate_batch_mapping(outputs, plans, prompts)

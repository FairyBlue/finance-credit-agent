"""Probability measurement boundaries and the fixed action-support population."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "support_probe_test", ROOT / "scripts/diagnostics/m2_action_support_probe.py"
)
assert spec and spec.loader
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def lp(value, rank=1):
    return SimpleNamespace(logprob=value, rank=rank)


def test_fixed_grid_has_1024_distinct_identities_and_paired_seeds():
    grid = P.grid_plan()
    assert len(grid) == 1024
    assert [r["request_index"] for r in grid] == list(range(1024))
    assert len({(r["condition"], r["window_index"], r["sibling"]) for r in grid}) == 1024
    for pi in range(8):
        rows = [r for r in grid if r["prompt_index"] == pi]
        assert len(rows) == 128 and [r["sibling"] for r in rows] == list(range(128))
        assert len({r["seed"] for r in rows}) == 128
    assert len({r["seed"] for r in grid}) == 256
    assert P.budget()["maximum_engine_requests"] == 1024 + 384 + 8
    assert P.budget()["maximum_generated_tokens"] == 1024 * 4096 + 392


def test_candidates_preserve_universe_and_exclude_illegal_cash_sells():
    symbols = [f"S{i}" for i in range(20)]
    candidates = P.candidate_actions(symbols)
    assert len(candidates) == 42
    assert [c["name"] for c in candidates[:2]] == ["hold", "abstain"]
    assert [c["arguments"]["symbol"] for c in candidates[2:22]] == symbols
    assert [c["arguments"]["symbol"] for c in candidates[22:]] == symbols
    assert all(c["arguments"]["target_weight"] == 0.1 for c in candidates[2:22])
    assert not any(c["name"] == "sell" for c in candidates)


@pytest.mark.parametrize("symbols", [[], ["X"] * 20, [str(i) for i in range(19)]])
def test_candidates_reject_selected_or_duplicate_universe(symbols):
    with pytest.raises(RuntimeError):
        P.candidate_actions(symbols)


def test_suffix_excludes_prompt_but_includes_final_eos():
    output = SimpleNamespace(
        prompt_token_ids=[10, 11, 20, 151645],
        prompt_logprobs=[None, {11: lp(-9)}, {20: lp(-0.2)}, {151645: lp(-0.3)}],
    )
    records = P.suffix_logprobs([10, 11], [20, 151645], output)
    assert [r["token_id"] for r in records] == [20, 151645]
    assert math.fsum(r["logprob"] for r in records) == -0.5


@pytest.mark.parametrize("mode", ["wrong_ids", "short", "none", "missing_token"])
def test_suffix_rejects_mapping_and_logprob_loss(mode):
    output = SimpleNamespace(
        prompt_token_ids=[10, 11, 20], prompt_logprobs=[None, {11: lp(-9)}, {20: lp(-0.2)}]
    )
    if mode == "wrong_ids":
        output.prompt_token_ids[-1] = 21
    elif mode == "short":
        output.prompt_logprobs.pop()
    elif mode == "none":
        output.prompt_logprobs = None
    else:
        output.prompt_logprobs[-1] = {21: lp(-0.2)}
    with pytest.raises(RuntimeError):
        P.suffix_logprobs([10, 11], [20], output)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 0.2])
def test_nonfinite_or_positive_logprob_rejected(value):
    with pytest.raises(RuntimeError):
        P.token_records([4], [{4: lp(value)}])


def test_logprob_keeps_actual_target_when_not_top_one():
    assert P.token_records([4], [{3: lp(-0.1), 4: lp(-8, 80)}]) == [
        {"token_id": 4, "logprob": -8.0, "rank": 80}
    ]


def test_calibration_detects_single_token_and_accumulated_error():
    a = [{"token_id": i, "logprob": -1.0} for i in range(20)]
    assert P.calibration(a, a)["status"] == "PASSED"
    b = [dict(r) for r in a]
    b[0]["logprob"] -= 0.11
    assert P.calibration(a, b)["status"] == "FAILED"
    b = [{**r, "logprob": r["logprob"] - 0.03} for r in a]
    result = P.calibration(a, b)
    assert result["status"] == "FAILED" and result["maximum_token_abs_difference"] < 0.1
    with pytest.raises(RuntimeError):
        P.calibration(a, b[:-1])


def test_incomplete_generation_is_not_eos_or_complete_episode():
    assert not P.H.response_complete([10] * 4096, "length", P.H.EOS_IDS)
    assert not P.H.response_complete([10], "stop", P.H.EOS_IDS)
    assert P.H.response_complete([10, P.EOS], "stop", P.H.EOS_IDS)
    assert P.budget()["full_trajectories"] == P.budget()["training_updates"] == 0


def test_logprob_observation_overrides_only_legacy_none_field():
    old = P.H.sampling_kwargs({"seed": 17})
    new = P.natural_kwargs({"seed": 17})
    assert old["logprobs"] is None and new["logprobs"] == 1
    assert {k: v for k, v in old.items() if k != "logprobs"} == {
        k: v for k, v in new.items() if k != "logprobs"
    }
    assert SimpleNamespace(**new).seed == 17

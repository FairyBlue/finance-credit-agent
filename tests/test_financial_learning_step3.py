"""The conditional MC run must use real training evidence and fixed resources."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "scripts/diagnostics"
if str(DIRECTORY) not in sys.path:
    sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location(
    "financial_step3_test", DIRECTORY / "run_financial_learning_step3.py"
)
STEP3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STEP3)


def write(path, value, *, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in value) if jsonl else json.dumps(value))


@pytest.mark.parametrize(
    ("action", "delta", "changed", "expected"),
    [
        ("calculate", 0.001, True, "PASS"),
        ("calculate", 0.0, True, "UNRESOLVED_SIGNAL"),
        ("solve", 0.001, True, "UNRESOLVED_SIGNAL"),
        ("calculate", 0.001, False, "UNRESOLVED_SIGNAL"),
    ],
)
def test_gate_requires_responsive_genuine_calculation_update(
    tmp_path, action, delta, changed, expected
):
    write(
        tmp_path / "baseline/sft_panel/trajectory_results.jsonl",
        [{"task_id": "a", "business_correct": bool(i % 2)} for i in range(4)],
        jsonl=True,
    )
    rd = tmp_path / "training/rloo_lr2e6/round_000"
    write(
        rd / "train/summary.json",
        {
            "trainable_sha256_before": "before",
            "trainable_sha256_after": "after" if changed else "before",
        },
    )
    write(
        rd / "train/optimizer_updates.jsonl",
        [{"status": "UPDATED", "gradient_norm_before_clip": 0.2, "trajectories": ["a/rep0"]}],
        jsonl=True,
    )
    write(
        rd / "collect/requests.jsonl",
        [
            {
                "request_key": "a/rep0/call3",
                "trajectory_key": "a/rep0",
                "role": "parent",
                "action": {"name": action},
            }
        ],
        jsonl=True,
    )
    write(
        rd / "train/request_probability_deltas.jsonl",
        [
            {
                "request_key": "a/rep0/call3",
                "trajectory_key": "a/rep0",
                "phase": "calculation",
                "actor_advantage": 1.0,
                "old_mean_response_logprob": -0.5,
                "new_mean_response_logprob": -0.5 + delta,
                "mean_response_logprob_delta": delta,
            }
        ],
        jsonl=True,
    )
    result = STEP3.evidence_gate(tmp_path)
    assert result["status"] == expected
    assert result["dev_or_holdout_score_used"] is False


@pytest.mark.parametrize(
    ("mean", "scope"),
    [
        (10_000, "PANEL_K4_AND_REPORT_HOLDOUT"),
        (26_318, "PANEL_K4_ONLY"),
        (50_000, "INSUFFICIENT_FIXED_ENDPOINT_ALLOWANCE"),
    ],
)
def test_full_endpoint_scope_is_chosen_from_pretraining_costs(tmp_path, mean, scope):
    write(
        tmp_path / "baseline/sft_panel/summary.json",
        {"total_tokens": mean * 64, "complete_trajectories": 64},
    )
    result = STEP3.evaluation_contract(tmp_path)
    assert result["scope"] == scope
    assert result["selected_before_any_stage3_model_generation"] is True
    assert result["dev_evaluated"] is False


def test_namespaced_parent_quota_and_auxiliary_rng_are_distinct():
    seeds = [
        STEP3.stream_seed(arm, index, purpose)
        for arm in STEP3.ARMS
        for index in range(4)
        for purpose in ("parent", "quota", "auxiliary")
    ]
    assert len(seeds) == len(set(seeds))
    assert STEP3.stream_seed("rloo", 0, "parent") == STEP3.stream_seed("rloo", 0, "parent")


def test_protocol_cannot_be_rewritten_after_freeze(tmp_path):
    path = STEP3.freeze_protocol(tmp_path)
    original = path.read_bytes()
    STEP3.freeze_protocol(tmp_path)
    assert path.read_bytes() == original
    changed = json.loads(path.read_text())
    changed["rules"]["gate"]["dev_or_holdout_improvement_required"] = True
    write(path, changed)
    with pytest.raises(RuntimeError, match="prespecified protocol changed"):
        STEP3.freeze_protocol(tmp_path)


def test_handoff_waits_for_holder_claim_to_clear_after_completion(tmp_path, monkeypatch):
    write(tmp_path / "state.json", {"status": "STEP2_COMPLETE", "attempt": 2})
    write(tmp_path / "completion.json", {"status": "STEP2_COMPLETE", "attempt": 2})
    write(tmp_path / "step2_decision.json", {"status": "COMPLETE"})
    source_root = tmp_path / "source"
    write(source_root / "runs/GPU_RESERVATION_CURRENT.json", {"directory": "holder"})
    monkeypatch.setattr(STEP3, "ROOT", source_root)
    states = iter([{"claim": {"controller": "still-exiting"}}, {"claim": None}])
    calls = []

    def status(*args, **kwargs):
        calls.append(args)
        return next(states)

    monkeypatch.setattr(STEP3.gpu, "request", status)
    monkeypatch.setattr(STEP3.time, "sleep", lambda seconds: None)
    STEP3.wait_stage2(tmp_path)
    assert len(calls) == 2

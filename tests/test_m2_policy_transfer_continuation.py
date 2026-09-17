"""Native request-cap observations remain incomplete during the bounded continuation."""

import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from src.agents.rollout_seed import derive_request_seed

SPEC = importlib.util.spec_from_file_location(
    "transfer_continuation",
    Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_policy_transfer_continuation.py",
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def fixture(cap=False):
    contexts = {
        "teacher_labels_included": False,
        "short": [],
        "full": [
            {
                "id": f"window-{i}",
                "prefix": [],
                "expected_start_session_index": 0,
                "env_kwargs": {"initial_cash": 1000000, "screening_calibration_sha256": "a" * 64},
            }
            for i in range(12)
        ],
    }
    rows, requests = [], []
    for index, context in enumerate(contexts["full"]):
        for sibling in range(5):
            is_cap = cap and index == sibling == 0
            count = 128 if is_cap else 20
            lane = "greedy" if sibling == 0 else f"sample{sibling - 1}"
            identity = {
                "step": 1,
                "sample_index": index,
                "rollout_n": max(0, sibling - 1),
                "validate": sibling == 0,
            }
            traces = [
                {
                    "step_index": i,
                    "action": "invalid" if is_cap else "hold",
                    "invalid_action": is_cap,
                    "done": not is_cap and i == count - 1,
                    "result_session_index": 0 if is_cap else i + 1,
                }
                for i in range(count)
            ]
            row = {
                "id": f"full-{context['id']}-{lane}",
                "context_id": context["id"],
                "context": context,
                "horizon": "full",
                "state_kind": "cash",
                "sampling": "greedy" if sibling == 0 else "sample",
                "identity": identity,
                "request_limit": 128,
                "requests": count,
                "max_response_tokens": 2,
                "max_prompt_tokens": 3,
                "response_tokens": count * 2,
                "prefix_fill_count": 0,
                "prefix_info": [],
                "takeover_equity": 1000000,
                "takeover_state": {"session_index": 0, "positions": {}},
                "steps_including_prefix": traces,
                "actions": dict(Counter(trace["action"] for trace in traces)),
                "invalid": count if is_cap else 0,
            }
            if is_cap:
                row.update(status="INCOMPLETE", failure="REQUEST_CAP")
            else:
                row.update(
                    status="COMPLETE",
                    final_state={"session_index": 20},
                    terminal_metrics={
                        "session_count": 21,
                        "terminal_score": -3,
                        "calibration_sha256": "a" * 64,
                    },
                    takeover_net_return=-0.01,
                )
            rows.append(row)
            for i, trace in enumerate(traces):
                requests.append(
                    {
                        "trajectory_id": row["id"],
                        "request_index": i,
                        "response_token_ids": [1, 2],
                        "prompt_token_ids": [3, 4, 5],
                        "finish_reason": "stop",
                        "trace": trace,
                        "sampling_params": {
                            "temperature": 0.0 if sibling == 0 else 1.0,
                            "top_p": 1.0,
                            "top_k": -1,
                            "repetition_penalty": 1.0,
                            "max_tokens": 4096,
                            "seed": derive_request_seed(42, identity, i),
                            "n": 1,
                        },
                    }
                )
    return (
        contexts,
        {"status": "OBSERVED", "action_protocol": entry.PROTOCOL, "grid": rows},
        requests,
    )


def write(directory, result, requests):
    entry.save(directory / "state.json", result)
    entry.save(directory / "result.json", result)
    (directory / "requests.jsonl").write_text(
        "".join(json.dumps(request) + "\n" for request in requests)
    )


@pytest.mark.parametrize("cap", [False, True])
def test_valid_native_observation_preserves_incomplete_without_terminal_imputation(tmp_path, cap):
    contexts, result, requests = fixture(cap)
    write(tmp_path, result, requests)
    before = (tmp_path / "result.json").read_bytes()
    validated = entry.validate_observed(tmp_path, contexts)
    assert validated["status"] == ("OBSERVED_WITH_INCOMPLETE" if cap else "COMPLETE")
    assert validated["observed"] == 60
    assert validated["complete"] == 60 - int(cap)
    assert validated["incomplete"] == int(cap)
    assert validated["terminal_outcomes_imputed"] is False
    assert (tmp_path / "result.json").read_bytes() == before
    if cap:
        assert "takeover_net_return" not in result["grid"][0]


@pytest.mark.parametrize(
    "damage",
    [
        "missing_row",
        "duplicate_row",
        "changed_context",
        "changed_seed",
        "short_budget",
        "fabricated_return",
        "fabricated_metrics",
        "false_terminal",
        "response_cap",
        "prompt_cap",
        "unknown_failure",
        "short_trace",
        "missing_request",
        "request_seed",
        "request_finish",
        "request_tokens",
        "token_count",
        "invalid_count",
        "state_disagrees",
    ],
)
def test_only_exact_native_request_cap_is_permitted(tmp_path, damage):
    contexts, result, requests = fixture(True)
    row = result["grid"][0]
    if damage == "missing_row":
        result["grid"].pop()
    elif damage == "duplicate_row":
        result["grid"][-1] = copy.deepcopy(row)
    elif damage == "changed_context":
        row["context"] = {**row["context"], "prefix": ["hold"]}
    elif damage == "changed_seed":
        row["identity"]["step"] = 8
    elif damage == "short_budget":
        row["requests"] = 127
    elif damage == "fabricated_return":
        row["takeover_net_return"] = 0.0
    elif damage == "fabricated_metrics":
        row["terminal_metrics"] = None
    elif damage == "false_terminal":
        row["steps_including_prefix"][-1]["done"] = True
    elif damage == "response_cap":
        row["failure"] = "RESPONSE_INCOMPLETE"
    elif damage == "prompt_cap":
        row["failure"] = "PROMPT_CAP"
    elif damage == "unknown_failure":
        row["failure"] = "OTHER"
    elif damage == "short_trace":
        row["steps_including_prefix"].pop()
    elif damage == "missing_request":
        requests.pop()
    elif damage == "request_seed":
        requests[0]["sampling_params"]["seed"] = -1
    elif damage == "request_finish":
        requests[0]["finish_reason"] = "length"
    elif damage == "request_tokens":
        requests[0]["response_token_ids"] = [1] * 4096
    elif damage == "token_count":
        row["response_tokens"] += 1
    elif damage == "invalid_count":
        row["invalid"] -= 1
    write(tmp_path, result, requests)
    if damage == "state_disagrees":
        entry.save(tmp_path / "state.json", {})
    with pytest.raises(RuntimeError):
        entry.validate_observed(tmp_path, contexts)


def test_existing_continuation_state_cannot_retry(tmp_path):
    entry.save(tmp_path / "state.json", {"status": "FAILED"})
    with pytest.raises(RuntimeError, match="no retry or resume"):
        entry.run(tmp_path)
    assert entry.CELL_IDS == ("m1", "m2_eta1", "m2_eta025")


def test_existing_continuation_directory_cannot_be_overwritten(tmp_path):
    with pytest.raises(RuntimeError, match="no overwrite"):
        entry.prepare(tmp_path)

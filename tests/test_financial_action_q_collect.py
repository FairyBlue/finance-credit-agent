"""Post-action Q sampling preserves the actual action, policy and K4 budget."""

import copy
import hashlib
from types import SimpleNamespace

import pytest
from test_financial_mc_probe import entry as mc_entry

from src.agents import financial_action_q_collect as probe
from src.agents.financial_evidence_env import FinancialEvidenceEnv
from src.agents.same_prefix_online_collect import canonical


@pytest.fixture
def entries():
    original = mc_entry.__wrapped__()
    task = original["task"]
    task["split"] = "train"
    env = FinancialEvidenceEnv(task)
    messages, history = env.initial_messages(), []
    for doc in ("mandate", "report"):
        raw = canonical(dict(name="read_document", arguments=dict(doc_id=doc)))
        result = env.step(raw)
        history.append(raw)
        messages.extend(
            [dict(role="assistant", content=raw), dict(role="tool", content=canonical(result))]
        )
    raw = canonical(dict(name="calculate", arguments=dict(expression="25/100")))
    expected = env.step(raw)
    result = []
    for replica in range(4):
        key = f"{task['task_id']}/rep{replica}"
        anchor = dict(
            task_id=task["task_id"],
            trajectory_key=key,
            replicate_index=replica,
            request_key=f"{key}/call2",
            call_index=2,
            messages=copy.deepcopy(messages),
            prefix_snapshot=dict(calls_used=2, action_history=list(history)),
            role="parent",
            response_text=raw,
            action=dict(name="calculate"),
            tool_result=copy.deepcopy(expected),
        )
        parent = dict(
            task_id=task["task_id"],
            trajectory_key=key,
            replicate_index=replica,
            role="parent",
            terminal_reward=0.98 if replica < 2 else -0.02,
            status="COMPLETE",
        )
        result.append(dict(task=task, anchor=anchor, parent=parent))
    return result


def test_replay_executes_exact_parent_action_and_keeps_invalid_json(entries):
    entry = entries[0]
    before = copy.deepcopy(entry)
    state = probe.restore_post_action(entry)
    assert state["env"].calls_used == 3
    assert state["messages"][-2]["content"] == entry["anchor"]["response_text"]
    assert entry == before
    invalid = copy.deepcopy(entry)
    invalid["anchor"].update(response_text='{"arguments": {', action=None)
    restored = FinancialEvidenceEnv(entry["task"])
    for raw in invalid["anchor"]["prefix_snapshot"]["action_history"]:
        restored.step(raw)
    invalid["anchor"]["tool_result"] = restored.step(invalid["anchor"]["response_text"])
    state = probe.restore_post_action(invalid)
    assert state["env"].calls_used == 3
    assert state["messages"][-2]["content"] == '{"arguments": {'
    assert state["messages"][-1]["content"] == canonical(invalid["anchor"]["tool_result"])


def test_selection_and_q_do_not_resample_action_or_select_on_parent_reward(entries):
    tasks = [entries[0]["task"]]
    parents = [e["parent"] for e in entries]
    requests = [e["anchor"] for e in entries]
    first = probe.select_groups(tasks, requests, parents)
    changed = copy.deepcopy(parents)
    for parent in changed:
        parent["terminal_reward"] = 500
    second = probe.select_groups(tasks, requests, changed)
    assert first[0]["unsupported_reason"] is None
    assert [e["anchor"]["request_key"] for e in first[0]["entries"]] == [
        e["anchor"]["request_key"] for e in second[0]["entries"]
    ]
    row = probe.action_q_record(
        entries[0], dict(terminal_reward=-0.02, trajectory_key="aux"), "policy", complete=True
    )
    assert row["q_mean"] == pytest.approx(0.48)
    assert row["parent_action_executed_once"] and not row["parent_action_resampled"]
    assert not row["auxiliary_enters_actor"]
    assert probe.stable_seed(42, requests[0]["request_key"], 3) != probe.stable_seed(
        42, requests[1]["request_key"], 3
    )


def test_full_k4_budget_reservation_prevents_any_partial_group_gpu_load(
    entries, tmp_path, monkeypatch
):
    from src.agents import mandate_repair_train

    monkeypatch.setattr(mandate_repair_train, "adapter_digest", lambda _: "test-policy")
    monkeypatch.setattr(probe, "_load_actor", lambda *_: pytest.fail("must not load GPU"))
    parents = tmp_path / "parents"
    parents.mkdir()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    data = tmp_path / "train.jsonl"
    data.write_text(canonical(entries[0]["task"]) + "\n")
    (parents / "requests.jsonl").write_text("".join(canonical(e["anchor"]) + "\n" for e in entries))
    (parents / "trajectory_results.jsonl").write_text(
        "".join(canonical(e["parent"]) + "\n" for e in entries)
    )
    (parents / "collection_started.json").write_text(
        canonical(
            dict(
                adapter_path=str(adapter),
                adapter_sha256="test-policy",
                base_model_path=str(tmp_path),
                environment="financial_evidence",
                temperature=1.0,
                decode_mode="native",
                data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
            )
        )
    )
    states = [probe.restore_post_action(e) for e in entries]
    required = probe.group_reservation(states, 16384)
    args = SimpleNamespace(
        data=data,
        parents=parents,
        adapter=adapter,
        base_model=tmp_path,
        output=tmp_path / "aux",
        seed=123,
        max_context_tokens=16384,
        token_budget=required - 1,
    )
    summary = probe.run(args)
    assert summary["status"] == "BUDGET_EXHAUSTED"
    assert summary["model_requests"] == summary["total_tokens"] == 0
    rows = probe.read_rows(args.output / "action_q_values.jsonl")
    assert len(rows) == 4
    assert all(r["status"] == "UNSUPPORTED" and r["q_mean"] is None for r in rows)

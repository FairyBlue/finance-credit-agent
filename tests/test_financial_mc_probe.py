"""MC state baselines must start before the parent's sampled action."""

import copy
import json
from types import SimpleNamespace

import pytest

from src.agents.financial_evidence_env import FinancialEvidenceEnv
from src.agents.financial_mc_probe import restore_branch, select_anchors, stable_seed
from src.agents.mandate_repair_verifier import solve_explicit
from src.agents.same_prefix_online_collect import canonical


@pytest.fixture
def entry():
    limits = dict(cash_min=100.0, exposure_max=1.0, symbol_max={"A": 0.4},
                  sector_max={}, sectors={}, cannot_sell=[], cannot_buy=[])
    task = dict(
        task_id="mc-train-1", base_scenario_id="mc-base-1", task_type="conditional_repair",
        as_of="2025-01-02T16:00:00+00:00",
        account=dict(cash=1000.0, positions={"A": 80.0}, initial_equity=9000.0),
        quotes={"A": 100.0}, execution=dict(commission_bps=10.0, slippage_bps=5.0),
        condition=dict(company="ACME", report_year="2018", question="2018 expense/revenue?",
                       period="2018", unit="ratio", comparison="greater", threshold=0.2,
                       controlled_symbol="A", strict_cap=0.4, relaxed_cap=0.8),
        documents=[
            dict(doc_id=key, title=title, text=text,
                 available_at="2020-01-01T00:00:00+00:00",
                 effective_at="2020-01-01T00:00:00+00:00")
            for key, title, text in (
                ("mandate", "Current conditional mandate", "Conditional cap."),
                ("report", "ACME annual report 2018, excerpt", "Expense25 revenue100."),
                ("copy", "ACME annual report 2018, excerpt copy", "Expense25 revenue100."),
            )
        ],
        gold=dict(constraints=limits, required_citations=["mandate"],
                  evidence_groups=[["report", "copy"]], metric_value=0.25,
                  metric_period="2018", metric_unit="ratio", triggered=True),
    )
    task["gold"]["oracle"] = solve_explicit(
        task["account"], task["quotes"], limits, execution=task["execution"],
    )
    env = FinancialEvidenceEnv(task)
    messages = env.initial_messages()
    first = canonical(dict(name="read_document", arguments=dict(doc_id="mandate")))
    response = env.step(first)
    messages += [dict(role="assistant", content=first),
                 dict(role="tool", content=canonical(response))]
    anchor = dict(task_id=task["task_id"], trajectory_key="mc-train-1/rep0",
                  request_key="mc-train-1/rep0/call1", role="parent", messages=messages,
                  prefix_snapshot=dict(calls_used=1, action_history=[first]),
                  action=dict(name="read_document", arguments=dict(doc_id="report")),
                  response_text="RECORDED PARENT ACTION MUST NEVER EXECUTE", parent_reward=-123)
    selection = dict(request_key=anchor["request_key"], selected=True,
                     selection_probability=0.3, pre_action_selection=True,
                     selection_policy="test_public_prefix_only", observed_calls_used=1)
    return dict(task=task, anchor=anchor, selection=selection)


def test_restore_keeps_parent_action_unexecuted_and_does_not_mutate_prefix(entry):
    original = copy.deepcopy(entry)
    state = restore_branch(entry, 0)
    assert state["env"].calls_used == 1
    assert state["env"].seen_documents == {"mandate"}
    assert state["messages"] == entry["anchor"]["messages"]
    assert "report" not in state["env"].seen_documents
    assert state["trajectory_key"] != entry["anchor"]["trajectory_key"]
    assert entry == original


def test_restore_rejects_an_extra_parent_action_in_messages(entry):
    entry["anchor"]["messages"].append(dict(role="assistant", content="parent action"))
    with pytest.raises(ValueError, match="before the parent action"):
        restore_branch(entry, 0)


def test_explicit_selection_never_uses_parent_action_or_outcome(entry, tmp_path):
    parents = tmp_path / "parents"
    parents.mkdir()
    (parents / "requests.jsonl").write_text(canonical(entry["anchor"]) + "\n")
    selected = tmp_path / "selections.jsonl"
    selected.write_text(canonical(entry["selection"]) + "\n")
    anchors, count = select_anchors([entry["task"]], parents, selected)
    assert count == 1 and anchors[0]["anchor"]["request_key"] == entry["anchor"]["request_key"]
    entry["selection"]["pre_action_selection"] = False
    selected.write_text(canonical(entry["selection"]) + "\n")
    with pytest.raises(ValueError, match="precedes parent action"):
        select_anchors([entry["task"]], parents, selected)


def args_for(entry, tmp_path, monkeypatch, budget=2_000_000):
    from src.agents import mandate_repair_train

    monkeypatch.setattr(mandate_repair_train, "adapter_digest", lambda _: "mc-test-policy")
    parents, adapter = tmp_path / "parents", tmp_path / "adapter"
    parents.mkdir()
    adapter.mkdir()
    (parents / "collection_started.json").write_text(canonical(dict(
        adapter_path=str(adapter), adapter_sha256="mc-test-policy",
        environment="financial_evidence", temperature=1.0, decode_mode="native",
    )))
    (parents / "requests.jsonl").write_text(canonical(entry["anchor"]) + "\n")
    # Deliberately no trajectory_results.jsonl: the worker must not read parent outcomes.
    selections = tmp_path / "selections.jsonl"
    selections.write_text(canonical(entry["selection"]) + "\n")
    data = tmp_path / "train.jsonl"
    data.write_text(canonical(entry["task"]) + "\n")
    return SimpleNamespace(data=data, parents=parents, selections=selections,
                           base_model=tmp_path, adapter=adapter, output=tmp_path / "probe",
                           seed=42, repetitions=2, token_budget=budget,
                           max_context_tokens=16384, max_anchors=32)


def fake_actor(entry, monkeypatch, probe, *, failure=False):
    actions = [
        dict(name="read_document", arguments=dict(doc_id="report")),
        dict(name="calculate", arguments=dict(expression="25/100")),
        dict(name="solve", arguments=dict(constraints=entry["task"]["gold"]["constraints"])),
        dict(name="submit", arguments=dict(
            plan_id="plan_1", citations=["mandate", "report"],
            metric=dict(value=0.25, period="2018", unit="ratio", triggered=True),
        )),
    ]

    class Actor:
        cursor = 0
        params = []

        def generate(self, prompts, params, **_):
            self.params.extend(params)
            text = canonical(actions[self.cursor])
            self.cursor += 1
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[7, 8])],
                                    test_text=None if failure else text) for _ in prompts]

    actor = Actor()
    tokenizer = SimpleNamespace(apply_chat_template=lambda *_, **__: [1, 2, 3])
    monkeypatch.setattr(
        probe, "_load_actor", lambda *_: (actor, tokenizer, lambda seed: seed, None),
    )
    monkeypatch.setattr(probe, "decoded_generation", lambda output, *_: dict(
        generation_status="MODEL_ERROR" if failure else "COMPLETE",
        response_text=output.test_text, response_token_ids=[] if failure else [7, 8],
    ))
    return actor


def test_multiturn_mc_baseline_uses_independent_complete_auxiliary_rewards(
    entry, tmp_path, monkeypatch,
):
    from src.agents import financial_mc_probe as probe

    args = args_for(entry, tmp_path, monkeypatch)
    actor = fake_actor(entry, monkeypatch, probe)
    summary = probe.run(args)
    assert summary["status"] == "COMPLETE" and summary["completed_baselines"] == 1
    assert summary["model_requests"] == 8 and summary["total_tokens"] == 40
    assert len(set(actor.params)) == 8
    assert stable_seed(42, entry["anchor"]["request_key"], 0, 1) != stable_seed(
        42, entry["anchor"]["request_key"], 1, 1,
    )
    row = json.loads((args.output / "state_value_baselines.jsonl").read_text())
    assert row["m"] == 2 and len(row["aux_returns"]) == 2
    assert row["mc_baseline"] == sum(row["aux_returns"]) / 2
    assert row["mc_baseline"] > 0.9 and row["mc_baseline"] != -123
    assert row["parent_action_excluded"] and row["parent_return_excluded"]
    assert row["aux_rng_independent"] and row["continuation_policy"] == "current_policy"
    assert json.loads((args.output / "progress.json").read_text()) == summary
    assert (args.output / "state_value_baselines.jsonl").read_bytes() == (
        args.output / "mc_baselines.jsonl"
    ).read_bytes()


def test_failed_validation_keeps_raw_generated_token_cost_and_no_baseline(
    entry, tmp_path, monkeypatch,
):
    from src.agents import financial_mc_probe as probe

    args = args_for(entry, tmp_path, monkeypatch)
    fake_actor(entry, monkeypatch, probe, failure=True)
    summary = probe.run(args)
    assert summary["status"] == "COMPLETE_WITH_FAILURES"
    assert summary["completed_baselines"] == 0 and summary["incomplete_baselines"] == 1
    assert summary["response_tokens"] == 4 and summary["total_tokens"] == 10
    row = json.loads((args.output / "state_value_baselines.jsonl").read_text())
    assert row["mc_baseline"] is None and row["aux_returns"] == [None, None]


def test_zero_or_insufficient_pair_budget_does_not_load_gpu(entry, tmp_path, monkeypatch):
    from src.agents import financial_mc_probe as probe

    args = args_for(entry, tmp_path, monkeypatch, budget=0)
    monkeypatch.setattr(probe, "_load_actor", lambda *_: pytest.fail("must not load GPU"))
    summary = probe.run(args)
    assert summary["status"] == "BUDGET_EXHAUSTED"
    assert summary["model_requests"] == summary["total_tokens"] == 0
    assert (args.output / "state_value_baselines.jsonl").read_text() == ""

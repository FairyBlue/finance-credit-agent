"""A response intervention must preserve honest recovery and reject noisy gates."""

import copy
import json
from types import SimpleNamespace

import pytest

from src.agents.financial_credit_probe import gate_result, restore_branch, select_anchors
from src.agents.financial_evidence_env import FinancialEvidenceEnv
from src.agents.mandate_repair_verifier import solve_explicit
from src.agents.same_prefix_online_collect import canonical


@pytest.fixture
def entry():
    limits = dict(
        cash_min=100.0,
        exposure_max=1.0,
        symbol_max={"A": 0.4},
        sector_max={},
        sectors={},
        cannot_sell=[],
        cannot_buy=[],
    )
    task = dict(
        task_id="fresh-1-v0",
        base_scenario_id="fresh-base-1",
        task_type="conditional_repair",
        as_of="2025-01-02T16:00:00+00:00",
        account=dict(cash=1000.0, positions={"A": 80.0}, initial_equity=9000.0),
        quotes={"A": 100.0},
        execution=dict(commission_bps=10.0, slippage_bps=5.0),
        condition=dict(
            company="ACME",
            report_year="2018",
            question="2018 expense/revenue?",
            period="2018",
            unit="ratio",
            comparison="greater",
            threshold=0.2,
            controlled_symbol="A",
            strict_cap=0.4,
            relaxed_cap=0.8,
        ),
        documents=[
            dict(
                doc_id=key,
                title=title,
                text=text,
                available_at="2020-01-01T00:00:00+00:00",
                effective_at="2020-01-01T00:00:00+00:00",
            )
            for key, title, text in (
                ("mandate", "Current conditional mandate", "Conditional cap."),
                ("report", "ACME annual report 2018, financial excerpt A", "Expense25 revenue100."),
                (
                    "copy",
                    "ACME annual report 2018, financial excerpt A (copy)",
                    "Expense25 revenue100.",
                ),
            )
        ],
        gold=dict(
            constraints=limits,
            required_citations=["mandate"],
            evidence_groups=[["report", "copy"]],
            metric_value=0.25,
            metric_period="2018",
            metric_unit="ratio",
            triggered=True,
        ),
    )
    task["gold"]["oracle"] = solve_explicit(
        task["account"], task["quotes"], limits, execution=task["execution"]
    )
    env = FinancialEvidenceEnv(task)
    messages = env.initial_messages()
    first = canonical(dict(name="read_document", arguments=dict(doc_id="mandate")))
    returned = env.step(first)
    messages.extend(
        [dict(role="assistant", content=first), dict(role="tool", content=canonical(returned))]
    )
    action = dict(name="read_document", arguments=dict(doc_id="report"))
    raw = canonical(action)
    anchor = dict(
        task_id=task["task_id"],
        trajectory_key="fresh-1-v0/rep0",
        request_key="fresh-1-v0/rep0/call1",
        role="parent",
        action=action,
        messages=messages,
        response_text=raw,
        tool_result=env.step(raw),
        prefix_snapshot=dict(calls_used=1, action_history=[first]),
    )
    return dict(task=task, anchor=anchor, base_scenario_id=task["base_scenario_id"])


def test_single_response_mask_keeps_same_later_read_and_duplicate_available(entry):
    original = copy.deepcopy(entry)
    factual, masked, placebo = [
        restore_branch(entry, condition, 0) for condition in ("factual", "masked", "placebo")
    ]
    assert {state["env"].calls_used for state in (factual, masked, placebo)} == {2}
    assert "report" in factual["env"].seen_documents
    assert "report" not in masked["env"].seen_documents
    assert "report" in placebo["env"].seen_documents
    assert json.loads(masked["messages"][-1]["content"])["information_unavailable"] is True
    placebo_response = json.loads(placebo["messages"][-1]["content"])
    assert placebo_response["text"] == entry["anchor"]["tool_result"]["text"]
    assert "available_at" not in placebo_response and "effective_at" not in placebo_response
    reread = masked["env"].step(entry["anchor"]["response_text"])
    assert reread == entry["anchor"]["tool_result"]
    duplicate = masked["env"].step(dict(name="read_document", arguments=dict(doc_id="copy")))
    assert duplicate["ok"] and duplicate["text"] == reread["text"]
    assert entry == original


def test_selector_uses_public_report_not_terminal_outcome(entry, tmp_path):
    rows = [dict(entry["anchor"], parent_success=False)]
    (tmp_path / "requests.jsonl").write_text("\n".join(canonical(row) for row in rows) + "\n")
    selected, eligible = select_anchors([entry["task"]], tmp_path)
    assert eligible == 1 and selected[0]["anchor"]["request_key"] == entry["anchor"]["request_key"]


def local(base, factual=(1, 1), masked=(1, 1), placebo=(1, 1)):
    return dict(
        base_scenario_id=base,
        status="COMPLETE",
        factual_target_values=list(factual),
        masked_target_values=list(masked),
        placebo_target_values=list(placebo),
    )


def test_gate_requires_multiple_repeated_effects_beyond_placebo():
    rows = [local(str(i), masked=(0, 0)) for i in range(3)]
    rows += [local(str(i)) for i in range(3, 8)]
    assert gate_result(rows)["status"] == "PASS"
    # One lucky replicate per question does not demonstrate reproducibility.
    for row in rows[:3]:
        row["masked_target_values"] = [0, 1]
    assert gate_result(rows)["status"] == "FAIL"
    # Nor does equally large sensitivity to cosmetic metadata.
    for row in rows[:3]:
        row["masked_target_values"] = [0, 0]
        row["placebo_target_values"] = [0, 0]
    assert gate_result(rows)["status"] == "FAIL"


def test_zero_probe_is_scoped_negative_and_small_probe_is_inconclusive():
    rows = [local(str(i)) for i in range(8)]
    result = gate_result(rows)
    assert result["status"] == "FAIL"
    assert result["reason"] == "NO_DETECTABLE_LOCAL_BUSINESS_SIGNAL_AT_THIS_PROBE_SCALE"
    assert result["no_true_action_advantage_claim"] is True
    assert gate_result(rows[:3])["status"] == "INCONCLUSIVE"
    assert gate_result([rows[0]] * 8)["status"] == "INCONCLUSIVE"
    assert gate_result(rows, conditions=("factual", "masked"))["status"] == "INCONCLUSIVE"


def probe_args(entry, tmp_path, monkeypatch, budget=2_000_000):
    from src.agents import mandate_repair_train

    monkeypatch.setattr(mandate_repair_train, "adapter_digest", lambda _: "test-policy")
    parents = tmp_path / "parents"
    parents.mkdir()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (parents / "collection_started.json").write_text(
        canonical(
            dict(
                adapter_path=str(adapter),
                adapter_sha256="test-policy",
                environment="financial_evidence",
            )
        )
    )
    (parents / "requests.jsonl").write_text(canonical(entry["anchor"]) + "\n")
    (parents / "trajectory_results.jsonl").write_text(
        canonical(
            dict(
                trajectory_key=entry["anchor"]["trajectory_key"],
                status="COMPLETE",
                verification=dict(business_correct=True),
                terminal_reward=0.98,
            )
        )
        + "\n"
    )
    data = tmp_path / "tasks.jsonl"
    data.write_text(canonical(entry["task"]) + "\n")
    return SimpleNamespace(
        data=data,
        parents=parents,
        adapter=adapter,
        base_model=tmp_path,
        output=tmp_path / "probe",
        seed=42,
        token_budget=budget,
        max_context_tokens=16384,
        repetitions=2,
        max_anchors=1,
        minimum_gate_anchors=8,
        conditions=["factual", "masked", "placebo"],
        local_credit_arm="m3",
    )


def test_multiturn_probe_replaces_progress_and_finishes_after_legitimate_reread(
    entry, tmp_path, monkeypatch
):
    from src.agents import financial_credit_probe as probe

    args = probe_args(entry, tmp_path, monkeypatch)
    actions = [
        dict(name="read_document", arguments=dict(doc_id="report")),
        dict(name="calculate", arguments=dict(expression="25/100")),
        dict(name="solve", arguments=dict(constraints=entry["task"]["gold"]["constraints"])),
        dict(
            name="submit",
            arguments=dict(
                plan_id="plan_1",
                citations=["mandate", "report"],
                metric=dict(value=0.25, period="2018", unit="ratio", triggered=True),
            ),
        ),
    ]

    class Actor:
        cursor = 0

        def generate(self, prompts, params, **kwargs):
            action = canonical(actions[self.cursor])
            self.cursor += 1
            return [action for _ in prompts]

    tokenizer = SimpleNamespace(apply_chat_template=lambda *_, **__: [1])
    monkeypatch.setattr(
        probe, "_load_actor", lambda *_: (Actor(), tokenizer, lambda seed: seed, None)
    )
    monkeypatch.setattr(
        probe,
        "decoded_generation",
        lambda output, *_: dict(
            generation_status="COMPLETE",
            response_text=output,
            response_token_ids=[1],
        ),
    )
    summary = probe.run(args)
    assert summary["status"] == "COMPLETE" and summary["model_requests"] == 24
    assert json.loads((args.output / "progress.json").read_text()) == summary
    rows = [
        json.loads(line) for line in (args.output / "local_credit.jsonl").read_text().splitlines()
    ]
    assert rows[0]["factual_target_values"] == rows[0]["masked_target_values"] == [1.0, 1.0]
    assert rows[0]["unscaled_local_difference"] == 0


def test_zero_auxiliary_tail_does_not_load_gpu(entry, tmp_path, monkeypatch):
    from src.agents import financial_credit_probe as probe

    args = probe_args(entry, tmp_path, monkeypatch, budget=0)

    def forbidden(*_):
        raise AssertionError("GPU must not load when no full paired anchor is affordable")

    monkeypatch.setattr(probe, "_load_actor", forbidden)
    summary = probe.run(args)
    assert summary["status"] == "BUDGET_EXHAUSTED"
    assert summary["model_requests"] == summary["total_tokens"] == 0
    assert (args.output / "local_credit.jsonl").read_text() == ""

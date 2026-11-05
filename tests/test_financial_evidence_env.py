"""Financial conditions must be correct even when a portfolio happens to be safe."""

import copy
from types import SimpleNamespace

import pytest

from src.agents.financial_evidence_env import FinancialEvidenceEnv
from src.agents.mandate_repair_collect import _safe_result, collect_groups, environment_factory
from src.agents.mandate_repair_env import MandateRepairEnv
from src.agents.mandate_repair_metrics import business_score, local_difference
from src.agents.mandate_repair_train import configure_request_checkpointing, recipe
from src.agents.mandate_repair_verifier import solve_explicit


@pytest.fixture
def task():
    constraints = dict(
        cash_min=1000.0, exposure_max=0.85, symbol_max={"A": 0.4, "B": 0.6},
        sector_max={}, sectors={}, cannot_sell=[], cannot_buy=[],
    )
    row = dict(
        task_id="financial-check", task_type="conditional_mixed",
        as_of="2025-01-02T16:00:00+00:00",
        account=dict(cash=100.0, positions={"A": 80.0, "B": 10.0}, initial_equity=9100.0),
        quotes={"A": 100.0, "B": 100.0},
        execution=dict(commission_bps=10.0, slippage_bps=5.0),
        condition=dict(
            company="ACME", report_year="2018", question="2018 expense / revenue?",
            period="2018", unit="ratio", comparison="greater", threshold=0.2,
            controlled_symbol="A", strict_cap=0.4, relaxed_cap=0.8,
        ),
        documents=[dict(
            doc_id=identifier, title=title,
            available_at="2020-01-01T00:00:00+00:00",
            effective_at="2020-01-01T00:00:00+00:00", text=text,
        ) for identifier, title, text in (
            ("mandate", "Current conditional mandate", "Cap A at0.4 if expense/revenue >0.2."),
            ("report", "ACME annual report 2018", "Expense 25; revenue100 in2018."),
            ("copy", "ACME annual report 2018 duplicate", "Expense25; revenue100 in2018."),
            ("wrong", "OTHER annual report 2017", "Other company/period evidence."),
        )],
    )
    row["gold"] = dict(
        constraints=constraints, required_citations=["mandate"],
        evidence_groups=[["report", "copy"]], metric_value=0.25, metric_period="2018",
        metric_unit="ratio", triggered=True,
        oracle=solve_explicit(
            row["account"], row["quotes"], constraints, execution=row["execution"]
        ),
    )
    return row


def completed(task, *, metric=None, source="report", mask=False, citations=None):
    env = FinancialEvidenceEnv(task)
    env.step(dict(name="read_document", arguments=dict(doc_id="mandate")))
    env.step(dict(name="read_document", arguments=dict(doc_id=source)), intervention=mask)
    plan = env.step(dict(name="solve", arguments=dict(constraints=task["gold"]["constraints"])))
    env.step(dict(name="submit", arguments=dict(
        plan_id=plan["plan_id"], citations=citations or ["mandate", source],
        metric=metric or dict(value=0.25, period="2018", unit="ratio", triggered=True),
    )))
    return _safe_result(dict(
        env=env, task_id=task["task_id"], trajectory_key="financial/rep0", replicate_index=0,
        request_keys=[], failure=None,
    ))


def test_correct_financial_claim_and_alternate_source_succeed(task):
    for source in ("report", "copy"):
        result = completed(task, source=source)
        assert result["success"] is True
        assert result["metric_correct"] is True and result["portfolio_correct"] is True
        assert business_score(result) == 1.0


@pytest.mark.parametrize("field,value", [
    ("value", 0.1), ("period", "2017"), ("unit", "percent"), ("triggered", False),
])
def test_wrong_financial_claim_fails_despite_legitimate_conservative_portfolio(task, field, value):
    metric = dict(value=0.25, period="2018", unit="ratio", triggered=True)
    metric[field] = value
    result = completed(task, metric=metric)
    assert result["verification"]["business_feasible"] is True
    assert result["verification"]["portfolio_correct"] is True
    assert result["verification"]["business_correct"] is False
    assert result["success"] is False and business_score(result) == 0.0


def test_citation_loss_alone_does_not_create_business_local_credit(task):
    parent, masked = completed(task), completed(task, mask=True)
    assert parent["success"] is True and masked["success"] is False
    assert masked["business_correct"] is True
    assert local_difference(parent, [masked], "business")["unscaled_local_difference"] == 0
    assert "missing_sufficient_financial_evidence" in masked["verification"]["evidence_errors"]


def test_new_environment_has_ten_calls_old_remains_eight(task, tmp_path):
    factory = environment_factory("financial_evidence")
    env = factory(task)
    for _ in range(9):
        env.step(dict(name="calculate", arguments=dict(expression="1+1")))
    assert not env.done and env.calls_used == 9
    env.step(dict(name="calculate", arguments=dict(expression="1+1")))
    assert env.done and env.calls_used == 10
    old = MandateRepairEnv(task)
    assert old.visible()["maximum_tool_calls"] == 8
    result = collect_groups(
        [task], factory, None, None, None, tmp_path,
        seed=43, temperature=1.0, repetitions=4, max_response=512,
        token_budget=4 * 10 * 16384 - 1, max_groups=None, model_path="base",
        max_context=16384,
    )
    assert result["status"] == "BUDGET_EXHAUSTED" and result["model_requests"] == 0
    assert result["worst_case_group_tokens"] == 4 * 10 * 16384


def test_mc_selection_is_persisted_before_parent_generation(task, tmp_path, monkeypatch):
    import json

    selection_path = tmp_path / "prefix_selections.jsonl"

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2]

        def decode(self, ids, **kwargs):
            return "public prefix"

    class Model:
        checked = 0

        def generate(self, prompts, params, **kwargs):
            rows = [json.loads(line) for line in selection_path.read_text().splitlines()]
            assert len(rows) == self.checked + len(prompts)
            assert all(row["pre_action_selection"] for row in rows)
            assert not any(row["parent_outcome_used"] for row in rows)
            self.checked += len(prompts)
            return [None] * len(prompts)

    response = '{"name":"calculate","arguments":{"expression":"1+1"}}'

    def decode(output, ids, maximum):
        return dict(
            generation_status="COMPLETE", generation_failure_reason=None,
            response_text=response, response_token_ids=[3], response_logprobs=[-1.0],
            response_logprobs_complete=True, action=json.loads(response),
        )

    monkeypatch.setattr("src.agents.mandate_repair_collect.decoded_generation", decode)
    model = Model()
    result = collect_groups(
        [task], environment_factory("financial_evidence"), model, Tokenizer(),
        lambda seed: None, tmp_path, seed=42, temperature=1.0, repetitions=4,
        max_response=512, token_budget=4 * 10 * 16384, max_groups=None,
        model_path="base", max_context=16384, mc_selection="priority",
        mc_selection_budget=327680, mc_max_anchors=1,
    )
    assert result["complete_trajectories"] == 4
    assert model.checked == 40
    assert result["mc_selected_anchors"] <= 1


def test_public_observations_hide_gold_and_diagnostic_facts_are_explicit(task):
    standard = FinancialEvidenceEnv(task)
    initial = standard.initial_messages()
    assert "gold" not in standard.visible()
    assert "metric_value" not in initial[1]["content"]
    supplied = FinancialEvidenceEnv(task, "oracle_facts")
    visible = supplied.visible()
    assert visible["privileged_correct_facts_diagnostic"] is True
    assert supplied.seen_documents == {"diagnostic-certified-financial-facts"}
    assert supplied.calls_used == 0
    fixed = FinancialEvidenceEnv(task, "fixed_retrieval")
    assert fixed.calls_used == 3 and len(fixed.visible()["initial_observed_evidence"]) == 2
    assert "mandate" in fixed.seen_documents
    assert not fixed.visible().get("privileged_correct_facts_diagnostic", False)


def test_conditional_trigger_and_metric_clauses_are_not_mutated_by_environment(task):
    before = copy.deepcopy(task)
    env = FinancialEvidenceEnv(task, "oracle_facts")
    assert task == before
    assert "diagnostic-certified-financial-facts" in env.task["gold"]["evidence_groups"][0]


def test_public_pair_identity_does_not_encode_private_condition_label(task):
    first, second = copy.deepcopy(task), copy.deepcopy(task)
    for variant, row in enumerate((first, second)):
        row.update(task_id=f"financial-0012-v{variant}", base_scenario_id="financial-base-0012")
    second["condition"]["threshold"] = 0.3
    second["gold"]["triggered"] = False
    env0, env1 = FinancialEvidenceEnv(first), FinancialEvidenceEnv(second)
    visible0, visible1 = env0.visible(), env1.visible()
    assert visible0["task_id"] == visible1["task_id"]
    assert visible0["task_id"].startswith("financial-public-")
    assert visible0["task_id"] not in (first["task_id"], second["task_id"])
    assert visible0["condition"]["threshold"] != visible1["condition"]["threshold"]
    del visible0["condition"], visible1["condition"]
    assert visible0 == visible1
    assert env0.task_id == first["task_id"] and env1.task_id == second["task_id"]
    assert env0.result()["task_id"] != env1.result()["task_id"]


def test_same_public_pair_identity_supports_opposite_labels_without_private_id_in_prompt(task):
    first, second = copy.deepcopy(task), copy.deepcopy(task)
    for variant, row in enumerate((first, second)):
        row.update(task_id=f"financial-0013-v{variant}", base_scenario_id="financial-base-0013")
        row["gold"]["triggered"] = variant == 0
    env0, env1 = FinancialEvidenceEnv(first), FinancialEvidenceEnv(second)
    assert env0.visible()["task_id"] == env1.visible()["task_id"]
    assert env0.task["gold"]["triggered"] is True and env1.task["gold"]["triggered"] is False
    for env in (env0, env1):
        assert env.task_id not in env.initial_messages()[1]["content"]
    # This controls all public content: changing a hidden label cannot change
    # the public ID or any other observation, so ID alone cannot identify it.
    assert env0.visible() == env1.visible()


def test_sft_auto_checkpointing_is_opt_in_and_toggles_only_long_training_requests(tmp_path):
    args = SimpleNamespace(
        command="sft", model=tmp_path, seed=43, learning_rate=5e-5,
        gradient_accumulation=16,
    )
    assert "sft_auto_checkpoint_input_threshold" not in recipe(args)
    args.auto_request_checkpointing = True
    assert recipe(args)["sft_auto_checkpoint_input_threshold"] == 4096

    class Model:
        training = True
        is_gradient_checkpointing = False

        def gradient_checkpointing_enable(self, **kwargs):
            self.is_gradient_checkpointing = True

        def gradient_checkpointing_disable(self):
            self.is_gradient_checkpointing = False

    model = Model()
    assert configure_request_checkpointing(model, 5000) is True
    assert model.is_gradient_checkpointing and model._mandate_auto_checkpoint_request_uses == 1
    assert configure_request_checkpointing(model, 2000) is False
    assert not model.is_gradient_checkpointing
    model.training = False
    assert configure_request_checkpointing(model, 9000) is False

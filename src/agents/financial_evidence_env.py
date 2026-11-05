"""Conditional mandates whose terminal business outcome includes financial facts.

Only public conditions and actual returned documents enter actor observations.
FinQA programs, answers and verifier evidence groups remain offline. Diagnostic
facts are explicitly privileged and never the standard training environment.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from datetime import datetime

from src.agents.mandate_repair_env import MandateRepairEnv, canonical

PROFILE = "financial_evidence_conditional_mandate_v2_opaque_public_ids"
MAX_CALLS = 10
DIAGNOSTIC_MODES = ("standard", "oracle_facts", "fixed_retrieval")
SYSTEM_PROMPT = """You are a portfolio mandate assistant in a simulated known-quote task.
Complete the client's conditional mandate using the accessible source documents.
Identify the specified company, financial report and reporting period. Read the
actual financial evidence, compute the requested metric with calculate, then
apply the specified condition to the portfolio constraints. Report units exactly
as requested in the public condition; ratio means a fraction, not a percentage.
The catalog also contains other companies/periods and duplicate evidence. You
may use any genuinely sufficient observed source, not every catalog document.
The solver receives only your explicit constraints and never infers missing
mandate or financial facts. Known quotes stay fixed; fees and slippage are paid.
You have at most 10 tool calls including submission. Do not predict future prices.
Output exactly one JSON object {"name": TOOL_NAME, "arguments": {...}} each turn.
Tools:
- search_docs: {"query": "text"}; searches public catalog metadata.
- read_document: {"doc_id": "id"}; returns original source text for citation.
- calculate: {"expression": "arithmetic"}; + - * / and cash/equity are available.
- solve: {"constraints": {"cash_min": dollars, "exposure_max": fraction,
  "symbol_max": {"SYMBOL": fraction}, "sector_max": {"SECTOR": fraction},
  "sectors": {"SYMBOL": "SECTOR"}, "cannot_sell": ["SYMBOL"],
  "cannot_buy": ["SYMBOL"]}}; returns a plan_id. Omitted limits are not inferred.
- submit: {"plan_id": "id", "citations": ["document_ids"],
  "metric": {"value": number, "period": "requested period", "unit": "requested unit",
  "triggered": true_or_false}}; terminal claim and executed portfolio submission.
To succeed the metric value, period, unit and conditional trigger must be correct,
and the executed plan (or a justified infeasibility proof) must satisfy the actual
effective mandate. A conservative portfolio with a wrong metric still fails.
Use genuine observed supporting evidence; alternate sufficient sources are valid.
"""


def _normalized(value):
    return str(value).strip().casefold()


def financial_verification(task, metric):
    """Check claims at termination, without exposing the hidden target values."""
    gold = task["gold"]
    if not isinstance(metric, dict):
        metric = {}
    try:
        value = metric.get("value")
        correct_value = (
            type(value) in (int, float)
            and math.isfinite(value)
            and math.isclose(
                float(value), float(gold["metric_value"]),
                rel_tol=float(gold.get("metric_rel_tolerance", 1e-4)),
                abs_tol=float(gold.get("metric_abs_tolerance", gold.get("metric_tolerance", 1e-5))),
            )
        )
    except (TypeError, ValueError, OverflowError):
        correct_value = False
    period_correct = _normalized(metric.get("period", "")) == _normalized(gold["metric_period"])
    unit_correct = _normalized(metric.get("unit", "")) == _normalized(gold["metric_unit"])
    condition_correct = (
        type(metric.get("triggered")) is bool and metric["triggered"] == gold["triggered"]
    )
    return dict(
        metric_correct=bool(correct_value), period_correct=period_correct,
        unit_correct=unit_correct, condition_correct=condition_correct,
        fact_correct=bool(correct_value and period_correct and unit_correct),
        financial_fact_correct=bool(correct_value and period_correct and unit_correct),
        financial_business_correct=bool(
            correct_value and period_correct and unit_correct and condition_correct
        ),
    )


class FinancialEvidenceEnv(MandateRepairEnv):
    maximum_calls = MAX_CALLS

    def __init__(self, task, diagnostic_mode="standard"):
        if diagnostic_mode not in DIAGNOSTIC_MODES:
            raise ValueError("unknown financial diagnostic mode")
        super().__init__(task)
        self.diagnostic_mode = diagnostic_mode
        self.metric_submission = None
        self._initial_evidence = []
        self._fixed_retrieval = []
        if diagnostic_mode == "oracle_facts":
            self._provide_oracle_facts()
        elif diagnostic_mode == "fixed_retrieval":
            self._provide_fixed_retrieval()

    def _provide_oracle_facts(self):
        gold = self.task["gold"]
        identifier = "diagnostic-certified-financial-facts"
        document = dict(
            doc_id=identifier, title="Privileged diagnostic: correct structured financial facts",
            available_at=self.task["as_of"], effective_at=self.task["as_of"],
            text=canonical(dict(
                value=gold["metric_value"], period=gold["metric_period"], unit=gold["metric_unit"],
                note="Privileged correct-facts diagnostic; no conditional trigger provided.",
            )),
        )
        self.task["documents"].append(document)
        self.seen_documents.add(identifier)
        self._initial_evidence.append(copy.deepcopy(document))
        self.task["gold"]["evidence_groups"] = [
            sorted(set(group) | {identifier}) for group in gold.get("evidence_groups", [])
        ]

    def _provide_fixed_retrieval(self):
        """A fixed public-metadata strategy; actual reads consume the tool budget."""
        condition = self.task["condition"]
        query = " ".join(str(condition.get(k, "")) for k in ("company", "report_year", "question"))
        search = super().step(dict(name="search_docs", arguments=dict(query=query)))
        catalog = super().visible()["document_catalog"]
        mandates = [d for d in catalog if "mandate" in d["title"].casefold()]
        candidates = [d for d in search.get("results", []) if d not in mandates]
        chosen = ([max(mandates, key=lambda d: d["effective_at"])] if mandates else [])
        chosen += candidates[:1]
        for document in chosen:
            response = super().step(dict(
                name="read_document", arguments=dict(doc_id=document["doc_id"])
            ))
            if response.get("ok"):
                self._initial_evidence.append(copy.deepcopy(response))
        self._fixed_retrieval = copy.deepcopy(self.trace)

    def visible(self):
        result = super().visible()
        # Private variant IDs remain useful for logs/seeds, but v0/v1 encode
        # opposite condition labels. Present one opaque identity for the pair.
        base = self.task.get("base_scenario_id")
        if base is None:
            base = re.sub(r"-v\d+$", "", self.task_id)
        public_id = hashlib.sha256(
            ("financial-evidence-public-v2|" + str(base)).encode()
        ).hexdigest()[:20]
        result.update(
            profile=PROFILE,
            task_id=f"financial-public-{public_id}",
            request="Compute the financial metric and execute the conditional mandate correctly.",
            condition=copy.deepcopy(self.task["condition"]),
            diagnostic_mode=self.diagnostic_mode,
        )
        if self._initial_evidence:
            result["initial_observed_evidence"] = copy.deepcopy(self._initial_evidence)
            result["initial_evidence_tool_calls"] = self.calls_used
        if self.diagnostic_mode == "oracle_facts":
            result["privileged_correct_facts_diagnostic"] = True
        return result

    def initial_messages(self):
        return [dict(role="system", content=SYSTEM_PROMPT),
                dict(role="user", content=canonical(self.visible()))]

    def _supporting_evidence(self, citations):
        if not isinstance(citations, list) or any(not isinstance(v, str) for v in citations):
            citations = []
        documents = {d["doc_id"]: d for d in self.task["documents"]}
        now = datetime.fromisoformat(self.task["as_of"])
        valid = {
            identifier for identifier in citations if identifier in self.seen_documents
            and identifier in documents
            and datetime.fromisoformat(documents[identifier]["available_at"]) <= now
            and datetime.fromisoformat(documents[identifier]["effective_at"]) <= now
            and not documents[identifier].get("superseded", False)
        }
        groups = self.task["gold"].get("evidence_groups", [])
        return all(bool(set(group) & valid) for group in groups)

    def step(self, action, *, intervention=False):
        try:
            parsed = self.parse_action(action)
        except (ValueError, TypeError, KeyError):
            parsed = None
        is_submission = parsed is not None and parsed["name"] == "submit"
        if is_submission:
            self.metric_submission = copy.deepcopy(parsed["arguments"].get("metric"))
        response = super().step(action, intervention=intervention)
        if is_submission and response.get("submitted"):
            verification = self.verification
            checks = financial_verification(self.task, self.metric_submission)
            execution_correct = bool(
                verification["business_feasible"] or verification["correct_infeasible"]
            )
            supported = self._supporting_evidence(parsed["arguments"].get("citations", []))
            if not supported:
                verification["evidence_errors"] = sorted(set(
                    verification["evidence_errors"] + ["missing_sufficient_financial_evidence"]
                ))
            verification.update(
                checks, execution_business_correct=execution_correct,
                portfolio_correct=execution_correct,
            )
            verification["evidence_valid"] = bool(verification["evidence_valid"] and supported)
            verification["business_correct"] = bool(
                execution_correct and checks["financial_business_correct"]
            )
            verification["success"] = bool(
                verification["business_correct"] and verification["evidence_valid"]
            )
            self.success = verification["success"]
            response.update(success=self.success, verification=verification)
            self.trace[-1]["response"] = response
        return response

    def result(self):
        result = super().result()
        result.update(
            profile=PROFILE, diagnostic_mode=self.diagnostic_mode,
            submitted_metric=copy.deepcopy(self.metric_submission),
            initial_fixed_retrieval_tool_calls=len(self._fixed_retrieval),
        )
        return result


def fixed_public_ranking(documents, condition):
    """Public-only ranking utility for diagnostics and unit tests."""
    terms = set(re.findall(r"[a-z0-9]+", canonical(condition).casefold()))
    return sorted(documents, key=lambda d: (
        -len(terms & set(re.findall(r"[a-z0-9]+", canonical(d).casefold()))), d["doc_id"]
    ))

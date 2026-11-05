"""Bounded evidence/mandate task with explicit solver inputs and terminal gold checks."""

from __future__ import annotations

import ast
import copy
import json
import math
import operator
import re
from datetime import datetime, timedelta

from src.agents.mandate_repair_verifier import evaluate_plan, solve_explicit, verify_submission
from src.env import ExecutionPolicy, PortfolioLedger

PROFILE = "mandate_repair_known_quote_v1"
MAX_CALLS = 8
INFORMATION_TOOLS = ("search_docs", "read_document", "calculate")
TOOLS = (*INFORMATION_TOOLS, "solve", "submit")

SYSTEM_PROMPT = """You are a portfolio mandate assistant in a simulated known-quote task.
Complete the client's currently effective mandate using the accessible documents.
The catalog may contain old amendments and irrelevant documents. Read the relevant
current evidence, interpret monetary units and weight fractions correctly, and
provide explicit constraints to the numerical solver. The solver does not know
the client's hidden mandate and cannot fill omitted constraints. Quote prices are
known and fixed during this task. Fees and slippage must be paid. You have at most
8 tool calls including submission. Do not predict future prices.
Output exactly one JSON object {"name": TOOL_NAME, "arguments": {...}} each turn.
Tools:
- search_docs: {"query": "text"}; searches the accessible catalog, not hidden answers.
- read_document: {"doc_id": "id"}; reads a document and makes it available for citation.
- calculate: {"expression": "arithmetic"}; + - * / and current cash/equity are available.
- solve: {"constraints": {"cash_min": dollars, "exposure_max": fraction,
  "symbol_max": {"SYMBOL": fraction}, "sector_max": {"SECTOR": fraction},
  "sectors": {"SYMBOL": "SECTOR"}, "cannot_sell": ["SYMBOL"],
  "cannot_buy": ["SYMBOL"]}}. Omitted constraints are not inferred. Sector limits
  require the sector classification for every quoted asset. Returns a plan_id.
- submit: {"plan_id": "id", "citations": ["document_ids"]}; final submission.
Submit a feasible solution or a justified infeasible solver result, citing the
effective documents you actually read. Multiple valid portfolios are accepted.
Calling tools or writing explanations by themselves earns no task success.
"""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _date(value):
    return datetime.fromisoformat(value)


def execute_known_quote(account, quotes, target_shares, execution, as_of):
    """Use the original ledger, selling first and targeting exact planned shares."""
    initial_equity = float(account["cash"]) + math.fsum(
        float(account.get("positions", {}).get(s, 0)) * float(p) for s, p in quotes.items()
    )
    ledger = PortfolioLedger(
        initial_cash=initial_equity,
        policy=ExecutionPolicy(execution["commission_bps"], execution["slippage_bps"]),
    )
    ledger._cash = float(account["cash"])
    ledger._positions = {s: float(account.get("positions", {}).get(s, 0)) for s in quotes}
    desired = {s: float(target_shares.get(s, ledger.positions[s])) for s in quotes}
    order = sorted(quotes, key=lambda s: (desired[s] >= ledger.positions[s], s))
    decided_at, executed_at = _date(as_of), _date(as_of) + timedelta(seconds=1)
    for symbol in order:
        if abs(desired[symbol] - ledger.positions[symbol]) * quotes[symbol] < 1e-7:
            continue
        equity = ledger.equity(quotes)
        weight = desired[symbol] * quotes[symbol] / equity
        ledger.queue_target_order(symbol, target_weight=weight, decision_at=decided_at)
        ledger.execute_next_open(executed_at, quotes)
    actual = evaluate_plan(account, quotes, dict(ledger.positions), execution)
    actual.update(
        cash=ledger.cash,
        final_equity=ledger.equity(quotes),
        transaction_cost=math.fsum(f.total_cost for f in ledger.fills),
        commission=math.fsum(f.commission for f in ledger.fills),
        slippage_cost=math.fsum(f.slippage_cost for f in ledger.fills),
        fill_count=len(ledger.fills),
    )
    return actual


class MandateRepairEnv:
    maximum_calls = MAX_CALLS

    def __init__(self, task):
        self.task = copy.deepcopy(task)
        self.task_id = task["task_id"]
        self.calls_used = 0
        self.done = False
        self.success = False
        self.seen_documents = set()
        self.plans = {}
        self.plan_inputs = {}
        self.trace = []
        self.verification = None
        self.execution_metrics = None
        self.failure_reason = None

    def visible(self):
        task = self.task
        as_of = _date(task["as_of"])
        catalog = [
            {key: doc[key] for key in ("doc_id", "title", "available_at", "effective_at")}
            for doc in task["documents"]
            if _date(doc["available_at"]) <= as_of
        ]
        return dict(
            profile=PROFILE,
            task_id=self.task_id,
            as_of=task["as_of"],
            request="Meet the client's effective mandate with minimum unnecessary turnover.",
            account=task["account"],
            quotes=task["quotes"],
            execution=task["execution"],
            document_catalog=catalog,
            maximum_tool_calls=self.maximum_calls,
            calls_used=self.calls_used,
            calls_remaining=self.maximum_calls - self.calls_used,
        )

    def initial_messages(self):
        return [
            dict(role="system", content=SYSTEM_PROMPT),
            dict(role="user", content=canonical(self.visible())),
        ]

    @staticmethod
    def choice_schema():
        return {
            "type": "object",
            "required": ["name", "arguments"],
            "properties": {"name": {"enum": list(TOOLS)}, "arguments": {"type": "object"}},
            "additionalProperties": False,
        }

    public_schema = choice_schema

    @staticmethod
    def parse_action(action):
        if isinstance(action, str):
            raw = action.strip()
            if raw.startswith("<tool_call>") and raw.endswith("</tool_call>"):
                raw = raw[len("<tool_call>") : -len("</tool_call>")].strip()
            action = json.loads(raw)
        if not isinstance(action, dict) or set(action) != {"name", "arguments"}:
            raise ValueError("Return one JSON object with only name and arguments")
        if action["name"] not in TOOLS or not isinstance(action["arguments"], dict):
            raise ValueError(f"Available tools: {', '.join(TOOLS)}; arguments must be an object")
        return action

    def _calculate(self, expression):
        operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }
        names = dict(
            cash=self.task["account"]["cash"], equity=self.task["account"]["initial_equity"]
        )

        def evaluate(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                value = node.value
            elif isinstance(node, ast.Name) and node.id in names:
                value = names[node.id]
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            elif isinstance(node, ast.BinOp) and type(node.op) in operations:
                value = operations[type(node.op)](evaluate(node.left), evaluate(node.right))
            else:
                raise ValueError("Only finite arithmetic with cash/equity and + - * / is supported")
            if not math.isfinite(value) or abs(value) > 1e15:
                raise ValueError("Arithmetic result outside supported finite range")
            return value

        if not isinstance(expression, str) or len(expression) > 500:
            raise ValueError("expression must be a short arithmetic string")
        return evaluate(ast.parse(expression, mode="eval").body)

    def step(self, action, *, intervention=False):
        if self.done:
            raise ValueError("Task already terminated")
        self.calls_used += 1
        seen_before = set(self.seen_documents)
        parsed = None
        try:
            parsed = self.parse_action(action)
            name, args = parsed["name"], parsed["arguments"]
            if name == "search_docs":
                query = str(args["query"])
                terms = set(re.findall(r"[a-z0-9]+", query.lower()))
                catalog = self.visible()["document_catalog"]
                ranked = sorted(
                    catalog,
                    key=lambda d: (
                        -len(terms & set(re.findall(r"[a-z0-9]+", canonical(d).lower()))),
                        d["doc_id"],
                    ),
                )
                result = dict(results=ranked[:6], catalog_size=len(catalog))
            elif name == "read_document":
                identifier = args["doc_id"]
                document = next(
                    (d for d in self.task["documents"] if d["doc_id"] == identifier), None
                )
                if document is None or _date(document["available_at"]) > _date(self.task["as_of"]):
                    raise ValueError("Document is not in the accessible catalog")
                self.seen_documents.add(identifier)
                result = {
                    key: document[key]
                    for key in ("doc_id", "title", "available_at", "effective_at", "text")
                }
            elif name == "calculate":
                result = dict(value=self._calculate(args["expression"]))
            elif name == "solve":
                constraints = args["constraints"]
                if not isinstance(constraints, dict):
                    raise ValueError("constraints must be an explicit object")
                allowed = {
                    "cash_min",
                    "exposure_max",
                    "symbol_max",
                    "sector_max",
                    "sectors",
                    "cannot_sell",
                    "cannot_buy",
                }
                if set(constraints) - allowed:
                    raise ValueError(f"Supported explicit constraint fields: {sorted(allowed)}")
                plan = solve_explicit(
                    self.task["account"],
                    self.task["quotes"],
                    constraints,
                    execution=self.task["execution"],
                )
                plan_id = f"plan_{len(self.plans) + 1}"
                self.plans[plan_id] = plan
                self.plan_inputs[plan_id] = copy.deepcopy(constraints)
                result = dict(plan_id=plan_id, **plan)
            else:
                plan_id = args["plan_id"]
                if plan_id not in self.plans:
                    raise ValueError("Submit a plan_id returned by solve")
                plan = dict(self.plans[plan_id], citations=args.get("citations", []))
                if plan["status"] == "feasible":
                    self.execution_metrics = execute_known_quote(
                        self.task["account"],
                        self.task["quotes"],
                        plan["target_shares"],
                        self.task["execution"],
                        self.task["as_of"],
                    )
                    plan["target_shares"] = self.execution_metrics["target_shares"]
                self.verification = verify_submission(
                    self.task,
                    plan,
                    self.seen_documents,
                    submitted_constraints=self.plan_inputs[plan_id],
                )
                self.success = bool(self.verification["success"])
                self.done = True
                result = dict(
                    submitted=True,
                    success=self.success,
                    verification=self.verification,
                    tool_calls=self.calls_used,
                )
            result = dict(ok=True, **result)
            if intervention and name in INFORMATION_TOOLS:
                self.seen_documents = seen_before
                result = dict(
                    ok=True,
                    information_unavailable=True,
                    note="No usable evidence returned by this information request.",
                )
        except (ValueError, TypeError, KeyError, SyntaxError, ArithmeticError) as error:
            result = dict(ok=False, error=str(error)[:400], available_tools=list(TOOLS))
        if self.calls_used >= self.maximum_calls and not self.done:
            self.done = True
            self.failure_reason = "TOOL_LIMIT_WITHOUT_CORRECT_SUBMISSION"
        self.trace.append(
            dict(
                call_index=self.calls_used,
                action=parsed,
                response=result,
                intervention=intervention,
            )
        )
        return result

    def result(self):
        cost = (self.execution_metrics or {}).get("transaction_cost", 0.0)
        equity = self.task["account"]["initial_equity"]
        reward = float(self.success) - 0.025 * self.calls_used / self.maximum_calls
        reward -= 0.025 * min(cost / equity / 0.005, 1.0)
        return dict(
            task_id=self.task_id,
            task_type=self.task["task_type"],
            status="COMPLETE" if self.done else "PARTIAL",
            success=self.success,
            reward=reward,
            calls_used=self.calls_used,
            failure_reason=self.failure_reason,
            verification=self.verification,
            metrics=self.execution_metrics or {},
            trace=self.trace,
            seen_document_ids=sorted(self.seen_documents),
        )

"""Real Train teacher traces and public-document CPU baselines for M3.

The teacher uses Train gold to choose steps and labels. The fair retrieval rule
reads only the environment catalog and returned document text; no gold enters
its parser or solver arguments. Reserved test tasks require the explicit final
baseline entry point and a completed frozen-models receipt before any access.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.agents.mandate_repair_env import (
    MandateRepairEnv,
    canonical,
    execute_known_quote,
)
from src.agents.mandate_repair_verifier import (
    constraint_violations,
    evaluate_plan,
    verify_submission,
)

PROFILE = "mandate_repair_cpu_teachers_baselines_v1"


def require_frozen_models(root):
    receipt = Path(root).resolve() / "frozen_models.json"
    if not receipt.is_file():
        raise PermissionError("test baselines require frozen_models.json with status COMPLETE")
    value = json.loads(receipt.read_text())
    if value.get("status") != "COMPLETE":
        raise PermissionError("test baselines require a completed frozen-models receipt")
    return receipt


def load_tasks(root, split, *, allow_frozen_test=False):
    if split == "test":
        if not allow_frozen_test:
            raise PermissionError(
                "test data require the explicit frozen-test-baselines entry point"
            )
        require_frozen_models(root)
    elif split not in ("train", "development"):
        raise ValueError("unknown mandate dataset partition")
    path = Path(root) / "data" / f"{split}.jsonl"
    tasks = [json.loads(line) for line in path.open()]
    expected = 800 if split == "train" else 100
    if len(tasks) != expected or any(t["split"] != split for t in tasks):
        raise ValueError("fixed v2 task split required")
    return tasks


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def teacher_actions(env, task, perform):
    """Gold chooses applicable Train evidence; every observation uses real tools."""
    perform("search_docs", {"query": "mandate amendment effective restriction classification"})
    for identifier in task["gold"]["required_citations"]:
        response = perform("read_document", {"doc_id": identifier})
        if not response.get("ok"):
            raise RuntimeError("teacher could not read its declared evidence")
    constraints = copy.deepcopy(task["gold"]["constraints"])
    if not constraints["sector_max"]:
        # Do not teach an unobserved classification table that was unnecessary
        # and was not retrieved. The public solver needs no sectors in this case.
        constraints["sectors"] = {}
    perform("calculate", {"expression": f"{constraints['cash_min']} - cash"})
    plan = perform("solve", {"constraints": constraints})
    if not plan.get("ok"):
        raise RuntimeError("teacher explicit solver failed")
    perform("submit", {"plan_id": plan["plan_id"], "citations": task["gold"]["required_citations"]})
    if not env.done or not env.success or env.calls_used > 8:
        raise RuntimeError(f"teacher failed: {task['task_id']}")


def build_teacher(root):
    root = Path(root).resolve()
    target = root / "teacher"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "summary.json").exists():
        return json.loads((target / "summary.json").read_text())
    counts, kinds = Counter(), Counter()
    with (
        (target / "train_requests.jsonl").open("x") as requests,
        (target / "train_trajectories.jsonl").open("x") as trajectories,
    ):
        for task in load_tasks(root, "train"):
            env = MandateRepairEnv(task)
            messages = env.initial_messages()

            def perform(name, arguments, *, task=task, env=env, messages=messages):
                completion = canonical(dict(name=name, arguments=arguments))
                record = dict(
                    id=f"{task['task_id']}/call{env.calls_used + 1}",
                    task_id=task["task_id"],
                    messages=copy.deepcopy(messages),
                    completion=completion,
                    metadata=dict(
                        source_split="train",
                        profile=PROFILE,
                        teacher_used_gold_to_choose_steps=True,
                        prefix_contains_only_actual_tool_observations=True,
                        call_index=env.calls_used + 1,
                        tool_name=name,
                    ),
                )
                requests.write(canonical(record) + "\n")
                messages.append(dict(role="assistant", content=completion))
                response = env.step(completion)
                messages.append(dict(role="tool", content=canonical(response)))
                counts["requests"] += 1
                return response

            teacher_actions(env, task, perform)
            trajectories.write(canonical(env.result()) + "\n")
            counts["tasks"] += 1
            counts["successes"] += int(env.success)
            kinds[task["task_type"]] += 1
            if counts["tasks"] % 100 == 0:
                requests.flush()
                trajectories.flush()
                print(canonical(dict(stage="TEACHER", counts=dict(counts))), flush=True)
    summary = dict(
        status="COMPLETE",
        profile=PROFILE,
        source_split="train",
        counts=dict(counts),
        task_type_counts=dict(kinds),
        all_tasks_successful=counts["tasks"] == counts["successes"],
        requests_path=str(target / "train_requests.jsonl"),
        trajectories_path=str(target / "train_trajectories.jsonl"),
        context_roles=["system", "user", "assistant", "tool"],
        labels="one exact tool-call JSON per row; current completion absent from prefix",
        gold_selection="Train teacher only; public observations come from actual env.step",
        maximum_tool_calls=8,
        formal_valid_access=False,
        formal_final_oos_access=False,
        reserved_test_opened=False,
    )
    write_json(target / "summary.json", summary)
    print(canonical(dict(stage="TEACHER_READY", **summary)), flush=True)
    return summary


def parse_public_documents(mandate, restrictions, classification=None):
    """Parse the supplied public natural-language text; never accepts a task."""
    text = mandate["text"]
    cash = re.search(r"Cash available must be at least USD ([\d,.]+)\.", text)
    exposure = re.search(r"Total stock exposure must not exceed ([\d.]+) percent", text)
    if not cash or not exposure:
        raise ValueError("public mandate cash/exposure clauses not found")
    symbol_caps = {
        symbol: float(cap) / 100
        for symbol, cap in re.findall(
            r"The holding in ([A-Z]+) is capped at ([\d.]+) percent", text
        )
    }
    sector_caps = {
        sector: float(cap) / 100
        for sector, cap in re.findall(
            r"Combined ([A-Za-z]+) holdings may not exceed ([\d.]+) percent", text
        )
    }
    sectors = {}
    if sector_caps:
        if classification is None:
            raise ValueError("a public sector-cap clause requires the classification document")
        sectors = dict(
            re.findall(
                r"([A-Z]+) belongs to the ([A-Za-z]+) industry bucket\.", classification["text"]
            )
        )
    return dict(
        cash_min=float(cash.group(1).replace(",", "")),
        exposure_max=float(exposure.group(1)) / 100,
        symbol_max=symbol_caps,
        sector_max=sector_caps,
        sectors=sectors,
        cannot_sell=re.findall(r"Do not sell any shares of ([A-Z]+)\.", restrictions["text"]),
        cannot_buy=re.findall(r"Do not buy any shares of ([A-Z]+)\.", restrictions["text"]),
    )


def public_rule_rollout(task, *, missing_key_document=False):
    """Deterministic same-tools baseline; all selections use only public data."""
    env = MandateRepairEnv(task)

    def perform(name, arguments, **kwargs):
        response = env.step(dict(name=name, arguments=arguments), **kwargs)
        if not response.get("ok"):
            raise ValueError(response.get("error", "public tool failed"))
        return response

    try:
        visible = env.visible()
        as_of = datetime.fromisoformat(visible["as_of"])
        search = perform(
            "search_docs", {"query": "mandate amendment effective restriction classification"}
        )
        catalog = search["results"]
        applicable = [d for d in catalog if datetime.fromisoformat(d["effective_at"]) <= as_of]
        mandates = sorted(
            [d for d in applicable if "mandate" in d["title"].lower()],
            key=lambda d: (d["effective_at"], d["doc_id"]),
            reverse=True,
        )
        chosen = mandates[0]
        mandate = perform(
            "read_document", {"doc_id": chosen["doc_id"]}, intervention=missing_key_document
        )
        if missing_key_document:
            # Same failed information response and budget consumption as a real
            # intervention. Fall back to the older publicly available mandate.
            chosen = mandates[1]
            mandate = perform("read_document", {"doc_id": chosen["doc_id"]})
        restriction_item = next(d for d in applicable if "restriction" in d["title"].lower())
        restriction = perform("read_document", {"doc_id": restriction_item["doc_id"]})
        citations = [chosen["doc_id"], restriction_item["doc_id"]]
        classification = None
        if re.search(r"Combined [A-Za-z]+ holdings may not exceed", mandate["text"]):
            item = next(d for d in applicable if "classification" in d["title"].lower())
            classification = perform("read_document", {"doc_id": item["doc_id"]})
            citations.append(item["doc_id"])
        constraints = parse_public_documents(mandate, restriction, classification)
        perform("calculate", {"expression": f"{constraints['cash_min']} - cash"})
        plan = perform("solve", {"constraints": constraints})
        perform("submit", {"plan_id": plan["plan_id"], "citations": citations})
    except (ValueError, KeyError, IndexError, StopIteration, RuntimeError) as error:
        env.done = True
        env.failure_reason = f"RULE_BASELINE_ERROR:{type(error).__name__}:{error}"
    result = env.result()
    result.update(
        policy="no_key_document" if missing_key_document else "public_rule_retrieval",
        policy_used_gold=False,
        information_access="current mandate response unavailable"
        if missing_key_document
        else "same catalog, documents, calculate, explicit solve and submit tools as agent",
    )
    return result


def privileged_oracle(task):
    """Explicitly privileged full-information reference, not a fair agent policy."""
    plan = copy.deepcopy(task["gold"]["oracle_plan"])
    metrics = {}
    if plan["status"] == "feasible":
        metrics = execute_known_quote(
            task["account"], task["quotes"], plan["target_shares"], task["execution"], task["as_of"]
        )
        plan["target_shares"] = metrics["target_shares"]
    verification = verify_submission(task, plan, submitted_constraints=task["gold"]["constraints"])
    return dict(
        task_id=task["task_id"],
        task_type=task["task_type"],
        status="COMPLETE",
        policy="privileged_full_information_oracle",
        policy_used_gold=True,
        information_access="privileged structured truth; full-information reference upper bound",
        success=verification["success"],
        calls_used=0,
        verification=verification,
        metrics=metrics,
        trace=[],
    )


def aggregate(rows):
    total = len(rows)
    feasible = [row for row in rows if row["success"] and row.get("metrics")]
    by_type = {}
    for kind in sorted({row["task_type"] for row in rows}):
        subset = [r for r in rows if r["task_type"] == kind]
        by_type[kind] = dict(tasks=len(subset), successes=sum(r["success"] for r in subset))
    return dict(
        tasks=total,
        successes=sum(row["success"] for row in rows),
        success_rate=sum(row["success"] for row in rows) / total,
        business_feasible=sum(
            bool((r.get("verification") or {}).get("business_feasible")) for r in rows
        ),
        evidence_valid=sum(bool((r.get("verification") or {}).get("evidence_valid")) for r in rows),
        correct_infeasible=sum(
            bool((r.get("verification") or {}).get("correct_infeasible")) for r in rows
        ),
        mean_tool_calls=math.fsum(row["calls_used"] for row in rows) / total,
        successful_feasible_tasks=len(feasible),
        mean_cost_successful_feasible=(
            math.fsum(r["metrics"]["transaction_cost"] for r in feasible) / len(feasible)
            if feasible
            else None
        ),
        task_types=by_type,
    )


def build_baselines(root, *, split="development", allow_frozen_test=False):
    root = Path(root).resolve()
    if split == "test":
        if not allow_frozen_test:
            raise PermissionError("use the explicit frozen-test-baselines entry point")
        # Check even before reading an existing test summary. A missing receipt
        # cannot silently expose previously cached test results or task data.
        frozen_receipt = require_frozen_models(root)
    elif split == "development":
        frozen_receipt = None
    else:
        raise ValueError("CPU baselines support development or explicitly frozen test")
    prefix = "test" if split == "test" else "dev"
    target = root / "baselines"
    target.mkdir(parents=True, exist_ok=True)
    if (target / f"{prefix}_summary.json").exists():
        return json.loads((target / f"{prefix}_summary.json").read_text())
    groups = {
        p: []
        for p in ("privileged_full_information_oracle", "public_rule_retrieval", "no_key_document")
    }
    with (target / f"{prefix}_trajectories.jsonl").open("x") as trajectories:
        for task in load_tasks(root, split, allow_frozen_test=allow_frozen_test):
            results = (
                privileged_oracle(task),
                public_rule_rollout(task),
                public_rule_rollout(task, missing_key_document=True),
            )
            for result in results:
                groups[result["policy"]].append(result)
                trajectories.write(canonical(result) + "\n")
    summary = dict(
        status="COMPLETE",
        profile=PROFILE,
        source_split=split,
        policies={name: aggregate(rows) for name, rows in groups.items()},
        fair_policy_uses_hidden_gold=False,
        privileged_oracle_marked=True,
        reserved_test_opened=split == "test",
        frozen_models_receipt=str(frozen_receipt) if frozen_receipt else None,
        formal_valid_access=False,
        formal_final_oos_access=False,
        trajectory_path=str(target / f"{prefix}_trajectories.jsonl"),
    )
    # The summary is the completion marker, written only after all three
    # policies' 100 full trajectories have been retained in the closed JSONL.
    write_json(target / f"{prefix}_summary.json", summary)
    print(canonical(summary), flush=True)
    return summary


def build_frozen_test_baselines(root):
    return build_baselines(root, split="test", allow_frozen_test=True)


def _execute_future_quote(task, quotes, targets, executed_at):
    """Existing shares/fees ledger, with actual next-open execution timestamp."""
    from src.env import ExecutionPolicy, PortfolioLedger

    account, policy = task["account"], task["execution"]
    equity = account["cash"] + math.fsum(account["positions"].get(s, 0) * quotes[s] for s in quotes)
    ledger = PortfolioLedger(
        initial_cash=equity,
        policy=ExecutionPolicy(policy["commission_bps"], policy["slippage_bps"]),
    )
    ledger._cash = account["cash"]
    ledger._positions = dict(account["positions"])
    order = sorted(quotes, key=lambda s: (targets[s] >= account["positions"].get(s, 0), s))
    for symbol in order:
        if abs(targets[symbol] - ledger.positions.get(symbol, 0)) * quotes[symbol] < 1e-7:
            continue
        weight = targets[symbol] * quotes[symbol] / ledger.equity(quotes)
        ledger.queue_target_order(
            symbol, target_weight=weight, decision_at=datetime.fromisoformat(task["as_of"])
        )
        ledger.execute_next_open(executed_at, quotes)
    actual = evaluate_plan(account, quotes, dict(ledger.positions), policy)
    actual.update(
        cash=ledger.cash,
        final_equity=ledger.equity(quotes),
        transaction_cost=math.fsum(fill.total_cost for fill in ledger.fills),
        fill_count=len(ledger.fills),
    )
    clipped = any(abs(ledger.positions.get(s, 0) - targets[s]) * quotes[s] > 0.05 for s in quotes)
    return actual, clipped


def build_next_open_diagnostic(root):
    """Future Train opens are isolated auxiliary evaluation, never actor input."""
    from src.agents.portfolio_management import load_train_markets

    root = Path(root).resolve()
    target = root / "baselines"
    summary_path = target / "dev_next_open_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    manifest = json.loads((root / "data/manifest.json").read_text())
    markets = load_train_markets(manifest["source_prepared"], target / "next_open_adapter")
    rows = []
    for task in load_tasks(root, "development"):
        oracle = task["gold"]["oracle"]
        row = dict(
            task_id=task["task_id"],
            task_type=task["task_type"],
            original_plan_status=oracle["status"],
            auxiliary_only=True,
            primary_fixed_quote_judgment_unchanged=True,
        )
        if oracle["status"] == "infeasible":
            row["status"] = "NO_PLAN_INFEASIBLE_AT_ORIGINAL_QUOTE"
            rows.append(row)
            continue
        market = markets[task["source"]["episode_id"]]
        j = task["source"]["session_index"]
        executed_at = datetime.fromisoformat(market.kwargs["episode_open_times"][j + 1])
        quotes = {s: market.opens[j + 1][s] for s in task["quotes"]}
        try:
            metrics, clipped = _execute_future_quote(
                task, quotes, oracle["target_shares"], executed_at
            )
            violations = constraint_violations(
                task["account"], quotes, task["gold"]["constraints"], metrics
            )
            # Separately flag a scheduled document becoming effective during the
            # interval; this never changes the original known-quote main task.
            mandates = [
                d
                for d in task["documents"]
                if "mandate" in d["title"].lower()
                and datetime.fromisoformat(d["effective_at"]) <= executed_at
            ]
            latest = max(mandates, key=lambda d: d["effective_at"])
            changed_version = latest["doc_id"] not in task["gold"]["required_citations"]
            later_violations = None
            if changed_version:
                restriction = next(d for d in task["documents"] if d["kind"] == "restrictions")
                classification = next(d for d in task["documents"] if d["kind"] == "classification")
                later = parse_public_documents(latest, restriction, classification)
                later_violations = constraint_violations(task["account"], quotes, later, metrics)
            row.update(
                status="EXECUTED",
                decision_at=task["as_of"],
                executed_at=executed_at.isoformat(),
                metrics=metrics,
                cash_clipped=clipped,
                original_mandate_violations=violations,
                original_mandate_feasible=not violations,
                later_version_effective=changed_version,
                later_version_document=latest["doc_id"] if changed_version else None,
                later_version_violations=later_violations,
                cost_delta_to_known_quote=metrics["transaction_cost"]
                - oracle["metrics"]["transaction_cost"],
                cash_delta_to_known_quote=metrics["cash"] - oracle["metrics"]["cash"],
            )
        except (ValueError, RuntimeError, KeyError) as error:
            row.update(status="AUXILIARY_EXECUTION_ERROR", error=f"{type(error).__name__}:{error}")
        rows.append(row)
    executed = [r for r in rows if r["status"] == "EXECUTED"]
    with (target / "dev_next_open_diagnostic.jsonl").open("x") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")
    summary = dict(
        status="COMPLETE",
        profile=PROFILE,
        source_split="development",
        tasks=len(rows),
        feasible_plans_at_original_quote=sum(r["original_plan_status"] == "feasible" for r in rows),
        actually_executed=len(executed),
        original_mandate_feasible_at_next_open=sum(
            r["original_mandate_feasible"] for r in executed
        ),
        cash_clipped=sum(r["cash_clipped"] for r in executed),
        scheduled_mandate_version_changed=sum(r["later_version_effective"] for r in executed),
        mean_cost_at_next_open=(
            math.fsum(r["metrics"]["transaction_cost"] for r in executed) / len(executed)
            if executed
            else None
        ),
        auxiliary_only=True,
        primary_fixed_quote_judgment_unchanged=True,
        future_quotes_never_exposed_to_actor=True,
        reserved_test_opened=False,
        formal_valid_access=False,
        formal_final_oos_access=False,
        path=str(target / "dev_next_open_diagnostic.jsonl"),
    )
    write_json(summary_path, summary)
    print(canonical(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--only", choices=("all", "teacher", "baselines", "next-open"), default="all"
    )
    parser.add_argument("--part", choices=("frozen-test-baselines",))
    args = parser.parse_args()
    if args.part == "frozen-test-baselines":
        if args.only != "all":
            parser.error("--part frozen-test-baselines must not be combined with --only")
        build_frozen_test_baselines(args.output)
        return
    if args.only in ("all", "teacher"):
        build_teacher(args.output)
    if args.only in ("all", "baselines"):
        build_baselines(args.output)
    if args.only == "next-open":
        build_next_open_diagnostic(args.output)


if __name__ == "__main__":
    main()

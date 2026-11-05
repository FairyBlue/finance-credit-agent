"""Known-quote mandate solver and terminal verifier for the simulated M3 task.

``solve_explicit`` and ``evaluate_plan`` accept only their public arguments;
neither reads a task or hidden truth. Only ``verify_submission`` consults gold.
Costs are exactly the existing ledger's proportional commission/slippage model.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

DEFAULT_EXECUTION = {"commission_bps": 10.0, "slippage_bps": 5.0}
MONEY_TOLERANCE = 0.05
WEIGHT_TOLERANCE = 1e-7


def _inputs(account, quotes, execution=None):
    symbols = tuple(quotes)
    prices = np.asarray([float(quotes[s]) for s in symbols], dtype=float)
    positions = account.get("positions", {})
    if set(positions) - set(symbols):
        raise ValueError("positions contain an unquoted asset")
    shares = np.asarray([float(positions.get(s, 0.0)) for s in symbols], dtype=float)
    cash = float(account["cash"])
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or not np.isfinite(prices).all()
        or (prices <= 0).any()
        or not np.isfinite(shares).all()
        or (shares < 0).any()
        or not math.isfinite(cash)
        or cash < 0
    ):
        raise ValueError("finite positive quotes, nonnegative shares/cash required")
    policy = {**DEFAULT_EXECUTION, **(execution or {})}
    commission, slippage = policy["commission_bps"] / 10000, policy["slippage_bps"] / 10000
    if not (0 <= commission < 1 and 0 <= slippage < 1):
        raise ValueError("invalid commission/slippage basis points")
    equity = cash + float(np.dot(shares, prices))
    if equity <= 0:
        raise ValueError("positive account equity required")
    return symbols, prices, shares, cash, equity, commission, slippage


def evaluate_plan(account, quotes, target_shares, execution=None):
    """Exact-share known-quote accounting; unspecified assets retain their shares."""
    symbols, prices, before, cash, equity, commission, slippage = _inputs(
        account, quotes, execution
    )
    if not isinstance(target_shares, dict) or set(target_shares) - set(symbols):
        raise ValueError("target_shares must map quoted symbols to nonnegative shares")
    after = np.asarray([float(target_shares.get(s, before[i])) for i, s in enumerate(symbols)])
    if not np.isfinite(after).all() or (after < 0).any():
        raise ValueError("nonnegative finite target shares required")
    delta = (after - before) * prices
    buys, sells = np.maximum(delta, 0), np.maximum(-delta, 0)
    buy_coefficient = slippage + commission * (1 + slippage)
    sell_coefficient = slippage + commission * (1 - slippage)
    commission_cost = float(np.sum(buys * (1 + slippage) + sells * (1 - slippage))) * commission
    slippage_cost = float(np.sum(buys + sells)) * slippage
    cost = commission_cost + slippage_cost
    cash_after = (
        cash
        - float(np.sum(buys * (1 + buy_coefficient)))
        + float(np.sum(sells * (1 - sell_coefficient)))
    )
    equity_after = equity - cost
    if not math.isfinite(equity_after) or equity_after <= 0:
        raise ValueError("plan has nonpositive or nonfinite post-fee equity")
    weights = after * prices / equity_after
    return dict(
        initial_equity=equity,
        final_equity=equity_after,
        cash=cash_after,
        target_shares={s: float(after[i]) for i, s in enumerate(symbols)},
        weights={s: float(weights[i]) for i, s in enumerate(symbols)},
        stock_exposure=float(np.sum(weights)),
        transaction_cost=cost,
        commission=commission_cost,
        slippage_cost=slippage_cost,
        turnover=float(np.sum(buys + sells)) / equity,
        traded_raw_notional=float(np.sum(buys + sells)),
        cash_feasible=cash_after >= -MONEY_TOLERANCE,
        finite_equity=math.isfinite(equity_after) and equity_after > 0,
    )


score_plan = evaluate_plan


def constraint_violations(account, quotes, constraints, metrics):
    """Independent business checks, accepting any feasible target vector."""
    violations = []
    if not metrics["cash_feasible"] or not metrics["finite_equity"]:
        violations.append("insufficient_cash_or_invalid_equity")
    if metrics["cash"] + MONEY_TOLERANCE < float(constraints.get("cash_min", 0)):
        violations.append("cash_min")
    if metrics["stock_exposure"] > float(constraints.get("exposure_max", 1)) + WEIGHT_TOLERANCE:
        violations.append("exposure_max")
    for symbol, cap in constraints.get("symbol_max", {}).items():
        if symbol not in quotes or metrics["weights"][symbol] > float(cap) + WEIGHT_TOLERANCE:
            violations.append(f"symbol_max:{symbol}")
    sectors = constraints.get("sectors", {})
    for sector, cap in constraints.get("sector_max", {}).items():
        total = math.fsum(w for s, w in metrics["weights"].items() if sectors.get(s) == sector)
        if total > float(cap) + WEIGHT_TOLERANCE:
            violations.append(f"sector_max:{sector}")
    for side, comparison in (("cannot_sell", -1), ("cannot_buy", 1)):
        for symbol in constraints.get(side, []):
            if symbol not in quotes:
                violations.append(f"unknown_restriction:{symbol}")
                continue
            change = metrics["target_shares"][symbol] - float(account["positions"].get(symbol, 0))
            if comparison * change * float(quotes[symbol]) > MONEY_TOLERANCE:
                violations.append(f"{side}:{symbol}")
    return violations


def solve_explicit(account, quotes, constraints, *, execution=None):
    """Minimum raw-turnover LP using only explicitly supplied constraints.

    Fractional shares are allowed. Weight limits use post-fee marked NAV.
    This tool deliberately does not infer omitted clauses or evidence.
    """
    from scipy.optimize import linprog

    symbols, prices, shares, cash, equity, commission, slippage = _inputs(
        account, quotes, execution
    )
    n = len(symbols)
    initial = shares * prices
    cb = slippage + commission * (1 + slippage)
    cs = slippage + commission * (1 - slippage)
    fee = np.asarray([cb] * n + [cs] * n)
    rows, rhs = [], []
    # x=(buy raw dollars, sell raw dollars), cash'=cash-(1+cb)b+(1-cs)s.
    rows.append(np.asarray([1 + cb] * n + [-(1 - cs)] * n))
    rhs.append(cash - max(0.0, float(constraints.get("cash_min", 0))))

    def cap_row(indexes, limit):
        limit = float(limit)
        if not math.isfinite(limit) or not 0 <= limit <= 1:
            raise ValueError("weight/exposure limits must be fractions in [0,1]")
        vector = limit * fee.copy()
        for i in indexes:
            vector[i] += 1
            vector[n + i] -= 1
        rows.append(vector)
        rhs.append(limit * equity - float(np.sum(initial[list(indexes)])))

    cap_row(list(range(n)), constraints.get("exposure_max", 1))
    for symbol, cap in constraints.get("symbol_max", {}).items():
        if symbol not in symbols:
            raise ValueError("single-asset constraint names an unquoted asset")
        cap_row([symbols.index(symbol)], cap)
    sectors = constraints.get("sectors", {})
    for sector, cap in constraints.get("sector_max", {}).items():
        if any(s not in sectors for s in symbols):
            raise ValueError("explicit sector classification is required for every asset")
        cap_row([i for i, s in enumerate(symbols) if sectors[s] == sector], cap)
    cannot_buy, cannot_sell = (
        set(constraints.get("cannot_buy", [])),
        set(constraints.get("cannot_sell", [])),
    )
    if (cannot_buy | cannot_sell) - set(symbols):
        raise ValueError("trade restriction names an unquoted asset")
    bounds = [(0, 0 if s in cannot_buy else None) for s in symbols]
    bounds += [(0, 0 if s in cannot_sell else float(initial[i])) for i, s in enumerate(symbols)]
    result = linprog(
        np.ones(2 * n), A_ub=np.asarray(rows), b_ub=np.asarray(rhs), bounds=bounds, method="highs"
    )
    if result.status == 2:
        return dict(status="infeasible", solver="scipy-highs-explicit-linear-cost-v1")
    if not result.success:
        raise RuntimeError(f"explicit LP failed: {result.message}")
    values = np.maximum(initial + result.x[:n] - result.x[n:], 0)
    target = {s: float(values[i] / prices[i]) for i, s in enumerate(symbols)}
    metrics = evaluate_plan(account, quotes, target, execution)
    violations = constraint_violations(account, quotes, constraints, metrics)
    if violations:
        raise RuntimeError(f"LP plan failed independent arithmetic verification: {violations}")
    return dict(
        status="feasible",
        target_shares=target,
        metrics=metrics,
        solver="scipy-highs-explicit-linear-cost-v1",
        objective="minimum_raw_turnover",
    )


def infeasibility_certificate(task, submitted_constraints):
    """An infeasible relaxation of the actual mandate is a valid certificate.

    A fabricated, stronger problem is not evidence that the client's mandate
    is impossible. Omitted upper limits and permissions remain relaxations.
    """
    if not isinstance(submitted_constraints, dict):
        return False, "missing_submitted_constraints"
    gold = task["gold"]["constraints"]
    candidate = submitted_constraints
    try:
        allowed = {
            "cash_min",
            "exposure_max",
            "symbol_max",
            "sector_max",
            "sectors",
            "cannot_buy",
            "cannot_sell",
        }
        if set(candidate) - allowed:
            return False, "unsupported_constraint_fields"

        def number(value):
            result = float(value)
            if not math.isfinite(result):
                raise ValueError("nonfinite certificate constraint")
            return result

        if number(candidate.get("cash_min", 0)) > number(gold.get("cash_min", 0)) + 1e-8:
            return False, "cash_min_stronger_than_mandate"
        if number(candidate.get("exposure_max", 1)) < number(gold.get("exposure_max", 1)) - 1e-12:
            return False, "exposure_max_stronger_than_mandate"
        for field in ("symbol_max", "sector_max"):
            limits = candidate.get(field, {})
            if not isinstance(limits, dict):
                return False, "invalid_cap_mapping"
            for key, limit in limits.items():
                if field == "symbol_max" and key not in task["quotes"]:
                    return False, "unknown_asset_constraint"
                if number(limit) < number(gold.get(field, {}).get(key, 1)) - 1e-12:
                    return False, f"{field}_stronger_than_mandate"
        for field in ("cannot_buy", "cannot_sell"):
            if not isinstance(candidate.get(field, []), list):
                return False, "invalid_trade_permissions"
            if not set(candidate.get(field, [])).issubset(gold.get(field, [])):
                return False, "trade_permissions_stronger_than_mandate"
        if candidate.get("sector_max"):
            mapping = candidate.get("sectors", {})
            if not isinstance(mapping, dict) or any(
                mapping.get(s) != gold.get("sectors", {}).get(s) for s in task["quotes"]
            ):
                return False, "sector_classification_mismatch"
        checked = solve_explicit(
            task["account"], task["quotes"], candidate, execution=task["execution"]
        )
        if checked["status"] != "infeasible":
            return False, "submitted_problem_is_feasible"
    except (TypeError, ValueError, KeyError, OverflowError, RuntimeError):
        return False, "invalid_submitted_constraints"
    return True, "infeasible_relaxation_of_effective_mandate"


def verify_submission(task, plan, seen_document_ids=None, *, submitted_constraints=None):
    """Terminal gold check only; never call this to reveal intermediate clauses."""
    gold, documents = task["gold"], {d["doc_id"]: d for d in task["documents"]}
    citations = plan.get("citations", []) if isinstance(plan, dict) else []
    if not isinstance(citations, list) or any(not isinstance(v, str) for v in citations):
        citations = []
    as_of = datetime.fromisoformat(task["as_of"])
    evidence_errors = []
    if not set(gold["required_citations"]).issubset(citations):
        evidence_errors.append("missing_required_evidence")
    for doc_id in citations:
        document = documents.get(doc_id)
        if document is None:
            evidence_errors.append("unknown_citation")
        elif (
            datetime.fromisoformat(document["available_at"]) > as_of
            or datetime.fromisoformat(document["effective_at"]) > as_of
            or document.get("superseded", False)
        ):
            evidence_errors.append("inapplicable_or_superseded_citation")
        if seen_document_ids is not None and doc_id not in seen_document_ids:
            evidence_errors.append("citation_not_observed")
    evidence_valid = not evidence_errors
    oracle_status = gold["oracle"]["status"]
    claimed = plan.get("status") if isinstance(plan, dict) else None
    metrics, violations = None, []
    classification_correct = claimed == "infeasible" and oracle_status == "infeasible"
    justified, justification = (
        infeasibility_certificate(task, submitted_constraints)
        if classification_correct
        else (False, "not_an_infeasible_gold_task")
    )
    correct_infeasible = classification_correct and justified
    if claimed == "feasible":
        try:
            metrics = evaluate_plan(
                task["account"], task["quotes"], plan["target_shares"], task["execution"]
            )
            violations = constraint_violations(
                task["account"], task["quotes"], gold["constraints"], metrics
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            violations = [f"invalid_plan:{type(error).__name__}"]
    elif claimed != "infeasible":
        violations = ["invalid_submission_status"]
    elif not classification_correct:
        violations = ["incorrect_infeasible_claim"]
    elif not justified:
        violations = ["unjustified_infeasible_claim"]
    business_feasible = claimed == "feasible" and not violations and oracle_status == "feasible"
    success = evidence_valid and (business_feasible or correct_infeasible)
    return dict(
        success=success,
        business_feasible=business_feasible,
        evidence_valid=evidence_valid,
        correct_infeasible=correct_infeasible,
        infeasible_classification_correct=classification_correct,
        infeasibility_justified=justified,
        infeasibility_justification=justification,
        violations=violations,
        evidence_errors=sorted(set(evidence_errors)),
        metrics=metrics,
        turnover_gap=(
            metrics["turnover"] - gold["oracle"]["metrics"]["turnover"]
            if business_feasible
            else None
        ),
        oracle_status=oracle_status,
    )

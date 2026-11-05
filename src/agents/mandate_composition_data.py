"""Diverse compositional mandates with grouped business-sensitive task pairs.

Synthetic mandates/accounts on known old-Train quotes. No test performance is
computed here; oracle solutions validate the task generator, not actor policies.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from src.agents.mandate_repair_data import ASSET_GROUPS, PREPARED, canonical
from src.agents.mandate_repair_verifier import (
    constraint_violations,
    evaluate_plan,
    solve_explicit,
    verify_submission,
)
from src.agents.portfolio_management import load_train_markets

PROFILE = "mandate_composition_simulated_train_quotes_v2"
SEED = 20261029


def cash_clause(value, style):
    value = f"{value:,.2f}"
    return (
        f"Cash available must be at least USD {value}.",
        f"Retain at least USD {value} in cash.",
        f"The closing cash balance must be no less than USD {value}.",
        f"Cash floor after fees: USD {value}.",
        f"End-of-task cash floor: USD {value}.",
        f"Available closing cash must reach USD {value} or more.",
        f"Minimum final cash required: USD {value}.",
        f"Set aside a final cash balance of at least USD {value}.",
    )[style]


def exposure_clause(value, style):
    percent = f"{100 * value:.1f}"
    return (
        f"Total stock exposure must not exceed {percent} percent of net account equity.",
        f"Aggregate equity exposure must be at most {percent} percent of net account equity.",
        f"Total equity exposure ceiling: {percent} percent of net account equity.",
        f"Keep combined stock positions within {percent} percent of net account equity.",
        f"Final aggregate stock exposure is limited to {percent} percent of net account equity.",
        f"The stock allocation ceiling is {percent} percent of net account equity.",
        (
            f"At completion all equities together may occupy no more than {percent} "
            "percent of net account equity."
        ),
        f"Total post-cost equity allocation: maximum {percent} percent of net account equity.",
    )[style]


def symbol_clause(symbol, value, style):
    percent = f"{100 * value:.1f}"
    if style % 3 == 0:
        return f"The holding in {symbol} is capped at {percent} percent of net account equity."
    if style % 3 == 1:
        return f"Limit {symbol} to {percent} percent of net account equity."
    return f"Single-name ceiling for {symbol}: {percent} percent of net account equity."


def sector_clause(sector, value, style):
    percent = f"{100 * value:.1f}"
    if style % 3 == 0:
        return f"Combined {sector} holdings may not exceed {percent} percent of net account equity."
    if style % 3 == 1:
        return f"Cap combined {sector} exposure at {percent} percent of net account equity."
    return f"Aggregate industry ceiling for {sector}: {percent} percent of net account equity."


def permissions_text(constraints, style):
    lines = [
        "SIMULATED current dealing permissions. This register applies at the account snapshot time."
    ]
    for side in ("sell", "buy"):
        for symbol in constraints[f"cannot_{side}"]:
            if style % 3 == 0:
                lines.append(f"Do not {side} any shares of {symbol}.")
            elif style % 3 == 1:
                lines.append(
                    f"No {'sales' if side == 'sell' else 'purchases'} of {symbol} are permitted."
                )
            else:
                lines.append(f"{symbol} may not be {'sold' if side == 'sell' else 'bought'}.")
    if not constraints["cannot_sell"] and not constraints["cannot_buy"]:
        lines.append("There are no additional buy or sell restrictions on listed assets.")
    return "\n".join(lines)


def mandate_text(constraints, style, *, appendix):
    lines = [
        (
            "SIMULATED CLIENT MANDATE. This complete restatement replaces all earlier "
            "mandates in full."
        ),
        "All limits are measured on final net account equity after every transaction fee.",
        cash_clause(constraints["cash_min"], style),
        exposure_clause(constraints["exposure_max"], style),
        *(symbol_clause(s, cap, style) for s, cap in constraints["symbol_max"].items()),
    ]
    if appendix:
        lines.append(
            "Aggregate industry limits are set in the current allocation schedule; "
            "read that schedule as part of this mandate."
        )
    elif constraints["sector_max"]:
        lines.extend(sector_clause(s, v, style) for s, v in constraints["sector_max"].items())
        lines.append("Use the current classification register for every industry calculation.")
    else:
        lines.append("There is no additional aggregate industry cap in this mandate.")
    lines.append(
        "Read the current dealing-restriction register. No borrowing, short sale or cash "
        "injection is allowed; fractional shares are allowed."
    )
    return "\n".join(lines)


def scenario(group, market, seed, attempt):
    rng = random.Random(seed + group * 104729 + attempt * 1000003)
    buckets = {
        sector: rng.sample(list(pool), 1 if sector in ("Consumer", "Healthcare") else 2)
        for sector, pool in ASSET_GROUPS.items()
    }
    symbols = [s for items in buckets.values() for s in items]
    sectors = {s: sector for sector, items in buckets.items() for s in items}
    quotes = {s: float(market.closes[10][s]) for s in symbols}
    family = "hold" if group % 10 == 0 else "infeasible" if group % 10 == 1 else "mixed"
    cash_fraction = rng.choice((0.08, 0.11, 0.14, 0.18, 0.24))
    amounts = [rng.uniform(0.45, 1.65) for _ in symbols]
    weights = {
        s: a / sum(amounts) * (1 - cash_fraction) for s, a in zip(symbols, amounts, strict=True)
    }
    equity = rng.choice((750_000.0, 1_000_000.0, 1_250_000.0))
    account = dict(
        cash=equity * cash_fraction,
        positions={s: equity * weights[s] / quotes[s] for s in symbols},
        initial_equity=equity,
        currency="USD",
    )
    constraints = dict(
        cash_min=round(equity * rng.choice((0.15, 0.20, 0.25, 0.30)), 2),
        exposure_max=rng.choice((0.60, 0.65, 0.70, 0.75, 0.80)),
        symbol_max={s: rng.choice((0.16, 0.20, 0.24, 0.28)) for s in symbols},
        sector_max={},
        sectors=sectors,
        cannot_sell=[],
        cannot_buy=[],
    )
    # All sectors, cap cardinalities and numerical ranges appear in both Train
    # portions. Account/combination instances and held-out surface forms differ.
    selected_sectors = rng.sample(list(buckets), 1 + (group // 10) % 3)
    for sector in selected_sectors:
        current = math.fsum(weights[s] for s in buckets[sector])
        constraints["sector_max"][sector] = max(
            0.025, math.floor(current * rng.choice((0.45, 0.60, 0.75)) * 200) / 200
        )
    movable = [s for s in symbols if sectors[s] not in selected_sectors]
    if movable and rng.random() < 0.8:
        constraints["cannot_sell"] = rng.sample(movable, min(len(movable), rng.choice((1, 2))))
    constraints["cannot_buy"] = rng.sample(symbols, rng.choice((0, 1, 2)))
    if family == "hold":
        constraints.update(
            cash_min=round(account["cash"] * 0.7, 2),
            exposure_max=0.98,
            symbol_max={s: 0.40 for s in symbols},
            sector_max={},
            cannot_sell=[],
            cannot_buy=[],
        )
    elif family == "infeasible":
        constraints.update(
            cash_min=round(account["cash"] + 0.20 * equity, 2),
            exposure_max=1.0,
            symbol_max={s: 0.40 for s in symbols},
            sector_max={},
            cannot_sell=list(symbols),
        )
    return rng, family, account, quotes, constraints


def make_task(group, variant, split, market, seed, account, quotes, base_constraints):
    constraints = copy.deepcopy(base_constraints)
    rng = random.Random(seed + group * 65537)
    family = "hold" if group % 10 == 0 else "infeasible" if group % 10 == 1 else "mixed"
    if variant:
        if family == "hold":
            constraints["cash_min"] = round(account["cash"] + 0.15 * account["initial_equity"], 2)
        elif family == "mixed":
            sector = next(iter(constraints["sector_max"]))
            constraints["sector_max"][sector] = max(
                0.005, constraints["sector_max"][sector] - 0.025
            )
        else:
            constraints["cash_min"] += 0.05 * account["initial_equity"]
    style = (
        rng.randrange(4)
        if split == "train"
        else 4 + group % 2
        if split == "development"
        else 6 + group % 2
    )
    task_id = f"composition-{group:04d}-v{variant}"
    base_id = f"composition-base-{group:04d}"
    now = datetime.fromisoformat(market.kwargs["episode_dates"][10])
    execution = dict(commission_bps=10.0, slippage_bps=5.0)
    documents = []
    ids = {
        name: "doc-" + hashlib.sha256(f"{seed}/{group}/{name}".encode()).hexdigest()[:12]
        for name in (
            "mandate",
            "old",
            "future",
            "classification",
            "restrictions",
            "allocation",
            "service",
            "outlook",
            "notice",
            "methodology",
        )
    }

    def add(name, title, text, available=-7, effective=-3, superseded=False):
        documents.append(
            dict(
                doc_id=ids[name],
                title=title,
                kind=name,
                available_at=(now + timedelta(days=available)).isoformat(),
                effective_at=(now + timedelta(days=effective)).isoformat(),
                text=text,
                superseded=superseded,
            )
        )

    appendix = len(constraints["sector_max"]) >= 2
    add(
        "mandate",
        "Investment mandate — current full restatement",
        mandate_text(constraints, style, appendix=appendix),
    )
    old = copy.deepcopy(constraints)
    old.update(cash_min=0.0, exposure_max=1.0, sector_max={}, symbol_max={s: 0.40 for s in quotes})
    add(
        "old",
        "Investment mandate — archived original issue",
        mandate_text(old, style, appendix=False),
        -40,
        -35,
        True,
    )
    future = copy.deepcopy(base_constraints)
    future.update(cash_min=0.45 * account["initial_equity"], exposure_max=0.5)
    add(
        "future",
        "Investment mandate — scheduled future restatement",
        mandate_text(future, style, appendix=False),
        -1,
        3,
    )
    add(
        "restrictions",
        "Current dealing-restriction register",
        permissions_text(constraints, style),
        -14,
        -10,
    )
    add(
        "classification",
        "Current portfolio classification register",
        "SIMULATED exercise classification, not an official index taxonomy.\n"
        + "\n".join(
            f"{s} belongs to the {sector} industry bucket."
            for s, sector in constraints["sectors"].items()
        ),
        -14,
        -10,
    )
    required = [ids["mandate"], ids["restrictions"]]
    if constraints["sector_max"]:
        required.append(ids["classification"])
    if appendix:
        add(
            "allocation",
            "Current aggregate allocation schedule",
            "SIMULATED current allocation schedule, incorporated by the mandate.\n"
            + "\n".join(
                sector_clause(s, cap, style) for s, cap in constraints["sector_max"].items()
            )
            + "\nUse the current classification register.",
        )
        required.append(ids["allocation"])
    add(
        "service",
        "Service preferences and meeting schedule",
        "SIMULATED administration notice. Electronic statements and quarterly meetings "
        "are preferred. No investment or dealing conditions change.",
        -5,
        -5,
    )
    extra = rng.randrange(0, 4 if appendix else 5)
    for name in ("outlook", "notice", "methodology")[:extra]:
        add(
            name,
            {
                "outlook": "Research commentary — not an account instruction",
                "notice": "Account contact details",
                "methodology": "Reference return calculation methodology",
            }[name],
            "SIMULATED optional reference. This document imposes no mandate, cash, concentration "
            "or trading restriction. No authority to change the effective client instructions.",
            -9,
            -8,
        )
    rng.shuffle(documents)
    oracle = solve_explicit(account, quotes, constraints, execution=execution)
    hold_metrics = evaluate_plan(account, quotes, account["positions"], execution)
    hold_violations = constraint_violations(account, quotes, constraints, hold_metrics)
    kind = (
        "infeasible"
        if oracle["status"] == "infeasible"
        else "hold"
        if not hold_violations
        else "mixed"
    )
    oracle_plan = {**oracle, "citations": required}
    task = dict(
        profile=PROFILE,
        task_id=task_id,
        base_scenario_id=base_id,
        version_chain_id=base_id,
        variant=variant,
        split=split,
        task_type=kind,
        as_of=now.isoformat(),
        request=(
            "Satisfy all currently applicable client instructions with minimum unnecessary "
            "turnover, actual execution and citations. Do not predict future prices."
        ),
        account=copy.deepcopy(account),
        quotes=quotes,
        execution=execution,
        documents=documents,
        tool_budget=8,
        source=dict(
            source_split="train",
            episode_id=market.episode_id,
            session_index=10,
            quote_time=now.isoformat(),
            quote_field="close",
            synthetic_mandate=True,
            synthetic_account=True,
            formal_valid_access=False,
            formal_final_oos_access=False,
        ),
        gold=dict(
            constraints=constraints,
            required_citations=required,
            oracle=oracle,
            oracle_plan=oracle_plan,
            hold_violations=hold_violations,
            minimum_information_reads=len(required),
            family=family,
            sector_cap_count=len(constraints["sector_max"]),
        ),
    )
    if not verify_submission(task, oracle_plan, submitted_constraints=constraints)["success"]:
        raise RuntimeError(f"oracle verification failed {task_id}")
    return task


def generate(output, *, prepared=PREPARED, seed=SEED):
    output = Path(output).resolve()
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    if (data / "manifest.json").exists():
        return json.loads((data / "manifest.json").read_text())
    markets = list(load_train_markets(prepared, data / "market_adapter").values())
    order = list(range(len(markets)))
    random.Random(seed).shuffle(order)
    names = ("train", "sft_train", "rl_train", "development", "test")
    streams = {s: (data / f"{s}.jsonl").open("x") for s in names}
    counts, types, cap_counts = Counter(), {}, {}
    pairing, group_ids, rejected = [], {s: [] for s in names}, 0
    try:
        for group in range(1200):
            split = "train" if group < 1000 else "development" if group < 1100 else "test"
            portion = "sft_train" if group < 400 else "rl_train" if group < 1000 else split
            market = markets[order[group % len(order)]]
            # Retry genuine infeasible random combinations; never use model
            # outcomes or held-out performance to choose a task.
            for attempt in range(30):
                _, family, account, quotes, constraints = scenario(group, market, seed, attempt)
                tasks = [
                    make_task(group, v, split, market, seed, account, quotes, constraints)
                    for v in (0, 1)
                ]
                if family != "mixed" or all(t["task_type"] != "infeasible" for t in tasks):
                    break
                rejected += 1
            else:
                raise RuntimeError(f"could not construct feasible pair {group}")
            for t in tasks:
                t["training_portion"] = portion if split == "train" else None
                t["gold"]["pair_id"] = t["base_scenario_id"]
                for destination in ["train", portion] if split == "train" else [split]:
                    streams[destination].write(canonical(t) + "\n")
                    counts[destination] += 1
                    types.setdefault(destination, Counter())[t["task_type"]] += 1
                    cap_counts.setdefault(destination, Counter())[
                        len(t["gold"]["constraints"]["sector_max"])
                    ] += 1
            a, b = tasks
            status_changed = a["gold"]["oracle"]["status"] != b["gold"]["oracle"]["status"]
            if a["gold"]["oracle"]["status"] == b["gold"]["oracle"]["status"] == "feasible":
                gap = abs(
                    a["gold"]["oracle"]["metrics"]["traded_raw_notional"]
                    - b["gold"]["oracle"]["metrics"]["traded_raw_notional"]
                )
                targets_changed = any(
                    abs(
                        a["gold"]["oracle"]["target_shares"][s]
                        - b["gold"]["oracle"]["target_shares"][s]
                    )
                    * quotes[s]
                    > 1.0
                    for s in quotes
                )
                original_fails_partner = bool(
                    constraint_violations(
                        account, quotes, b["gold"]["constraints"], a["gold"]["oracle"]["metrics"]
                    )
                )
            else:
                gap, targets_changed, original_fails_partner = None, False, False
            pairing.append(
                dict(
                    base_scenario_id=a["base_scenario_id"],
                    split=split,
                    training_portion=portion,
                    family=family,
                    task_ids=[a["task_id"], b["task_id"]],
                    statuses=[a["gold"]["oracle"]["status"], b["gold"]["oracle"]["status"]],
                    status_changed=status_changed,
                    reference_targets_changed=targets_changed,
                    first_reference_violates_partner=original_fails_partner,
                    reference_turnover_notional_gap=gap,
                    business_sensitive=status_changed
                    or original_fails_partner
                    or gap is not None
                    and gap > 1.0,
                )
            )
            for destination in ["train", portion] if split == "train" else [split]:
                group_ids[destination].append(a["base_scenario_id"])
            if group % 50 == 49:
                for f in streams.values():
                    f.flush()
                print(
                    canonical(dict(stage="GENERATING", groups=group + 1, tasks=dict(counts))),
                    flush=True,
                )
    finally:
        for stream in streams.values():
            stream.close()
    (data / "pairing.jsonl").write_text("".join(canonical(p) + "\n" for p in pairing))
    manifest = dict(
        status="COMPLETE",
        profile=PROFILE,
        seed=seed,
        tasks=2400,
        counts=dict(counts),
        task_type_counts={s: dict(c) for s, c in types.items()},
        sector_cap_cardinalities={s: dict(c) for s, c in cap_counts.items()},
        base_group_count=1200,
        variants_per_base=2,
        variant_semantics="one cash floor or industry cap changes; shared text style/doc ids",
        grouped_splits=True,
        base_scenario_groups=group_ids,
        source_prepared=str(Path(prepared).resolve()),
        source_split="train",
        quote_rule="old Train session10 already-known close",
        source_account_seed_disjoint_from_v1=True,
        teacher_training_portion="sft_train",
        rl_training_portion="rl_train",
        training_portions_share_atomic_clauses_and_generation_rules=True,
        heldout_styles=dict(train=[0, 1, 2, 3], development=[4, 5], test=[6, 7]),
        maximum_tool_calls=8,
        required_document_reads=[2, 4],
        catalog_documents=[6, 10],
        public_document_ids=(
            "pseudorandom opaque doc ids; catalog titles/times supply ordinary retrieval clues"
        ),
        synthetic_mandates=True,
        synthetic_accounts=True,
        synthetic_sector_register=True,
        gold_hidden_from_agent=True,
        formal_valid_access=False,
        formal_final_oos_access=False,
        generation_oracle_validation_only=True,
        reserved_test_model_or_baseline_evaluation=False,
        business_sensitive_pair_counts={
            s: sum(p["business_sensitive"] for p in pairing if p["split"] == s)
            for s in ("train", "development", "test")
        },
        rejected_infeasible_random_mixed_candidates=rejected,
        paths={s: str(data / f"{s}.jsonl") for s in names},
    )
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(canonical(dict(stage="DATA_READY", **manifest)), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output, prepared=args.prepared, seed=args.seed)


if __name__ == "__main__":
    main()

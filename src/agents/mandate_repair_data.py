"""Simulated versioned mandates on real historical Train quote snapshots.

Natural-language documents are public through retrieval. Gold and task type are
offline-only. Both paraphrases and every version of a base scenario stay in the
same partition. No Formal Valid or Final OOS source is used.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from src.agents.mandate_repair_verifier import (
    constraint_violations,
    evaluate_plan,
    solve_explicit,
    verify_submission,
)
from src.agents.portfolio_management import ROOT, load_train_markets

PROFILE = "mandate_repair_simulated_train_quotes_v1"
PREPARED = ROOT / "runs/m2-portfolio-teacher-sft-20261028-v1/data/all_train.parquet"
KINDS = ("cash", "exposure", "restriction", "mixed", "hold", "infeasible")
ASSET_GROUPS = {
    "Technology": ("MSFT", "CRM", "IBM", "AAPL"),
    "Financials": ("GS", "V", "AXP", "JPM", "TRV"),
    "Industrials": ("SHW", "CAT", "HON", "BA"),
    "Consumer": ("HD", "MCD", "AMZN", "PG"),
    "Healthcare": ("UNH", "AMGN", "JNJ"),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def clauses(constraints, symbols, style):
    introductions = (
        "The following limits govern this account after execution and all fees.",
        "Apply these instructions to final holdings, measured after transaction charges.",
        "Assess this replacement mandate on the completed account's net marked NAV.",
        "The following end-state conditions include all trading fees.",
        "This restatement sets post-cost cash and concentration requirements.",
        "At completion enforce these client limits using net portfolio equity.",
    )
    lines = [introductions[style]]
    lines.extend(
        [
            f"Cash available must be at least USD {constraints['cash_min']:,.2f}.",
            f"Total stock exposure must not exceed {constraints['exposure_max'] * 100:.1f} "
            "percent of net account equity.",
        ]
    )
    lines.extend(
        f"The holding in {s} is capped at {constraints['symbol_max'][s] * 100:.1f} "
        "percent of net account equity."
        for s in symbols
    )
    for sector, limit in constraints["sector_max"].items():
        lines.append(
            f"Combined {sector} holdings may not exceed {limit * 100:.1f} percent of net "
            "account equity. Use the applicable classification register."
        )
    if not constraints["sector_max"]:
        lines.append("There is no additional aggregate industry cap in this mandate.")
    lines.append(
        "Trade permissions are governed separately by the current dealing-restriction register. "
        "No cash injection, borrowing or short sale is permitted. Fractional shares are permitted."
    )
    return "\n".join(lines)


def scenario(index, market, seed):
    rng = random.Random(seed + index * 104729)
    kind = KINDS[index % len(KINDS)]
    buckets = {
        sector: rng.sample(list(pool), 1 if sector in ("Consumer", "Healthcare") else 2)
        for sector, pool in ASSET_GROUPS.items()
    }
    symbols = [s for items in buckets.values() for s in items]
    sectors = {s: sector for sector, items in buckets.items() for s in items}
    quotes = {s: float(market.closes[10][s]) for s in symbols}
    cash_fraction = 0.25 if kind == "hold" else rng.choice((0.05, 0.08, 0.10))
    amounts = [rng.uniform(0.7, 1.3) for s in symbols]
    if kind == "restriction":
        amounts[0] = 3.0
    weights = {
        s: a / sum(amounts) * (1 - cash_fraction) for s, a in zip(symbols, amounts, strict=True)
    }
    account = dict(
        cash=1_000_000 * cash_fraction,
        positions={s: weights[s] * 1_000_000 / quotes[s] for s in symbols},
        initial_equity=1_000_000.0,
        currency="USD",
    )
    constraints = dict(
        cash_min=0.0,
        exposure_max=1.0,
        symbol_max={s: 0.40 for s in symbols},
        sector_max={},
        sectors=sectors,
        cannot_sell=[],
        cannot_buy=[],
    )
    if kind == "cash":
        constraints["cash_min"] = rng.choice((200_000.0, 250_000.0, 300_000.0))
    elif kind == "exposure":
        constraints.update(
            exposure_max=rng.choice((0.65, 0.70, 0.75)),
            symbol_max={s: 0.15 for s in symbols},
            sector_max={"Technology": 0.18},
        )
    elif kind == "restriction":
        constraints.update(cash_min=250_000.0, cannot_sell=[symbols[0]])
    elif kind == "mixed":
        constraints.update(
            cash_min=200_000.0,
            exposure_max=0.70,
            symbol_max={s: 0.20 for s in symbols},
            sector_max={"Technology": 0.16},
            cannot_sell=[buckets["Healthcare"][0]],
            cannot_buy=[buckets["Technology"][0]],
        )
        if 400 <= index < 450:
            # Declared before any model evaluation: development uses an unseen
            # industry/permission combination, readable in the same documents.
            constraints["sector_max"] = {"Financials": 0.16}
            constraints["cannot_buy"] = [buckets["Technology"][0], buckets["Industrials"][0]]
        elif index >= 450:
            # Test adds a second simultaneous industry cap. Its data stay held
            # out after generation, including from teacher construction.
            constraints["sector_max"] = {"Technology": 0.16, "Consumer": 0.06}
    elif kind == "hold":
        constraints.update(
            cash_min=100_000.0,
            exposure_max=0.90,
            symbol_max={s: 0.20 for s in symbols},
            sector_max={"Technology": 0.35},
            cannot_sell=[symbols[-1]],
        )
    else:
        constraints.update(cash_min=account["cash"] + 200_000, cannot_sell=list(symbols))
    return kind, symbols, quotes, account, constraints


def make_task(group, variant, split, market, seed):
    kind, symbols, quotes, account, constraints = scenario(group, market, seed)
    style = {"train": 0, "development": 2, "test": 4}[split] + variant
    task_id, base = f"mandate-{group:04d}-v{variant}", f"mandate-base-{group:04d}"
    now = datetime.fromisoformat(market.kwargs["episode_dates"][10])
    execution = dict(commission_bps=10.0, slippage_bps=5.0)
    ids = {
        name: f"{task_id}/{name}"
        for name in (
            "original",
            "amendment",
            "future_amendment",
            "classification",
            "restrictions",
            "notice",
        )
    }
    documents = []

    def doc(name, title, text, available, effective, superseded=False):
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

    original = copy.deepcopy(constraints)
    original.update(cash_min=400_000.0 if kind == "hold" else 0.0, exposure_max=1.0)
    original["sector_max"] = {}
    original["symbol_max"] = {s: 0.40 for s in symbols}
    doc(
        "original",
        "Account mandate, original issue, version 1",
        "SIMULATED CLIENT DOCUMENT. Original mandate, replaced by a later effective "
        "full restatement if one exists.\n" + clauses(original, symbols, style),
        -30,
        -28,
        True,
    )
    doc(
        "amendment",
        "Account mandate amendment, full restatement, version 2",
        f"SIMULATED CLIENT DOCUMENT. Version 2 takes effect "
        f"{(now - timedelta(days=3)).isoformat()} and supersedes version 1 in full.\n"
        + clauses(constraints, symbols, style),
        -7,
        -3,
    )
    future = copy.deepcopy(constraints)
    future.update(cash_min=500_000.0, exposure_max=0.50)
    doc(
        "future_amendment",
        "Scheduled mandate amendment, version 3, not yet effective",
        f"SIMULATED CLIENT DOCUMENT. Version 3 only takes effect "
        f"{(now + timedelta(days=3)).isoformat()}, not before that time.\n"
        + clauses(future, symbols, style),
        -1,
        3,
    )
    doc(
        "classification",
        "Current portfolio classification register",
        "SIMULATED CLASSIFICATION REGISTER for this exercise, not an official index taxonomy.\n"
        + "\n".join(
            f"{s} belongs to the {constraints['sectors'][s]} industry bucket." for s in symbols
        ),
        -20,
        -20,
    )
    lines = ["SIMULATED DEALING-RESTRICTION REGISTER, applicable at this execution time."]
    lines.extend(f"Do not sell any shares of {s}." for s in constraints["cannot_sell"])
    lines.extend(f"Do not buy any shares of {s}." for s in constraints["cannot_buy"])
    if not constraints["cannot_sell"] and not constraints["cannot_buy"]:
        lines.append("There are no additional buy or sell restrictions on listed assets.")
    doc("restrictions", "Current dealing-restriction register", "\n".join(lines), -20, -20)
    doc(
        "notice",
        "Service notice: quarterly communication preferences",
        "SIMULATED SERVICE NOTICE. The customer prefers electronic statements and an annual "
        "review call. This administrative notice changes no cash, dealing or investment terms.",
        -5,
        -5,
    )
    required = [ids["amendment"], ids["restrictions"]]
    if constraints["sector_max"]:
        required.append(ids["classification"])
    oracle = solve_explicit(account, quotes, constraints, execution=execution)
    if (oracle["status"] == "infeasible") != (kind == "infeasible"):
        raise RuntimeError(f"unexpected generated feasibility: {task_id}")
    hold_metrics = evaluate_plan(account, quotes, account["positions"], execution)
    hold_violations = constraint_violations(account, quotes, constraints, hold_metrics)
    if (not hold_violations) != (kind == "hold"):
        raise RuntimeError(f"inconsistent generated hold category: {task_id}")
    oracle_plan = {**oracle, "citations": required}
    task = dict(
        profile=PROFILE,
        task_id=task_id,
        base_scenario_id=base,
        version_chain_id=base,
        variant=variant,
        split=split,
        task_type=kind,
        as_of=now.isoformat(),
        request=(
            "Using instructions effective at the account snapshot time, submit a low-turnover "
            "portfolio repair with supporting document citations. Retain holdings if already "
            "compliant. If the effective terms cannot be met, report infeasible with evidence. "
            "Main verification uses only the fixed known quotes."
        ),
        account=account,
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
        ),
    )
    if not verify_submission(task, oracle_plan, submitted_constraints=constraints)["success"]:
        raise RuntimeError("oracle failed independent terminal verification")
    return task


def generate(output, *, prepared=PREPARED, seed=43):
    if seed != 43:
        raise ValueError("the initial task batch uses fixed seed43")
    output = Path(output).resolve()
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    if (data / "manifest.json").exists():
        raise FileExistsError("task data already complete")
    markets = list(load_train_markets(prepared, data / "market_adapter").values())
    indexes = list(range(len(markets)))
    random.Random(seed).shuffle(indexes)
    files = {s: (data / f"{s}.jsonl").open("x") for s in ("train", "development", "test")}
    counts, types = Counter(), {}
    groups = {s: [] for s in files}
    try:
        for group in range(500):
            split = "train" if group < 400 else "development" if group < 450 else "test"
            market = markets[indexes[group % len(indexes)]]
            groups[split].append(f"mandate-base-{group:04d}")
            types.setdefault(split, Counter())
            for variant in range(2):
                task = make_task(group, variant, split, market, seed)
                files[split].write(canonical(task) + "\n")
                counts[split] += 1
                types[split][task["task_type"]] += 1
            if group % 50 == 49:
                for stream in files.values():
                    stream.flush()
                print(
                    canonical(dict(status="GENERATING", groups=group + 1, tasks=dict(counts))),
                    flush=True,
                )
    finally:
        for stream in files.values():
            stream.close()
    manifest = dict(
        status="COMPLETE",
        data_build_version=2,
        profile=PROFILE,
        seed=seed,
        tasks=sum(counts.values()),
        counts=dict(counts),
        task_type_counts={s: dict(v) for s, v in types.items()},
        base_scenario_groups=groups,
        base_group_count=500,
        variants_per_base=2,
        related_versions_and_paraphrases_never_cross_partitions=True,
        template_styles=dict(train=[0, 1], development=[2, 3], test=[4, 5]),
        constraint_combination_holdouts=dict(
            development_mixed="Financials cap plus two cannot-buy assets",
            test_mixed="Technology and Consumer simultaneous industry caps",
            chosen_before_any_model_evaluation=True,
        ),
        source_prepared=str(Path(prepared).resolve()),
        source_split="train",
        quote_rule="existing historical Train, session10 already-known close; no future quote",
        synthetic_mandates=True,
        synthetic_accounts=True,
        synthetic_sector_register=True,
        assets_per_task=8,
        initial_account_equity=1_000_000,
        tool_budget=8,
        maximum_required_document_reads=3,
        main_execution="known quote; common proportional commission/slippage",
        gold_hidden_from_agent=True,
        metadata_hidden_from_agent=["gold", "task_type", "base_scenario_id", "version_chain_id"],
        document_metadata_hidden_from_agent=["superseded"],
        oracle="minimum raw-turnover explicit LP, post-fee weight denominator",
        formal_valid_access=False,
        formal_final_oos_access=False,
        paths={s: str(data / f"{s}.jsonl") for s in files},
    )
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(canonical(manifest), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    generate(args.output, prepared=args.prepared, seed=args.seed)


if __name__ == "__main__":
    main()

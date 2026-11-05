"""Real FinQA evidence and simulated, independently verifiable conditional mandates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import urllib.request
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from src.agents.mandate_repair_data import PREPARED
from src.agents.mandate_repair_verifier import solve_explicit
from src.agents.portfolio_management import ROOT, load_train_markets

PROFILE = "financial_evidence_conditional_mandate_v1"
DEFAULT = ROOT / "runs/m3-financial-evidence-20261030-v1"
SEED = 20261030


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def number(value):
    text = str(value).strip().replace(",", "")
    if text.startswith("const_"):
        text = text[6:].replace("m1", "-1")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", text):
        raise ValueError("unsupported_operand")
    return float(text.rstrip("%")) / (100 if text.endswith("%") else 1)


def financial_candidate(row):
    """Accept only arithmetic programs consistent with both published references."""
    qa = row["qa"]
    program = qa.get("program", "")
    matches = re.findall(
        r"(add|subtract|multiply|divide)\(\s*([^,()]+)\s*,\s*([^,()]+)\s*\)", program
    )
    residue = re.sub(r"(add|subtract|multiply|divide)\([^()]*\)", "", program)
    if not 1 <= len(matches) <= 3 or re.sub(r"\s|,", "", residue):
        raise ValueError("unsupported_program")
    values, expressions = [], []
    for operation, left, right in matches:
        args, encoded = [], []
        for token in (left.strip(), right.strip()):
            if token.startswith("#"):
                index = int(token[1:])
                args.append(values[index])
                encoded.append(expressions[index])
            else:
                value = number(token)
                args.append(value)
                encoded.append(repr(value))
        a, b = args
        value = (
            a + b if operation == "add" else a - b if operation == "subtract"
            else a * b if operation == "multiply" else a / b
        )
        if not math.isfinite(value) or abs(value) > 1e10:
            raise ValueError("nonfinite_or_large_program")
        symbol = dict(add="+", subtract="-", multiply="*", divide="/")[operation]
        expressions.append(f"({encoded[0]} {symbol} {encoded[1]})")
        values.append(value)
    value = values[-1]
    declared = qa.get("exe_ans")
    if type(declared) not in (float, int) or not math.isclose(
        value, declared, rel_tol=2e-4, abs_tol=1e-4
    ):
        raise ValueError("program_execution_reference_mismatch")
    answer = str(qa["answer"]).strip().replace(",", "").replace("$", "")
    match = re.fullmatch(
        r"\(?([-+]?\d+(?:\.\d+)?)\)?\s*(%|million|billion|thousand|years|days|months|times)?",
        answer, re.I,
    )
    if not match:
        raise ValueError("ambiguous_written_answer")
    written = float(match[1]) * (-1 if answer.startswith("(") else 1)
    suffix = (match[2] or "").lower()
    unit = "native_report_units"
    if suffix == "%":
        if math.isclose(value, written / 100, rel_tol=0.02, abs_tol=0.0003):
            unit = "ratio"
        elif math.isclose(value, written, rel_tol=0.02, abs_tol=0.03):
            value /= 100
            expressions[-1] = f"({expressions[-1]} / 100.0)"
            unit = "ratio"
        else:
            raise ValueError("written_answer_mismatch")
    elif abs(written - value) > max(abs(written) * 0.02, 0.03):
        raise ValueError("written_answer_mismatch")
    years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", qa["question"])))
    if not years:
        raise ValueError("no_explicit_question_period")
    # Original table cells and paragraph strings are retained, never gold retrieval text.
    text = (
        "\n".join(row["pre_text"]) + "\nTABLE (original supplied cell strings):\n"
        + json.dumps(row["table"], ensure_ascii=False) + "\n"
        + "\n".join(row["post_text"])
    )
    if len(text) > 11000 or len(expressions[-1]) > 450:
        raise ValueError("long_evidence_or_expression")
    company, report_year = row["id"].split("/")[:2]
    return dict(
        value=value, expression=expressions[-1], period="/".join(years), unit=unit,
        company=company, report_year=report_year, report_id=f"{company}/{report_year}",
        text=text, source=row,
    )


def load_source(output):
    directory = Path(output) / "source/finqa"
    directory.mkdir(parents=True, exist_ok=True)
    records, sources = [], []
    for split in ("train", "dev", "test"):
        url = f"https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/{split}.json"
        path = directory / f"{split}.json"
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        payload = path.read_bytes()
        rows = json.loads(payload)
        sources.append(dict(
            original_split=split, url=url, path=str(path.resolve()), records=len(rows),
            sha256=hashlib.sha256(payload).hexdigest(),
        ))
        for row in rows:
            records.append((split, row))
    return records, sources


def task_pair(index, split, candidate, distractors, market):
    rng = random.Random(SEED + index * 104729)
    quoted = list(market.closes[10])
    symbols = rng.sample(quoted, 8)
    controlled = symbols[0]
    quotes = {symbol: float(market.closes[10][symbol]) for symbol in symbols}
    equity = rng.choice((750_000.0, 1_000_000.0, 1_250_000.0))
    positions = {
        symbol: equity * (0.30 if i == 0 else 0.55 / 7) / quotes[symbol]
        for i, symbol in enumerate(symbols)
    }
    account = dict(cash=equity * 0.15, positions=positions,
                   initial_equity=equity, currency="USD")
    base = dict(
        cash_min=equity * 0.12, exposure_max=1.0,
        symbol_max={symbol: 0.40 for symbol in symbols},
        sector_max={}, sectors={}, cannot_sell=[], cannot_buy=[],
    )
    as_of = datetime.fromisoformat(market.kwargs["episode_dates"][10])
    task_base = f"financial-base-{index:04d}"
    ids = {
        name: "doc-" + hashlib.sha256(f"{SEED}/{index}/{name}".encode()).hexdigest()[:12]
        for name in ("mandate", "report", "report_copy", *[f"other{i}" for i in range(10)])
    }

    def document(name, title, text, source_id=None):
        return dict(
            doc_id=ids[name], title=title, text=text, kind=name,
            available_at=(as_of - timedelta(days=2)).isoformat(),
            effective_at=(as_of - timedelta(days=2)).isoformat(), superseded=False,
            source_id=source_id, library_availability_is_simulated=True,
        )

    report_title = (
        f"{candidate['company']} annual report {candidate['report_year']}, "
        f"financial excerpt {candidate['source']['filename'].split('/')[-1]}"
    )
    evidence = [
        document("report", report_title, candidate["text"], candidate["source"]["id"]),
        document("report_copy", report_title + " (duplicate archive copy)",
                 candidate["text"], candidate["source"]["id"]),
    ]
    for i, other in enumerate(distractors):
        title = (
            f"{other['company']} annual report {other['report_year']}, "
            f"financial excerpt {other['source']['filename'].split('/')[-1]}"
        )
        evidence.append(document(f"other{i}", title, other["text"], other["source"]["id"]))
    margin = max(abs(candidate["value"]) * 0.05, 0.01)
    pair = []
    for variant in (0, 1):
        threshold = candidate["value"] + (-margin if variant == 0 else margin)
        triggered = candidate["value"] > threshold
        condition = dict(
            company=candidate["company"], report_year=candidate["report_year"],
            question=candidate["source"]["qa"]["question"],
            period=candidate["period"], unit=candidate["unit"], comparison="greater",
            threshold=threshold, controlled_symbol=controlled, strict_cap=0.10, relaxed_cap=0.40,
        )
        mandate_text = (
            "SIMULATED CLIENT CONDITIONAL MANDATE. All limits refer to final equity after "
            "fees. The reference issuer is contractually specified; it need not issue the "
            "portfolio asset. This is not a return prediction.\n"
            f"Cash available must be at least USD {base['cash_min']:,.2f}.\n"
            "Total stock exposure must not exceed 100.0 percent of net account equity.\n"
            + "\n".join(
                f"The holding in {s} is capped at 40.0 percent of net account equity."
                for s in symbols
            )
            + "\nThere is no aggregate industry cap, and no additional buy/sell restriction. "
            "No borrowing, shorting or cash injection is permitted.\n"
            "CONDITIONAL OVERRIDE (use the specified original report and requested periods):\n"
            + canonical(condition)
            + "\nAnswer the financial question in the declared unit. ratio means a fraction "
            "(e.g. 12 percent is 0.12), not percentage points. native_report_units means "
            "the same amount/count units used by the original report and question. "
            "report_year identifies the source report, while period lists the years "
            "explicitly requested by the financial question.\n"
            "If metric.value > threshold, replace the controlled asset cap with strict_cap; "
            "otherwise use relaxed_cap. Submit the numeric metric, exact period/unit, "
            "triggered boolean and plan, with the current mandate and sufficient report "
            "evidence. A conservative portfolio alone is not a correct metric answer."
        )
        documents = [document("mandate", "Current conditional investment mandate", mandate_text),
                     *copy.deepcopy(evidence)]
        rng_for_order = random.Random(SEED + index)
        rng_for_order.shuffle(documents)
        effective = copy.deepcopy(base)
        effective["symbol_max"][controlled] = 0.10 if triggered else 0.40
        oracle = solve_explicit(account, quotes, effective,
                                execution=dict(commission_bps=10.0, slippage_bps=5.0))
        if oracle["status"] != "feasible":
            raise RuntimeError("conditional task oracle unexpectedly infeasible")
        pair.append(dict(
            profile=PROFILE, task_id=f"financial-{index:04d}-v{variant}",
            base_scenario_id=task_base, version_chain_id=task_base, variant=variant, split=split,
            task_type="conditional_repair" if triggered else "conditional_hold",
            as_of=as_of.isoformat(), account=copy.deepcopy(account), quotes=quotes,
            execution=dict(commission_bps=10.0, slippage_bps=5.0), documents=documents,
            condition=condition, tool_budget=10,
            request=(
                "Read the applicable conditional mandate, find and calculate the reference "
                "financial metric from the specified report, evaluate its threshold, then "
                "execute a compliant low-turnover plan and submit metric plus evidence."
            ),
            source=dict(
                financial_source="FinQA official public release",
                finqa_id=candidate["source"]["id"], company=candidate["company"],
                report_year=candidate["report_year"], report_id=candidate["report_id"],
                quote_source_split="train", quote_episode=market.episode_id,
                quote_session=10, reference_issuer_need_not_match_controlled_asset=True,
                synthetic_mandate=True, synthetic_account=True,
                source_report_values_unmodified=True, library_dates_are_simulated=True,
                no_real_filing_timestamp_claim=True,
            ),
            gold=dict(
                constraints=effective, base_constraints=copy.deepcopy(base),
                required_citations=[ids["mandate"]],
                evidence_groups=[[ids["report"], ids["report_copy"]]],
                metric_value=candidate["value"], metric_period=candidate["period"],
                metric_unit=candidate["unit"], metric_tolerance=max(
                    1e-4, abs(candidate["value"]) * 1e-4
                ),
                triggered=triggered, expression=candidate["expression"],
                program=candidate["source"]["qa"]["program"],
                original_answer=candidate["source"]["qa"]["answer"],
                original_exe_ans=candidate["source"]["qa"]["exe_ans"],
                supporting_facts=candidate["source"]["qa"].get("gold_inds", {}),
                oracle=oracle, oracle_plan={**oracle, "citations": [ids["mandate"], ids["report"]]},
                supporting_document_ids=[ids["report"], ids["report_copy"]],
            ),
        ))
    return pair


def generate(output, *, prepared=PREPARED):
    output = Path(output).resolve()
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    if (data / "manifest.json").exists():
        return json.loads((data / "manifest.json").read_text())
    raw, provenance = load_source(output)
    candidates, rejected = [], Counter()
    for original_split, row in raw:
        try:
            candidate = financial_candidate(row)
            candidate["original_split"] = original_split
            candidates.append(candidate)
        except (ValueError, ZeroDivisionError, IndexError, OverflowError) as error:
            rejected[str(error)] += 1
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    # Exactly one source question per complete company-year report. Its two
    # threshold variants stay together; all company reports stay in one split.
    reports = {}
    for candidate in candidates:
        reports.setdefault(candidate["report_id"], candidate)
    companies = sorted({c["company"] for c in reports.values()})
    rng = random.Random(SEED)
    rng.shuffle(companies)
    assignments = dict(
        development=companies[:20], test=companies[20:40], train=companies[40:]
    )
    chosen = {}
    for split, count in (("train", 400), ("development", 50), ("test", 50)):
        pool = [c for c in reports.values() if c["company"] in assignments[split]]
        rng.shuffle(pool)
        if len(pool) < count:
            raise RuntimeError(f"not enough independent reports for {split}: {len(pool)}")
        chosen[split] = pool[:count]
    markets = list(load_train_markets(prepared, data / "market_adapter").values())
    task_counts, units = Counter(), Counter()
    pair_records, report_groups = [], {}
    index = 0
    for split in ("train", "development", "test"):
        report_groups[split] = []
        with (data / f"{split}.jsonl").open("x") as stream:
            for candidate in chosen[split]:
                others = [
                    c for c in candidates
                    if c["report_id"] != candidate["report_id"]
                    and c["company"] in assignments[split]
                ]
                rng_for_task = random.Random(SEED + index)
                distractors = rng_for_task.sample(others, 10)
                pair = task_pair(
                    index, split, candidate, distractors, markets[index % len(markets)]
                )
                for task in pair:
                    stream.write(canonical(task) + "\n")
                    task_counts[split] += 1
                    units[task["condition"]["unit"]] += 1
                report_groups[split].append(candidate["report_id"])
                pair_records.append(dict(
                    base_scenario_id=pair[0]["base_scenario_id"], split=split,
                    task_ids=[t["task_id"] for t in pair],
                    source_report_id=candidate["report_id"],
                    controlled_symbol=pair[0]["condition"]["controlled_symbol"],
                    same_report_numbers=True, changed_field="condition.threshold",
                    triggered=[t["gold"]["triggered"] for t in pair], business_sensitive=True,
                ))
                index += 1
    (data / "pairing.jsonl").write_text("".join(canonical(p) + "\n" for p in pair_records))
    manifest = dict(
        status="COMPLETE", profile=PROFILE, seed=SEED, data_build_version=1,
        counts=dict(task_counts), tasks=sum(task_counts.values()),
        source_records=len(raw), accepted_candidates=len(candidates),
        source_provenance=provenance, rejected=dict(rejected),
        company_partition=assignments, report_groups=report_groups,
        independent_source_reports={k: len(v) for k, v in report_groups.items()},
        variants_per_report=2, source_question_per_report=1,
        original_finqa_split_is_not_experiment_split=True,
        original_public_train_dev_test_combined_then_company_group_split=True,
        canonical_metric_unit_counts=dict(units),
        source_table_and_paragraph_strings_unmodified=True,
        public_evidence_fields=["pre_text", "table", "post_text"],
        answers_programs_gold_support_are_offline_only=True,
        maximum_tool_calls=10, catalog_documents=13,
        financial_sufficient_sources="either of two complete identical report copies",
        required_mandate_citation=True, fixed_all_doc_citation_requirement=False,
        report_year_distinct_from_measurement_period=True,
        library_availability_dates_are_synthetic=True, real_pit_latest_report_claim=False,
        reference_issuer_and_traded_asset_relation="explicit simulated contractual trigger",
        source_quote_split="old market Train only", synthetic_mandates=True,
        formal_valid_access=False, formal_final_oos_access=False,
        reserved_test_performance_evaluation=False,
        teacher_adaptation_tasks=64,
        paths={s: str(data / f"{s}.jsonl") for s in task_counts},
    )
    write_json(data / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    args = parser.parse_args()
    value = generate(args.output, prepared=args.prepared)
    print(canonical({k: value[k] for k in ("status", "counts", "accepted_candidates")}))


if __name__ == "__main__":
    main()

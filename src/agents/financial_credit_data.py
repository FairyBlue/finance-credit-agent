"""Fresh financial questions for the bounded, strong-SFT credit experiment.

The financial task and verifier are unchanged. New target questions exclude
every source ID previously exposed as a target OR a distractor. Development
tasks are explicitly reused; fresh test questions remain behind model freeze.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from src.agents.financial_evidence_data import (
    canonical,
    financial_candidate,
    task_pair,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "runs/m3-financial-evidence-20261030-v2"
DEFAULT = ROOT / "runs/m3-financial-credit-20261030-v1"
SEED = 202610301
PROFILE = "financial_credit_fresh_questions_v1"
OBSERVATIONS = "financial_evidence_conditional_mandate_v2_opaque_public_ids"


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing data: {path}")
    path.write_text("".join(canonical(row) + "\n" for row in rows))


def load_inputs(parent):
    parent = Path(parent).resolve()
    manifest = json.loads((parent / "data/manifest.json").read_text())
    previous = {
        split: read_rows(parent / "data" / f"{split}.jsonl")
        for split in ("train", "development", "test")
    }
    used_targets = {task["source"]["finqa_id"] for rows in previous.values() for task in rows}
    exposed = used_targets | {
        doc["source_id"] for rows in previous.values() for task in rows
        for doc in task["documents"] if doc.get("source_id")
    }
    candidates, rejected = [], Counter()
    for split in ("train", "dev", "test"):
        path = parent / "source/finqa" / f"{split}.json"
        for row in json.loads(path.read_text()):
            try:
                value = financial_candidate(row)
            except (ValueError, ZeroDivisionError, IndexError, OverflowError) as error:
                rejected[str(error)] += 1
                continue
            value["original_split"] = split
            candidates.append(value)
    return manifest, previous, used_targets, exposed, candidates, rejected


def recorded_quote_markets(previous_train):
    """Reuse recorded old-Train quotes rather than loading market adapters."""
    markets = []
    seen = set()
    for task in previous_train:
        if task["base_scenario_id"] in seen:
            continue
        seen.add(task["base_scenario_id"])
        markets.append(SimpleNamespace(
            closes={10: copy.deepcopy(task["quotes"])},
            kwargs={"episode_dates": [task["as_of"]] * 11},
            episode_id=task["source"]["quote_episode"],
        ))
    return markets


def select_questions(candidates, companies, exposed, split, count):
    pool = [
        candidate for candidate in candidates
        if candidate["company"] in companies and candidate["source"]["id"] not in exposed
    ]
    random.Random(SEED + (0 if split == "train" else 1)).shuffle(pool)
    # Distinct reports within each fresh split, while old Train reports may be
    # reused for a new question. Company partitions still never cross regions.
    reports = {}
    for candidate in pool:
        reports.setdefault(candidate["report_id"], candidate)
    chosen = list(reports.values())[:count]
    if len(chosen) != count:
        raise RuntimeError(f"not enough fresh reports/questions in {split}: {len(reports)}")
    return chosen, len(pool), len(reports)


def build_split(split, candidates, chosen, companies, markets):
    result, pairings = [], []
    offset = 10000 if split == "train" else 20000
    for ordinal, candidate in enumerate(chosen):
        index = offset + ordinal
        distractor_pool = [
            other for other in candidates
            if other["company"] in companies and other["report_id"] != candidate["report_id"]
        ]
        distractors = random.Random(SEED + index).sample(distractor_pool, 10)
        pair = task_pair(index, split, candidate, distractors, markets[ordinal % len(markets)])
        for task in pair:
            task["task_id"] = f"financial-credit-{index:05d}-v{task['variant']}"
            task["base_scenario_id"] = f"financial-credit-base-{index:05d}"
            task["version_chain_id"] = task["base_scenario_id"]
            task["source"].update(
                fresh_source_question=True,
                source_id_not_in_parent_targets_or_distractors=True,
                quote_reuse="parent recorded Train eight-asset quote snapshot",
            )
        result.extend(pair)
        pairings.append(dict(
            split=split, base_scenario_id=pair[0]["base_scenario_id"],
            source_id=candidate["source"]["id"], source_report_id=candidate["report_id"],
            task_ids=[task["task_id"] for task in pair],
            changed_field="condition.threshold", same_report_values=True,
            public_observation_profile=OBSERVATIONS,
        ))
    return result, pairings


def validate_rows(rows, companies, excluded_sources):
    ids = [task["task_id"] for task in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate fresh task IDs")
    for task in rows:
        if task["source"]["finqa_id"] in excluded_sources:
            raise RuntimeError("fresh target was previously exposed")
        if task["source"]["company"] not in companies:
            raise RuntimeError("target company crossed its partition")
        for doc in task["documents"]:
            if doc.get("source_id") and doc["source_id"].split("/")[0] not in companies:
                raise RuntimeError("distractor company crossed its partition")


def generate(output=DEFAULT, parent=PARENT, part="all"):
    output, parent = Path(output).resolve(), Path(parent).resolve()
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    target_manifest = data / "manifest.json"
    if target_manifest.exists():
        current = json.loads(target_manifest.read_text())
        if current["status"] == "COMPLETE" or part == "train":
            return current
    old, previous, used_targets, exposed, candidates, rejected = load_inputs(parent)
    companies = {key: set(values) for key, values in old["company_partition"].items()}
    markets = recorded_quote_markets(previous["train"])
    source_dir = output / "source/finqa"
    source_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "dev", "test"):
        destination = source_dir / f"{name}.json"
        if not destination.exists():
            os.link(parent / "source/finqa" / f"{name}.json", destination)
    if not target_manifest.exists():
        chosen, pool_size, reports = select_questions(candidates, companies["train"], exposed,
                                                    "train", 128)
        train, pairings = build_split("train", candidates, chosen, companies["train"], markets)
        validate_rows(train, companies["train"], exposed)
        write_rows(data / "train.jsonl", train)
        probe = random.Random(SEED + 32).sample(train, 32)
        write_rows(data / "probe32.jsonl", probe)
        # Development is deliberately the same already-used development region.
        (data / "development.jsonl").write_bytes((parent / "data/development.jsonl").read_bytes())
        write_rows(data / "pairing.jsonl", pairings)
        current = dict(
            status="TRAIN_READY", training_ready=True, fresh_test_ready=False,
            profile=PROFILE, observation_profile=OBSERVATIONS, seed=SEED,
            parent_data=str(parent / "data"), counts=dict(train=256, development=100, test=0),
            company_partition=old["company_partition"],
            prior_target_source_questions=len(used_targets),
            prior_exposed_target_or_distractor_source_ids=len(exposed),
            fresh_target_exclusion="all parent target and distractor source IDs",
            train_source_questions=128, train_distinct_reports=128,
            available_fresh_train_questions=pool_size, available_fresh_train_reports=reports,
            selected_train_source_ids=[c["source"]["id"] for c in chosen],
            probe_tasks=32,
            probe_source_questions=len({t["source"]["finqa_id"] for t in probe}),
            probe_selection="single random draw from all fresh Train tasks; no model filtering",
            development_reused=True, development_already_used_for_model_selection=True,
            development_source=str(parent / "data/development.jsonl"),
            task_environment_and_terminal_reward_unchanged=True,
            company_and_distractor_partitions_unchanged=True,
            fresh_train_report_may_have_other_old_sft_questions=True,
            official_finqa_split_is_not_experiment_split=True,
            source_provenance=old["source_provenance"], source_rejections=dict(rejected),
            new_teacher_sft_generated=False, quote_adapter_rebuilt=False,
            quote_source="recorded old-Train eight-asset quote snapshots",
            reserved_test_model_evaluation=False,
            formal_valid_access=False, formal_final_oos_access=False,
            paths={s: str(data / f"{s}.jsonl") for s in ("train", "development", "test")},
            probe_path=str(data / "probe32.jsonl"),
        )
        write_json(target_manifest, current)
    if part == "train":
        return current
    chosen, pool_size, reports = select_questions(candidates, companies["test"], exposed,
                                                "test", 50)
    test, pairings = build_split("test", candidates, chosen, companies["test"], markets)
    validate_rows(test, companies["test"], exposed)
    train_ids = set(current["selected_train_source_ids"])
    if train_ids.intersection(c["source"]["id"] for c in chosen):
        raise RuntimeError("new Train/Test source question overlap")
    write_rows(data / "test.jsonl", test)
    with (data / "pairing.jsonl").open("a") as stream:
        stream.writelines(canonical(row) + "\n" for row in pairings)
    current.update(
        status="COMPLETE", fresh_test_ready=True,
        counts=dict(train=256, development=100, test=100),
        test_source_questions=50, test_distinct_reports=50,
        available_fresh_test_questions=pool_size, available_fresh_test_reports=reports,
        selected_test_source_ids=[c["source"]["id"] for c in chosen],
        fresh_test_questions_not_used_for_training_or_selection=True,
    )
    write_json(target_manifest, current)
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--part", choices=("train", "test", "all"), default="all")
    args = parser.parse_args()
    result = generate(args.output, args.parent, args.part)
    print(canonical({k: result[k] for k in ("status", "counts", "training_ready")}))


if __name__ == "__main__":
    main()

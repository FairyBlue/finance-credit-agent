#!/usr/bin/env python3
"""Descriptive readout of the bounded stochastic-continuation feasibility pilot.

Reads persisted JSON evidence only. Two paths per cell are repeated pilot
continuations, never separate estimation/reference samples or market episodes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROFILE = "stochastic_continuation_feasibility_readout_v1"
EXPECTED_CELLS = 77
EXPECTED_PATHS = 154
REQUEST_LIMIT = 6904
DAYS = (0, 5, 10, 15)


def require(condition: bool, description: str) -> None:
    if not condition:
        raise ValueError(description)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict]:
    return list(iter_rows(path))


def iter_rows(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: object) -> bool:
    return type(value) in (float, int) and math.isfinite(value)


def indexed(items: list[dict], field: str) -> dict:
    result = {row[field]: row for row in items}
    require(len(result) == len(items), f"duplicate {field}")
    return result


def write_csv(path: Path, items: list[dict]) -> None:
    require(bool(items), "nonempty CSV output")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(items[0]))
        writer.writeheader()
        for row in items:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def summarize(planned: list[dict], results: list[dict]) -> tuple[list, list, list, dict]:
    """Accept normalized driver records; derive coverage without dropping failures."""
    specifications = indexed(planned, "trajectory_key")
    observed = indexed(results, "trajectory_key")
    require(len(specifications) == EXPECTED_PATHS, "all 154 planned trajectories")
    require(set(observed) == set(specifications), "all planned terminal path records retained")
    groups: dict[str, list[dict]] = defaultdict(list)
    path_output = []
    for key, specification in specifications.items():
        result = observed[key]
        for name in ("trajectory_key", "cell_id", "anchor_id", "replicate", "start_session_index"):
            require(result[name] == specification[name], f"path identity: {key}/{name}")
        day = specification["start_session_index"]
        require(day in DAYS, "original decision session")
        require(type(result["model_calls"]) is int and result["model_calls"] >= 0, "path calls")
        require(
            type(result["session_advances"]) is int
            and 0 <= result["session_advances"] <= 19 - day
            and result["model_calls"] >= result["session_advances"],
            "one model request advances at most one session",
        )
        require(
            result["terminal_session_index"] == day + 1 + result["session_advances"],
            "recorded last session follows the fixed first action and retained advances",
        )
        require(result["model_calls"] <= 4 * (19 - day), "per-path fixed remaining-session budget")
        complete = result["status"] == "COMPLETE"
        if complete:
            require(finite(result["Y"]), "complete path has finite terminal financial Y")
            require(result["terminal_session_index"] == 20, "original terminal session")
            require(result["session_advances"] == 19 - day, "all remaining advances completed")
            metrics = result["terminal_metrics"]
            require(
                isinstance(metrics, dict)
                and metrics["session_count"] == 21
                and finite(metrics["terminal_score"])
                and metrics["terminal_score"] == result["Y"],
                "complete original-window financial score",
            )
        else:
            require(
                result["Y"] is None and result["terminal_metrics"] is None,
                "failed or capped path has no invented terminal label",
            )
        row = {
            "trajectory_key": key,
            "cell_id": result["cell_id"],
            "anchor_id": result["anchor_id"],
            "replicate": result["replicate"],
            "start_session_index": day,
            "status": result["status"],
            "failure_reason": result.get("failure_reason"),
            "model_calls": result["model_calls"],
            "maximum_model_calls": 4 * (19 - day),
            "session_advances": result["session_advances"],
            "terminal_session_index": result["terminal_session_index"],
            "Y": result["Y"],
        }
        path_output.append(row)
        groups[row["cell_id"]].append(row)
    require(len(groups) == EXPECTED_CELLS, "all 77 original cells")
    cell_output = []
    for cell_id, paths in sorted(groups.items()):
        paths.sort(key=lambda row: row["replicate"])
        require(len(paths) == 2 and {p["replicate"] for p in paths} == {0, 1}, "two original paths")
        require(
            len({(p["anchor_id"], p["start_session_index"]) for p in paths}) == 1,
            "cell paths share original anchor",
        )
        paired = all(p["status"] == "COMPLETE" for p in paths)
        difference = paths[1]["Y"] - paths[0]["Y"] if paired else None
        cell_output.append(
            {
                "cell_id": cell_id,
                "anchor_id": paths[0]["anchor_id"],
                "start_session_index": paths[0]["start_session_index"],
                "planned_paths": 2,
                "complete_paths": sum(p["status"] == "COMPLETE" for p in paths),
                "both_complete": paired,
                "replicate0_status": paths[0]["status"],
                "replicate1_status": paths[1]["status"],
                "replicate0_Y": paths[0]["Y"],
                "replicate1_Y": paths[1]["Y"],
                "paired_Y_difference_rep1_minus_rep0": difference,
                "paired_absolute_Y_difference": abs(difference) if paired else None,
                "paired_exactly_equal_Y": difference == 0 if paired else None,
                "model_calls": sum(p["model_calls"] for p in paths),
            }
        )

    def aggregate(paths: list[dict], cells: list[dict]) -> dict:
        return {
            "planned_paths": len(paths),
            "planned_cells": len(cells),
            "anchors": len({p["anchor_id"] for p in paths}),
            "complete_paths": sum(p["status"] == "COMPLETE" for p in paths),
            "noncomplete_paths": sum(p["status"] != "COMPLETE" for p in paths),
            "both_complete_cells": sum(c["both_complete"] for c in cells),
            "pair_difference_NA_cells": sum(not c["both_complete"] for c in cells),
            "complete_pairs_exactly_equal_Y": sum(
                c["paired_exactly_equal_Y"] is True for c in cells
            ),
            "complete_pairs_different_Y": sum(c["paired_exactly_equal_Y"] is False for c in cells),
            "model_calls": sum(p["model_calls"] for p in paths),
            "maximum_model_calls": sum(p["maximum_model_calls"] for p in paths),
        }

    daily = [
        {
            "start_session_index": day,
            **aggregate(
                [p for p in path_output if p["start_session_index"] == day],
                [c for c in cell_output if c["start_session_index"] == day],
            ),
        }
        for day in DAYS
    ]
    counts = aggregate(path_output, cell_output)
    counts["path_status_counts"] = dict(Counter(p["status"] for p in path_output))
    require(counts["anchors"] == 32, "all 32 original anchors retained")
    require(counts["maximum_model_calls"] == REQUEST_LIMIT, "fixed 6904 maximum model calls")
    return path_output, cell_output, daily, counts


def normalized_driver_records(manifest: dict, results: list[dict]) -> tuple[list, list]:
    """Driver schema adapter; identities are checked before descriptive arithmetic."""
    planned = [
        {**row, "start_session_index": row["decision_session_index"]}
        for row in manifest["trajectories"]
    ]
    observed = [
        {
            **row,
            "start_session_index": row["decision_session_index"],
            "terminal_session_index": row["last_session_index"],
        }
        for row in results
    ]
    return planned, observed


def generation_complete(record: dict) -> bool:
    tokens = record["response_token_ids"]
    stop = record["stop_reason"]
    return (
        0 < len(tokens) < 4096
        and record["returned_finished"] is True
        and record["finish_reason"] == "stop"
        and tokens[-1] in (151645, 151643)
        and not any(token in (151645, 151643) for token in tokens[:-1])
        and (stop is None or (type(stop) is int and stop == tokens[-1]))
    )


def token_hash(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode("ascii")).hexdigest()


def validate_request_logs(directory: Path, manifest: dict, results: list[dict]) -> dict:
    """Stream large prompt logs, retaining only request identities and scalar costs."""
    seed_path = directory / "request_seed_table.jsonl"
    require(sha(seed_path) == manifest["request_seed_table_sha256"], "frozen seed-table SHA")
    seeds = indexed(rows(seed_path), "request_key")
    require(len(seeds) == REQUEST_LIMIT, "all 6904 potential request slots retained")
    require(
        len({r["request_seed"] for r in seeds.values()}) == REQUEST_LIMIT, "fixed seed uniqueness"
    )
    terminal = indexed(results, "trajectory_key")
    specifications = indexed(manifest["trajectories"], "trajectory_key")
    cells = indexed(manifest["cells"], "cell_id")
    require(len(cells) == EXPECTED_CELLS, "fixed 77 manifest cells")
    by_path: dict[str, list[str]] = defaultdict(list)
    submitted = {}
    fields = (
        "request_key",
        "trajectory_key",
        "cell_id",
        "anchor_id",
        "replicate",
        "decision_session_index",
        "attempt",
        "request_seed",
        "prompt_sha256",
        "prompt_token_count",
        "max_tokens",
    )
    for request in iter_rows(directory / "driver_requests.jsonl"):
        key = request["request_key"]
        require(key not in submitted and key in seeds, "one actual request per frozen slot")
        require(all(request[k] == v for k, v in seeds[key].items()), "exact frozen request seed")
        spec = specifications[request["trajectory_key"]]
        require(
            all(request[k] == spec[k] for k in ("cell_id", "anchor_id", "replicate")),
            "request path and cell identity",
        )
        require(
            request["prompt_token_count"] == len(request["prompt_token_ids"]) <= 8192
            and request["prompt_sha256"] == token_hash(request["prompt_token_ids"])
            and request["max_tokens"] == 4096,
            "full prompt identity and original response cap",
        )
        submitted[key] = {k: request[k] for k in fields}
        by_path[request["trajectory_key"]].append(key)
    generated = {}
    finish_counts: Counter = Counter()
    status_counts: Counter = Counter()
    response_counts = []
    for number, generation in enumerate(iter_rows(directory / "generations.jsonl")):
        key = generation["request_key"]
        require(key not in generated and key in submitted, "one generation for actual request")
        require(
            all(generation[k] == v for k, v in submitted[key].items()),
            "worker generation matches full driver request identity",
        )
        require(
            token_hash(generation["returned_prompt_token_ids"]) == submitted[key]["prompt_sha256"],
            "native returned prompt identity",
        )
        require(str(generation["vllm_request_id"]) == str(number), "contiguous native IDs")
        complete = generation_complete(generation)
        require(
            (generation["generation_status"] == "COMPLETE") is complete,
            "independent native generation completeness",
        )
        require(
            generation["actual_sampling_params"]["seed"] == submitted[key]["request_seed"],
            "native request-specific seed",
        )
        count = len(generation["response_token_ids"])
        require(count == generation["response_token_count"], "actual response token cost")
        generated[key] = {
            "complete": complete,
            "generation_status": generation["generation_status"],
            "failure_reason": generation["failure_reason"],
            "response_hash": hashlib.sha256(generation["response_text"].encode()).hexdigest(),
            "response_ids_hash": token_hash(generation["response_token_ids"]),
            "response_token_count": count,
        }
        finish_counts[str(generation["finish_reason"])] += 1
        status_counts[generation["generation_status"]] += 1
        response_counts.append(count)
    require(set(generated) == set(submitted), "every actual request has a native generation")
    position = {key: specifications[key]["decision_session_index"] + 1 for key in terminal}
    attempts = dict.fromkeys(terminal, 0)
    applied = dict.fromkeys(terminal, 0)
    advances = dict.fromkeys(terminal, 0)
    seen = set()
    env_counts: Counter = Counter()
    for evaluation in iter_rows(directory / "driver_evaluations.jsonl"):
        key = evaluation["request_key"]
        require(key not in seen and key in generated, "one driver classification per response")
        seen.add(key)
        request, generation = submitted[key], generated[key]
        require(all(evaluation[k] == v for k, v in request.items()), "evaluation request identity")
        trajectory = request["trajectory_key"]
        require(
            request["decision_session_index"] == position[trajectory]
            and request["attempt"] == attempts[trajectory] < 4
            and evaluation["session_before"] == position[trajectory],
            "original session and at most four requests before advancement",
        )
        require(
            hashlib.sha256(evaluation["response_text"].encode()).hexdigest()
            == generation["response_hash"]
            and token_hash(evaluation["response_token_ids"]) == generation["response_ids_hash"]
            and evaluation["generation_status"] == generation["generation_status"],
            "unrepaired original generated response applied",
        )
        if not generation["complete"]:
            require(
                evaluation["env_status"] == "NOT_APPLIED"
                and evaluation["info"] is evaluation["trace"] is None
                and evaluation["session_after"] == position[trajectory],
                "incomplete response never enters the environment",
            )
        else:
            info, trace = evaluation["info"], evaluation["trace"]
            require(
                trace["step_index"]
                == terminal[trajectory]["prefix_requests"] + 1 + applied[trajectory],
                "complete original prefix and fixed first action preserved",
            )
            applied[trajectory] += 1
            after = evaluation["session_after"]
            require(
                after in (position[trajectory], position[trajectory] + 1), "single-step advance"
            )
            expected = (
                "INVALID"
                if info.get("invalid_action")
                else ("ADVANCING" if after > position[trajectory] else "NONADVANCING")
            )
            require(evaluation["env_status"] == expected, "actual environment classification")
            if after > position[trajectory]:
                require(not info.get("invalid_action"), "rejection never advances")
                advances[trajectory] += 1
                attempts[trajectory] = 0
                position[trajectory] = after
            else:
                attempts[trajectory] += 1
        env_counts[evaluation["env_status"]] += 1
    require(seen == set(submitted), "all actual responses classified")
    path_cost = {}
    for trajectory, result in terminal.items():
        cell = cells[result["cell_id"]]
        for field in (
            "anchor_id",
            "source_group",
            "first_action_text",
            "economic_order_key",
            "source_slot",
            "added_hold",
            "financial_state_sha256",
            "observation_messages_sha256",
            "prefix_requests",
        ):
            require(result[field] == cell[field], f"fixed cell and prefix identity: {field}")
        require(
            result["policy_snapshot_sha256"] == manifest["policy_snapshot_sha256"],
            "fixed common recovered SFT identity",
        )
        keys = by_path[trajectory]
        require(
            len(keys) == result["model_calls"]
            and position[trajectory] == result["last_session_index"]
            and advances[trajectory] == result["session_advances"],
            "path calls and retained actual advances",
        )
        if result["status"] == "REQUEST_LIMIT":
            require(attempts[trajectory] == 4, "four actual nonadvancing requests at limit")
        if result["status"] == "INCOMPLETE":
            require(bool(keys) and not generated[keys[-1]]["complete"], "native incomplete path")
        if result["status"] == "COMPLETE":
            require(
                result["terminal_metrics"]["step_count"]
                == result["prefix_requests"] + 1 + applied[trajectory],
                "terminal history includes original prefix, fixed action and all applied replies",
            )
        path_cost[trajectory] = {
            "prompt_tokens": sum(submitted[k]["prompt_token_count"] for k in keys),
            "response_tokens": sum(generated[k]["response_token_count"] for k in keys),
            "response_sequence_sha256": hashlib.sha256(
                json.dumps([generated[k]["response_hash"] for k in keys]).encode()
            ).hexdigest(),
        }
    return {
        "model_requests": len(submitted),
        "environment_applied_model_responses": sum(applied.values()),
        "expected_total_environment_requests": sum(r["prefix_requests"] + 1 for r in results)
        + sum(applied.values()),
        "generation_status_counts": dict(status_counts),
        "native_finish_reason_counts": dict(finish_counts),
        "environment_response_counts": dict(env_counts),
        "prompt_tokens": sum(r["prompt_token_count"] for r in submitted.values()),
        "response_tokens": sum(response_counts),
        "response_token_count_min": min(response_counts) if response_counts else None,
        "response_token_count_median": statistics.median(response_counts)
        if response_counts
        else None,
        "response_token_count_max": max(response_counts) if response_counts else None,
        "path_cost": path_cost,
    }


def run(directory: Path) -> None:
    manifest_path = directory / "continuation_manifest.json"
    completion_path = directory / "driver_completion.json"
    result_path = directory / "trajectory_results.jsonl"
    request_path = directory / "driver_requests.jsonl"
    manifest, completion = read(manifest_path), read(completion_path)
    require(completion["status"] == "COMPLETE", "driver classified all planned paths")
    require(
        completion["continuation_manifest_sha256"] == sha(manifest_path)
        and manifest["expected_cells"] == EXPECTED_CELLS
        and manifest["expected_trajectories"] == EXPECTED_PATHS
        and manifest["max_model_requests"] == REQUEST_LIMIT
        and manifest["max_attempts_per_session"] == 4
        and manifest["terminal_session_index"] == 20
        and manifest["automatic_retry"] is False
        and manifest["fallback_hold"] is False,
        "fixed bounded pilot manifest",
    )
    for relative, digest in completion["output_sha256"].items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "relative driver output path")
        require(sha(directory / path) == digest, f"driver artifact changed: {relative}")
    original_results = rows(result_path)
    planned, results = normalized_driver_records(manifest, original_results)
    paths, cells, daily, counts = summarize(planned, results)
    log_summary = validate_request_logs(directory, manifest, original_results)
    path_cost = log_summary.pop("path_cost")
    require(
        log_summary["model_requests"] == counts["model_calls"] == completion["model_requests"]
        and completion["trajectories_classified"] == EXPECTED_PATHS
        and completion["fixed_first_actions"] == EXPECTED_PATHS
        and completion["environments_constructed"] == EXPECTED_PATHS
        and completion["environment_requests"]
        == log_summary["expected_total_environment_requests"],
        "actual path, model and environment request accounting",
    )
    require(
        completion["trajectory_status_counts"] == counts["path_status_counts"]
        and completion["optimizer_updates"] == 0
        and completion["formal_valid_oos_access"] is False
        and completion["automatic_retry"] is False,
        "driver outcomes and zero-update scope",
    )
    worker_path = directory / "worker_completion.json"
    worker = read(worker_path)
    require(
        worker["status"] == "COMPLETE"
        and worker["model_requests"] == counts["model_calls"]
        and worker["complete_generations"]
        == log_summary["generation_status_counts"].get("COMPLETE", 0)
        and worker["incomplete_generations"]
        == log_summary["generation_status_counts"].get("INCOMPLETE", 0),
        "native worker completion and request accounting",
    )
    for relative, digest in worker["output_sha256"].items():
        require(sha(directory / relative) == digest, f"native output unchanged: {relative}")
    for path in paths:
        path.update(path_cost[path["trajectory_key"]])
    cells_by_id = indexed(manifest["cells"], "cell_id")
    costs_by_cell: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        costs_by_cell[path["cell_id"]].append(path)
    for cell in cells:
        specification = cells_by_id[cell["cell_id"]]
        cell["economic_order_key"] = specification["economic_order_key"]
        cell["added_hold"] = specification["added_hold"]
        pair = costs_by_cell[cell["cell_id"]]
        cell["paired_exact_response_sequence_equal"] = (
            pair[0]["response_sequence_sha256"] == pair[1]["response_sequence_sha256"]
            if cell["both_complete"]
            else None
        )
    for day in daily:
        day_paths = [p for p in paths if p["start_session_index"] == day["start_session_index"]]
        day["path_status_counts"] = dict(Counter(p["status"] for p in day_paths))
        day["prompt_tokens"] = sum(p["prompt_tokens"] for p in day_paths)
        day["response_tokens"] = sum(p["response_tokens"] for p in day_paths)
    require(counts["model_calls"] <= REQUEST_LIMIT, "total model request cap")
    names = ("continuation_paths.csv", "continuation_cells.csv", "continuation_days.csv")
    for name, items in zip(names, (paths, cells, daily), strict=True):
        write_csv(directory / name, items)
    report = {
        "profile": PROFILE,
        "status": "PASS",
        "scope": "bounded continuation feasibility only",
        "at_UTC": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "request_cost": log_summary,
        "day_summary": daily,
        "definitions": {
            "Y": (
                "original terminal financial score after the fixed first action "
                "and stochastic common-SFT continuation"
            ),
            "paired_difference": (
                "replicate1 minus replicate0 Y, only when both original planned paths complete"
            ),
            "NA": (
                "failed/capped path terminal Y and every unsupported pair difference "
                "stay NA, never zero"
            ),
        },
        "limits": [
            (
                "The two paths per cell are repeated feasibility continuations, "
                "not independent estimation/reference sets."
            ),
            (
                "Two terminal draws do not establish precise Q^pi, A^pi, denoising, "
                "MSE, ranking or financial superiority."
            ),
            (
                "Cells share Train windows/dates and historical M2 prefixes; "
                "they are not independent market samples or on-policy state coverage."
            ),
            (
                "No candidate pooling, actor update, formal Valid or Final OOS inference "
                "is performed by this reader."
            ),
        ],
        "input_sha256": {
            p.name: sha(p)
            for p in (
                manifest_path,
                completion_path,
                worker_path,
                result_path,
                request_path,
                directory / "driver_evaluations.jsonl",
                directory / "generations.jsonl",
                directory / "request_seed_table.jsonl",
            )
        },
        "output_sha256": {name: sha(directory / name) for name in names},
        "readout_source_sha256": sha(Path(__file__).resolve()),
    }
    with (directory / "continuation_readout.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "counts": counts}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run(args.run.resolve())


if __name__ == "__main__":
    main()

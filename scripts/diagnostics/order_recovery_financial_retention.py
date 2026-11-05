#!/usr/bin/env python3
"""Describe paired old/new SFT finance after the independent 400-trajectory audit.

Reads evidence only. The sole output is NEW/financial_retention.json, created
exclusively. No GPU, model weights, market snapshot, inference or profit gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROFILE = "common_sft_paired_financial_description_v1"
FULL_PROFILE = "r5_order_recovery_fullreset_v1"
OBSERVATION_PROFILE = "history_features_recovery_v2"
PROTOCOL = "single_tool_portfolio_v1"
CASE_IDS = tuple(f"formal-train-{n:04d}" for n in (210, 252, 336, 399, 420, 441, 231, 273))
NORMAL_IDS = CASE_IDS[-2:]
IDENTITIES = [{"id": f"seed{s}_sft", "base_seed": s} for s in (42, 43)]
METRICS = (
    "buy_fills",
    "sell_fills",
    "commission",
    "slippage_cost",
    "total_cost",
    "total_turnover",
    "full_window_net_return",
    "terminal_score",
)
DEFAULT_OLD = Path("runs/m2-order-recovery-episodes-20261011-v2")
DEFAULT_NEW = Path("runs/m2-order-recovery-sft-acceptance-20261011-v1")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


class Evidence:
    def __init__(self):
        self.sources = {}

    def pin(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or digest == expected, f"Source hash changed: {path}")
        require(
            self.sources.get(str(path), digest) == digest, f"Source changed during read: {path}"
        )
        self.sources[str(path)] = digest
        return digest

    def read(self, path):
        self.pin(path)
        return json.loads(Path(path).read_text())

    def audit_sources(self, audit, required):
        sources = audit.get("source_sha256")
        require(isinstance(sources, dict) and sources, "Raw audit lacks source hashes")
        require(
            {str(Path(p).resolve()) for p in required} <= set(sources),
            "Raw audit omits compared manifest or worker evidence",
        )
        for filename, digest in sources.items():
            require(Path(filename).is_absolute(), "Audit source path must be absolute")
            self.pin(filename, digest)

    def unchanged(self):
        for filename, digest in self.sources.items():
            require(sha(filename) == digest, f"Evidence changed during comparison: {filename}")


def worker_paths(directory):
    return [
        directory / "workers/sft" / name
        for name in (
            "state.json",
            "result.json",
            "config.json",
            "requests.jsonl",
            "raw.jsonl",
            "submissions.jsonl",
        )
    ]


def verify_legacy_archive(evidence, manifest):
    """Use the completed run's archive, never compare its source to live src."""
    selection = Path(manifest["selection_manifest"])
    selected = evidence.read(selection)
    directory = selection.parent
    receipt = evidence.read(directory / "completion_runtime_receipt.json")
    archive = directory / "runtime_sources.zip"
    require(
        receipt.get("status") == "PASS"
        and receipt.get("controller_status") == "COMPLETE"
        and receipt.get("runtime_files_verified") == len(manifest["runtime_sha256"])
        and selected["runtime_sha256"] == manifest["runtime_sha256"]
        and evidence.pin(archive) == receipt["runtime_archive_sha256"],
        "Legacy source archive/completion receipt differs",
    )
    with zipfile.ZipFile(archive) as bundle:
        for filename, expected in manifest["runtime_sha256"].items():
            member = Path(filename).relative_to(directory.parent.parent).as_posix()
            require(
                hashlib.sha256(bundle.read(member)).hexdigest() == expected,
                f"Legacy archived source changed: {filename}",
            )


def load_side(evidence, directory, *, new):
    full = directory / "full" if new else directory
    audit = evidence.read(directory / "raw_audit.json")
    manifest = evidence.read(full / "manifest.json")
    readout = evidence.read(directory / "readout.json")
    state = evidence.read(full / "workers/sft/state.json")
    result = evidence.read(full / "workers/sft/result.json")
    require(audit.get("status") == "PASS", "Independent raw integrity audit is not PASS")
    require(
        state == result
        and state.get("status") == "OBSERVED"
        and state.get("profile") == FULL_PROFILE
        and state.get("model_key") == "sft"
        and state.get("start_mode") == "fullreset"
        and state.get("full_episode_evaluation") is True,
        "Full worker state/result differs",
    )
    require(
        manifest.get("profile") == FULL_PROFILE
        and manifest.get("start_mode") == "fullreset"
        and manifest.get("observation_profile") == OBSERVATION_PROFILE
        and manifest.get("action_protocol") == PROTOCOL
        and manifest.get("enable_thinking") is False,
        "Fullreset inference protocol differs",
    )
    require(manifest["models"]["sft"]["identities"] == IDENTITIES, "SFT request identities differ")
    require(readout.get("status") == "OBSERVED", "Population has not finished observation")
    required = [full / "manifest.json", *worker_paths(full)]
    if new:
        parent = evidence.read(directory / "manifest.json")
        require(
            parent.get("profile") == "common_sft_order_recovery_acceptance_v1"
            and parent.get("mode") == "sft"
            and set(parent["models"]) == {"sft"}
            and parent.get("observation_profile") == OBSERVATION_PROFILE
            and parent.get("action_protocol") == PROTOCOL
            and parent.get("enable_thinking") is False
            and parent["full_cases"] == manifest["cases"]
            and parent["models"] == manifest["models"],
            "New acceptance parent differs",
        )
        budget = parent["budget"]
        require(
            all(
                budget.get(k) == v
                for k, v in {
                    "short_trajectories": 320,
                    "full_trajectories": 80,
                    "planned_trajectories": 400,
                    "full_request_cap": 256,
                    "recovery_followup_cap": 4,
                    "prompt_cap": 8192,
                    "response_cap": 4096,
                    "automatic_retry": False,
                    "automatic_actions": False,
                }.items()
            ),
            "New acceptance population/caps changed",
        )
        for phase in ("short", "full"):
            path = directory / phase / "manifest.json"
            require(
                evidence.pin(path) == parent["phase_manifest_sha256"][phase],
                "New phase manifest differs from parent",
            )
        require(
            audit.get("profile") == "common_sft_order_recovery_raw_audit_v1"
            and audit.get("mode") == "sft"
            and audit.get("audit_scope") == "complete_population"
            and audit.get("manifest_sha256") == evidence.pin(directory / "manifest.json")
            and audit.get("short_trajectories") == 320
            and audit.get("full_trajectories") == 80
            and audit.get("trajectories") == 400
            and set(audit.get("workers", {})) == {"sft"}
            and type(audit.get("acceptance_pass")) is bool,
            "New raw audit has the wrong population or manifest",
        )
        short = evidence.read(directory / "short/workers/sft/state.json")
        require(
            short == evidence.read(directory / "short/workers/sft/result.json")
            and short.get("status") == "OBSERVED"
            and len(short["grid"]) == len({r["id"] for r in short["grid"]}) == 320
            and audit["workers"]["sft"]["short"]["trajectories"] == 320,
            "New short population is incomplete",
        )
        require(
            readout.get("manifest_sha256") == audit["manifest_sha256"]
            and readout.get("raw_audit_status") == "PASS"
            and readout.get("raw_audit_sha256") == evidence.pin(directory / "raw_audit.json")
            and readout["short_grid"] == short["grid"]
            and len(readout["full_grid"]) == 80
            and readout["full_grid"] == result["grid"]
            and readout.get("acceptance") == ("PASS" if audit["acceptance_pass"] else "FAIL"),
            "New readout is stale or drops cells",
        )
        outer = evidence.read(directory / "workers/sft/state.json")
        require(
            outer == evidence.read(directory / "workers/sft/result.json")
            and outer.get("status") == "OBSERVED",
            "Combined SFT worker is incomplete",
        )
        required += [
            directory / "manifest.json",
            directory / "short/manifest.json",
            *worker_paths(directory / "short"),
            directory / "workers/sft/state.json",
            directory / "workers/sft/result.json",
        ]
        audited = audit["workers"]["sft"]["full"]
        behavior = {k: readout.get(k) for k in ("behavioral_gate", "acceptance")}
    else:
        verify_legacy_archive(evidence, manifest)
        require(
            audit.get("profile") == "r5_order_recovery_episode_independent_raw_audit_v2"
            and audit.get("mode") == "all_workers"
            and audit.get("trajectories") == 240,
            "Old full raw audit differs",
        )
        selected = [w for w in audit["workers"] if w["model_key"] == "sft"]
        require(len(selected) == 1, "Old audit lacks unique SFT worker")
        audited = selected[0]
        evidence.audit_sources({"source_sha256": audited["source_sha256"]}, worker_paths(full))
        require(
            readout.get("manifest_sha256") == evidence.pin(full / "manifest.json"),
            "Old readout manifest differs",
        )
        require(
            [r for r in readout["grid"] if r["model_identity"].endswith("_sft")] == result["grid"],
            "Old readout differs from SFT worker",
        )
        budget = manifest["budget"]
        require(
            all(
                budget.get(k) == v
                for k, v in {
                    "requests_per_trajectory": 256,
                    "recovery_followup_requests": 4,
                    "prompt_cap": 8192,
                    "response_cap": 4096,
                    "automatic_retry": False,
                    "automatic_actions": False,
                }.items()
            ),
            "Old episode caps changed",
        )
        # The legacy audit stores worker hashes separately from top-level hashes.
        required = [full / "manifest.json", directory / "readout.json"]
        behavior = None
    evidence.audit_sources(audit, required)
    require(
        audited.get("trajectories") == 80
        and audited.get("request_count") == sum(r["requests"] for r in result["grid"])
        and audited.get("statuses") == dict(Counter(r["status"] for r in result["grid"])),
        "Audited SFT counts differ from result",
    )
    return {
        "manifest": manifest,
        "rows": result["grid"],
        "full": full,
        "behavior": behavior,
        "model": manifest["models"]["sft"],
    }


def pair_population(old, new):
    old_cases = {c["id"]: c for c in old["manifest"]["cases"]}
    new_cases = {c["id"]: c for c in new["manifest"]["cases"]}
    require(
        len(old_cases) == len(new_cases) == 8 and set(old_cases) == set(new_cases) == set(CASE_IDS),
        "Eight-window population differs",
    )
    for cid in CASE_IDS:
        a, b = old_cases[cid], new_cases[cid]
        require(a["kind"] == ("normal" if cid in NORMAL_IDS else "rejection"), "Case kind differs")
        for field in ("kind", "env_kwargs", "raw_prompt", "source_identity", "initial_ledger"):
            require(
                a[field] == b[field], f"{cid}: initial/data/environment contract changed: {field}"
            )
        require(
            a["initial_ledger"]
            == {
                "cash": a["env_kwargs"]["initial_cash"],
                "fills": [],
                "positions": {},
                "session_index": 0,
            },
            "Initial ledger is not reset",
        )
    expected = {}
    for identity in IDENTITIES:
        for cid in CASE_IDS:
            for sibling in range(5):
                key = f"{identity['id']}:{cid}:{OBSERVATION_PROFILE}:{sibling}"
                expected[key] = {
                    "model_identity": identity["id"],
                    "base_seed": identity["base_seed"],
                    "case_id": cid,
                    "case_kind": old_cases[cid]["kind"],
                    "profile": OBSERVATION_PROFILE,
                    "start_mode": "fullreset",
                    "request_limit": 256,
                    "sampling": "greedy" if sibling == 0 else "sampled",
                    "rollout_identity": {
                        **old_cases[cid]["source_identity"],
                        "rollout_n": sibling,
                        "validate": True,
                    },
                }
    pairs = []
    indexed = []
    for side in (old, new):
        by_id = {r["id"]: r for r in side["rows"]}
        require(
            len(by_id) == len(side["rows"]) == 80 and set(by_id) == set(expected),
            "Missing, duplicate or unpaired fullreset trajectory",
        )
        for key, fields in expected.items():
            require(
                all(by_id[key].get(k) == v for k, v in fields.items()),
                f"{key}: request identity/sampling/episode schema differs",
            )
        indexed.append(by_id)
    for key in expected:
        a, b = (index[key] for index in indexed)
        require(
            a["initial_ledger"] == b["initial_ledger"] == old_cases[a["case_id"]]["initial_ledger"]
            and a["initial_state"] == b["initial_state"],
            f"{key}: fullreset observation differs",
        )
        pairs.append((a, b, old_cases[a["case_id"]]))
    return pairs


def request_seed(row, index):
    payload = {
        "base_seed": row["base_seed"],
        "env_step": index,
        **row["rollout_identity"],
        "scheme": "finance-rollout-seed-v1",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF


def check_request_seeds(side):
    rows = {r["id"]: r for r in side["rows"]}
    counts = Counter()
    with (side["full"] / "workers/sft/requests.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            row = rows[record["trajectory_id"]]
            if record.get("submitted") is False:
                require(record.get("failure") == "PROMPT_CAP", "Unknown unsubmitted request marker")
                continue
            index = counts[row["id"]]
            expected = {
                "n": 1,
                "temperature": 0.0 if row["sampling"] == "greedy" else 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.0,
                "max_tokens": 4096,
                "seed": request_seed(row, index),
            }
            require(
                record["request_index"] == record["seed_env_step"] == index
                and record["base_seed"] == row["base_seed"]
                and record["rollout_identity"] == row["rollout_identity"]
                and record["sampling_params"] == expected,
                "Logged request seed/sampling changed",
            )
            counts[row["id"]] += 1
    require(all(counts[r["id"]] == r["requests"] for r in side["rows"]), "Request count differs")


def near(actual, expected, message):
    require(
        math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8),
        message,
    )


def finance(row, case, metric_schema, fill_schema):
    require(row["status"] in {"COMPLETE", "INCOMPLETE"}, "Unobserved trajectory status")
    require("final_ledger" in row, "Observed trajectory lacks actual final ledger")
    fills = row["final_ledger"]["fills"]
    require(all(set(fill) == fill_schema for fill in fills), "Fill schema differs from old result")
    require(
        all(
            f["side"] in {"buy", "sell"} and f["commission"] >= 0 and f["slippage_cost"] >= 0
            for f in fills
        ),
        "Invalid fill fee/side schema",
    )
    result = {
        "status": row["status"],
        "failure_reason": row.get("failure", row.get("failure_reason")),
        "requests": row["requests"],
        "invalid_requests": row["invalid_requests"],
        "final_session_index": row["final_ledger"]["session_index"],
        "buy_fills": sum(f["side"] == "buy" for f in fills),
        "sell_fills": sum(f["side"] == "sell" for f in fills),
        "commission": math.fsum(f["commission"] for f in fills),
        "slippage_cost": math.fsum(f["slippage_cost"] for f in fills),
        "total_cost": math.fsum(f["commission"] + f["slippage_cost"] for f in fills),
        "total_turnover": None,
        "full_window_net_return": None,
        "terminal_score": None,
    }
    if row["status"] != "COMPLETE":
        require(
            not {"terminal_metrics", "full_window_net_return"} & set(row),
            "Incomplete trajectory publishes terminal finance",
        )
        return result
    metrics, env = row["terminal_metrics"], case["env_kwargs"]
    require(set(metrics) == metric_schema, "Terminal financial metric schema changed")
    require(
        row["final_ledger"]["session_index"]
        == row["final_state"]["session_index"]
        == len(env["episode_dates"]) - 1
        and row["final_state"]["sessions_remaining"] == 0
        and metrics["session_count"] == len(env["episode_dates"])
        and metrics["scorer_id"] == env["finance_scorer_id"]
        and metrics["calibration_sha256"] == env["screening_calibration_sha256"],
        "COMPLETE lacks actual common financial terminal/schema",
    )
    for field in ("commission", "slippage_cost"):
        near(metrics[field], result[field], f"Terminal {field} differs from actual fills")
    near(
        metrics["transaction_cost"], result["total_cost"], "Terminal costs differ from actual fills"
    )
    near(row["full_window_net_return"], metrics["raw_net_return"], "Net return fields differ")
    near(
        row["full_window_net_return"],
        row["final_state"]["portfolio_equity"] / env["initial_cash"] - 1,
        "Net return differs from terminal equity",
    )
    result.update(
        total_turnover=metrics["turnover"],
        full_window_net_return=metrics["raw_net_return"],
        terminal_score=metrics["terminal_score"],
    )
    require(all(math.isfinite(result[k]) for k in METRICS), "Non-finite finance")
    return result


def aggregate(rows):
    complete = sum(row["status"] == "COMPLETE" for row in rows)
    all_complete = complete == len(rows)
    observed = {k: math.fsum(row[k] for row in rows) for k in METRICS[:5]}
    return {
        "planned": len(rows),
        "complete": complete,
        "completion_rate": complete / len(rows),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "no_fill_trajectories": sum(r["buy_fills"] + r["sell_fills"] == 0 for r in rows),
        "with_buy_fill_trajectories": sum(r["buy_fills"] > 0 for r in rows),
        "with_sell_fill_trajectories": sum(r["sell_fills"] > 0 for r in rows),
        "complete_no_fill_trajectories": sum(
            r["status"] == "COMPLETE" and r["buy_fills"] + r["sell_fills"] == 0 for r in rows
        ),
        "actual_observed_activity_totals": observed,
        "activity_horizon": "full_window" if all_complete else "mixed_incomplete_horizons",
        "financial_mean_status": "COMPLETE" if all_complete else "NOT_COMPUTED_INCOMPLETE",
        "financial_means": {
            k: math.fsum(row[k] for row in rows) / len(rows) if all_complete else None
            for k in METRICS
        },
    }


def comparison(rows, **selection):
    chosen = [row for row in rows if all(row.get(k) == v for k, v in selection.items())]
    old, new = (aggregate([r[side] for r in chosen]) for side in ("old", "new"))
    both_complete = old["complete"] == new["complete"] == len(chosen)
    delta = {
        k: new["financial_means"][k] - old["financial_means"][k] if both_complete else None
        for k in METRICS
    }
    delta["net_return_percentage_points"] = (
        100 * delta["full_window_net_return"] if both_complete else None
    )
    return {
        "selection": selection,
        "planned_pairs": len(chosen),
        "old": old,
        "new": new,
        "completion_rate_delta": new["completion_rate"] - old["completion_rate"],
        "mean_delta_new_minus_old": delta,
    }


def compare(old_directory, new_directory):
    evidence = Evidence()
    evidence.pin(Path(__file__))
    old = load_side(evidence, old_directory, new=False)
    new = load_side(evidence, new_directory, new=True)
    pairs = pair_population(old, new)
    for side in (old, new):
        check_request_seeds(side)
    schemas = {
        tuple(sorted(r["terminal_metrics"])) for r in old["rows"] if r["status"] == "COMPLETE"
    }
    require(len(schemas) == 1, "Old evidence lacks a unique complete financial metric schema")
    metric_schema = set(next(iter(schemas)))
    fill_schemas = {
        tuple(sorted(fill)) for row in old["rows"] for fill in row["final_ledger"]["fills"]
    }
    require(len(fill_schemas) == 1, "Old evidence lacks a unique actual Fill schema")
    fill_schema = set(next(iter(fill_schemas)))
    grid = []
    for a, b, case in pairs:
        old_finance, new_finance = (finance(r, case, metric_schema, fill_schema) for r in (a, b))
        complete = a["status"] == b["status"] == "COMPLETE"
        grid.append(
            {
                "id": a["id"],
                "model_identity": a["model_identity"],
                "case_id": a["case_id"],
                "case_kind": a["case_kind"],
                "sampling": a["sampling"],
                "rollout_identity": a["rollout_identity"],
                "first_request_seed": request_seed(a, 0),
                "old": old_finance,
                "new": new_finance,
                "delta_new_minus_old": {
                    k: new_finance[k] - old_finance[k] if complete else None for k in METRICS
                },
            }
        )
    groups = {"all8": comparison(grid), "normal2": comparison(grid, case_kind="normal")}
    stratified = [
        comparison(grid, **selection)
        for kind in (None, "normal")
        for identity in (None, "seed42_sft", "seed43_sft")
        for sampling in (None, "greedy", "sampled")
        if (
            selection := {
                k: v
                for k, v in (
                    ("case_kind", kind),
                    ("model_identity", identity),
                    ("sampling", sampling),
                )
                if v is not None
            }
        )
    ]
    per_case = [comparison(grid, case_id=cid) for cid in CASE_IDS]
    result = {
        "profile": PROFILE,
        "status": "DESCRIBED",
        "created_at_UTC": datetime.now(timezone.utc).isoformat(),
        "old_directory": str(old_directory.resolve()),
        "new_directory": str(new_directory.resolve()),
        "paired_trajectories": 80,
        "new_acceptance_status": new["behavior"],
        "contract_checks": {
            "fullreset": True,
            "same_data_env_kwargs": True,
            "same_initial_state": True,
            "same_request_identity_and_seed_scheme": True,
            "same_financial_metric_schema": True,
            "raw_integrity_audits": "PASS",
        },
        "model_paths": {"old": old["model"]["model_path"], "new": new["model"]["model_path"]},
        "terminal_metric_schema": sorted(metric_schema),
        "fill_schema": sorted(fill_schema),
        "groups": groups,
        "stratified": stratified,
        "per_case": per_case,
        "paired_grid": grid,
        "source_sha256": evidence.sources,
        "interpretation": [
            "Describes the new common-SFT policy on the same eight selected Train windows.",
            "Two request identities use one checkpoint; they are not independent training seeds.",
            "Equal seeds pair identities and request ordinals, not tokenwise outputs.",
            "Normal2 covers two selected windows; it cannot establish general financial retention.",
            "All trajectories are retained; incomplete groups have no financial mean or delta.",
            "Observed activity totals can have partial horizons, not full-window performance.",
            "No financial threshold, no no-degradation conclusion, and no M1/M2 advantage claim.",
        ],
    }
    evidence.unchanged()
    return result


def write_exclusive(destination, result):
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-directory", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--new-directory", type=Path, default=DEFAULT_NEW)
    parser.add_argument(
        "--check-only", action="store_true", help="Read and print; create no artifact"
    )
    args = parser.parse_args()
    destination = args.new_directory / "financial_retention.json"
    require(args.check_only or not destination.exists(), "Financial description refuses overwrite")
    result = compare(args.old_directory.resolve(), args.new_directory.resolve())
    if not args.check_only:
        write_exclusive(destination, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "paired_trajectories": 80,
                "groups": result["groups"],
                "output": None if args.check_only else str(destination),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

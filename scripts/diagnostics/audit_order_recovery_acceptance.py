#!/usr/bin/env python3
"""Independent stdlib raw audit for the frozen common-SFT acceptance wrapper.

No inference runtime, model weights or market snapshot is loaded. Complete mode
requires every prescribed worker and creates raw_audit.json exclusively. A
--model-key selection is read-only and cannot authorize the SFT training gate.
The full-episode checks are a separately pinned unchanged historical auditor;
new manifest/config population checks and short checks are implemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROFILE = "common_sft_order_recovery_acceptance_v1"
SHORT_PROFILE = "common_sft_order_recovery_short_v1"
AUDIT_PROFILE = "common_sft_order_recovery_raw_audit_v1"
OBSERVATION_PROFILE = "history_features_recovery_v2"
PROTOCOL = "single_tool_portfolio_v1"
ALL_KEYS = ("sft", "seed42_m1", "seed42_m2", "seed43_m1", "seed43_m2")
KNOWN_IDS = tuple(f"formal-train-{value:04d}" for value in (210, 252, 336, 399, 420, 441, 231, 273))
GATE_SHA256 = "7d728b0b008f5867a61c3770cf8019e00d249e048419ec37a59484add5f89125"
REQUEST_SEED_NAMESPACE = "fresh-after-max-source-sample-index-v1"
FULL_CHECKS_SHA256 = "1b2ce58b5add3bfc12ed4ef8c889934ab519744db01fa8e9f60547117a9800c1"
FULL_CHECKS_PATH = Path(__file__).resolve().with_name("order_recovery_acceptance_full_checks.py")


def require(value: Any, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def lines(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def full_checks() -> Any:
    require(sha(FULL_CHECKS_PATH) == FULL_CHECKS_SHA256, "Full auditor helper source changed")
    spec = importlib.util.spec_from_file_location("frozen_full_recovery_checks", FULL_CHECKS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_budget(mode: str) -> dict:
    require(mode in {"sft", "all"}, "Unregistered acceptance mode")
    identities, engines = (2, 1) if mode == "sft" else (6, 5)
    short, full = 32 * identities * 5, 8 * identities * 5
    return {
        "mode": mode,
        "known_cases": 8,
        "fresh_cases": 24,
        "short_cases": 32,
        "full_cases": 8,
        "model_engines": engines,
        "model_identities": identities,
        "profiles": [OBSERVATION_PROFILE],
        "greedy_per_case": 1,
        "sampled_per_case": 4,
        "short_trajectories": short,
        "full_trajectories": full,
        "planned_trajectories": short + full,
        "short_request_cap": 4,
        "full_request_cap": 256,
        "recovery_followup_cap": 4,
        "maximum_requests": short * 4 + full * 256,
        "maximum_response_tokens": (short * 4 + full * 256) * 4096,
        "prompt_cap": 8192,
        "response_cap": 4096,
        "training_updates": 0,
        "automatic_retry": False,
        "automatic_actions": False,
    }


def expected_engine(model: str) -> dict:
    return {
        "model": model,
        "tokenizer": model,
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "tensor_parallel_size": 1,
        "max_model_len": 12288,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 8192,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "gpu_memory_utilization": 0.35,
        "disable_custom_all_reduce": True,
        "enable_sleep_mode": False,
        "enforce_eager": False,
        "seed": 42,
        "compilation_config": {"cudagraph_capture_sizes": [1, 2, 4, 8]},
        "override_generation_config": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "max_new_tokens": 4096,
        },
    }


def check_request_seed_namespace(manifest: dict) -> None:
    """Independently derive the fresh namespace from immutable source provenance."""
    namespace = manifest["request_seed_namespace"]
    require(namespace.get("scheme") == REQUEST_SEED_NAMESPACE, "Request seed namespace differs")
    banks = []
    for field in ("known_manifest", "acceptance_bank"):
        filename, expected = namespace[field], namespace[field + "_sha256"]
        require(
            Path(filename).is_absolute()
            and manifest["evidence_sha256"].get(filename) == expected
            and sha(Path(filename)) == expected,
            "Request namespace source bank binding differs",
        )
        banks.append(read(Path(filename)))
    known, fresh = banks
    originals = [*known["cases"], *fresh["cases"]]
    require(
        [case["id"] for case in known["cases"]] == list(KNOWN_IDS)
        and len(fresh["cases"]) == 24
        and [case["id"] for case in manifest["short_cases"]] == [case["id"] for case in originals],
        "Request namespace source population/order differs",
    )
    for original in originals:
        identity = original["source_identity"]
        require(
            "request_seed_identity" not in original
            and set(identity) == {"step", "sample_index", "rollout_n", "validate"}
            and identity["validate"] is True
            and all(
                type(identity[field]) is int and identity[field] >= 0
                for field in ("step", "sample_index", "rollout_n")
            ),
            "Source bank identity has changed",
        )
    start = max(original["source_identity"]["sample_index"] for original in originals) + 1
    require(namespace.get("sample_index_start") == start, "Request namespace start differs")
    anchors = set()
    for index, (case, original) in enumerate(zip(manifest["short_cases"], originals, strict=True)):
        require(
            all(
                case.get(field) == original.get(field)
                for field in ("source_identity", "source_base_seed", "source")
            ),
            "Request namespace changed original source provenance",
        )
        identity = original["source_identity"]
        if index < 8:
            require("request_seed_identity" not in case, "Known request seed override is forbidden")
        else:
            identity = {**identity, "sample_index": start + index - 8}
            require(
                case.get("request_seed_identity") == identity,
                "Fresh request seed namespace differs from its source bank",
            )
        anchor = identity["step"], identity["sample_index"]
        require(anchor not in anchors, "Acceptance cases collide in request-seed identity")
        anchors.add(anchor)
    for case, original in zip(manifest["full_cases"], known["cases"], strict=True):
        require(
            case["source_identity"] == original["source_identity"]
            and "request_seed_identity" not in case,
            "Fullreset changed legacy request seeds",
        )
    # The frozen helper derives seeds independently of the inference runtime.
    checks, seen, derived = full_checks(), {}, 0
    for cases, cap, short in (
        (manifest["short_cases"], 4, True),
        (manifest["full_cases"], 256, False),
    ):
        for base in (42, 43):
            for case in cases:
                source = case.get("request_seed_identity", case["source_identity"])
                for sibling in range(5):
                    identity = {**source, "rollout_n": sibling, "validate": True}
                    first = len(case["prefix"]) if short else 0
                    for index in range(first, first + cap):
                        payload = (base, tuple(sorted(identity.items())), index)
                        seed = checks.request_seed(base, identity, index)
                        require(
                            seen.setdefault(seed, payload) == payload,
                            "Distinct request payloads collide in 63-bit seed",
                        )
                        derived += 1
    require(
        namespace.get("seed_collision_check")
        == {
            "derived_requests": derived,
            "unique_payloads_and_seeds": len(seen),
            "identical_payload_reuses": derived - len(seen),
            "distinct_payload_collisions": 0,
        },
        "Request namespace collision check differs",
    )


def manifest_contract(directory: Path, checks: Any) -> tuple[dict, dict, dict, dict]:
    manifest = read(directory / "manifest.json")
    mode = manifest["mode"]
    keys = ("sft",) if mode == "sft" else ALL_KEYS
    require(
        manifest.get("profile") == PROFILE
        and manifest.get("observation_profile") == OBSERVATION_PROFILE
        and manifest.get("action_protocol") == PROTOCOL
        and manifest.get("enable_thinking") is False
        and manifest.get("budget") == expected_budget(mode)
        and set(manifest["models"]) == set(keys)
        and manifest.get("endpoint_gate_sha256") == GATE_SHA256
        and fingerprint(manifest["endpoint_gate"]) == GATE_SHA256,
        "Parent acceptance protocol, budget, models or endpoint gate differs",
    )
    sources = {
        str((directory / "manifest.json").resolve()): sha(directory / "manifest.json"),
        str(Path(__file__).resolve()): sha(Path(__file__)),
        str(FULL_CHECKS_PATH): sha(FULL_CHECKS_PATH),
    }
    for filename, digest in {**manifest["evidence_sha256"], **manifest["runtime_sha256"]}.items():
        require(
            Path(filename).is_absolute() and sha(Path(filename)) == digest,
            f"Frozen acceptance evidence changed: {filename}",
        )
        sources[filename] = digest
    snapshot = directory / "runtime_sources.zip"
    require(sha(snapshot) == manifest["runtime_snapshot_sha256"], "Runtime snapshot changed")
    sources[str(snapshot.resolve())] = sha(snapshot)
    short_cases, full_cases = manifest["short_cases"], manifest["full_cases"]
    require(
        len(short_cases) == len({case["id"] for case in short_cases}) == 32
        and [case["id"] for case in short_cases[:8]] == list(KNOWN_IDS)
        and [case["id"] for case in full_cases] == list(KNOWN_IDS)
        and all(
            case["bank_partition"] == ("known" if index < 8 else "fresh")
            for index, case in enumerate(short_cases)
        ),
        "Acceptance case population differs",
    )
    check_request_seed_namespace(manifest)
    for index, case in enumerate(short_cases):
        require(
            case["kind"] == ("normal" if index in {6, 7} else "rejection")
            and case["takeover_state"]["session_index"]
            == case["takeover_ledger"]["session_index"]
            == case["expected_start_session_index"],
            "Short takeover state differs",
        )
        checks.check_observation(
            case["takeover_state"], case["takeover_ledger"], case["env_kwargs"], case["id"]
        )
        if index >= 8:
            expected = endpoint_expected(case["takeover_state"], checks)
            require(
                case.get("training_example") is False
                and case.get("teacher_call_is_submitted") is False
                and case["endpoint_expected_call"] == expected
                and endpoint_matches(case["cpu_expected_next_call"], expected, checks),
                "Fresh endpoint label or teacher isolation differs",
            )
    for key in keys:
        model = manifest["models"][key]
        seeds = (42, 43) if key == "sft" else (int(key[4:6]),)
        require(
            model["identities"]
            == [{"id": f"seed{s}_sft" if key == "sft" else key, "base_seed": s} for s in seeds]
            and model["fingerprint"] == fingerprint(model["files"]),
            f"{key}: dynamic model fingerprint or request identity differs",
        )
    phase_manifests = {}
    for phase, cases in (("short", short_cases), ("full", full_cases)):
        path = directory / phase / "manifest.json"
        digest = sha(path)
        require(
            digest == manifest["phase_manifest_sha256"][phase],
            f"{phase}: child manifest digest differs",
        )
        child = read(path)
        require(
            child["models"] == manifest["models"]
            and child["cases"] == cases
            and child["profile"] == (SHORT_PROFILE if phase == "short" else checks.PROFILE)
            and child.get("enable_thinking") is False
            and child.get("action_protocol") == PROTOCOL,
            f"{phase}: child manifest model/profile/cases differs",
        )
        if phase == "short":
            require(
                child["budget"] == {"profiles": [OBSERVATION_PROFILE]},
                "Short profile population differs",
            )
        else:
            require(
                child.get("start_mode") == "fullreset"
                and child.get("observation_profile") == OBSERVATION_PROFILE
                and all(
                    "prefix" not in case
                    and case["initial_ledger"]["session_index"] == 0
                    and case["initial_ledger"]["positions"] == {}
                    and case["initial_ledger"]["fills"] == []
                    for case in cases
                ),
                "Full-reset child reused historical state",
            )
        phase_manifests[phase] = child
        sources[str(path.resolve())] = digest
    return manifest, phase_manifests["short"], phase_manifests["full"], sources


def endpoint_expected(state: dict, checks: Any) -> dict:
    rejection = state["order_rejection"]
    original = rejection["rejected_call"]
    require(original["name"] in {"buy", "sell"}, "Fresh rejected call is not an order")
    args = original["arguments"]
    if isinstance(args, str):
        args = json.loads(args)
    current, target = float(rejection["current_weight"]), float(args["target_weight"])
    require(
        math.isfinite(current)
        and math.isfinite(target)
        and rejection["comparison_tolerance"] == 1e-12,
        "Fresh raw current weight or validator tolerance differs",
    )
    name = "buy" if target > current + 1e-12 else ("sell" if target < current - 1e-12 else "hold")
    return {
        "name": name,
        "arguments": {}
        if name == "hold"
        else {"symbol": args["symbol"].strip().upper(), "target_weight": target},
    }


def endpoint_matches(text: str, expected: dict, checks: Any) -> bool:
    try:
        call = checks.legal_call(
            {"response": text, "trace": {"action": expected["name"]}}, "endpoint"
        )
        if call["name"] != expected["name"]:
            return False
        if call["name"] == "hold":
            return call["arguments"] == {}
        args = call["arguments"]
        target = float(args["target_weight"])
        return (
            args["symbol"].strip().upper() == expected["arguments"]["symbol"]
            and math.isfinite(target)
            and abs(target - expected["arguments"]["target_weight"]) <= 1e-12
        )
    except (ValueError, KeyError, TypeError, AttributeError):
        return False


def config_contract(directory: Path, manifest: dict, phase: str, key: str, sources: dict) -> dict:
    folder = directory / phase / "workers" / key
    config_path = folder / "config.json"
    config = read(config_path)
    child = read(directory / phase / "manifest.json")
    require(
        config.get("profile") == child["profile"]
        and config.get("model_key") == key
        and config.get("engine") == expected_engine(manifest["models"][key]["model_path"])
        and config.get("action_protocol") == PROTOCOL
        and config.get("enable_thinking") is False
        and config.get("observation_profile") == OBSERVATION_PROFILE
        and config.get("start_mode") == ("historical_prefix" if phase == "short" else "fullreset")
        and config.get("manifest_sha256") == sha(directory / phase / "manifest.json")
        and config.get("acceptance_manifest_sha256") == sha(directory / "manifest.json")
        and config.get("model_fingerprint") == manifest["models"][key]["fingerprint"],
        f"{phase}/{key}: phase config, engine or parent manifest binding differs",
    )
    sources[str(config_path.resolve())] = sha(config_path)
    return config


def short_summary(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model_identity"], row["profile"], row["case_kind"], row["sampling"])].append(
            row
        )
    return {
        "financial_evaluation": False,
        "planned": len(rows),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "requests": sum(row["requests"] for row in rows),
        "groups": [
            {
                "model_identity": key[0],
                "profile": key[1],
                "case_kind": key[2],
                "sampling": key[3],
                "planned": len(members),
                "statuses": dict(Counter(row["status"] for row in members)),
                "advanced": sum(row["status"] in {"RECOVERED", "ADVANCED"} for row in members),
                "invalid_requests": sum(row["invalid_requests"] for row in members),
                "requests": sum(row["requests"] for row in members),
            }
            for key, members in sorted(groups.items())
        ],
    }


def mapped_requests(folder: Path) -> tuple[list, list, dict]:
    files = {
        name: folder / name
        for name in (
            "state.json",
            "result.json",
            "config.json",
            "requests.jsonl",
            "submissions.jsonl",
            "raw.jsonl",
        )
    }
    hashes = {str(path.resolve()): sha(path) for path in files.values()}
    evidence, batches, raw = (
        lines(files[name]) for name in ("requests.jsonl", "submissions.jsonl", "raw.jsonl")
    )
    requests = [r for r in evidence if "request_index" in r]
    markers = [r for r in evidence if "request_index" not in r]
    require(
        len(requests)
        == sum(len(batch["requests"]) for batch in batches)
        == sum(len(batch["outputs"]) for batch in raw)
        and len(batches) == len(raw),
        "Short submission/raw/request counts differ",
    )
    indexed = {(r["trajectory_id"], r["request_index"]): r for r in requests}
    require(len(indexed) == len(requests), "Repeated short request identity")
    seen, engine_ids = set(), set()
    for batch, returned in zip(batches, raw, strict=True):
        require(
            0 < len(batch["requests"]) == len(returned["outputs"]) <= 8,
            "Short batch size or output mapping differs",
        )
        for sent, received in zip(batch["requests"], returned["outputs"], strict=True):
            identity = (sent["trajectory_id"], sent["request_index"])
            require(
                identity in indexed and identity not in seen, "Orphan/repeated short submission"
            )
            seen.add(identity)
            record = indexed[identity]
            require(
                all(record.get(field) == value for field, value in sent.items())
                and sent["prompt_token_ids"] == received["prompt_token_ids"]
                and len(received["outputs"]) == 1,
                "Short raw prompt or submitted request mapping differs",
            )
            response = received["outputs"][0]
            require(
                all(
                    response[field] == record[target]
                    for field, target in (
                        ("token_ids", "response_token_ids"),
                        ("finish_reason", "finish_reason"),
                        ("stop_reason", "stop_reason"),
                    )
                )
                and received["request_id"] == record["vllm_request_id"]
                and received["request_id"] not in engine_ids,
                "Short raw response finish/token/engine identity differs",
            )
            engine_ids.add(received["request_id"])
    require(seen == set(indexed), "Short request missing raw/submission evidence")
    return requests, markers, hashes


def short_worker(directory: Path, manifest: dict, key: str, checks: Any) -> dict:
    folder = directory / "workers" / key
    state, result = read(folder / "state.json"), read(folder / "result.json")
    require(
        state == result
        and state.get("status") == "OBSERVED"
        and state.get("profile") == SHORT_PROFILE
        and state.get("model_key") == key
        and state.get("full_episode_evaluation") is False,
        f"short/{key}: state/result protocol or completion differs",
    )
    # Keep the frozen full auditor's legacy seed contract unchanged. Only this
    # short manifest projection uses the independently verified fresh namespace.
    planning = {
        **manifest,
        "cases": [
            {**case, "source_identity": case.get("request_seed_identity", case["source_identity"])}
            for case in manifest["cases"]
        ],
    }
    planned = checks.expected_grid(planning, key)
    cases = {case["id"]: case for case in manifest["cases"]}
    rows = state["grid"]
    requests, markers, hashes = mapped_requests(folder)
    by_id = {row["id"]: row for row in rows}
    require(
        len(rows) == len(by_id) == len(planned) and set(by_id) == set(planned),
        "Short planned population missing, duplicated or substituted",
    )
    for identity, expected in planned.items():
        # Full and short use the same request identity; only start_mode and cap differ.
        expected = {
            field: value
            for field, value in expected.items()
            if field not in {"start_mode", "request_limit"}
        }
        require(
            all(by_id[identity].get(field) == value for field, value in expected.items())
            and by_id[identity]["request_limit"] == 4,
            f"{identity}: immutable short trajectory identity differs",
        )
    grouped, failures = defaultdict(list), defaultdict(list)
    for record in requests:
        require(record["trajectory_id"] in planned, "Orphan short request")
        grouped[record["trajectory_id"]].append(record)
    for marker in markers:
        require(
            marker["trajectory_id"] in planned and marker.get("failure") == "PROMPT_CAP",
            "Orphan/unregistered short prompt-cap marker",
        )
        failures[marker["trajectory_id"]].append(marker)
    totals, endpoint_grid = Counter(), []
    for row in rows:
        identity, records = row["id"], grouped[row["id"]]
        case = cases[row["case_id"]]
        require(
            type(row["requests"]) is int
            and 0 <= row["requests"] <= 4
            and len(records) == row["requests"]
            and [r["request_index"] for r in records] == list(range(row["requests"]))
            and row["prefix_length"] == len(case["prefix"])
            and row["takeover_session"] == case["expected_start_session_index"]
            and row["response_tokens"] == sum(len(r["response_token_ids"]) for r in records),
            f"{identity}: short request sequence, budget or seed prefix differs",
        )
        ledger, current_state = case["takeover_ledger"], case["takeover_state"]
        traces, invalid, advanced, expected_failure, last_response = [], 0, False, None, ""
        for index, record in enumerate(records):
            label = f"{identity}:{index}"
            require(not advanced and expected_failure is None, f"{label}: request after short stop")
            state_text = json.dumps(
                current_state,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            require(
                state_text in record["prompt"]
                or json.dumps(state_text, ensure_ascii=True)[1:-1] in record["prompt"],
                f"{label}: prompt lacks actual current observation",
            )
            require(
                "cpu_expected_next_call" not in record["prompt"]
                and "endpoint_expected_call" not in record["prompt"],
                f"{label}: offline endpoint teacher entered prompt",
            )
            env_step = row["prefix_length"] + index
            params = {
                "n": 1,
                "temperature": 0.0 if row["sampling"] == "greedy" else 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.0,
                "max_tokens": 4096,
                "seed": checks.request_seed(row["base_seed"], row["rollout_identity"], env_step),
            }
            tokens = record["response_token_ids"]
            require(
                record["ledger_before"] == ledger
                and record["base_seed"] == row["base_seed"]
                and record["rollout_identity"] == row["rollout_identity"]
                and record["seed_env_step"] == env_step
                and record["decision_session_index"] == ledger["session_index"]
                and record["sampling_params"] == params,
                f"{label}: ledger chain, derived seed or sampling budget differs",
            )
            require(
                0 < len(record["prompt_token_ids"]) <= 8192
                and len(tokens) <= 4096
                and all(
                    type(token) is int and token >= 0
                    for token in tokens + record["prompt_token_ids"]
                ),
                f"{label}: prompt/response token budget differs",
            )
            response_complete = (
                bool(tokens) and len(tokens) < 4096 and record["finish_reason"] == "stop"
            )
            if not record["environment_applied"]:
                require(
                    not response_complete
                    and not {"feedback", "ledger_after", "trace", "terminal_reached"} & set(record),
                    f"{label}: unapplied response has effects or lacks a stop failure",
                )
                expected_failure = "RESPONSE_INCOMPLETE"
                totals["unapplied"] += 1
                continue
            require(
                response_complete and record["stop_reason"] not in {"length", "abort", "aborted"},
                f"{label}: truncated/aborted generation applied",
            )
            trace, feedback, after = record["trace"], record["feedback"], record["ledger_after"]
            rejected, done = bool(feedback.get("invalid_action")), record["terminal_reached"]
            require(
                type(done) is bool
                and trace["step_index"] == env_step
                and trace["decision_session_index"] == ledger["session_index"]
                and trace["result_session_index"] == after["session_index"]
                and trace["invalid_action"] == rejected
                and trace["done"] == done,
                f"{label}: short credit/session/invalid trace differs",
            )
            traces.append(trace)
            current_state = checks.parse_state(record["observation_messages"])
            checks.check_observation(current_state, after, case["env_kwargs"], label)
            delta = after["session_index"] - ledger["session_index"]
            require(delta in {0, 1}, f"{label}: short session advance exceeds one")
            if rejected:
                require(
                    after == ledger and trace["action"] == "invalid" and not done,
                    f"{label}: rejected order changed ledger or was terminal",
                )
                invalid += 1
                totals["invalid"] += 1
            elif delta:
                require(
                    trace["action"] in checks.ADVANCING,
                    f"{label}: nonfinancial action advanced a session",
                )
                checks.check_accounting(ledger, after, record, case["env_kwargs"], label)
                advanced, last_response = True, record["response"]
                totals["advances"] += 1
            else:
                require(
                    after == ledger and trace["action"] in {"retrieve", "compute"} and not done,
                    f"{label}: nonadvancing valid request changed finance or terminated",
                )
            ledger = after
        prompt_markers = failures[identity]
        if prompt_markers:
            require(
                len(prompt_markers) == 1
                and not advanced
                and expected_failure is None
                and row["requests"] < 4
                and len(prompt_markers[0]["prompt_token_ids"]) > 8192,
                f"{identity}: prompt cap marker follows stop or is within budget",
            )
            expected_failure = "PROMPT_CAP"
        require(
            row["continuation_trace"] == traces
            and row["final_ledger"] == ledger
            and row["invalid_requests"] == invalid,
            f"{identity}: short final ledger/trace/invalid count differs",
        )
        if advanced:
            require(
                row["status"] == ("RECOVERED" if row["case_kind"] == "rejection" else "ADVANCED")
                and row["advanced_to_session"]
                == ledger["session_index"]
                == row["takeover_session"] + 1
                and row["advance_action"] == traces[-1]["action"],
                f"{identity}: short false recovery/advance",
            )
        else:
            expected_failure = expected_failure or ("REQUEST_CAP" if row["requests"] == 4 else None)
            require(
                row["status"] == "INCOMPLETE"
                and expected_failure is not None
                and row["failure"] == expected_failure
                and ledger["session_index"] == row["takeover_session"],
                f"{identity}: short incomplete reason or session differs",
            )
        require(
            not {
                "net_return",
                "takeover_net_return",
                "terminal_metrics",
                "terminal_score",
                "full_window_net_return",
            }
            & set(row),
            f"{identity}: short probe fabricates terminal financial outcomes",
        )
        if case["bank_partition"] == "fresh":
            expected = endpoint_expected(case["takeover_state"], checks)
            matched = advanced and endpoint_matches(last_response, expected, checks)
            endpoint_grid.append(
                {
                    "id": identity,
                    "case_id": case["id"],
                    "legal_advance": advanced,
                    "endpoint_adherence": matched,
                    "expected_call": expected,
                    "actual_response": last_response,
                }
            )
    require(
        state["summary"] == short_summary(rows)
        and sum(row["requests"] for row in rows) == len(requests),
        f"short/{key}: independently calculated summary differs",
    )
    require(
        hashes == {filename: sha(Path(filename)) for filename in hashes},
        f"short/{key}: logs changed during audit",
    )
    return {
        "status": "PASS",
        "model_key": key,
        "trajectories": len(rows),
        "request_count": len(requests),
        "submitted_count": len(requests),
        "raw_output_count": len(requests),
        "prompt_cap_markers": len(markers),
        "rejections_with_unchanged_ledger": totals["invalid"],
        "legal_session_advances": totals["advances"],
        "unapplied_incomplete_responses": totals["unapplied"],
        "statuses": dict(Counter(row["status"] for row in rows)),
        "failure_reasons": dict(
            Counter(row.get("failure") for row in rows if row["status"] == "INCOMPLETE")
        ),
        "endpoint_planned": len(endpoint_grid),
        "endpoint_adherent": sum(row["endpoint_adherence"] for row in endpoint_grid),
        "endpoint_grid": endpoint_grid,
        "source_sha256": hashes,
    }


def audit(directory: Path, selected: tuple[str, ...] | None = None) -> dict:
    directory = directory.resolve()
    checks = full_checks()
    manifest, short_manifest, full_manifest, sources = manifest_contract(directory, checks)
    keys = ("sft",) if manifest["mode"] == "sft" else ALL_KEYS
    complete_population = selected is None
    selected = keys if selected is None else selected
    require(
        set(selected) <= set(keys) and len(selected) == len(set(selected)),
        "Selected model is not in the frozen acceptance population",
    )
    workers, all_short, all_full = {}, [], []
    for key in selected:
        short_config = config_contract(directory, manifest, "short", key, sources)
        full_config = config_contract(directory, manifest, "full", key, sources)
        require(short_config["engine"] == full_config["engine"], "Shared engine config changed")
        short = short_worker(directory / "short", short_manifest, key, checks)
        full = checks.audit_worker(directory / "full", full_manifest, key)
        workers[key] = {"short": short, "full": full}
        sources.update(short["source_sha256"])
        sources.update(full["source_sha256"])
        outer = directory / "workers" / key
        state, result = read(outer / "state.json"), read(outer / "result.json")
        short_state = read(directory / "short/workers" / key / "state.json")
        full_state = read(directory / "full/workers" / key / "state.json")
        require(
            state == result
            and state.get("status") == "OBSERVED"
            and state.get("profile") == PROFILE
            and state.get("model_key") == key
            and state.get("manifest_sha256") == sha(directory / "manifest.json")
            and state.get("phase") == "complete"
            and state.get("short") == short_state["summary"]
            and state.get("full") == full_state["summary"],
            f"{key}: parent/phase state and result disagree",
        )
        behavioral_pass = (
            short["legal_session_advances"] == short["trajectories"]
            and short["endpoint_adherent"] == short["endpoint_planned"]
            and full["statuses"] == {"COMPLETE": full["trajectories"]}
        )
        require(
            state.get("behavioral_gate") == ("PASS" if behavioral_pass else "FAIL"),
            f"{key}: parent behavior gate disagrees with independently audited requests",
        )
        endpoint_summary = state.get("endpoint_adherence", {})
        require(
            endpoint_summary.get("planned") == short["endpoint_planned"]
            and endpoint_summary.get("endpoint_adherent") == short["endpoint_adherent"]
            and endpoint_summary.get("legal_advances")
            == sum(row["legal_advance"] for row in short["endpoint_grid"]),
            f"{key}: parent endpoint counts differ",
        )
        raw_ids = [
            record["vllm_request_id"]
            for phase in ("short", "full")
            for record in lines(directory / phase / "workers" / key / "requests.jsonl")
            if "vllm_request_id" in record
        ]
        require(
            len(raw_ids) == len(set(raw_ids)), f"{key}: engine request ID repeated across phases"
        )
        for name in ("state.json", "result.json"):
            path = outer / name
            sources[str(path.resolve())] = sha(path)
        all_short.extend(short_state["grid"])
        all_full.extend(full_state["grid"])
    short_count = sum(worker["short"]["trajectories"] for worker in workers.values())
    full_count = sum(worker["full"]["trajectories"] for worker in workers.values())
    if complete_population:
        require(
            short_count == manifest["budget"]["short_trajectories"]
            and full_count == manifest["budget"]["full_trajectories"],
            "Independent audit omitted prescribed identities",
        )
    readout_path = directory / "readout.json"
    if readout_path.exists():
        readout = read(readout_path)
        require(
            readout.get("manifest_sha256") == sha(directory / "manifest.json"),
            "Readout belongs to another acceptance manifest",
        )
        for phase, expected_count in (
            ("short", manifest["budget"]["short_trajectories"]),
            ("full", manifest["budget"]["full_trajectories"]),
        ):
            rows = readout[f"{phase}_grid"]
            require(
                len(rows) == len({row["id"] for row in rows}) == expected_count,
                f"Readout drops planned {phase} population",
            )
            audited = all_short if phase == "short" else all_full
            by_id = {row["id"]: row for row in rows}
            require(
                all(by_id.get(row["id"]) == row for row in audited),
                f"Readout {phase} cells differ from raw-audited worker state",
            )
    require(
        sources == {filename: sha(Path(filename)) for filename in sources},
        "Acceptance source or raw evidence changed during independent audit",
    )
    acceptance_pass = complete_population and all(
        worker["short"]["legal_session_advances"] == worker["short"]["trajectories"]
        and worker["short"]["endpoint_adherent"] == worker["short"]["endpoint_planned"]
        and worker["full"]["statuses"] == {"COMPLETE": worker["full"]["trajectories"]}
        for worker in workers.values()
    )
    return {
        "profile": AUDIT_PROFILE,
        "status": "PASS",
        "mode": manifest["mode"],
        "audit_scope": "complete_population" if complete_population else "read_only_partial",
        "manifest_sha256": sha(directory / "manifest.json"),
        "endpoint_gate_sha256": GATE_SHA256,
        "acceptance_pass": acceptance_pass,
        "short_trajectories": short_count,
        "full_trajectories": full_count,
        "trajectories": short_count + full_count,
        "requests": sum(
            worker[phase]["request_count"]
            for worker in workers.values()
            for phase in ("short", "full")
        ),
        "workers": workers,
        "source_sha256": sources,
        "claims": [
            "All raw batches map to submitted prompts and recorded requests.",
            "All rejections preserve the ledger; advances are exactly one session.",
            "Incomplete outputs are not applied or assigned financial returns.",
            "Fresh endpoint adherence is scored offline; no teacher is submitted.",
            "Actual terminal finance is recomputed from complete episode evidence.",
            "Raw integrity PASS alone implies neither behavior PASS nor no policy degradation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--model-key", choices=ALL_KEYS)
    args = parser.parse_args()
    partial = args.model_key is not None
    destination = args.directory / "raw_audit.json"
    if not partial:
        require(not destination.exists(), "Independent raw audit refuses overwrite")
    result = audit(args.directory, (args.model_key,) if partial else None)
    if not partial:
        descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"workers", "source_sha256"}
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

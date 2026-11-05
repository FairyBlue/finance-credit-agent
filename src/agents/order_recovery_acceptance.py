"""Draft common-SFT recovery acceptance wrapper; CPU preparation is separate from GPU workers.

Models must already exist. A single SFT engine covers two request-seed identities.
Known eight plus fresh 24 states produce 320 short + 80 full SFT trajectories;
five engines produce 960 short + 240 full trajectories. No training is launched.

Model specification JSON maps sft (or all five model keys) to model_path and
lineage_receipt paths. The lineage schema below binds real exported HF files,
training evidence, the common initialization, method/seed and update budget.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import os
import re
import signal
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from src.agents import order_recovery_episode as episode
from src.agents import order_recovery_probe as probe
from src.agents.coldstart_data import parse_state
from src.agents.rollout_seed import normalize_rollout_identity

PROFILE = "common_sft_order_recovery_acceptance_v1"
SHORT_PROFILE = "common_sft_order_recovery_short_v1"
OBSERVATION_PROFILE = "history_features_recovery_v2"
PROTOCOL = "single_tool_portfolio_v1"
MODEL_KEYS = probe.MODEL_KEYS
KNOWN_IDS = probe.FAILURE_IDS + probe.NORMAL_IDS
ROOT = probe.ROOT
MODES = ("sft", "all")
ENDPOINT_GATE = {
    "version": "common_sft_recovery_endpoint_gate_v1",
    "observation_profile": OBSERVATION_PROFILE,
    "comparison_tolerance": 1e-12,
    "fresh_gate": "all first legal advances preserve the last rejected total endpoint",
    "direction_comparison": (
        "target > current + tolerance => buy; target < current - tolerance => sell; otherwise hold"
    ),
    "target_comparison": "abs(actual target_weight - rejected target_weight) <= 1e-12",
    "symbol_comparison": "strip whitespace and uppercase",
    "no_trade_action": "hold",
    "known_gate": "all legal advances; no endpoint adherence requirement",
    "full_gate": "all original eight fullreset trajectories reach actual financial terminal",
    "lanes": "greedy1 + sampled4 for every model request identity and case",
    "teacher_fields_model_visible": False,
    "stronger_than_general_legal_advance": True,
    "financial_optimality_claim": False,
    "raw_audit_pass_required": True,
}
RAW_AUDIT_PROFILE = "common_sft_order_recovery_raw_audit_v1"
REQUEST_SEED_NAMESPACE = "fresh-after-max-source-sample-index-v1"


def require(value: Any, message: str) -> None:
    if not value:
        raise ValueError(message)


def fingerprint(files: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def parse_call(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        call = value
    else:
        matches = re.findall(r"<tool_call>(.*?)</tool_call>", value, flags=re.DOTALL)
        calls = []
        for block in matches:
            try:
                candidate = json.loads(block)
                name, args = candidate["name"], candidate["arguments"]
                if not isinstance(args, dict):
                    args = json.loads(args) if isinstance(args, str) else {}
                calls.append({"name": name, "arguments": args})
            except (ValueError, KeyError, TypeError):
                continue
        require(len(calls) == 1, "Endpoint evidence requires exactly one parsed tool call")
        call = calls[0]
    require(
        isinstance(call, dict) and isinstance(call.get("name"), str),
        "Endpoint evidence call is malformed",
    )
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    require(isinstance(arguments, dict), "Endpoint evidence arguments are malformed")
    return {"name": call["name"], "arguments": arguments}


def endpoint_label(state: dict[str, Any]) -> dict[str, Any]:
    rejection = state["order_rejection"]
    original = parse_call(rejection["rejected_call"])
    require(original["name"] in {"buy", "sell"}, "Fresh rejection is not an order")
    current = float(rejection["current_weight"])
    target = float(original["arguments"]["target_weight"])
    tolerance = ENDPOINT_GATE["comparison_tolerance"]
    require(
        math.isfinite(current)
        and math.isfinite(target)
        and rejection["comparison_tolerance"] == tolerance,
        "Endpoint direction evidence or validator tolerance differs",
    )
    # Match the validator's comparisons exactly; subtracting first can alter
    # the representable boundary around current +/- tolerance.
    name = (
        "buy"
        if target > current + tolerance
        else ("sell" if target < current - tolerance else "hold")
    )
    return {
        "name": name,
        "arguments": {}
        if name == "hold"
        else {"symbol": original["arguments"]["symbol"].strip().upper(), "target_weight": target},
    }


def endpoint_matches(actual: str | dict[str, Any], expected: dict[str, Any]) -> bool:
    try:
        actual = parse_call(actual)
        if actual["name"] != expected["name"]:
            return False
        if expected["name"] == "hold":
            return actual["arguments"] == {}
        args = actual["arguments"]
        target = float(args["target_weight"])
        return (
            isinstance(args.get("symbol"), str)
            and args["symbol"].strip().upper() == expected["arguments"]["symbol"]
            and math.isfinite(target)
            and abs(target - expected["arguments"]["target_weight"])
            <= ENDPOINT_GATE["comparison_tolerance"]
        )
    except (ValueError, KeyError, TypeError, AttributeError):
        return False


def endpoint_readout(cases: list[dict], rows: list[dict], requests: list[dict]) -> dict[str, Any]:
    by_case = {case["id"]: case for case in cases}
    final = {}
    for request in requests:
        if request.get("environment_applied"):
            final[request["trajectory_id"]] = request
    details = []
    for row in rows:
        case = by_case[row["case_id"]]
        if case["bank_partition"] != "fresh":
            continue
        legal = row["status"] in {"RECOVERED", "ADVANCED"}
        # The short runner stops immediately at its first legal advance.
        record = final.get(row["id"], {})
        text = record.get("response", "")
        matches = legal and endpoint_matches(text, case["endpoint_expected_call"])
        details.append(
            {
                "id": row["id"],
                "case_id": row["case_id"],
                "model_identity": row["model_identity"],
                "sampling": row["sampling"],
                "legal_advance": legal,
                "endpoint_adherence": matches,
                "expected_call": case["endpoint_expected_call"],
                "actual_call": parse_call(text) if matches else text,
            }
        )
    return {
        "planned": len(details),
        "legal_advances": sum(r["legal_advance"] for r in details),
        "endpoint_adherent": sum(r["endpoint_adherence"] for r in details),
        "grid": details,
    }


def keys_for(mode: str) -> tuple[str, ...]:
    require(mode in MODES, "Acceptance mode must be sft or all")
    return ("sft",) if mode == "sft" else MODEL_KEYS


def budget(mode: str) -> dict[str, Any]:
    identities = 2 if mode == "sft" else 6
    short, full = 32 * identities * 5, 8 * identities * 5
    return {
        "mode": mode,
        "known_cases": 8,
        "fresh_cases": 24,
        "short_cases": 32,
        "full_cases": 8,
        "model_engines": len(keys_for(mode)),
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


def hf_inventory(model: Path, *, excluded: set[Path]) -> dict[str, Any]:
    """Hash actual export files; no manifest can stand in for absent weights."""
    model = model.resolve()
    require(model.is_dir(), f"Model does not exist: {model}")
    require(
        (model / "config.json").is_file() and (model / "tokenizer_config.json").is_file(),
        "HF model config and tokenizer config are required",
    )
    files = {}
    for path in sorted(model.rglob("*")):
        require(not path.is_symlink(), f"Symlink in model export: {path}")
        if path.is_dir():
            continue
        require(not path.is_symlink() and path.is_file(), f"Non-regular model file: {path}")
        if path.resolve() in excluded:
            continue
        files[str(path.relative_to(model))] = {
            "bytes": path.stat().st_size,
            "sha256": probe.sha(path),
        }
    weights = {name for name in files if name.endswith(".safetensors")}
    require(bool(weights), "Actual safetensors model weights are required")
    index = model / "model.safetensors.index.json"
    if index.exists():
        indexed = set(probe.read(index)["weight_map"].values())
        require(indexed == weights, "HF weight index and actual shards differ")
    return files


def model_contracts(spec_path: Path, mode: str) -> dict[str, Any]:
    """Read genuine model lineage receipts, then verify their exported fingerprints.

    Required receipt fields: status COMPLETE, model_key, model_fingerprint,
    action_protocol, observation_profile, acceptance_gate_sha256,
    training_updates > 0, and a nonempty
    training_evidence_sha256 map. Actors additionally name method, training_seed,
    common_sft_fingerprint, common_training_protocol_sha256 and actor_learning_rate.
    SFT names role common_sft. Evidence hashes pin real training/export receipts;
    supplying this schema never replaces actual training evidence.
    """
    specification = probe.read(spec_path)
    require(set(specification) == set(keys_for(mode)), "Model specification population differs")
    contracts = {}
    for key in keys_for(mode):
        entry = specification[key]
        model = Path(entry["model_path"]).expanduser().resolve()
        receipt_path = Path(entry["lineage_receipt"]).expanduser().resolve()
        require(
            receipt_path.is_file() and not receipt_path.is_symlink(),
            f"Model lineage receipt does not exist: {receipt_path}",
        )
        receipt = probe.read(receipt_path)
        files = hf_inventory(model, excluded={receipt_path})
        digest = fingerprint(files)
        require(
            receipt.get("status") == "COMPLETE"
            and receipt.get("model_key") == key
            and receipt.get("model_fingerprint") == digest
            and receipt.get("action_protocol") == PROTOCOL
            and receipt.get("observation_profile") == OBSERVATION_PROFILE
            and receipt.get("acceptance_gate_sha256") == fingerprint(ENDPOINT_GATE)
            and type(receipt.get("training_updates")) is int
            and receipt["training_updates"] > 0,
            f"{key}: incomplete, mismatched or incompatible training lineage",
        )
        evidence = receipt.get("training_evidence_sha256")
        require(isinstance(evidence, dict) and evidence, f"{key}: real training evidence required")
        evidence = dict(evidence)
        for filename, expected in evidence.items():
            require(
                Path(filename).is_absolute() and probe.sha(Path(filename)) == expected,
                f"{key}: training evidence changed: {filename}",
            )
        evidence[str(receipt_path)] = probe.sha(receipt_path)
        seeds = (42, 43) if key == "sft" else (int(key[4:6]),)
        identities = [
            {"id": f"seed{seed}_sft" if key == "sft" else key, "base_seed": seed} for seed in seeds
        ]
        if key == "sft":
            require(receipt.get("role") == "common_sft", "SFT must be the shared initialization")
        else:
            require(
                receipt.get("method") == key[-2:].upper()
                and receipt.get("training_seed") == seeds[0]
                and receipt.get("common_sft_fingerprint") == contracts["sft"]["fingerprint"]
                and isinstance(receipt.get("common_training_protocol_sha256"), str)
                and len(receipt["common_training_protocol_sha256"]) == 64
                and receipt["common_training_protocol_sha256"] in evidence.values()
                and receipt.get("actor_learning_rate") == 1e-7,
                f"{key}: method/seed/common initialization or training contract differs",
            )
        contracts[key] = {
            "model_path": str(model),
            "fingerprint": digest,
            "identities": identities,
            "files": files,
            "evidence": evidence,
            "lineage_receipt": str(receipt_path),
            "lineage": receipt,
        }
    if mode == "all":
        actors = [contracts[key]["lineage"] for key in MODEL_KEYS if key != "sft"]
        require(
            len({row["training_updates"] for row in actors}) == 1
            and len({row["common_training_protocol_sha256"] for row in actors}) == 1,
            "All actors must share the predeclared training protocol and update budget",
        )
        require(
            len({contracts[key]["fingerprint"] for key in MODEL_KEYS}) == len(MODEL_KEYS),
            "Fresh actors must be actual distinct updated policy checkpoints",
        )
    return contracts


def request_seed_identities(known: dict, fresh: dict) -> tuple[int, dict[str, dict]]:
    """Keep source provenance intact and put fresh requests after every source index."""
    originals = [*known["cases"], *fresh["cases"]]
    for case in originals:
        require("request_seed_identity" not in case, "Source bank contains a request override")
        identity = normalize_rollout_identity(case["source_identity"]).as_dict()
        require(identity["validate"] is True, "Acceptance source seed identity differs")
    start = max(case["source_identity"]["sample_index"] for case in originals) + 1
    return start, {
        case["id"]: {**case["source_identity"], "sample_index": start + index}
        for index, case in enumerate(fresh["cases"])
    }


def check_request_seed_collisions(short_cases: list[dict], full_cases: list[dict]) -> dict:
    """Reject a 63-bit hash collision while retaining intentional common requests."""
    seen, derived = {}, 0
    for cases, cap, short in (
        (short_cases, probe.REQUEST_CAP, True),
        (full_cases, episode.REQUEST_CAP, False),
    ):
        for base_seed in (42, 43):
            for case in cases:
                source = case.get("request_seed_identity", case["source_identity"])
                for sibling in range(5):
                    identity = {**source, "rollout_n": sibling, "validate": True}
                    start = len(case["prefix"]) if short else 0
                    for env_step in range(start, start + cap):
                        payload = (base_seed, tuple(sorted(identity.items())), env_step)
                        seed = probe.derive_request_seed(base_seed, identity, env_step)
                        require(
                            type(seed) is int and 0 <= seed <= 0x7FFFFFFFFFFFFFFF,
                            "Request seed is outside the frozen 63-bit contract",
                        )
                        previous = seen.setdefault(seed, payload)
                        require(
                            previous == payload, "Distinct request payloads collide in 63-bit seed"
                        )
                        derived += 1
    return {
        "derived_requests": derived,
        "unique_payloads_and_seeds": len(seen),
        "identical_payload_reuses": derived - len(seen),
        "distinct_payload_collisions": 0,
    }


def verify_request_seed_namespace(contract: dict) -> None:
    """Recompute request identities from the pinned, unchanged source banks."""
    namespace = contract["request_seed_namespace"]
    require(namespace.get("scheme") == REQUEST_SEED_NAMESPACE, "Request seed namespace differs")
    banks = []
    for field in ("known_manifest", "acceptance_bank"):
        filename = namespace[field]
        expected = namespace[field + "_sha256"]
        require(
            Path(filename).is_absolute()
            and contract["evidence_sha256"].get(filename) == expected
            and probe.sha(Path(filename)) == expected,
            "Request namespace source bank binding differs",
        )
        banks.append(probe.read(Path(filename)))
    known, fresh = banks
    start, assigned = request_seed_identities(known, fresh)
    require(namespace.get("sample_index_start") == start, "Request namespace start differs")
    originals = [*known["cases"], *fresh["cases"]]
    require(
        [case["id"] for case in contract["short_cases"]] == [case["id"] for case in originals],
        "Request namespace source case order differs",
    )
    anchors = set()
    for index, (case, original) in enumerate(zip(contract["short_cases"], originals, strict=True)):
        require(
            all(
                case.get(field) == original.get(field)
                for field in ("source_identity", "source_base_seed", "source")
            ),
            "Request namespace changed original source provenance",
        )
        if index < len(known["cases"]):
            require("request_seed_identity" not in case, "Known request seed override is forbidden")
            identity = original["source_identity"]
        else:
            identity = assigned[original["id"]]
            require(
                case.get("request_seed_identity") == identity,
                "Fresh request seed namespace differs from its source bank",
            )
        anchor = identity["step"], identity["sample_index"]
        require(anchor not in anchors, "Acceptance cases collide in request-seed identity")
        anchors.add(anchor)
    for case, original in zip(contract["full_cases"], known["cases"], strict=True):
        require(
            case["source_identity"] == original["source_identity"]
            and "request_seed_identity" not in case,
            "Fullreset changed legacy request seeds",
        )
    require(
        namespace.get("seed_collision_check")
        == check_request_seed_collisions(contract["short_cases"], contract["full_cases"]),
        "Request namespace collision check differs",
    )


async def validate_bank(
    known: dict[str, Any],
    fresh: dict[str, Any],
    *,
    replay: Any = probe.replay_case,
    reset: Any = episode.reset_case,
) -> tuple[list[dict], list[dict]]:
    require(
        [case["id"] for case in known["cases"]] == list(KNOWN_IDS),
        "Known bank must preserve the original eight cases",
    )
    require(
        fresh.get("observation_profile") == OBSERVATION_PROFILE
        and fresh.get("action_protocol") == PROTOCOL
        and fresh.get("teacher_labels_force_actions") is False
        and fresh.get("formal_Valid_or_OOS_used") is False
        and fresh.get("source_split") == "train"
        and fresh.get("training_acceptance_known_failure_dates_disjoint") is True
        and len(fresh["cases"]) == 24,
        "Fresh acceptance bank contract differs",
    )
    dates = {
        name: set(fresh[name])
        for name in (
            "train_decision_dates",
            "recovery_training_source_dates",
            "acceptance_source_dates",
            "known_failure_source_dates",
        )
    }
    require(
        all(dates.values())
        and not (
            (dates["train_decision_dates"] | dates["recovery_training_source_dates"])
            & (dates["acceptance_source_dates"] | dates["known_failure_source_dates"])
        )
        and not (dates["acceptance_source_dates"] & dates["known_failure_source_dates"]),
        "Training, acceptance and known failure source dates overlap",
    )
    _, assigned = request_seed_identities(known, fresh)
    cases = []
    seeds = set()
    for partition, members in (("known", known["cases"]), ("fresh", fresh["cases"])):
        for original in members:
            case = copy.deepcopy(original)
            case["bank_partition"] = partition
            require(case["kind"] in {"rejection", "normal"}, "Unregistered acceptance case kind")
            if partition == "fresh":
                require(
                    case.get("training_example") is False
                    and case["kind"] == "rejection"
                    and case.get("teacher_call_is_submitted") is False
                    and case["source"]["episode_id"] not in KNOWN_IDS,
                    "Fresh recovery case overlaps known windows or teaching data",
                )
                case["request_seed_identity"] = assigned[case["id"]]
            identity = case.get("request_seed_identity", case["source_identity"])
            anchor = (identity["step"], identity["sample_index"])
            require(anchor not in seeds, "Acceptance cases collide in request-seed identity")
            seeds.add(anchor)
            env, old_obs = await replay(case, probe.OLD)
            ledger = probe.ledger_state(env)
            require(
                case.get("takeover_ledger", ledger) == ledger, "Prepared takeover ledger differs"
            )
            case["takeover_ledger"] = ledger
            _, obs = await replay(case, OBSERVATION_PROFILE)
            state = parse_state(obs)
            case["takeover_state"] = state
            require(
                ("order_rejection" in state) == (case["kind"] == "rejection"),
                "Acceptance final rejection feedback/kind differs",
            )
            require(
                case["prefix_interactions"][-1]["done"] is False,
                "Acceptance takeover is already terminal",
            )
            if partition == "fresh" and "raw_state" in case:
                require(case["raw_state"] == state, "Fresh raw state differs from actual replay")
            if partition == "fresh":
                expected = endpoint_label(state)
                require(
                    endpoint_matches(case["cpu_expected_next_call"], expected),
                    "Offline expected endpoint disagrees with the actual validator state",
                )
                case["endpoint_expected_call"] = expected
                case_dates = {str(day)[:10] for day in case["env_kwargs"]["episode_dates"]}
                require(
                    case_dates <= dates["acceptance_source_dates"],
                    "Fresh case contains dates outside the declared acceptance holdout",
                )
            # Observation generation remains probe.replay_case; offline teacher
            # fields in the bank are never appended to model-visible messages.
            require(
                parse_state(old_obs)["session_index"] == state["session_index"],
                "Recovery profile advanced a session during preparation",
            )
            cases.append(case)
    require(len(cases) == len({case["id"] for case in cases}) == 32, "Duplicate/missing cases")
    full = []
    for historical in cases[:8]:
        case = {
            key: copy.deepcopy(historical[key])
            for key in (
                "id",
                "kind",
                "env_kwargs",
                "raw_prompt",
                "source_identity",
                "bank_partition",
            )
        }
        case["env_kwargs"].update(
            observation_profile=OBSERVATION_PROFILE,
            capture_interactions=True,
            reward_mode="finance_trajectory",
        )
        env, _ = await reset(case)
        case["initial_ledger"] = probe.ledger_state(env)
        require(
            case["initial_ledger"]["session_index"] == 0
            and case["initial_ledger"]["positions"] == {}
            and case["initial_ledger"]["fills"] == [],
            "Fullreset reused historical positions",
        )
        full.append(case)
    return cases, full


def runtime_sources() -> dict[str, str]:
    paths = set((ROOT / "src").rglob("*.py")) | set(
        (ROOT / "third_party/agent-r1/agent_r1").rglob("*.py")
    )
    paths.add(Path(__file__).resolve())
    return {str(path.resolve()): probe.sha(path) for path in sorted(paths)}


def prepare(
    directory: Path,
    known_manifest: Path,
    acceptance_bank: Path,
    model_spec: Path,
    mode: str,
    *,
    sft_gate_readout: Path | None = None,
) -> dict[str, Any]:
    directory = directory.resolve()
    require(not directory.exists(), "A new acceptance directory is required; no overwrite")
    # Hash models before creating any prepared directory. Missing future models
    # cannot yield a PREPARED contract or launchable placeholder.
    models = model_contracts(model_spec, mode)
    known, fresh = probe.read(known_manifest), probe.read(acceptance_bank)
    require(
        fresh["training_dataset_sha256"] in models["sft"]["evidence"].values(),
        "Shared SFT lineage does not bind the acceptance bank's training dataset",
    )
    short_cases, full_cases = asyncio.run(validate_bank(known, fresh))
    namespace_start, _ = request_seed_identities(known, fresh)
    seed_collision_check = check_request_seed_collisions(short_cases, full_cases)
    evidence = {
        str(path.resolve()): probe.sha(path)
        for path in (known_manifest, acceptance_bank, model_spec)
    }
    for contract in models.values():
        evidence.update(contract["evidence"])
    previous = None
    if mode == "all":
        require(sft_gate_readout is not None, "All-model acceptance requires a passed new-SFT gate")
        prior = probe.read(sft_gate_readout)
        require(
            prior.get("status") == "OBSERVED"
            and prior.get("acceptance") == "PASS"
            and prior.get("mode") == "sft",
            "New-SFT acceptance gate has not passed",
        )
        prior_manifest_path = sft_gate_readout.resolve().parent / "manifest.json"
        prior_manifest = verify_completed(
            prior_manifest_path.parent, expected_sha256=prior.get("manifest_sha256")
        )
        prior_audit = raw_audit_gate(
            prior_manifest_path.parent, prior_manifest, completed_runtime=True
        )
        require(
            prior.get("manifest_sha256") == probe.sha(prior_manifest_path)
            and prior_audit["status"] == "PASS"
            and prior_audit["acceptance_pass"] is True
            and prior.get("raw_audit_sha256") == prior_audit["sha256"]
            and prior_manifest["models"]["sft"]["fingerprint"] == models["sft"]["fingerprint"]
            and prior_manifest["evidence_sha256"].get(str(acceptance_bank.resolve()))
            == probe.sha(acceptance_bank)
            and prior_manifest["short_cases"] == short_cases
            and prior_manifest["full_cases"] == full_cases,
            "New-SFT gate uses different weights, bank, profiles or cases",
        )
        evidence[str(sft_gate_readout.resolve())] = probe.sha(sft_gate_readout)
        evidence[str(prior_manifest_path)] = probe.sha(prior_manifest_path)
        evidence[str(prior_manifest_path.parent / "raw_audit.json")] = prior_audit["sha256"]
        evidence.update(
            {
                filename: digest
                for filename, digest in probe.read(prior_manifest_path.parent / "raw_audit.json")[
                    "source_sha256"
                ].items()
                if filename not in prior_manifest["runtime_sha256"]
            }
        )
        for name in (
            "controller_state.json",
            "completion_runtime_receipt.json",
            "runtime_sources.zip",
        ):
            path = prior_manifest_path.parent / name
            evidence[str(path.resolve())] = probe.sha(path)
        previous = str(sft_gate_readout.resolve())
    sources = runtime_sources()
    contract = {
        "profile": PROFILE,
        "status": "PREPARED",
        "mode": mode,
        "budget": budget(mode),
        "action_protocol": PROTOCOL,
        "observation_profile": OBSERVATION_PROFILE,
        "enable_thinking": False,
        "short_cases": short_cases,
        "full_cases": full_cases,
        "request_seed_namespace": {
            "scheme": REQUEST_SEED_NAMESPACE,
            "known_manifest": str(known_manifest.resolve()),
            "known_manifest_sha256": probe.sha(known_manifest),
            "acceptance_bank": str(acceptance_bank.resolve()),
            "acceptance_bank_sha256": probe.sha(acceptance_bank),
            "sample_index_start": namespace_start,
            "seed_collision_check": seed_collision_check,
        },
        "endpoint_gate": ENDPOINT_GATE,
        "endpoint_gate_sha256": fingerprint(ENDPOINT_GATE),
        "models": models,
        "evidence_sha256": evidence,
        "runtime_sha256": sources,
        "previous_sft_gate": previous,
        "limitations": [
            "Known8 are development regressions; fresh24 are predeclared Train holdouts.",
            "SFT request seeds share one checkpoint and are not independent trained policies.",
            "All-model mode repeats SFT acceptance; repeats are not independent samples.",
            "The fullreset suite covers only the original eight windows.",
            "Acceptance establishes neither M2 advantage nor other update-budget stability.",
            "Fresh endpoint adherence exceeds legal advance and tests API intent recovery.",
            "API acceptance cannot exclude policy degradation; no profit threshold is imposed.",
        ],
    }
    short = {
        "profile": SHORT_PROFILE,
        "budget": {"profiles": [OBSERVATION_PROFILE]},
        "action_protocol": PROTOCOL,
        "enable_thinking": False,
        "cases": short_cases,
        "models": models,
    }
    full = {
        "profile": episode.PROFILE,
        "start_mode": "fullreset",
        "observation_profile": OBSERVATION_PROFILE,
        "action_protocol": PROTOCOL,
        "enable_thinking": False,
        "cases": full_cases,
        "models": models,
    }
    directory.mkdir(parents=True, exist_ok=False)
    for phase, manifest in (("short", short), ("full", full)):
        (directory / phase / "workers").mkdir(parents=True)
        probe.save(directory / phase / "manifest.json", manifest)
    contract["phase_manifest_sha256"] = {
        phase: probe.sha(directory / phase / "manifest.json") for phase in ("short", "full")
    }
    with zipfile.ZipFile(directory / "runtime_sources.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for filename in sources:
            archive.write(filename, arcname=filename.lstrip("/"))
    contract["runtime_snapshot_sha256"] = probe.sha(directory / "runtime_sources.zip")
    probe.save(directory / "manifest.json", contract)
    digest = probe.sha(directory / "manifest.json")
    probe.save(
        directory / "state.json",
        {"status": "PREPARED", "profile": PROFILE, "manifest_sha256": digest},
    )
    return {"status": "PREPARED", "manifest_sha256": digest, "budget": budget(mode)}


def _verify(
    directory: Path,
    expected_sha256: str | None = None,
    *,
    model_key: str | None = None,
    check_models: bool = True,
    archived_runtime: dict[str, str] | None = None,
) -> dict[str, Any]:
    path = directory / "manifest.json"
    if expected_sha256 is not None:
        require(probe.sha(path) == expected_sha256, "Acceptance manifest digest differs")
    contract = probe.read(path)
    mode = contract["mode"]
    require(
        contract.get("profile") == PROFILE
        and contract["budget"] == budget(mode)
        and contract.get("observation_profile") == OBSERVATION_PROFILE
        and contract.get("action_protocol") == PROTOCOL
        and contract.get("enable_thinking") is False
        and contract.get("endpoint_gate") == ENDPOINT_GATE
        and contract.get("endpoint_gate_sha256") == fingerprint(ENDPOINT_GATE)
        and set(contract["models"]) == set(keys_for(mode))
        and len(contract["short_cases"]) == 32
        and len(contract["full_cases"]) == 8
        and [case["id"] for case in contract["short_cases"][:8]] == list(KNOWN_IDS)
        and [case["id"] for case in contract["full_cases"]] == list(KNOWN_IDS),
        "Acceptance contract, profiles or population differs",
    )
    # Evidence is always live, even if an entry also appears in the runtime inventory.
    if archived_runtime is not None:
        require(
            archived_runtime == contract["runtime_sha256"], "Archived runtime inventory differs"
        )
    for sources in (
        contract["evidence_sha256"],
        contract["runtime_sha256"] if archived_runtime is None else {},
    ):
        for filename, digest in sources.items():
            require(
                probe.sha(Path(filename)) == digest,
                f"Frozen acceptance source changed: {filename}",
            )
    verify_request_seed_namespace(contract)
    require(
        probe.sha(directory / "runtime_sources.zip") == contract["runtime_snapshot_sha256"],
        "Frozen runtime snapshot changed",
    )
    for phase in ("short", "full"):
        phase_path = directory / phase / "manifest.json"
        require(
            probe.sha(phase_path) == contract["phase_manifest_sha256"][phase],
            f"Frozen {phase} manifest changed",
        )
        manifest = probe.read(phase_path)
        require(
            manifest["models"] == contract["models"]
            and manifest["cases"] == contract["short_cases" if phase == "short" else "full_cases"],
            f"{phase}: models or cases changed",
        )
    selected = (model_key,) if model_key is not None else keys_for(mode)
    require(set(selected) <= set(keys_for(mode)), "Worker model is not in the frozen population")
    if check_models:
        for key in selected:
            model = contract["models"][key]
            actual = hf_inventory(
                Path(model["model_path"]), excluded={Path(model["lineage_receipt"]).resolve()}
            )
            require(
                actual == model["files"] and fingerprint(actual) == model["fingerprint"],
                f"{key}: actual model inventory or content changed after freeze",
            )
    return contract


def verify(
    directory: Path,
    expected_sha256: str | None = None,
    *,
    model_key: str | None = None,
    check_models: bool = True,
) -> dict[str, Any]:
    """Verify an active gate against its exact live runtime."""
    return _verify(directory, expected_sha256, model_key=model_key, check_models=check_models)


def _completed_runtime(directory: Path, contract: dict[str, Any]) -> dict[str, str]:
    """Verify a historical PASS and every pinned archive member without extraction."""
    require(probe.read(directory / "manifest.json") == contract, "Historical manifest differs")
    digest = probe.sha(directory / "manifest.json")
    controller = probe.read(directory / "controller_state.json")
    receipt = probe.read(directory / "completion_runtime_receipt.json")
    readout = probe.read(directory / "readout.json")
    raw_digest = probe.sha(directory / "raw_audit.json")
    runtime = contract.get("runtime_sha256")
    require(isinstance(runtime, dict) and runtime, "Historical runtime inventory is missing")
    require(
        controller.get("status") == "COMPLETE"
        and controller.get("manifest_sha256") == digest
        and {row.get("model_key") for row in controller.get("models", [])}
        == set(contract["models"])
        and len(controller.get("models", [])) == len(contract["models"])
        and all(row.get("status") == "COMPLETE" for row in controller["models"]),
        "Historical acceptance controller is not COMPLETE for this manifest",
    )
    require(
        receipt.get("status") == "PASS"
        and receipt.get("controller_status") == "COMPLETE"
        and receipt.get("acceptance") == "PASS"
        and receipt.get("manifest_sha256") == digest
        and receipt.get("raw_audit_sha256") == raw_digest
        and receipt.get("runtime_archive_sha256") == contract["runtime_snapshot_sha256"]
        and type(receipt.get("runtime_files_verified")) is int
        and receipt["runtime_files_verified"] == len(runtime),
        "Historical acceptance completion receipt differs",
    )
    require(
        readout.get("profile") == PROFILE
        and readout.get("mode") == contract["mode"]
        and readout.get("status") == "OBSERVED"
        and readout.get("acceptance") == "PASS"
        and readout.get("behavioral_gate") == "PASS"
        and readout.get("manifest_sha256") == digest
        and readout.get("endpoint_gate_sha256") == fingerprint(ENDPOINT_GATE)
        and readout.get("raw_audit_required") is True
        and readout.get("raw_audit_status") == "PASS"
        and readout.get("raw_audit_sha256") == raw_digest
        and readout.get("short_workers") == dict.fromkeys(contract["models"], "OBSERVED")
        and readout.get("full_workers") == dict.fromkeys(contract["models"], "OBSERVED")
        and len(readout.get("short_grid", [])) == contract["budget"]["short_trajectories"]
        and len(readout.get("full_grid", [])) == contract["budget"]["full_trajectories"],
        "Historical acceptance readout is not an independently audited observed PASS",
    )
    archive_path = directory / "runtime_sources.zip"
    require(
        probe.sha(archive_path) == contract["runtime_snapshot_sha256"],
        "Historical runtime archive changed",
    )
    expected = {}
    for filename, source_digest in runtime.items():
        require(
            isinstance(filename, str)
            and filename.startswith("/")
            and "\\" not in filename
            and all(part not in {"", ".", ".."} for part in filename.split("/")[1:])
            and str(PurePosixPath(filename)) == filename
            and isinstance(source_digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", source_digest) is not None,
            "Historical runtime inventory has a noncanonical path or digest",
        )
        expected[filename[1:]] = source_digest
    verified = {}
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        require(
            len(names) == len(set(names)) and set(names) == set(expected),
            "Historical runtime archive inventory is missing, duplicated or out of bounds",
        )
        for entry in entries:
            require(
                not entry.is_dir()
                and not stat.S_ISLNK(entry.external_attr >> 16)
                and not (entry.flag_bits & 1),
                "Historical runtime archive member is not a regular unencrypted source",
            )
            content_digest = hashlib.sha256()
            with archive.open(entry) as member:
                while chunk := member.read(1024 * 1024):
                    content_digest.update(chunk)
            require(
                content_digest.hexdigest() == expected[entry.filename],
                f"Historical runtime archive member changed: {entry.filename}",
            )
            verified["/" + entry.filename] = content_digest.hexdigest()
    return verified


def verify_completed(
    directory: Path,
    expected_sha256: str | None = None,
    *,
    model_key: str | None = None,
    check_models: bool = True,
) -> dict[str, Any]:
    """Verify a completed PASS using its frozen runtime and live nonruntime evidence."""
    directory = directory.resolve()
    contract = probe.read(directory / "manifest.json")
    runtime = _completed_runtime(directory, contract)
    contract = _verify(
        directory,
        expected_sha256,
        model_key=model_key,
        check_models=check_models,
        archived_runtime=runtime,
    )
    audit = _raw_audit_gate(directory, contract, archived_runtime=runtime)
    require(
        audit["status"] == "PASS" and audit["acceptance_pass"] is True,
        "Historical independent raw audit did not pass acceptance",
    )
    return contract


def raw_audit_gate(
    directory: Path, contract: dict[str, Any], *, completed_runtime: bool = False
) -> dict[str, Any]:
    if completed_runtime:
        verified = verify_completed(directory, check_models=False)
        require(verified == contract, "Historical raw audit manifest differs")
        audit = probe.read(directory / "raw_audit.json")
        return {
            "status": "PASS",
            "acceptance_pass": audit["acceptance_pass"],
            "sha256": probe.sha(directory / "raw_audit.json"),
        }
    return _raw_audit_gate(directory, contract)


def _raw_audit_gate(
    directory: Path,
    contract: dict[str, Any],
    *,
    archived_runtime: dict[str, str] | None = None,
) -> dict[str, Any]:
    path = directory / "raw_audit.json"
    if not path.exists():
        return {"status": "MISSING", "acceptance_pass": False, "sha256": None}
    audit = probe.read(path)
    require(
        audit.get("profile") == RAW_AUDIT_PROFILE
        and audit.get("status") == "PASS"
        and audit.get("mode") == contract["mode"]
        and audit.get("audit_scope") == "complete_population"
        and audit.get("manifest_sha256") == probe.sha(directory / "manifest.json")
        and audit.get("endpoint_gate_sha256") == fingerprint(ENDPOINT_GATE)
        and audit.get("short_trajectories") == contract["budget"]["short_trajectories"]
        and audit.get("full_trajectories") == contract["budget"]["full_trajectories"]
        and set(audit.get("workers", {})) == set(keys_for(contract["mode"]))
        and type(audit.get("acceptance_pass")) is bool,
        "Independent raw audit population, gate or manifest differs",
    )
    sources = audit.get("source_sha256")
    require(isinstance(sources, dict) and sources, "Independent audit lacks source hashes")
    for filename, digest in sources.items():
        if archived_runtime is not None and filename in archived_runtime:
            actual = archived_runtime[filename]
        else:
            actual = probe.sha(Path(filename))
        require(
            Path(filename).is_absolute() and actual == digest,
            f"Independently audited source changed: {filename}",
        )
    required = {
        str((directory / phase / "workers" / key / name).resolve())
        for phase in ("short", "full")
        for key in keys_for(contract["mode"])
        for name in (
            "state.json",
            "result.json",
            "config.json",
            "requests.jsonl",
            "raw.jsonl",
            "submissions.jsonl",
        )
    }
    required |= {
        str((directory / "workers" / key / name).resolve())
        for key in keys_for(contract["mode"])
        for name in ("state.json", "result.json")
    }
    required |= {
        str((directory / name).resolve())
        for name in ("manifest.json", "short/manifest.json", "full/manifest.json")
    }
    require(required <= set(sources), "Independent audit does not bind all raw worker evidence")
    return {
        "status": "PASS",
        "acceptance_pass": audit["acceptance_pass"],
        "sha256": probe.sha(path),
    }


async def execute(
    llm: Any,
    tokenizer: Any,
    sampling_class: Any,
    contract: dict[str, Any],
    model_key: str,
    directory: Path,
    *,
    replay: Any = probe.replay_case,
    reset: Any = episode.reset_case,
    initialized_output: bool = False,
    engine_config: dict[str, Any] | None = None,
    compiler_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reuse unchanged inference runners; never patch their globals or action text."""
    output = directory / "workers" / model_key
    if initialized_output:
        require(
            probe.read(output / "state.json")["status"] == "INITIALIZING",
            "Acceptance worker output is not a fresh initialization",
        )
    else:
        output.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "RUNNING",
        "profile": PROFILE,
        "model_key": model_key,
        "manifest_sha256": probe.sha(directory / "manifest.json"),
        "phase": "short",
    }
    probe.save(output / "state.json", state)
    try:
        short_output, full_output = (
            directory / phase / "workers" / model_key for phase in ("short", "full")
        )
        short_output.mkdir(exist_ok=False)
        full_output.mkdir(exist_ok=False)
        for phase, phase_output in (("short", short_output), ("full", full_output)):
            manifest = probe.read(directory / phase / "manifest.json")
            probe.save(
                phase_output / "config.json",
                {
                    "profile": manifest["profile"],
                    "model_key": model_key,
                    "engine": engine_config
                    or probe.engine_settings(Path(contract["models"][model_key]["model_path"])),
                    "action_protocol": PROTOCOL,
                    "enable_thinking": False,
                    "observation_profile": OBSERVATION_PROFILE,
                    "start_mode": "historical_prefix" if phase == "short" else "fullreset",
                    "manifest_sha256": probe.sha(directory / phase / "manifest.json"),
                    "acceptance_manifest_sha256": state["manifest_sha256"],
                    "model_fingerprint": contract["models"][model_key]["fingerprint"],
                    "compiler": compiler_config,
                },
            )
        short = await probe.execute(
            llm,
            tokenizer,
            sampling_class,
            probe.read(directory / "short/manifest.json"),
            model_key,
            short_output,
            replay=replay,
        )
        if not (short_output / "raw.jsonl").exists() and short["summary"]["requests"] == 0:
            (short_output / "raw.jsonl").touch(exist_ok=False)
        state["phase"] = "full"
        probe.save(output / "state.json", state)
        full = await episode.execute(
            llm,
            tokenizer,
            sampling_class,
            probe.read(directory / "full/manifest.json"),
            model_key,
            full_output,
            reset=reset,
        )
        if not (full_output / "raw.jsonl").exists() and full["summary"]["requests"] == 0:
            (full_output / "raw.jsonl").touch(exist_ok=False)
        endpoints = endpoint_readout(
            contract["short_cases"], short["grid"], json_lines(short_output / "requests.jsonl")
        )
        state.update(
            status="OBSERVED",
            phase="complete",
            short=short["summary"],
            full=full["summary"],
            endpoint_adherence=endpoints,
            behavioral_gate="PASS"
            if all(row["status"] in {"RECOVERED", "ADVANCED"} for row in short["grid"])
            and endpoints["endpoint_adherent"] == endpoints["planned"]
            and all(row["status"] == "COMPLETE" for row in full["grid"])
            else "FAIL",
        )
        probe.save(output / "result.json", state)
        return state
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        probe.save(output / "state.json", state)


def worker(directory: Path, model_key: str, expected_sha256: str) -> dict[str, Any]:
    contract = verify(directory, expected_sha256, model_key=model_key)
    model = Path(contract["models"][model_key]["model_path"])
    if any(
        key.startswith(("FINANCE_SCREENING_", "FINANCE_LEARNING_", "FINANCE_CRITIC_"))
        for key in os.environ
    ):
        raise ValueError("Inference worker inherited training hooks")
    output = directory / "workers" / model_key
    output.mkdir(parents=True, exist_ok=False)
    initialization = {
        "status": "INITIALIZING",
        "profile": PROFILE,
        "model_key": model_key,
        "manifest_sha256": expected_sha256,
        "phase": "engine_initialization",
    }
    probe.save(output / "state.json", initialization)
    try:
        return initialize_worker(directory, model_key, expected_sha256, contract, model)
    except BaseException as error:
        current = probe.read(output / "state.json")
        current.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        probe.save(output / "state.json", current)
        raise


def initialize_worker(
    directory: Path, model_key: str, expected_sha256: str, contract: dict[str, Any], model: Path
) -> dict[str, Any]:
    from src.agents.launcher import _resolve_c_compiler

    compiler, version = _resolve_c_compiler(parent=dict(os.environ))
    os.environ.update(
        CC=compiler,
        VLLM_USE_V1="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    engine = probe.engine_settings(model)

    # The parent owns the holder lease and process-group telemetry; this runner
    # performs no reservation release, service startup, retry or fallback action.
    def interrupted(signum: int, frame: Any) -> None:
        raise InterruptedError(f"Acceptance worker interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    llm = LLM(**engine)
    result = asyncio.run(
        execute(
            llm,
            tokenizer,
            SamplingParams,
            contract,
            model_key,
            directory,
            initialized_output=True,
            engine_config=engine,
            compiler_config={"CC": compiler, "version": version},
        )
    )
    verify(directory, expected_sha256, check_models=False)
    return result


def json_lines(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def phase_rows(directory: Path, contract: dict[str, Any], phase: str) -> tuple[list[dict], dict]:
    manifest = probe.read(directory / phase / "manifest.json")
    rows, workers = [], {}
    for key in keys_for(contract["mode"]):
        planned = probe.grid(manifest, key) if phase == "short" else episode.grid(manifest, key)
        path = directory / phase / "workers" / key / "result.json"
        if not path.exists():
            rows.extend(planned)
            workers[key] = "UNOBSERVED"
            continue
        observed = probe.read(path)
        state_path = path.parent / "state.json"
        require(
            probe.read(state_path) == observed
            and observed.get("status") == "OBSERVED"
            and observed.get("model_key") == key
            and observed.get("profile") == manifest["profile"],
            f"{phase}/{key}: state/result not identical OBSERVED evidence",
        )
        actual = observed["grid"]
        require(len(actual) == len(planned), f"{phase}/{key}: missing trajectories")
        immutable = (
            "id",
            "model_identity",
            "base_seed",
            "case_id",
            "case_kind",
            "profile",
            "sampling",
            "rollout_identity",
            "request_limit",
        )
        for expected, row in zip(planned, actual, strict=True):
            require(
                all(row.get(field) == expected[field] for field in immutable),
                f"{phase}/{key}: immutable trajectory identity differs",
            )
            allowed = (
                {"RECOVERED", "ADVANCED", "INCOMPLETE"}
                if phase == "short"
                else {"COMPLETE", "INCOMPLETE"}
            )
            require(
                row["status"] in allowed
                and type(row["requests"]) is int
                and 0 <= row["requests"] <= (4 if phase == "short" else 256),
                f"{phase}/{key}: unfinished or over-budget trajectory",
            )
            if phase == "short":
                require(
                    not {"terminal_metrics", "full_window_net_return"} & set(row),
                    "Short recovery cannot publish terminal finance",
                )
                if row["status"] in {"RECOVERED", "ADVANCED"}:
                    require(
                        row["advanced_to_session"] == row["takeover_session"] + 1,
                        "Short success is not exactly one session advance",
                    )
            else:
                require(
                    all(
                        type(event["followup_requests"]) is int
                        and 0 <= event["followup_requests"] <= 4
                        and event["status"] in {"RECOVERED", "INCOMPLETE"}
                        for event in row["recovery_events"]
                    ),
                    "Full recovery event cap differs",
                )
                if row["status"] == "INCOMPLETE":
                    require(
                        not {"terminal_metrics", "full_window_net_return"} & set(row),
                        "Incomplete fullreset trajectory has terminal finance",
                    )
        rows.extend(actual)
        workers[key] = "OBSERVED"
    return rows, workers


def readout(directory: Path) -> dict[str, Any]:
    contract = verify(directory, check_models=False)
    short, short_workers = phase_rows(directory, contract, "short")
    full, full_workers = phase_rows(directory, contract, "full")
    expected = contract["budget"]
    require(
        len(short) == expected["short_trajectories"] and len(full) == expected["full_trajectories"],
        "Acceptance drops planned trajectories",
    )
    complete = all(
        value == "OBSERVED" for value in [*short_workers.values(), *full_workers.values()]
    )
    success = complete and all(row["status"] in {"RECOVERED", "ADVANCED"} for row in short)
    success = success and all(row["status"] == "COMPLETE" for row in full)
    requests = [
        record
        for key in keys_for(contract["mode"])
        if (directory / "short/workers" / key / "requests.jsonl").exists()
        for record in json_lines(directory / "short/workers" / key / "requests.jsonl")
    ]
    endpoints = endpoint_readout(contract["short_cases"], short, requests)
    success = success and endpoints["endpoint_adherent"] == endpoints["planned"]
    audit = raw_audit_gate(directory, contract)
    acceptance = (
        "PENDING"
        if not complete
        else (
            "FAIL"
            if not success
            else (
                "PASS"
                if audit["status"] == "PASS" and audit["acceptance_pass"]
                else "PENDING_AUDIT"
            )
        )
    )
    partitions = {case["id"]: case["bank_partition"] for case in contract["short_cases"]}
    result = {
        "profile": PROFILE,
        "mode": contract["mode"],
        "status": "OBSERVED" if complete else "INCOMPLETE",
        "behavioral_gate": "PASS" if success else ("FAIL" if complete else "PENDING"),
        "acceptance": acceptance,
        "manifest_sha256": probe.sha(directory / "manifest.json"),
        "short_workers": short_workers,
        "full_workers": full_workers,
        "short_summary": probe.summary(short),
        "full_summary": episode.summary(full),
        "short_by_partition": {
            partition: probe.summary(
                [row for row in short if partitions[row["case_id"]] == partition]
            )
            for partition in ("known", "fresh")
        },
        "short_grid": short,
        "full_grid": full,
        "endpoint_adherence": endpoints,
        "full_financial_activity": financial_activity(full),
        "endpoint_gate_sha256": fingerprint(ENDPOINT_GATE),
        "repeated_sft_acceptance": contract["mode"] == "all",
        "limitations": contract["limitations"],
        "raw_audit_required": True,
        "raw_audit_status": audit["status"],
        "raw_audit_sha256": audit["sha256"],
    }
    probe.save(directory / "readout.json", result)
    return result


def financial_activity(rows: list[dict]) -> list[dict[str, Any]]:
    """Expose actual fills and terminal performance without adding a profit gate."""
    details = []
    for row in rows:
        observed = "final_ledger" in row
        fills = row.get("final_ledger", {}).get("fills", [])
        metrics = row.get("terminal_metrics", {})
        details.append(
            {
                "id": row["id"],
                "case_id": row["case_id"],
                "case_kind": row["case_kind"],
                "model_identity": row["model_identity"],
                "sampling": row["sampling"],
                "status": row["status"],
                "observed_ledger": observed,
                "buy_fills": sum(fill["side"] == "buy" for fill in fills) if observed else None,
                "sell_fills": sum(fill["side"] == "sell" for fill in fills) if observed else None,
                "commission": math.fsum(fill["commission"] for fill in fills) if observed else None,
                "slippage_cost": math.fsum(fill["slippage_cost"] for fill in fills)
                if observed
                else None,
                "total_cost": math.fsum(
                    fill["commission"] + fill["slippage_cost"] for fill in fills
                )
                if observed
                else None,
                "total_turnover": metrics.get("turnover"),
                "full_window_net_return": row.get("full_window_net_return"),
                "terminal_score": metrics.get("terminal_score"),
            }
        )
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare", "verify", "worker", "readout"):
        command = commands.add_parser(name)
        command.add_argument("--directory", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--known-manifest", type=Path, required=True)
            command.add_argument("--acceptance-bank", type=Path, required=True)
            command.add_argument("--models", type=Path, required=True)
            command.add_argument("--mode", choices=MODES, required=True)
            command.add_argument("--sft-gate-readout", type=Path)
        if name in {"verify", "worker"}:
            command.add_argument("--expected-manifest-sha256", required=name == "worker")
        if name == "worker":
            command.add_argument("--model-key", choices=MODEL_KEYS, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(
            args.directory,
            args.known_manifest,
            args.acceptance_bank,
            args.models,
            args.mode,
            sft_gate_readout=args.sft_gate_readout,
        )
    elif args.action == "verify":
        contract = verify(args.directory, args.expected_manifest_sha256)
        result = {"status": "VERIFIED", "budget": contract["budget"]}
    elif args.action == "worker":
        result = worker(args.directory, args.model_key, args.expected_manifest_sha256)
    else:
        result = readout(args.directory)
    print(probe.canonical(result))


if __name__ == "__main__":
    main()

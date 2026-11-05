"""Bounded, paired order-rejection diagnostics for frozen R5 actors.

Preparation replays historical evidence on CPU. Workers run inference only and
stop each continuation after its first legal session advance or four requests.
RECOVERED/ADVANCED are local recovery outcomes, never full-episode performance.
The controller owns GPU reservation handoff; this module never controls a holder.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import signal
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from agent_r1.env.base import Action

from src.agents.coldstart_data import parse_state, render_prompt
from src.agents.rollout_seed import derive_request_seed
from src.env.trading import TradingEnv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_R5 = ROOT / "runs/m2-parameter-grid-20261008-r5"
DEFAULT = ROOT / "runs/m2-order-recovery-20261011"
PROFILE = "r5_order_recovery_short_v1"
OLD = "history_features_v1"
NEW = "history_features_recovery_v1"
PROFILES = (OLD, NEW)
ENDPOINT_PROFILE = "r5_order_recovery_endpoint_v2"
ENDPOINT_OBSERVATION = "history_features_recovery_v2"
EXPERIMENT_PROFILES = {
    PROFILE: PROFILES,
    ENDPOINT_PROFILE: (NEW, ENDPOINT_OBSERVATION),
}
PROTOCOL = "single_tool_portfolio_v1"
FAILURE_IDS = tuple(f"formal-train-{n:04d}" for n in (210, 252, 336, 399, 420, 441))
NORMAL_IDS = ("formal-train-0231", "formal-train-0273")
MODEL_KEYS = ("sft", "seed42_m1", "seed42_m2", "seed43_m1", "seed43_m2")
REQUEST_CAP, PROMPT_CAP, RESPONSE_CAP = 4, 8192, 4096
MAX_BATCH = 8
DIRECTION_ERRORS = {
    "buy target_weight must be above the current weight",
    "sell target_weight must be below the current weight",
}
ECONOMIC_FIELDS = (
    "cash",
    "positions",
    "portfolio_equity",
    "session_index",
    "current_date",
    "sessions_remaining",
)


def json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Unsupported evidence value: {type(value).__name__}")


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False, default=json_default
    )


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=json_default) + "\n")
    temporary.replace(path)


def read(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def full_text(document: dict[str, Any], name: str) -> str:
    text = document.get("text")
    if (
        document.get("truncated") is not False
        or not isinstance(text, str)
        or document.get("char_count") != len(text)
        or document.get("sha256") != hashlib.sha256(text.encode()).hexdigest()
    ):
        raise ValueError(f"Historical {name} is truncated or its identity differs")
    return text


def ledger_state(env: Any) -> dict[str, Any]:
    return json.loads(
        canonical(
            {
                "cash": env.ledger.cash,
                "positions": dict(env.ledger.positions),
                "fills": [asdict(fill) for fill in env.ledger.fills],
                "session_index": env.current_session_index,
            }
        )
    )


def source_case(
    record: dict[str, Any],
    row: dict[str, Any],
    *,
    kind: str,
    source: Path,
    line_number: int,
    line_sha256: str,
) -> dict[str, Any]:
    """Select a prefix by a fixed error rule, never by financial outcomes."""
    interactions, steps = record["interactions"], record["steps"]
    if len(interactions) != len(steps) or not interactions:
        raise ValueError("Historical interaction/step population differs")
    if kind == "rejection":
        indices = [
            i
            for i, item in enumerate(interactions)
            if item["info"].get("error") in DIRECTION_ERRORS
        ]
        if not indices:
            raise ValueError("No order-direction rejection in source")
        stop = indices[0] + 1
    else:
        stop = 1
        if interactions[0]["info"].get("invalid_action"):
            raise ValueError("Normal prefix must be legal")
    selected = copy.deepcopy(interactions[:stop])
    for interaction in selected:
        full_text(interaction["response"], "response")
        full_text(interaction["observation"], "observation")
    kwargs = (
        json.loads(row["env_kwargs"])
        if isinstance(row["env_kwargs"], str)
        else dict(row["env_kwargs"])
    )
    kwargs.update(
        observation_profile=OLD, capture_interactions=True, reward_mode="finance_trajectory"
    )
    if row["extra_info"].get("source_split") != "train":
        raise ValueError("Recovery diagnostics are restricted to historical Train")
    return {
        "id": record["episode_id"],
        "kind": kind,
        "env_kwargs": kwargs,
        "raw_prompt": list(row["prompt"]),
        "prefix_interactions": selected,
        "prefix_steps": copy.deepcopy(steps[:stop]),
        "prefix": [item["response"]["text"] for item in selected],
        "source_base_seed": record["base_seed"],
        "source_identity": copy.deepcopy(record["identity"]),
        "source": {
            "path": str(source.resolve()),
            "line_number": line_number,
            "line_sha256": line_sha256,
        },
        "expected_start_session_index": selected[-1]["result_session_index"],
    }


async def replay_case(
    case: dict[str, Any], profile: str = OLD, *, env_class: Any = TradingEnv
) -> tuple[Any, Any]:
    """Require exact old observations and identical accounting under both profiles."""
    if profile not in {OLD, NEW, ENDPOINT_OBSERVATION}:
        raise ValueError("Unregistered recovery observation profile")
    kwargs = dict(case["env_kwargs"])
    kwargs.update(observation_profile=profile, capture_interactions=True)
    env = env_class(**kwargs)
    obs = env.reset(raw_prompt=case["raw_prompt"])
    for index, text in enumerate(case["prefix"]):
        expected = case["prefix_interactions"][index]
        if text != full_text(expected["response"], "response"):
            raise ValueError("Prefix response differs from historical evidence")
        before = ledger_state(env)
        obs, _, done, info = await env.step(Action(text=text))
        actual = env.interaction_trace[-1]
        trace = json.loads(canonical(asdict(env.credit_step_trace[-1])))
        if trace != case["prefix_steps"][index]:
            raise ValueError(f"Prefix credit/session trace drift at {case['id']}:{index}")
        if actual["info"] != expected["info"] or actual["done"] != expected["done"]:
            raise ValueError(f"Prefix invalid/error/accounting drift at {case['id']}:{index}")
        if done:
            raise ValueError("Recovery prefix must remain before the episode terminal")
        if info.get("invalid_action") and ledger_state(env) != before:
            raise ValueError("Rejected prefix changed ledger or advanced the session")
        if profile == OLD and actual != expected:
            raise ValueError(f"Old-profile historical observation drift at {case['id']}:{index}")
        old_messages = json.loads(full_text(expected["observation"], "observation"))
        old_state = parse_state(type("HistoricalObservation", (), {"messages": old_messages})())
        state = parse_state(obs)
        if any(state.get(field) != old_state.get(field) for field in ECONOMIC_FIELDS):
            raise ValueError("Observation intervention changed economic state")
        if profile != OLD:
            permitted = {"order_rejection", "recent_events"}
            if {key: value for key, value in state.items() if key not in permitted} != {
                key: value for key, value in old_state.items() if key not in permitted
            }:
                raise ValueError("Recovery profile changed original features or tools")
    if env.current_session_index != case["expected_start_session_index"]:
        raise ValueError("Recovery takeover session differs from historical prefix")
    if "takeover_ledger" in case and ledger_state(env) != case["takeover_ledger"]:
        raise ValueError("Recovery takeover ledger differs from frozen bank")
    return env, obs


def select_bank(
    r5: Path, rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    by_id = {row["extra_info"]["episode_id"]: row for row in rows}
    wanted = set(FAILURE_IDS + NORMAL_IDS)
    if not wanted <= set(by_id):
        raise ValueError("R5 reserved Train windows do not contain the registered cases")
    selected: dict[str, Any] = {}
    sources: dict[str, str] = {}
    cell_order = (
        "sft0",
        "m1_lr1e7",
        "m2_lr1e7_eta025",
        "m2_lr1e7_eta0125",
        "m1_lr2e7",
        "m2_lr2e7_eta0125",
        "m2_lr2e7_eta025",
    )
    for seed in (42, 43):
        for cell in cell_order:
            folder = r5 / f"seed{seed}" / cell / "artifacts/screening_evidence"
            for source in sorted(folder.glob("screening-*.jsonl")):
                used = False
                with source.open() as handle:
                    for line_number, line in enumerate(handle, 1):
                        if not any(episode in line for episode in wanted - set(selected)):
                            continue
                        record = json.loads(line)
                        episode = record.get("episode_id")
                        identity = record.get("identity", {})
                        if episode not in wanted - set(selected) or not identity.get("validate"):
                            continue
                        if identity.get("rollout_n") != 0:
                            continue
                        rejection = episode in FAILURE_IDS
                        if rejection and record.get("complete") is not False:
                            continue
                        if not rejection and record.get("complete") is not True:
                            continue
                        if rejection and not any(
                            item["info"].get("error") in DIRECTION_ERRORS
                            for item in record.get("interactions", [])
                        ):
                            continue
                        selected[episode] = source_case(
                            record,
                            by_id[episode],
                            kind="rejection" if rejection else "normal",
                            source=source,
                            line_number=line_number,
                            line_sha256=hashlib.sha256(line.encode()).hexdigest(),
                        )
                        used = True
                if used:
                    sources[str(source.resolve())] = sha(source)
            if set(selected) == wanted:
                break
        if set(selected) == wanted:
            break
    if set(selected) != wanted:
        raise ValueError(f"Missing registered source cases: {sorted(wanted - set(selected))}")
    return [selected[key] for key in FAILURE_IDS + NORMAL_IDS], sources


def model_contracts(r5: Path) -> dict[str, Any]:
    models: dict[str, Any] = {}
    settings_path = r5 / "seed42/settings.json"
    settings = read(settings_path)
    sft = Path(settings["models"]["screening"]["name"])
    receipt = sft / "merge_receipt.json"
    if sha(receipt) != settings["models"]["screening"]["merge_receipt_sha256"]:
        raise ValueError("R5 shared SFT receipt differs")
    files = read(receipt)["files"]
    models["sft"] = {
        "model_path": str(sft),
        "identities": [{"id": f"seed{seed}_sft", "base_seed": seed} for seed in (42, 43)],
        "files": files,
        "evidence": {str(receipt): sha(receipt)},
    }
    for seed in (42, 43):
        other = read(r5 / f"seed{seed}/settings.json")["models"]["screening"]
        if other != settings["models"]["screening"]:
            raise ValueError("R5 seeds do not share the pinned SFT")
        for method, cell in (("m1", "m1_lr1e7"), ("m2", "m2_lr1e7_eta025")):
            directory = r5 / f"seed{seed}" / cell
            archive = directory / "actor_archive/manifest.json"
            receipt = directory / "actor_archive_receipt.json"
            manifest, retirement = read(archive), read(receipt)
            if retirement.get("status") != "COMPLETE" or retirement.get("manifest_sha256") != (
                sha(archive)
            ):
                raise ValueError("R5 actor archive retirement is not verified")
            original = manifest["original_hf"]
            models[f"seed{seed}_{method}"] = {
                "model_path": None,
                "archive_cell": str(directory),
                "identities": [{"id": f"seed{seed}_{method}", "base_seed": seed}],
                "files": {
                    name: {"bytes": size, "sha256": original["sha256"][name]}
                    for name, size in original["files"].items()
                },
                "evidence": {str(archive): sha(archive), str(receipt): sha(receipt)},
            }
    return models


def budget(profile: str = PROFILE) -> dict[str, Any]:
    return {
        "cases": 8,
        "rejection_cases": 6,
        "normal_cases": 2,
        "model_identities": 6,
        "model_engines": 5,
        "profiles": list(EXPERIMENT_PROFILES[profile]),
        "greedy_per_case": 1,
        "sampled_per_case": 4,
        "planned_trajectories": 480,
        "requests_per_trajectory": REQUEST_CAP,
        "maximum_requests": 1920,
        "maximum_response_tokens": 1920 * RESPONSE_CAP,
        "prompt_cap": PROMPT_CAP,
        "response_cap": RESPONSE_CAP,
        "training_updates": 0,
        "full_episode_evaluation": False,
        "automatic_retry": False,
    }


def prepare(directory: Path, r5: Path = DEFAULT_R5, profile: str = PROFILE) -> dict[str, Any]:
    import pandas as pd

    directory, r5 = directory.resolve(), r5.resolve()
    if directory.exists():
        raise FileExistsError("A new recovery directory is required; no overwrite")
    parquet = r5 / "seed42/prepared/dev.parquet"
    rows = pd.read_parquet(parquet).to_dict("records")
    cases, evidence = select_bank(r5, rows)
    evidence[str(parquet)] = sha(parquet)
    for case in cases:
        env, _ = asyncio.run(replay_case(case, OLD))
        case["takeover_ledger"] = ledger_state(env)
        for observation_profile in EXPERIMENT_PROFILES[profile]:
            _, obs = asyncio.run(replay_case(case, observation_profile))
            state = parse_state(obs)
            if (
                case["kind"] == "rejection"
                and "order_rejection" not in state
                and observation_profile != OLD
            ):
                raise ValueError("New observation profile did not expose order-rejection feedback")
            if case["kind"] == "normal" and "order_rejection" in state:
                raise ValueError("Normal prefix acquired rejection feedback")
    models = model_contracts(r5)
    for model in models.values():
        evidence.update(model["evidence"])
    runtime_paths = set((ROOT / "src").rglob("*.py")) | set(
        (ROOT / "third_party/agent-r1/agent_r1").rglob("*.py")
    )
    runtime = {str(path.resolve()): sha(path) for path in sorted(runtime_paths)}
    manifest = {
        "profile": profile,
        "status": "PREPARED",
        "r5_directory": str(r5),
        "budget": budget(profile),
        "action_protocol": PROTOCOL,
        "enable_thinking": False,
        "cases": cases,
        "models": models,
        "evidence_sha256": evidence,
        "runtime_sha256": runtime,
        "limitations": [
            "Fixed error-selected Train states; this does not estimate failure prevalence.",
            "Historical actions are scripted prefixes, not new model experience.",
            "New batching, engine and seed identities do not imply tokenwise R5 replay.",
            "Recovery is the first legal session advance within four requests, not an episode.",
            "No terminal financial metrics or synthesized returns are reported.",
        ],
    }
    directory.mkdir(parents=True, exist_ok=False)
    save(directory / "manifest.json", manifest)
    save(
        directory / "state.json",
        {
            "status": "PREPARED",
            "profile": profile,
            "manifest_sha256": sha(directory / "manifest.json"),
        },
    )
    return {
        "status": "PREPARED",
        "directory": str(directory),
        "budget": budget(profile),
        "manifest_sha256": sha(directory / "manifest.json"),
    }


def verify(directory: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    digest = sha(manifest_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Recovery manifest digest differs")
    manifest = read(manifest_path)
    profile = manifest.get("profile")
    if profile not in EXPERIMENT_PROFILES or manifest.get("budget") != budget(profile):
        raise ValueError("Recovery manifest protocol or budget differs")
    if [case["id"] for case in manifest["cases"]] != list(FAILURE_IDS + NORMAL_IDS):
        raise ValueError("Recovery case population differs")
    if set(manifest["models"]) != set(MODEL_KEYS):
        raise ValueError("Recovery model population differs")
    for path, expected in {**manifest["evidence_sha256"], **manifest["runtime_sha256"]}.items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Frozen recovery source changed: {path}")
    return manifest


def validate_model(model: Path, contract: dict[str, Any]) -> None:
    if not model.is_dir():
        raise ValueError("Restore the archived actor to a new HF directory before running")
    for name, evidence in contract["files"].items():
        path = model / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != evidence["bytes"]:
            raise ValueError(f"Frozen model file inventory differs: {name}")
        if sha(path) != evidence["sha256"]:
            raise ValueError(f"Frozen model file content differs: {name}")


def grid(manifest: dict[str, Any], model_key: str) -> list[dict[str, Any]]:
    rows = []
    for identity in manifest["models"][model_key]["identities"]:
        for case in manifest["cases"]:
            for profile in manifest.get("budget", {}).get("profiles", PROFILES):
                for sibling in range(5):
                    source_identity = copy.deepcopy(
                        case["request_seed_identity"]
                        if "request_seed_identity" in case
                        else case["source_identity"]
                    )
                    source_identity.update(rollout_n=sibling, validate=True)
                    rows.append(
                        {
                            "id": f"{identity['id']}:{case['id']}:{profile}:{sibling}",
                            "model_identity": identity["id"],
                            "base_seed": identity["base_seed"],
                            "case_id": case["id"],
                            "case_kind": case["kind"],
                            "profile": profile,
                            "sampling": "greedy" if sibling == 0 else "sampled",
                            "rollout_identity": source_identity,
                            "status": "UNSTARTED",
                            "requests": 0,
                            "invalid_requests": 0,
                            "response_tokens": 0,
                            "request_limit": REQUEST_CAP,
                        }
                    )
    return rows


def sampling_parameters(row: dict[str, Any], prefix_length: int) -> dict[str, Any]:
    env_step = prefix_length + row["requests"]
    return {
        "n": 1,
        "temperature": 0.0 if row["sampling"] == "greedy" else 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "max_tokens": RESPONSE_CAP,
        "seed": derive_request_seed(row["base_seed"], row["rollout_identity"], env_step),
    }


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
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


async def execute(
    llm: Any,
    tokenizer: Any,
    sampling_class: Any,
    manifest: dict[str, Any],
    model_key: str,
    output: Path,
    *,
    replay: Any = replay_case,
) -> dict[str, Any]:
    rows = grid(manifest, model_key)
    cases = {case["id"]: case for case in manifest["cases"]}
    state = {
        "profile": manifest.get("profile", PROFILE),
        "status": "RUNNING",
        "model_key": model_key,
        "full_episode_evaluation": False,
        "grid": rows,
    }
    save(output / "state.json", state)
    try:
        with (
            (output / "requests.jsonl").open("x") as evidence,
            (output / "submissions.jsonl").open("x") as submitted,
        ):
            for start in range(0, len(rows), MAX_BATCH):
                members, runtime = rows[start : start + MAX_BATCH], {}
                for row in members:
                    env, obs = await replay(cases[row["case_id"]], row["profile"])
                    row.update(
                        status="ACTIVE",
                        takeover_session=env.current_session_index,
                        prefix_length=len(cases[row["case_id"]]["prefix"]),
                    )
                    runtime[row["id"]] = [env, obs]
                while any(row["status"] == "ACTIVE" for row in members):
                    submissions = []
                    for row in members:
                        if row["status"] != "ACTIVE":
                            continue
                        if row["requests"] >= REQUEST_CAP:
                            row.update(status="INCOMPLETE", failure="REQUEST_CAP")
                            continue
                        env, obs = runtime[row["id"]]
                        prompt = render_prompt(tokenizer, env, obs, action_protocol=PROTOCOL)
                        ids = list(tokenizer.encode(prompt, add_special_tokens=False))
                        if len(ids) > PROMPT_CAP:
                            row.update(status="INCOMPLETE", failure="PROMPT_CAP")
                            evidence.write(
                                canonical(
                                    {
                                        "trajectory_id": row["id"],
                                        "failure": "PROMPT_CAP",
                                        "prompt": prompt,
                                        "prompt_token_ids": ids,
                                    }
                                )
                                + "\n"
                            )
                            evidence.flush()
                            continue
                        params = sampling_parameters(row, row["prefix_length"])
                        request = {
                            "trajectory_id": row["id"],
                            "request_index": row["requests"],
                            "base_seed": row["base_seed"],
                            "rollout_identity": row["rollout_identity"],
                            "seed_env_step": row["prefix_length"] + row["requests"],
                            "decision_session_index": env.current_session_index,
                            "prompt": prompt,
                            "prompt_token_ids": ids,
                            "sampling_params": params,
                            "ledger_before": ledger_state(env),
                        }
                        submissions.append((row, request, sampling_class(**params)))
                    if not submissions:
                        break
                    submitted.write(
                        canonical({"requests": [item[1] for item in submissions]}) + "\n"
                    )
                    submitted.flush()
                    for row, _, _ in submissions:
                        row["requests"] += 1
                    save(output / "state.json", state)
                    outputs = llm.generate(
                        [{"prompt_token_ids": item[1]["prompt_token_ids"]} for item in submissions],
                        [item[2] for item in submissions],
                        use_tqdm=False,
                    )
                    raw = [
                        {
                            "request_id": returned.request_id,
                            "prompt_token_ids": list(returned.prompt_token_ids),
                            "outputs": [
                                {
                                    "token_ids": list(response.token_ids),
                                    "finish_reason": response.finish_reason,
                                    "stop_reason": getattr(response, "stop_reason", None),
                                }
                                for response in returned.outputs
                            ],
                        }
                        for returned in outputs
                    ]
                    # Persist the entire returned batch before mapping or decoding can fail.
                    with (output / "raw.jsonl").open("a") as raw_handle:
                        raw_handle.write(canonical({"outputs": raw}) + "\n")
                        raw_handle.flush()
                    if len(outputs) != len(submissions) or len(
                        {item["request_id"] for item in raw}
                    ) != (len(raw)):
                        raise ValueError("Generation response count or request IDs differ")
                    for (row, request, _), returned in zip(submissions, outputs, strict=True):
                        if (
                            list(returned.prompt_token_ids) != request["prompt_token_ids"]
                            or len(returned.outputs) != 1
                        ):
                            raise ValueError("Generation response mapping differs")
                        response = returned.outputs[0]
                        tokens = list(response.token_ids)
                        text = tokenizer.decode(tokens, skip_special_tokens=True)
                        record = {
                            **request,
                            "vllm_request_id": returned.request_id,
                            "response": text,
                            "response_token_ids": tokens,
                            "finish_reason": response.finish_reason,
                            "stop_reason": getattr(response, "stop_reason", None),
                        }
                        row["response_tokens"] += len(tokens)
                        env, _ = runtime[row["id"]]
                        if (
                            not tokens
                            or len(tokens) >= RESPONSE_CAP
                            or response.finish_reason != "stop"
                        ):
                            row.update(status="INCOMPLETE", failure="RESPONSE_INCOMPLETE")
                            record["environment_applied"] = False
                        else:
                            before = env.current_session_index
                            obs, _, done, info = await env.step(Action(text=text, token_ids=tokens))
                            runtime[row["id"]][1] = obs
                            trace = asdict(env.credit_step_trace[-1])
                            row["invalid_requests"] += int(bool(info.get("invalid_action")))
                            record.update(
                                environment_applied=True,
                                feedback=info,
                                trace=trace,
                                ledger_after=ledger_state(env),
                                observation_messages=obs.messages,
                                terminal_reached=done,
                            )
                            if (
                                info.get("invalid_action")
                                and ledger_state(env) != request["ledger_before"]
                            ):
                                raise ValueError("Rejected model action changed ledger or session")
                            advanced = env.current_session_index > before
                            if advanced and (
                                info.get("invalid_action")
                                or env.current_session_index != before + 1
                            ):
                                raise ValueError(
                                    "Recovery advance is not a single legal session transition"
                                )
                            if advanced:
                                row.update(
                                    status="RECOVERED"
                                    if row["case_kind"] == "rejection"
                                    else "ADVANCED",
                                    advance_action=trace["action"],
                                    advanced_to_session=env.current_session_index,
                                )
                            elif done:
                                row.update(status="INCOMPLETE", failure="TERMINAL_WITHOUT_ADVANCE")
                        evidence.write(canonical(record) + "\n")
                        evidence.flush()
                    save(output / "state.json", state)
                for row in members:
                    env, _ = runtime[row["id"]]
                    row["continuation_trace"] = [
                        asdict(step) for step in env.credit_step_trace[row["prefix_length"] :]
                    ]
                    row["final_ledger"] = ledger_state(env)
        state.update(status="OBSERVED", summary=summary(rows))
        save(output / "result.json", state)
        return state
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(output / "state.json", state)


def engine_settings(model: Path) -> dict[str, Any]:
    return {
        "model": str(model),
        "tokenizer": str(model),
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "tensor_parallel_size": 1,
        "max_model_len": PROMPT_CAP + RESPONSE_CAP,
        "max_num_seqs": MAX_BATCH,
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
            "max_new_tokens": RESPONSE_CAP,
        },
    }


def worker(
    directory: Path,
    model_key: str,
    model_path: Path | None,
    expected_sha256: str | None = None,
    episode_directory: Path | None = None,
    expected_episode_sha256: str | None = None,
) -> dict[str, Any]:
    manifest = verify(directory, expected_sha256)
    episode_manifest = None
    if episode_directory is not None:
        from src.agents import order_recovery_episode as episode

        episode_manifest = episode.verify(episode_directory, expected_episode_sha256)
        if episode_manifest["models"] != manifest["models"]:
            raise ValueError("Short and complete-window evaluations must share model contracts")
    contract = manifest["models"][model_key]
    path = model_path or (Path(contract["model_path"]) if contract["model_path"] else None)
    if path is None:
        raise ValueError("--model is required for an actor restored from its R5 archive")
    path = path.resolve()
    validate_model(path, contract)
    output = directory / "workers" / model_key
    output.mkdir(parents=True, exist_ok=False)
    from src.agents.launcher import _resolve_c_compiler

    compiler, compiler_version = _resolve_c_compiler(parent=dict(os.environ))
    os.environ.update(
        CC=compiler,
        VLLM_USE_V1="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    if any(
        key.startswith(("FINANCE_SCREENING_", "FINANCE_LEARNING_", "FINANCE_CRITIC_"))
        for key in os.environ
    ):
        raise ValueError("Inference worker inherited training hooks")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    engine = engine_settings(path)
    save(
        output / "config.json",
        {
            "engine": engine,
            "model_key": model_key,
            "manifest_sha256": sha(directory / "manifest.json"),
            "compiler": {"CC": compiler, "version": compiler_version},
            "action_protocol": PROTOCOL,
            "enable_thinking": False,
        },
    )

    def interrupted(signum: int, frame: Any) -> None:
        raise InterruptedError(f"Recovery probe interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        llm = LLM(**engine)
        result = asyncio.run(execute(llm, tokenizer, SamplingParams, manifest, model_key, output))
        if episode_manifest is not None:
            episode_output = episode_directory / "workers" / model_key
            episode_output.mkdir(parents=True, exist_ok=False)
            save(
                episode_output / "config.json",
                {
                    "engine": engine,
                    "model_key": model_key,
                    "manifest_sha256": sha(episode_directory / "manifest.json"),
                    "shared_engine_short_manifest_sha256": sha(directory / "manifest.json"),
                    "action_protocol": PROTOCOL,
                    "enable_thinking": False,
                    "observation_profile": episode.OBSERVATION_PROFILE,
                    "start_mode": episode.START_MODE,
                },
            )
            episode_result = asyncio.run(
                episode.execute(
                    llm, tokenizer, SamplingParams, episode_manifest, model_key, episode_output
                )
            )
            result["summary"]["complete_window_summary"] = episode_result["summary"]
            episode.verify(episode_directory, expected_episode_sha256)
        verify(directory, expected_sha256)
        return result["summary"]
    except BaseException as error:
        if not (output / "state.json").exists():
            save(
                output / "state.json",
                {
                    "status": "FAILED",
                    "model_key": model_key,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        raise


def readout(directory: Path) -> dict[str, Any]:
    manifest = verify(directory)
    rows, workers = [], {}
    for model_key in MODEL_KEYS:
        planned_rows = grid(manifest, model_key)
        path = directory / "workers" / model_key / "state.json"
        if not path.exists():
            workers[model_key] = "UNSTARTED"
            rows.extend(planned_rows)
            continue
        state = read(path)
        workers[model_key] = state["status"]
        expected = {row["id"] for row in planned_rows}
        actual = state.get("grid", [])
        if actual and ({row["id"] for row in actual} != expected or len(actual) != len(expected)):
            raise ValueError("Recovery worker trajectory population differs")
        if state["status"] == "OBSERVED" and (
            not actual
            or any(row["status"] not in {"RECOVERED", "ADVANCED", "INCOMPLETE"} for row in actual)
        ):
            raise ValueError("Observed worker contains unfinished recovery slots")
        rows.extend(actual if actual else planned_rows)
    observed = summary(rows)
    result = {
        "profile": manifest.get("profile", PROFILE),
        "status": "OBSERVED"
        if all(value == "OBSERVED" for value in workers.values())
        else "INCOMPLETE",
        "budget": budget(manifest.get("profile", PROFILE)),
        "workers": workers,
        "summary": observed,
        "limitations": manifest["limitations"],
    }
    save(directory / "readout.json", result)
    save(
        directory / "state.json",
        {
            "status": result["status"],
            "workers": workers,
            "manifest_sha256": sha(directory / "manifest.json"),
        },
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify", "worker", "readout"):
        action = sub.add_parser(name)
        action.add_argument("--directory", type=Path, default=DEFAULT)
        if name == "prepare":
            action.add_argument("--r5-directory", type=Path, default=DEFAULT_R5)
            action.add_argument("--profile", choices=EXPERIMENT_PROFILES, default=PROFILE)
        if name in {"verify", "worker"}:
            action.add_argument("--expected-manifest-sha256")
        if name == "worker":
            action.add_argument("--model-key", choices=MODEL_KEYS, required=True)
            action.add_argument("--model", type=Path)
            action.add_argument("--episode-directory", type=Path)
            action.add_argument("--expected-episode-sha256")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.directory, args.r5_directory, args.profile)
    elif args.command == "verify":
        verify(args.directory, args.expected_manifest_sha256)
        result = {"status": "VERIFIED", "new_model_requests": 0}
    elif args.command == "worker":
        result = worker(
            args.directory.resolve(),
            args.model_key,
            args.model,
            args.expected_manifest_sha256,
            args.episode_directory,
            args.expected_episode_sha256,
        )
    else:
        result = readout(args.directory.resolve())
    print(json.dumps(result, allow_nan=False, default=json_default), flush=True)


if __name__ == "__main__":
    main()

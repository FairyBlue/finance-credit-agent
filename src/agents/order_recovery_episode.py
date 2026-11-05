"""Full-reset, inference-only evaluation on the fixed order-recovery Train windows.

The short probe tests recovery from a historical prefix. This separate protocol
starts every policy from the original initial state and runs to the real terminal.
All interactions share the original 256-request cap; recovery never injects actions.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import math
import os
import signal
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent_r1.env.base import Action

from src.agents import order_recovery_probe as probe
from src.agents.coldstart_data import parse_state, render_prompt
from src.env.trading import TradingEnv

PROFILE = "r5_order_recovery_fullreset_v1"
OBSERVATION_PROFILE = "history_features_recovery_v2"
START_MODE = "fullreset"
REQUEST_CAP = 256
RECOVERY_CAP = 4
PROMPT_CAP, RESPONSE_CAP, MAX_BATCH = probe.PROMPT_CAP, probe.RESPONSE_CAP, probe.MAX_BATCH
MODEL_KEYS = probe.MODEL_KEYS


def budget() -> dict[str, Any]:
    return {
        "cases": 8,
        "historical_rejection_cases": 6,
        "normal_cases": 2,
        "model_identities": 6,
        "model_engines": 5,
        "observation_profile": OBSERVATION_PROFILE,
        "start_mode": START_MODE,
        "greedy_per_case": 1,
        "sampled_per_case": 4,
        "planned_trajectories": 240,
        "requests_per_trajectory": REQUEST_CAP,
        "recovery_followup_requests": RECOVERY_CAP,
        "maximum_requests": 240 * REQUEST_CAP,
        "maximum_response_tokens": 240 * REQUEST_CAP * RESPONSE_CAP,
        "prompt_cap": PROMPT_CAP,
        "response_cap": RESPONSE_CAP,
        "training_updates": 0,
        "full_episode_evaluation": True,
        "automatic_retry": False,
        "automatic_actions": False,
    }


async def reset_case(case: dict[str, Any], *, env_class: Any = None) -> tuple[Any, Any]:
    """Reset the original row, without applying any historical or scripted action."""
    kwargs = dict(case["env_kwargs"])
    kwargs.update(
        observation_profile=OBSERVATION_PROFILE,
        capture_interactions=True,
        reward_mode="finance_trajectory",
    )
    env = (env_class or TradingEnv)(**kwargs)
    obs = env.reset(raw_prompt=case["raw_prompt"])
    if env.current_session_index != 0 or env.credit_step_trace or env.episode_complete:
        raise ValueError("Full-reset evaluation did not start at the original initial state")
    if "initial_ledger" in case and probe.ledger_state(env) != case["initial_ledger"]:
        raise ValueError("Full-reset initial ledger differs from the prepared contract")
    return env, obs


def prepare(directory: Path, recovery_directory: Path) -> dict[str, Any]:
    """Inherit the short probe's frozen selection and actors, never its mutable state."""
    directory, recovery_directory = directory.resolve(), recovery_directory.resolve()
    if directory.exists():
        raise FileExistsError("A new full-reset evaluation directory is required; no overwrite")
    source = probe.verify(recovery_directory)
    cases = []
    for historical in source["cases"]:
        case = {
            name: copy.deepcopy(historical[name])
            for name in ("id", "kind", "env_kwargs", "raw_prompt", "source_identity")
        }
        case["selection_source"] = copy.deepcopy(historical["source"])
        case["historical_prefix_length"] = len(historical["prefix"])
        case["env_kwargs"].update(
            observation_profile=OBSERVATION_PROFILE,
            capture_interactions=True,
            reward_mode="finance_trajectory",
        )
        env, obs = asyncio.run(reset_case(case))
        case["initial_ledger"] = probe.ledger_state(env)
        old_kwargs = {**case["env_kwargs"], "observation_profile": probe.OLD}
        old = TradingEnv(**old_kwargs)
        old_obs = old.reset(raw_prompt=case["raw_prompt"])
        state, old_state = parse_state(obs), parse_state(old_obs)
        permitted = {"order_rejection", "recent_events"}
        if (
            probe.ledger_state(old) != case["initial_ledger"]
            or "order_rejection" in state
            or {key: value for key, value in state.items() if key not in permitted}
            != {key: value for key, value in old_state.items() if key not in permitted}
        ):
            raise ValueError("Recovery observation profile changed the original initial state")
        cases.append(case)
    evidence = copy.deepcopy(source["evidence_sha256"])
    evidence[str(recovery_directory / "manifest.json")] = probe.sha(
        recovery_directory / "manifest.json"
    )
    manifest = {
        "profile": PROFILE,
        "status": "PREPARED",
        "start_mode": START_MODE,
        "observation_profile": OBSERVATION_PROFILE,
        "budget": budget(),
        "action_protocol": probe.PROTOCOL,
        "enable_thinking": False,
        "selection_manifest": str(recovery_directory / "manifest.json"),
        "cases": cases,
        "models": copy.deepcopy(source["models"]),
        "evidence_sha256": evidence,
        "runtime_sha256": copy.deepcopy(source["runtime_sha256"]),
        "limitations": [
            "Eight fixed error-selected Train windows; no estimate of failure prevalence.",
            "Historical prefixes select the windows but are never executed in this protocol.",
            "The shared historical seed anchors do not imply tokenwise replay of R5.",
            "Any invalid action opens a recovery event with at most four followup requests.",
            "Only a legal session advance closes a recovery event; no automatic hold is used.",
            "Incomplete trajectories retain evidence and have no terminal financial metrics.",
        ],
    }
    own_source = str(Path(__file__).resolve())
    if manifest["runtime_sha256"].get(own_source) != probe.sha(Path(own_source)):
        raise ValueError("Prepare the short manifest after completing all full-reset source edits")
    directory.mkdir(parents=True, exist_ok=False)
    probe.save(directory / "manifest.json", manifest)
    digest = probe.sha(directory / "manifest.json")
    probe.save(
        directory / "state.json",
        {
            "status": "PREPARED",
            "profile": PROFILE,
            "manifest_sha256": digest,
        },
    )
    return {
        "status": "PREPARED",
        "directory": str(directory),
        "budget": budget(),
        "manifest_sha256": digest,
    }


def verify(directory: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    path = directory / "manifest.json"
    if expected_sha256 is not None and probe.sha(path) != expected_sha256:
        raise ValueError("Full-reset manifest digest differs")
    manifest = probe.read(path)
    if (
        manifest.get("profile") != PROFILE
        or manifest.get("budget") != budget()
        or manifest.get("start_mode") != START_MODE
        or manifest.get("observation_profile") != OBSERVATION_PROFILE
        or manifest.get("action_protocol") != probe.PROTOCOL
        or manifest.get("enable_thinking") is not False
    ):
        raise ValueError("Full-reset protocol, profile or budget differs")
    if [case["id"] for case in manifest["cases"]] != list(probe.FAILURE_IDS + probe.NORMAL_IDS):
        raise ValueError("Full-reset case population differs")
    if set(manifest["models"]) != set(MODEL_KEYS):
        raise ValueError("Full-reset model population differs")
    for key in MODEL_KEYS:
        seeds = (42, 43) if key == "sft" else (int(key[4:6]),)
        expected_identities = [
            {"id": f"seed{seed}_sft" if key == "sft" else key, "base_seed": seed} for seed in seeds
        ]
        if manifest["models"][key]["identities"] != expected_identities:
            raise ValueError("Full-reset model/seed identity differs")
    for case in manifest["cases"]:
        if (
            "prefix" in case
            or case["kind"] != ("rejection" if case["id"] in probe.FAILURE_IDS else "normal")
            or case["env_kwargs"].get("observation_profile") != OBSERVATION_PROFILE
            or case["env_kwargs"].get("reward_mode") != "finance_trajectory"
            or case["env_kwargs"].get("capture_interactions") is not True
            or case["initial_ledger"]["session_index"] != 0
            or case["initial_ledger"]["fills"]
        ):
            raise ValueError("Full-reset case contains a prefix or a changed initial contract")
    for filename, digest in {
        **manifest["evidence_sha256"],
        **manifest["runtime_sha256"],
    }.items():
        if probe.sha(Path(filename)) != digest:
            raise ValueError(f"Frozen full-reset source changed: {filename}")
    if sum(len(grid(manifest, key)) for key in MODEL_KEYS) != budget()["planned_trajectories"]:
        raise ValueError("Full-reset identity population differs")
    return manifest


def grid(manifest: dict[str, Any], model_key: str) -> list[dict[str, Any]]:
    rows = []
    for identity in manifest["models"][model_key]["identities"]:
        for case in manifest["cases"]:
            for sibling in range(5):
                anchor = {**case["source_identity"], "rollout_n": sibling, "validate": True}
                rows.append(
                    {
                        "id": f"{identity['id']}:{case['id']}:{OBSERVATION_PROFILE}:{sibling}",
                        "model_identity": identity["id"],
                        "base_seed": identity["base_seed"],
                        "case_id": case["id"],
                        "case_kind": case["kind"],
                        "profile": OBSERVATION_PROFILE,
                        "start_mode": START_MODE,
                        "sampling": "greedy" if sibling == 0 else "sampled",
                        "rollout_identity": anchor,
                        "status": "UNSTARTED",
                        "requests": 0,
                        "invalid_requests": 0,
                        "response_tokens": 0,
                        "request_limit": REQUEST_CAP,
                        "recovery_events": [],
                    }
                )
    return rows


def _complete(row: dict[str, Any], env: Any, obs: Any) -> None:
    if not env.episode_complete or env.current_session_index != env.episode_session_count - 1:
        raise ValueError("Full-reset terminal flags or final session disagree")
    metrics = env.screening_terminal_metrics()
    ledger = probe.ledger_state(env)
    observed_cost = math.fsum(
        fill["commission"] + fill["slippage_cost"] for fill in ledger["fills"]
    )
    if not math.isclose(metrics["transaction_cost"], observed_cost, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("Terminal transaction cost disagrees with actual ledger fills")
    row.update(
        status="COMPLETE",
        terminal_metrics=metrics,
        final_state=parse_state(obs),
        full_window_net_return=metrics["raw_net_return"],
    )


def _abort(row: dict[str, Any], reason: str) -> None:
    row.update(status="INCOMPLETE", failure=reason)
    if row["recovery_events"] and row["recovery_events"][-1]["status"] == "ACTIVE":
        row["recovery_events"][-1].update(status="INCOMPLETE", failure=reason)


def _observe_step(
    row: dict[str, Any],
    env: Any,
    obs: Any,
    info: dict[str, Any],
    *,
    before: int,
    request_index: int,
    done: bool,
) -> None:
    """One event spans all requests after an invalid until a legal session advance."""
    invalid = bool(info.get("invalid_action"))
    advanced = env.current_session_index > before
    if advanced and (invalid or env.current_session_index != before + 1):
        raise ValueError("Full-reset advance is not a single legal session transition")
    active = (
        row["recovery_events"][-1]
        if row["recovery_events"] and row["recovery_events"][-1]["status"] == "ACTIVE"
        else None
    )
    if invalid and active is None:
        active = {
            "event_index": len(row["recovery_events"]),
            "session_index": before,
            "trigger_request_index": request_index,
            "trigger_error": info.get("error"),
            "followup_requests": 0,
            "invalid_requests": 0,
            "status": "ACTIVE",
            "direction_rejection": info.get("error") in probe.DIRECTION_ERRORS,
        }
        row["recovery_events"].append(active)
    if active is not None:
        active["invalid_requests"] += int(invalid)
        if advanced:
            active.update(
                status="RECOVERED",
                advance_request_index=request_index,
                advanced_to_session=env.current_session_index,
                advance_action=env.credit_step_trace[-1].action,
            )
        elif active["followup_requests"] >= RECOVERY_CAP:
            _abort(row, "RECOVERY_CAP")
    if done:
        if not advanced or invalid:
            raise ValueError("Full-reset terminal was reached without a legal advance")
        _complete(row, env, obs)


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model_identity"], row["case_kind"], row["sampling"])].append(row)
    details = []
    for key, members in sorted(groups.items()):
        complete = [row for row in members if row["status"] == "COMPLETE"]
        all_complete = len(complete) == len(members)
        events = [event for row in members for event in row["recovery_events"]]
        details.append(
            {
                "model_identity": key[0],
                "case_kind": key[1],
                "sampling": key[2],
                "planned": len(members),
                "complete": len(complete),
                "statuses": dict(Counter(row["status"] for row in members)),
                "requests": sum(row["requests"] for row in members),
                "invalid_requests": sum(row["invalid_requests"] for row in members),
                "recovery_events": len(events),
                "recovered_events": sum(event["status"] == "RECOVERED" for event in events),
                "trajectories_with_later_recovery_events": sum(
                    len(row["recovery_events"]) > 1 for row in members
                ),
                "performance_status": "COMPLETE" if all_complete else "INCOMPLETE",
                "mean_full_window_net_return": (
                    sum(row["full_window_net_return"] for row in complete) / len(complete)
                    if all_complete and complete
                    else None
                ),
                "mean_terminal_score": (
                    sum(row["terminal_metrics"]["terminal_score"] for row in complete)
                    / len(complete)
                    if all_complete and complete
                    else None
                ),
                "transaction_cost": (
                    math.fsum(row["terminal_metrics"]["transaction_cost"] for row in complete)
                    if all_complete
                    else None
                ),
            }
        )
    return {
        "full_episode_evaluation": True,
        "start_mode": START_MODE,
        "planned": len(rows),
        "complete": sum(row["status"] == "COMPLETE" for row in rows),
        "performance_status": (
            "COMPLETE"
            if rows and all(row["status"] == "COMPLETE" for row in rows)
            else "INCOMPLETE"
        ),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "requests": sum(row["requests"] for row in rows),
        "groups": details,
    }


async def execute(
    llm: Any,
    tokenizer: Any,
    sampling_class: Any,
    manifest: dict[str, Any],
    model_key: str,
    output: Path,
    *,
    reset: Any = reset_case,
) -> dict[str, Any]:
    rows = grid(manifest, model_key)
    cases = {case["id"]: case for case in manifest["cases"]}
    state = {
        "profile": PROFILE,
        "status": "RUNNING",
        "model_key": model_key,
        "full_episode_evaluation": True,
        "start_mode": START_MODE,
        "grid": rows,
    }
    probe.save(output / "state.json", state)
    seen_request_ids: set[str] = set()
    try:
        with (
            (output / "requests.jsonl").open("x") as evidence,
            (output / "submissions.jsonl").open("x") as submitted,
            (output / "raw.jsonl").open("x") as raw_handle,
        ):
            for start in range(0, len(rows), MAX_BATCH):
                members, runtime = rows[start : start + MAX_BATCH], {}
                for row in members:
                    env, obs = await reset(cases[row["case_id"]])
                    row.update(
                        status="ACTIVE",
                        initial_ledger=probe.ledger_state(env),
                        initial_state=parse_state(obs),
                    )
                    runtime[row["id"]] = [env, obs]
                while any(row["status"] == "ACTIVE" for row in members):
                    submissions = []
                    for row in members:
                        if row["status"] != "ACTIVE":
                            continue
                        if row["requests"] >= REQUEST_CAP:
                            _abort(row, "REQUEST_CAP")
                            continue
                        env, obs = runtime[row["id"]]
                        prompt = render_prompt(tokenizer, env, obs, action_protocol=probe.PROTOCOL)
                        ids = list(tokenizer.encode(prompt, add_special_tokens=False))
                        if len(ids) > PROMPT_CAP:
                            _abort(row, "PROMPT_CAP")
                            evidence.write(
                                probe.canonical(
                                    {
                                        "trajectory_id": row["id"],
                                        "request_index": row["requests"],
                                        "failure": "PROMPT_CAP",
                                        "submitted": False,
                                        "prompt": prompt,
                                        "prompt_token_ids": ids,
                                    }
                                )
                                + "\n"
                            )
                            evidence.flush()
                            continue
                        active = (
                            row["recovery_events"][-1]
                            if row["recovery_events"]
                            and row["recovery_events"][-1]["status"] == "ACTIVE"
                            else None
                        )
                        if active is not None and active["followup_requests"] >= RECOVERY_CAP:
                            raise ValueError("Recovery request escaped the four-followup cap")
                        params = probe.sampling_parameters(row, 0)
                        request = {
                            "trajectory_id": row["id"],
                            "request_index": row["requests"],
                            "base_seed": row["base_seed"],
                            "rollout_identity": row["rollout_identity"],
                            "seed_env_step": row["requests"],
                            "decision_session_index": env.current_session_index,
                            "prompt": prompt,
                            "prompt_token_ids": ids,
                            "sampling_params": params,
                            "ledger_before": probe.ledger_state(env),
                            "recovery_event_index": active["event_index"] if active else None,
                            "recovery_followup_index": (
                                active["followup_requests"] + 1 if active else None
                            ),
                        }
                        submissions.append((row, request, sampling_class(**params)))
                    if not submissions:
                        break
                    submitted.write(
                        probe.canonical(
                            {
                                "requests": [item[1] for item in submissions],
                            }
                        )
                        + "\n"
                    )
                    submitted.flush()
                    for row, request, _ in submissions:
                        row["requests"] += 1
                        if request["recovery_event_index"] is not None:
                            row["recovery_events"][-1]["followup_requests"] += 1
                    probe.save(output / "state.json", state)
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
                    raw_handle.write(probe.canonical({"outputs": raw}) + "\n")
                    raw_handle.flush()
                    request_ids = [item["request_id"] for item in raw]
                    if (
                        len(outputs) != len(submissions)
                        or len(set(request_ids)) != len(raw)
                        or seen_request_ids.intersection(request_ids)
                    ):
                        raise ValueError("Generation response count or request IDs differ")
                    seen_request_ids.update(request_ids)
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
                            _abort(row, "RESPONSE_INCOMPLETE")
                            record["environment_applied"] = False
                        else:
                            before = env.current_session_index
                            obs, reward, done, info = await env.step(
                                Action(text=text, token_ids=tokens)
                            )
                            runtime[row["id"]][1] = obs
                            row["invalid_requests"] += int(bool(info.get("invalid_action")))
                            record.update(
                                environment_applied=True,
                                feedback=info,
                                trace=asdict(env.credit_step_trace[-1]),
                                reward=reward,
                                ledger_after=probe.ledger_state(env),
                                observation_messages=obs.messages,
                                terminal_reached=done,
                            )
                            if (
                                info.get("invalid_action")
                                and record["ledger_after"] != (request["ledger_before"])
                            ):
                                raise ValueError("Rejected model action changed ledger or session")
                            _observe_step(
                                row,
                                env,
                                obs,
                                info,
                                before=before,
                                request_index=request["request_index"],
                                done=done,
                            )
                        evidence.write(probe.canonical(record) + "\n")
                        evidence.flush()
                    probe.save(output / "state.json", state)
                for row in members:
                    env, _ = runtime[row["id"]]
                    row["episode_trace"] = [asdict(step) for step in env.credit_step_trace]
                    row["final_ledger"] = probe.ledger_state(env)
                    row["interaction_trace"] = env.interaction_trace
        state.update(status="OBSERVED", summary=summary(rows))
        probe.save(output / "result.json", state)
        return state
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        probe.save(output / "state.json", state)


def worker(
    directory: Path,
    model_key: str,
    model_path: Path | None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    manifest = verify(directory, expected_sha256)
    contract = manifest["models"][model_key]
    model = model_path or (Path(contract["model_path"]) if contract["model_path"] else None)
    if model is None:
        raise ValueError("--model is required for an actor restored from its R5 archive")
    model = model.resolve()
    probe.validate_model(model, contract)
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

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    engine = probe.engine_settings(model)
    probe.save(
        output / "config.json",
        {
            "engine": engine,
            "model_key": model_key,
            "manifest_sha256": probe.sha(directory / "manifest.json"),
            "compiler": {"CC": compiler, "version": compiler_version},
            "action_protocol": probe.PROTOCOL,
            "enable_thinking": False,
            "profile": PROFILE,
            "observation_profile": OBSERVATION_PROFILE,
            "start_mode": START_MODE,
        },
    )

    def interrupted(signum: int, frame: Any) -> None:
        raise InterruptedError(f"Full-reset evaluation interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        llm = LLM(**engine)
        result = asyncio.run(execute(llm, tokenizer, SamplingParams, manifest, model_key, output))
        verify(directory, expected_sha256)
        return result["summary"]
    except BaseException as error:
        if not (output / "state.json").exists():
            probe.save(
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
    for key in MODEL_KEYS:
        planned = grid(manifest, key)
        path = directory / "workers" / key / "result.json"
        if not path.exists():
            rows.extend(planned)
            workers[key] = "UNOBSERVED"
            continue
        observed = probe.read(path)
        if (
            observed.get("status") != "OBSERVED"
            or observed.get("profile") != PROFILE
            or observed.get("start_mode") != START_MODE
            or observed.get("model_key") != key
            or observed.get("full_episode_evaluation") is not True
            or len(observed["grid"]) != len(planned)
        ):
            raise ValueError("Observed full-reset worker protocol or population differs")
        for expected, actual in zip(planned, observed["grid"], strict=True):
            immutable = (
                "id",
                "model_identity",
                "base_seed",
                "case_id",
                "case_kind",
                "profile",
                "start_mode",
                "sampling",
                "rollout_identity",
                "request_limit",
            )
            if any(actual.get(name) != expected[name] for name in immutable):
                raise ValueError("Observed full-reset identity differs")
            if (
                actual.get("status") not in {"COMPLETE", "INCOMPLETE"}
                or type(actual.get("requests")) is not int
                or not 0 <= actual["requests"] <= REQUEST_CAP
                or any(
                    not 0 <= event["followup_requests"] <= RECOVERY_CAP
                    or event["status"] not in {"RECOVERED", "INCOMPLETE"}
                    for event in actual["recovery_events"]
                )
            ):
                raise ValueError("Observed full-reset trajectory escaped its budgets")
            if actual["status"] == "INCOMPLETE" and any(
                name in actual for name in ("terminal_metrics", "full_window_net_return")
            ):
                raise ValueError("Incomplete full-reset trajectory contains terminal metrics")
        rows.extend(observed["grid"])
        workers[key] = "OBSERVED"
    result = {
        "profile": PROFILE,
        "start_mode": START_MODE,
        "status": "OBSERVED" if all(v == "OBSERVED" for v in workers.values()) else "INCOMPLETE",
        "manifest_sha256": probe.sha(directory / "manifest.json"),
        "workers": workers,
        "summary": summary(rows),
        "grid": rows,
        "limitations": manifest["limitations"],
    }
    probe.save(directory / "readout.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare", "verify", "worker", "readout"):
        action = actions.add_parser(name)
        action.add_argument("--directory", type=Path, required=True)
        if name == "prepare":
            action.add_argument("--recovery-directory", type=Path, required=True)
        if name in {"verify", "worker"}:
            action.add_argument("--expected-manifest-sha256")
        if name == "worker":
            action.add_argument("--model-key", choices=MODEL_KEYS, required=True)
            action.add_argument("--model", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.directory, args.recovery_directory)
    elif args.action == "verify":
        manifest = verify(args.directory, args.expected_manifest_sha256)
        result = {"status": "VERIFIED", "budget": manifest["budget"]}
    elif args.action == "worker":
        result = worker(args.directory, args.model_key, args.model, args.expected_manifest_sha256)
    else:
        result = readout(args.directory)
    print(probe.canonical(result))


if __name__ == "__main__":
    main()

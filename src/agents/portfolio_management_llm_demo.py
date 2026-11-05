"""Train-only zero-shot choice among four visible portfolio-management templates.

The common SFT model has not been trained for this interface. Constrained action
IDs demonstrate inference and the new environment, not a learned improvement.
GPU imports and execution happen only when this module's CLI is invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import math
import os
import time
from collections import Counter
from pathlib import Path

from src.agents.same_prefix_online_collect import _helper, canonical, require, write

PROFILE = "portfolio_management_zero_shot_demo_v1"
ENV_PROFILE = "portfolio_management_residual_v1"
ACTION_IDS = ("hold", "momentum", "reversal", "defensive")
DECISION_SESSIONS = (1, 6, 11, 16)
MAX_MODEL_LEN, MAX_RESPONSE, MAX_BATCH = 8192, 32, 32

SYSTEM_PROMPT = """You manage an existing stock portfolio at four decision dates.
Choose exactly one of the four action IDs: hold, momentum, reversal, defensive.
The observation contains only information available at the current decision date,
including the current portfolio, historical return features, constraints, and the
exact target portfolio for each action. The environment executes the chosen
template at the next trading-session open, applies transaction costs, and holds
it until the next decision. Hold keeps the current stock shares. Momentum and
reversal move toward fully invested templates based on the past 20-day return
ranking: momentum gives the top five stocks 7% each, the bottom five 3% each,
and the other ten 5% each; reversal reverses these rankings. Each such move is
capped at 0.20 in total absolute portfolio-weight change. Defensive reduces
stock exposure by 20 percentage points, with zero as the lower bound. Momentum
and reversal can gradually reinvest cash after defensive actions. You cannot
change the templates or individual orders.
Choose a prudent portfolio using the visible evidence, expected return, risk,
and trading costs. Future returns are unknown. Output only the action ID with no
explanation, JSON, tool call, or other text."""


def request_seed(seed: int, episode_id: str, session_index: int) -> int:
    payload = canonical([PROFILE, seed, episode_id, session_index])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def choice_sampling_factory(sampling_params_module, seed: int):
    """Select the installed vLLM choice API; never fall back to free decoding."""
    sampling_class = sampling_params_module.SamplingParams
    fields = set(inspect.signature(sampling_class).parameters)
    fields.update(getattr(sampling_class, "__struct_fields__", ()))
    if (
        hasattr(sampling_params_module, "StructuredOutputsParams")
        and "structured_outputs" in fields
    ):
        choice_class = sampling_params_module.StructuredOutputsParams
        keyword = "structured_outputs"
    elif hasattr(sampling_params_module, "GuidedDecodingParams") and "guided_decoding" in fields:
        choice_class = sampling_params_module.GuidedDecodingParams
        keyword = "guided_decoding"
    else:
        raise RuntimeError("installed vLLM has no supported strict choice-decoding API")

    def factory(rs):
        return sampling_class(
            temperature=0.0,
            max_tokens=MAX_RESPONSE,
            seed=rs,
            **{keyword: choice_class(choice=list(ACTION_IDS))},
        )

    # Validate the actual installed constructor before loading any GPU weights.
    probe = factory(seed)
    require(
        getattr(probe, keyword).choice == list(ACTION_IDS), "strict four-choice sampling"
    )
    return factory, dict(
        api=keyword,
        choice_parameter_class=choice_class.__name__,
        choices=list(ACTION_IDS),
        module_path=sampling_params_module.__file__,
    )


def prompt_messages(visible: dict) -> list[dict]:
    """Use only the environment's point-in-time public observation."""
    require(visible.get("profile") == ENV_PROFILE, "portfolio-management visible profile")
    return [
        dict(role="system", content=SYSTEM_PROMPT),
        dict(
            role="user",
            content="Current visible portfolio-management observation:\n" + canonical(visible),
        ),
    ]


def decode_choice(result, prompt_ids: list[int]) -> dict:
    """Keep malformed, capped, or unfinished generations as failed requests."""
    record = dict(
        status="GENERATION_ERROR",
        failure_reason=None,
        response_text=None,
        action_id=None,
        response_token_ids=[],
        returned_finished=getattr(result, "finished", None),
        finish_reason=None,
        stop_reason=None,
    )
    if list(getattr(result, "prompt_token_ids", [])) != prompt_ids:
        record["failure_reason"] = "PROMPT_TOKEN_MISMATCH"
        return record
    outputs = getattr(result, "outputs", [])
    if len(outputs) != 1:
        record["failure_reason"] = "EXPECTED_SINGLE_OUTPUT"
        return record
    answer = outputs[0]
    tokens = list(answer.token_ids)
    text = answer.text
    action_id = text.strip()
    record.update(
        response_text=text,
        response_token_ids=tokens,
        finish_reason=answer.finish_reason,
        stop_reason=answer.stop_reason,
    )
    if not result.finished or answer.finish_reason != "stop":
        record["failure_reason"] = "UNFINISHED_OR_TOKEN_CAPPED"
    elif not tokens or len(tokens) > MAX_RESPONSE:
        record["failure_reason"] = "INVALID_RESPONSE_TOKEN_COUNT"
    elif action_id not in ACTION_IDS:
        record["failure_reason"] = "INVALID_ACTION_ID"
    else:
        record.update(status="COMPLETE", action_id=action_id)
    return record


def aggregate_metrics(results: list[dict]) -> dict:
    complete = [row for row in results if row["status"] == "COMPLETE"]
    numeric_keys = {
        key
        for row in complete
        for key, value in row.get("metrics", {}).items()
        if type(value) in (int, float) and math.isfinite(value)
    }
    means = {}
    for key in sorted(numeric_keys):
        values = [
            row["metrics"][key]
            for row in complete
            if type(row.get("metrics", {}).get(key)) in (int, float)
            and math.isfinite(row["metrics"][key])
        ]
        means[key] = math.fsum(values) / len(values) if values else None
    return dict(
        complete_trajectories=len(complete),
        failed_trajectories=len(results) - len(complete),
        metric_means_over_complete=means,
        failed_trajectories_imputed=False,
    )


def run_waves(
    environments: dict,
    llm,
    tokenizer,
    sampling_factory,
    out: Path,
    seed: int,
    *,
    strategy: str = "llm_zero_shot",
    zero_shot: bool = True,
) -> dict:
    """Execute one batch per decision wave, without retries or substituted actions."""
    planned_model_requests = len(environments) * len(DECISION_SESSIONS)
    states = {
        episode_id: dict(env=env, failure=None, action_ids=[], decision_sessions=[])
        for episode_id, env in environments.items()
    }
    counts, action_counts = Counter(), Counter()
    logs = {
        name: (out / name).open("x")
        for name in ("requests.jsonl", "transitions.jsonl", "trajectory_results.jsonl")
    }

    def append(name: str, value: dict) -> None:
        logs[name].write(canonical(value) + "\n")
        logs[name].flush()

    def progress(wave: int) -> None:
        temporary = out / "collection_state.tmp"
        temporary.write_text(
            canonical(
                dict(
                    status="RUNNING",
                    completed_decision_waves=wave,
                    planned_decision_waves=4,
                    planned_model_requests=planned_model_requests,
                    failed_trajectories=sum(s["failure"] is not None for s in states.values()),
                    **counts,
                )
            )
            + "\n"
        )
        temporary.replace(out / "collection_state.json")

    try:
        progress(0)
        for wave, session_index in enumerate(DECISION_SESSIONS):
            pending = []
            for episode_id, state in states.items():
                if state["failure"] is not None:
                    continue
                env = state["env"]
                if env.done or env.session_index != session_index:
                    state["failure"] = dict(
                        status="ENVIRONMENT_ERROR",
                        failure_reason="UNEXPECTED_DECISION_SESSION",
                        decision_wave=wave,
                        expected_session_index=session_index,
                        observed_session_index=env.session_index,
                    )
                    continue
                visible = env.visible()
                messages = prompt_messages(visible)
                ids = list(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
                record = dict(
                    episode_id=episode_id,
                    decision_wave=wave,
                    decision_session_index=session_index,
                    request_seed=request_seed(seed, episode_id, session_index),
                    visible=visible,
                    messages=messages,
                    prompt_text=tokenizer.decode(ids, skip_special_tokens=False),
                    prompt_token_ids=ids,
                    prompt_tokens=len(ids),
                    temperature=0.0,
                    max_tokens=MAX_RESPONSE,
                    structured_output_choices=list(ACTION_IDS),
                    zero_shot=zero_shot,
                    constrained_decoding=True,
                    model_learning_claim=False,
                )
                if len(ids) + MAX_RESPONSE > MAX_MODEL_LEN:
                    record.update(
                        status="PROMPT_LIMIT",
                        failure_reason="PROMPT_EXCEEDS_CONTEXT_BUDGET",
                        response_text=None,
                        response_token_ids=[],
                        action_id=None,
                        model_request_submitted=False,
                    )
                    state["failure"] = dict(
                        status=record["status"],
                        failure_reason=record["failure_reason"],
                        decision_wave=wave,
                    )
                    counts["failed_requests"] += 1
                    append("requests.jsonl", record)
                    continue
                pending.append((episode_id, state, record))

            for start in range(0, len(pending), MAX_BATCH):
                chunk = pending[start : start + MAX_BATCH]
                counts["model_requests"] += len(chunk)
                counts["prompt_tokens"] += sum(r["prompt_tokens"] for _, _, r in chunk)
                try:
                    outputs = llm.generate(
                        [{"prompt_token_ids": r["prompt_token_ids"]} for _, _, r in chunk],
                        [sampling_factory(r["request_seed"]) for _, _, r in chunk],
                        use_tqdm=False,
                    )
                    require(len(outputs) == len(chunk), "one output per submitted model request")
                    decoded = [
                        decode_choice(result, record["prompt_token_ids"])
                        for (_, _, record), result in zip(chunk, outputs, strict=True)
                    ]
                except Exception as error:
                    decoded = [
                        dict(
                            status="MODEL_ERROR",
                            failure_reason=f"{type(error).__name__}: {error}",
                            response_text=None,
                            response_token_ids=[],
                            action_id=None,
                        )
                        for _ in chunk
                    ]
                for (episode_id, state, record), answer in zip(chunk, decoded, strict=True):
                    record.update(answer, model_request_submitted=True)
                    record["response_tokens"] = len(record["response_token_ids"])
                    counts["response_tokens"] += record["response_tokens"]
                    append("requests.jsonl", record)
                    if record["status"] != "COMPLETE":
                        counts["failed_requests"] += 1
                        state["failure"] = dict(
                            status=record["status"],
                            failure_reason=record["failure_reason"],
                            decision_wave=wave,
                        )
                        continue
                    counts["complete_requests"] += 1
                    action_id = record["action_id"]
                    action_counts[action_id] += 1
                    state["action_ids"].append(action_id)
                    state["decision_sessions"].append(session_index)
                    try:
                        transition = state["env"].step(action_id)
                        append(
                            "transitions.jsonl",
                            dict(
                                episode_id=episode_id,
                                decision_wave=wave,
                                action_id=action_id,
                                status="COMPLETE",
                                transition=transition,
                            ),
                        )
                    except Exception as error:
                        state["failure"] = dict(
                            status="ENVIRONMENT_ERROR",
                            failure_reason=f"{type(error).__name__}: {error}",
                            decision_wave=wave,
                        )
                        append(
                            "transitions.jsonl",
                            dict(episode_id=episode_id, action_id=action_id, **state["failure"]),
                        )
            progress(wave + 1)
            print(
                f"Portfolio choice wave {wave + 1}/4 complete: "
                f"{counts['model_requests']}/{planned_model_requests} model requests, "
                f"{sum(s['failure'] is not None for s in states.values())} failed trajectories",
                flush=True,
            )

        results = []
        for episode_id, state in states.items():
            env = state["env"]
            if state["failure"] is None and (not env.done or len(state["action_ids"]) != 4):
                state["failure"] = dict(
                    status="ENVIRONMENT_ERROR", failure_reason="INCOMPLETE_FINAL_TRAJECTORY"
                )
            if state["failure"] is None:
                try:
                    result = env.result()
                except Exception as error:
                    state["failure"] = dict(
                        status="ENVIRONMENT_ERROR",
                        failure_reason=f"{type(error).__name__}: {error}",
                    )
            if state["failure"] is not None:
                # The shared environment deliberately refuses terminal metrics
                # before session20. Keep the partial path without inventing Y.
                daily_states = list(getattr(env, "daily_states", []))
                result = dict(
                    episode_id=episode_id,
                    status=state["failure"]["status"],
                    metrics={},
                    terminal_metrics_available=False,
                    decisions=list(getattr(env, "decisions", [])),
                    daily_states=daily_states,
                    equity_curve=[row["equity"] for row in daily_states],
                )
            result.update(
                episode_id=episode_id,
                strategy=strategy,
                zero_shot=zero_shot,
                constrained_decoding=True,
                model_learning_claim=False,
                action_ids=state["action_ids"],
                decision_session_indices=state["decision_sessions"],
            )
            if state["failure"] is not None:
                result.update(status=state["failure"]["status"], failure=state["failure"])
            results.append(result)
            append("trajectory_results.jsonl", result)
        return dict(
            status=(
                "COMPLETE" if all(r["status"] == "COMPLETE" for r in results)
                else "COMPLETE_WITH_FAILURES"
            ),
            planned_episodes=len(environments),
            planned_model_requests=planned_model_requests,
            **{key: counts[key] for key in (
                "model_requests", "prompt_tokens", "response_tokens", "complete_requests",
                "failed_requests",
            )},
            action_counts={action_id: action_counts[action_id] for action_id in ACTION_IDS},
            results=results,
            **aggregate_metrics(results),
        )
    finally:
        for stream in logs.values():
            stream.close()


def run(args) -> dict:
    from src.agents.portfolio_management import PortfolioManagement, load_train_markets

    out, model = args.output.resolve(), args.model.resolve()
    prepared = args.prepared.resolve()
    require(args.seed == 43, "fixed demo seed43")
    require(model.is_dir() and prepared.is_file(), "existing common SFT and Train parquet")
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / "collection_started.json").exists(), "fresh demo output required")
    started = time.monotonic()
    write(
        out / "collection_started.json",
        dict(
            profile=PROFILE,
            model_path=str(model),
            prepared_path=str(prepared),
            seed=args.seed,
            planned_episodes=32,
            planned_model_requests=128,
            decision_session_indices=list(DECISION_SESSIONS),
            zero_shot=True,
            constrained_decoding=True,
            llm_optimizer_updates=0,
            model_learning_claim=False,
            training_split="train",
            valid_or_final_oos_access=False,
        ),
    )
    markets = load_train_markets(prepared, out)
    require(len(markets) == 32, "existing broad32 Train windows")
    require(
        all(eid.startswith("formal-train-") for eid in markets), "Train-only market episodes"
    )
    environments = {
        episode_id: PortfolioManagement(market, initialize_equal_weight=True)
        for episode_id, market in markets.items()
    }
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        USE_TORCH="1",
        USE_TF="0",
        USE_FLAX="0",
        VLLM_USE_V1="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    from vllm import LLM, sampling_params

    sampling_factory, choice_settings = choice_sampling_factory(sampling_params, args.seed)
    write(out / "constrained_decoding.json", choice_settings)

    helper = _helper("portfolio_management_native_worker", "m2_continuation_gpu_worker.py")
    compiler, version = helper.resolve_compiler()
    os.environ["CC"] = compiler
    settings = helper.engine_settings(str(model))
    settings.update(max_model_len=MAX_MODEL_LEN, seed=args.seed, gpu_memory_utilization=0.35)
    settings["override_generation_config"].update(temperature=0.0, max_new_tokens=MAX_RESPONSE)
    write(out / "engine_settings.json", settings)
    write(out / "compiler.json", dict(CC=compiler, version=version))
    print("Loading common SFT for constrained zero-shot portfolio choices", str(model), flush=True)
    llm = LLM(**settings)
    tokenizer = llm.get_tokenizer()

    summary = run_waves(environments, llm, tokenizer, sampling_factory, out, args.seed)
    summary.update(
        profile=PROFILE,
        environment_profile=ENV_PROFILE,
        strategy="llm_zero_shot",
        seed=args.seed,
        model_path=str(model),
        prepared_path=str(prepared),
        episode_ids=list(environments),
        planned_episodes=32,
        planned_model_requests=128,
        action_ids=list(ACTION_IDS),
        decision_session_indices=list(DECISION_SESSIONS),
        trajectory_results_path=str(out / "trajectory_results.jsonl"),
        elapsed_seconds=time.monotonic() - started,
        zero_shot=True,
        constrained_decoding=True,
        constrained_decoding_api=choice_settings["api"],
        zero_shot_interface_unadapted=True,
        llm_optimizer_updates=0,
        model_learning_claim=False,
        training_split="train",
        valid_or_final_oos_access=False,
        future_information_in_model_prompt=False,
        automatic_hold_or_retries=False,
    )
    write(out / "summary.json", summary)
    temporary = out / "collection_state.tmp"
    temporary.write_text(
        canonical({key: value for key, value in summary.items() if key != "results"}) + "\n"
    )
    temporary.replace(out / "collection_state.json")
    print(canonical({key: value for key, value in summary.items() if key != "results"}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

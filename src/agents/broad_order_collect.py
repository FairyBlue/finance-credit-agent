"""One broad Train-only batch of visible-order interventions for a frozen scorer."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from src.agents.order_ranker import FEATURE_NAMES, candidate_orders
from src.agents.same_prefix_online_collect import (
    MAX_ATTEMPTS,
    MAX_BATCH,
    MAX_PROMPT,
    PROTOCOL,
    ROOT,
    _helper,
    _install_adapter,
    canonical,
    object_sha,
    require,
    write,
)

SESSIONS = (0, 6, 12, 18)


def candidate_estimates(anchor, rows):
    estimates = []
    for candidate in anchor.get("candidates", []):
        group = sorted(
            [row for row in rows if row["candidate_id"] == candidate["candidate_id"]],
            key=lambda row: row["continuation_replicate"],
        )
        complete = len(group) == 4 and [r["continuation_replicate"] for r in group] == list(
            range(4)
        )
        complete = complete and all(r["status"] == "COMPLETE" for r in group)
        estimates.append(
            dict(
                anchor_id=anchor["anchor_id"],
                episode_id=anchor["episode_id"],
                session_index=anchor["session_index"],
                candidate_id=candidate["candidate_id"],
                action_key=candidate["action_key"],
                family=candidate["family"],
                symbol=candidate["symbol"],
                target_weight=candidate["target_weight"],
                features=candidate["features"],
                status="COMPLETE" if complete else "NA",
                planned_continuations=4,
                complete_continuations=sum(r["status"] == "COMPLETE" for r in group),
                mean_Q=math.fsum(r["Q"] for r in group) / 4 if complete else None,
                Q_draws=[r["Q"] for r in group],
                mean_raw_net_return=math.fsum(
                    r["terminal_metrics"]["raw_net_return"] for r in group
                )
                / 4
                if complete
                else None,
            )
        )
    supported = [row for row in estimates if row["status"] == "COMPLETE"]
    center = (
        math.fsum(row["mean_Q"] for row in supported) / len(supported)
        if len(supported) >= 2
        else None
    )
    for row in estimates:
        row["within_state_Q_center"] = center
        row["target_Q_centered"] = (
            row["mean_Q"] - center if row["mean_Q"] is not None and center is not None else None
        )
        row["training_supported"] = row["target_Q_centered"] is not None
    return estimates


async def collect(args):
    out, model, source = args.output.resolve(), args.model.resolve(), args.source.resolve()
    config = json.loads(args.experiment_config.read_text())
    require(
        config["profile"] == "broad_rloo_order_main_v1" and config["seed"] == args.seed == 43,
        "approved broad Train profile",
    )
    episode_ids = config["train_episode_ids"]
    require(
        len(episode_ids) == 32 and all(eid.startswith("formal-train-") for eid in episode_ids),
        "fixed32 Train-only episodes",
    )
    prepared_path = (
        args.prepared.resolve() if args.prepared else Path(config["prepared_train_path"]).resolve()
    )
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / "collection_started.json").exists(), "fresh collection required")
    source_summary = json.loads((source / "collection_summary.json").read_text())
    require(
        source_summary["seed"] == 43 and source_summary["episode_ids"] == episode_ids,
        "fixed seed43 broad Train source",
    )
    require(
        Path(source_summary["model_path"]).resolve() == model, "source must come from frozen model"
    )
    parents = [
        json.loads(line) for line in (source / "trajectory_results.jsonl").read_text().splitlines()
    ]
    parent_map = {
        r["episode_id"]: r
        for r in parents
        if r["role"] == "parent" and r.get("replicate_index") == 0
    }
    require(
        set(parent_map) == set(episode_ids),
        "all thirty-two replica0 parent slots retained",
    )
    write(
        out / "collection_started.json",
        dict(
            model_path=str(model),
            source=str(source),
            seed=args.seed,
            planned_anchors=128,
            maximum_branches=2048,
            parent_generation_reused=True,
            new_parent_requests=0,
            formal_valid_oos_access=False,
        ),
    )
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
    sys.path.insert(0, str(ROOT / "third_party/agent-r1"))
    sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
    import pyarrow.parquet as pq
    from agent_r1.env.base import Action

    os.environ["USE_TORCH"] = "1"
    from vllm import LLM, SamplingParams

    from src.agents.screening_prompt import apply_screening_action_protocol
    from src.env.trading import TradingEnv

    prepared = {}
    for row in pq.read_table(prepared_path).to_pylist():
        eid = row["extra_info"]["episode_id"]
        if eid in episode_ids:
            require(row["extra_info"]["source_split"] == "train", "Train-only source")
            kwargs = json.loads(row["env_kwargs"])
            # Request tokens, actual info and credit traces are recorded below.
            # Avoid duplicating every before/after observation in env diagnostics.
            kwargs.update(capture_interactions=False, reward_mode="finance_trajectory")
            prepared[eid] = dict(kwargs=kwargs, prompt=row["prompt"])
    require(set(prepared) == set(episode_ids), "all selected Train episodes prepared")
    _install_adapter(prepared, out)
    helper = _helper("broad_order_native_worker", "m2_continuation_gpu_worker.py")
    compiler, _ = helper.resolve_compiler()
    os.environ["CC"] = compiler
    settings = helper.engine_settings(str(model))
    settings["seed"] = args.seed
    write(out / "engine_settings.json", settings)
    print("Loading frozen RLOO vLLM engine", str(model), flush=True)
    llm = LLM(**settings)
    tokenizer = llm.get_tokenizer()
    logs = {
        name: (out / name).open("x")
        for name in (
            "anchors.jsonl",
            "branch_results.jsonl",
            "requests.jsonl",
            "candidate_rows.jsonl",
            "anchor_readout.jsonl",
        )
    }
    counts, results, anchors = Counter(), [], []

    def append(name, row):
        logs[name].write(canonical(row) + "\n")
        logs[name].flush()

    def progress(phase):
        p = out / "collection_state.tmp"
        p.write_text(
            canonical(
                dict(
                    status="RUNNING",
                    phase=phase,
                    anchors=len(anchors),
                    branches_finished=len(results),
                    **counts,
                )
            )
            + "\n"
        )
        p.replace(out / "collection_state.json")

    def prompt(state):
        return tokenizer.apply_chat_template(
            apply_screening_action_protocol(state["obs"].messages, PROTOCOL),
            tools=state["env"].tool_schemas,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    async def replay(anchor, key, candidate=None, rep=None):
        data = prepared[anchor["episode_id"]]
        env = TradingEnv(**data["kwargs"])
        state = dict(
            env=env,
            obs=env.reset(raw_prompt=data["prompt"]),
            timesteps=[],
            attempt=0,
            trajectory_key=key,
            anchor_id=anchor["anchor_id"],
            episode_id=anchor["episode_id"],
            candidate_id=candidate,
            continuation_replicate=rep,
        )
        parent = parent_map[anchor["episode_id"]]
        for step in parent["timesteps"][: anchor["parent_request_index"]]:
            state["obs"], _, _, info = await env.step(
                Action(text=step["response_text"], token_ids=step["response_token_ids"])
            )
            require(json.loads(canonical(info)) == step["info"], "source prefix replay identity")
        require(env.current_session_index == anchor["session_index"], "original session identity")
        require(
            prompt(state)
            == parent["timesteps"][anchor["parent_request_index"]]["prompt_token_ids"],
            "original full prompt identity",
        )
        return state

    async def apply(state, response):
        env = state["env"]
        before = env.current_session_index
        state["obs"], reward, _, info = await env.step(
            Action(text=response["response_text"], token_ids=response["response_token_ids"])
        )
        after = env.current_session_index
        invalid = bool(info.get("invalid_action"))
        require(
            after in (before, before + 1) and not (invalid and after != before),
            "single session transition",
        )
        state["timesteps"].append(
            dict(
                prompt_token_ids=response["prompt_token_ids"],
                response_token_ids=response["response_token_ids"],
                response_text=response["response_text"],
                request_seed=response.get("request_seed"),
                decision_session_index=before,
                result_session_index=after,
                advancing=after == before + 1,
                invalid=invalid,
                info=json.loads(canonical(info)),
                trace=asdict(env.credit_step_trace[-1]),
                reward=float(reward),
                forced_head=response.get("forced_head", False),
            )
        )
        if not response.get("forced_head"):
            counts["invalid_requests"] += invalid
        state["attempt"] = 0 if after != before else state["attempt"] + 1
        if env.episode_complete:
            return "COMPLETE"
        return "REQUEST_LIMIT" if state["attempt"] >= MAX_ATTEMPTS else None

    def finish(state, status):
        metrics = state["env"].screening_terminal_metrics() if status == "COMPLETE" else None
        y = metrics["terminal_score"] if metrics else None
        require(y is None or math.isfinite(y), "finite terminal Y")
        invalids = sum(s["invalid"] for s in state["timesteps"])
        row = {
            k: state[k]
            for k in (
                "trajectory_key",
                "anchor_id",
                "episode_id",
                "candidate_id",
                "continuation_replicate",
            )
        }
        row.update(
            status=status,
            Y=y,
            terminal_financial_Y=y,
            terminal_metrics=metrics,
            Q=y - 0.05 * invalids if y is not None else None,
            remaining_invalid_count=invalids,
            last_session_index=state["env"].current_session_index,
            timesteps=state["timesteps"],
            forced_head_requests=1,
            new_model_requests=len(state["timesteps"]) - 1,
        )
        results.append(row)
        append("branch_results.jsonl", row)
        counts["branches_complete" if status == "COMPLETE" else "branches_failed"] += 1
        print("FINISHED", row["trajectory_key"], status, y, flush=True)

    try:
        active = []
        for eid in episode_ids:
            parent = parent_map[eid]
            for session in SESSIONS:
                index = next(
                    (
                        i
                        for i, step in enumerate(parent["timesteps"])
                        if step["decision_session_index"] == session
                    ),
                    None,
                )
                if index is None:
                    anchor = dict(
                        anchor_id=f"{parent['trajectory_key']}/j{session}/missing",
                        episode_id=eid,
                        session_index=session,
                        parent_trajectory_key=parent["trajectory_key"],
                        parent_request_index=None,
                        status="MISSING_SOURCE_STATE",
                        candidates=[],
                    )
                    anchors.append(anchor)
                    append("anchors.jsonl", anchor)
                    counts["missing_source_states"] += 1
                    continue
                anchor = dict(
                    anchor_id=f"{parent['trajectory_key']}/j{session}/request{index}",
                    episode_id=eid,
                    session_index=session,
                    parent_trajectory_key=parent["trajectory_key"],
                    parent_request_index=index,
                )
                state = await replay(anchor, "snapshot")
                visible = json.loads(state["env"]._state_text())
                first = parent["timesteps"][index]
                candidates, proposal_legal, reason = candidate_orders(
                    visible, first["response_text"], seed=args.seed, key=anchor["anchor_id"]
                )
                original = candidates[0] if proposal_legal else None
                if original is not None:
                    original["response_token_ids"] = first["response_token_ids"]
                for cid, candidate in enumerate(candidates):
                    candidate["candidate_id"] = cid
                    if "response_token_ids" not in candidate:
                        candidate["response_token_ids"] = tokenizer.encode(
                            candidate["response_text"] + tokenizer.eos_token,
                            add_special_tokens=False,
                        )
                    require(
                        candidate["response_token_ids"][-1] == tokenizer.eos_token_id,
                        "complete supervised EOS",
                    )
                anchor.update(
                    prompt_token_ids=prompt(state),
                    status="CAPTURED",
                    visible_state=visible,
                    proposal_legality_reason=reason,
                    original_request_legal=not first["invalid"] and first["advancing"],
                    original_candidate_id=0 if proposal_legal else None,
                    original_response_text=first["response_text"],
                    candidates=candidates,
                    source_prompt_sha256=object_sha(first["prompt_token_ids"]),
                )
                anchors.append(anchor)
                append("anchors.jsonl", anchor)
                for candidate in candidates:
                    for rep in range(4):
                        branch = await replay(
                            anchor,
                            f"counterfactual/{anchor['anchor_id']}/c{candidate['candidate_id']}/r{rep}",
                            candidate["candidate_id"],
                            rep,
                        )
                        response = dict(
                            prompt_token_ids=anchor["prompt_token_ids"],
                            response_token_ids=candidate["response_token_ids"],
                            response_text=candidate["response_text"],
                            forced_head=True,
                        )
                        status = await apply(branch, response)
                        if (
                            branch["timesteps"][0]["invalid"]
                            or not branch["timesteps"][0]["advancing"]
                        ):
                            finish(branch, "INVALID_CONSTRUCTED_HEAD")
                        elif status:
                            finish(branch, status)
                        else:
                            active.append(branch)
                progress("PREPARING_BRANCHES")
        planned = sum(len(a["candidates"]) * 4 for a in anchors)
        require(len(anchors) == 128 and planned <= 2048, "bounded planned branches")
        print("Prepared", len(anchors), "anchors", planned, "branches", flush=True)
        while active:
            for start in range(0, len(active), MAX_BATCH):
                chunk = active[start : start + MAX_BATCH]
                pending = []
                for state in chunk:
                    ids = prompt(state)
                    if len(ids) > MAX_PROMPT:
                        finish(state, "PROMPT_LIMIT")
                        state["finished"] = True
                        continue
                    payload = [
                        "broad-order-main-v1",
                        args.seed,
                        state["anchor_id"],
                        state["continuation_replicate"],
                        state["env"].current_session_index,
                        state["attempt"],
                    ]
                    rs = int(object_sha(payload)[:16], 16) & ((1 << 63) - 1)
                    pending.append((state, ids, rs))
                if not pending:
                    continue
                outputs = llm.generate(
                    [{"prompt_token_ids": ids} for _, ids, _ in pending],
                    [SamplingParams(**helper.SAMPLING, seed=rs) for _, _, rs in pending],
                    use_tqdm=False,
                )
                require(len(outputs) == len(pending), "native batch length")
                for (state, ids, rs), output in zip(pending, outputs, strict=True):
                    require(
                        list(output.prompt_token_ids) == ids and len(output.outputs) == 1,
                        "native exact prompt",
                    )
                    answer = output.outputs[0]
                    tokens = list(answer.token_ids)
                    raw = dict(
                        response_token_ids=tokens,
                        returned_finished=output.finished,
                        finish_reason=answer.finish_reason,
                        stop_reason=answer.stop_reason,
                    )
                    status, reason = helper.generation_integrity(raw)
                    response = dict(
                        **raw,
                        status=status,
                        failure_reason=reason,
                        prompt_token_ids=ids,
                        response_text=tokenizer.decode(tokens, skip_special_tokens=True),
                        request_seed=rs,
                    )
                    append(
                        "requests.jsonl",
                        dict(
                            trajectory_key=state["trajectory_key"],
                            anchor_id=state["anchor_id"],
                            candidate_id=state["candidate_id"],
                            continuation_replicate=state["continuation_replicate"],
                            decision_session_index=state["env"].current_session_index,
                            attempt=state["attempt"],
                            **response,
                        ),
                    )
                    counts["model_requests"] += 1
                    counts["prompt_tokens"] += len(ids)
                    counts["response_tokens"] += len(tokens)
                    if status == "COMPLETE":
                        status = await apply(state, response)
                    if status:
                        finish(state, status)
                        state["finished"] = True
                progress("SAMPLING_BRANCHES")
            active = [state for state in active if not state.get("finished")]
        for anchor in anchors:
            rows = [row for row in results if row["anchor_id"] == anchor["anchor_id"]]
            estimates = candidate_estimates(anchor, rows)
            supported = sum(row["training_supported"] for row in estimates)
            counts["training_supported_candidates"] += supported
            counts["supported_anchors"] += supported >= 2
            for row in estimates:
                append("candidate_rows.jsonl", row)
            append(
                "anchor_readout.jsonl",
                dict(
                    anchor_id=anchor["anchor_id"],
                    episode_id=anchor["episode_id"],
                    session_index=anchor["session_index"],
                    candidate_count=len(estimates),
                    training_supported_candidates=supported,
                    source_state_status=anchor["status"],
                ),
            )
        summary = dict(
            status="COMPLETE"
            if counts["branches_failed"] == 0 and counts["missing_source_states"] == 0
            else "COMPLETE_WITH_MODEL_FAILURES",
            model_path=str(model),
            source_directory=str(source),
            seed=args.seed,
            episode_ids=episode_ids,
            sessions=list(SESSIONS),
            planned_anchors=128,
            captured_anchors=sum(anchor["status"] == "CAPTURED" for anchor in anchors),
            recorded_anchor_slots=len(anchors),
            maximum_branches=2048,
            planned_branches=planned,
            actual_branches=len(results),
            branches_complete=counts["branches_complete"],
            branches_failed=counts["branches_failed"],
            supported_anchors=counts["supported_anchors"],
            training_supported_candidates=counts["training_supported_candidates"],
            missing_source_states=counts["missing_source_states"],
            feature_names=list(FEATURE_NAMES),
            training_split="train",
            experiment_config=str(args.experiment_config.resolve()),
            prepared_train_path=str(prepared_path),
            redundant_environment_interaction_capture=False,
            model_requests=counts["model_requests"],
            prompt_tokens=counts["prompt_tokens"],
            response_tokens=counts["response_tokens"],
            invalid_requests=counts["invalid_requests"],
            parent_requests=0,
            parent_generation_reused=True,
            forced_candidate_heads=planned,
            request_cap_per_session=MAX_ATTEMPTS,
            terminal_session_index=20,
            optimizer_updates=0,
            formal_valid_oos_access=False,
            original_legal_anchors=sum(a.get("original_request_legal", False) for a in anchors),
            return_definition="terminal_financial_Y - 0.05 * remaining_invalid_count",
            estimation_rule=(
                "four complete continuations per candidate, meanQ, state-centred across "
                "supported candidates; all supported candidates train; no winner-only filtering"
            ),
        )
        require(len(results) == planned, "all planned branch slots accounted")
        write(out / "collection_summary.json", summary)
        (out / "collection_state.json").write_text(
            canonical(dict(summary, phase="COMPLETE")) + "\n"
        )
        return summary
    finally:
        for stream in logs.values():
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    args = parser.parse_args()
    require(args.seed == 43, "fixed demo seed43")
    print(canonical(asyncio.run(collect(args))), flush=True)


if __name__ == "__main__":
    main()

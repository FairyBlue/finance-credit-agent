"""Bounded same-prefix legal-order interventions and supervised winner labels."""

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

from src.agents.same_prefix_online_collect import (
    EIGHT_EPISODES,
    MAX_ATTEMPTS,
    MAX_BATCH,
    MAX_PROMPT,
    PREPARED,
    PROTOCOL,
    ROOT,
    _helper,
    _install_adapter,
    canonical,
    object_sha,
    require,
    write,
)

DEFAULT_SOURCE = ROOT / "runs/m2-rloo-eight-window-20261025-v1/seed43/round4/eval"
DEFAULT_MODEL = ROOT / "runs/m2-rloo-eight-window-20261025-v1/seed43/current_actor"
SESSIONS = (3, 7, 11, 15)


def order_key(name, arguments):
    args = dict(arguments)
    if name in ("buy", "sell"):
        args = dict(
            symbol=args["symbol"].strip().upper(), target_weight=float(args["target_weight"])
        )
    return json.dumps(dict(name=name, arguments=args), sort_keys=True, separators=(",", ":"))


def select_candidates(menu, original, visible):
    """Choose economic diversity using visible orders/holdings, never returns."""
    selected = []

    def add(row):
        if row is not None and row["action_key"] not in {x["action_key"] for x in selected}:
            selected.append(row)

    add(original)
    add(next(x for x in menu if x["family"] == "hold"))
    positions = visible["positions"]
    for family in ("sell", "buy"):
        choices = [x for x in menu if x["family"] == family]
        choices.sort(
            key=lambda x: (
                -float(positions.get(x["symbol"], {}).get("weight", 0)) if family == "sell" else 0,
                x["symbol"],
                x["target_weight"],
            )
        )
        add(
            next(
                (x for x in choices if x["action_key"] not in {s["action_key"] for s in selected}),
                None,
            )
        )
        if len(selected) == 4:
            return selected
    for row in menu:
        add(row)
        if len(selected) == 4:
            break
    return selected


def select_winner(anchor, rows):
    """First two draws select; last two must independently beat original."""
    original = anchor.get("original_candidate_id")
    if original is None:
        return dict(status="UNSUPPORTED_ORIGINAL", winner_candidate_id=None)
    by_candidate = {}
    for candidate in anchor["candidates"]:
        group = sorted(
            [r for r in rows if r["candidate_id"] == candidate["candidate_id"]],
            key=lambda r: r["continuation_replicate"],
        )
        complete = (
            len(group) == 4
            and [r["continuation_replicate"] for r in group] == list(range(4))
            and all(r["status"] == "COMPLETE" for r in group)
        )
        if complete:
            q = [r["Q"] for r in group]
            by_candidate[candidate["candidate_id"]] = dict(
                selection_mean=sum(q[:2]) / 2, reference_mean=sum(q[2:]) / 2
            )
    if original not in by_candidate:
        return dict(
            status="INCOMPLETE_ORIGINAL", winner_candidate_id=None, candidate_estimates=by_candidate
        )
    baseline = by_candidate[original]
    choices = [
        cid
        for cid, row in by_candidate.items()
        if cid != original and row["selection_mean"] > baseline["selection_mean"] + 1e-12
    ]
    if not choices:
        return dict(
            status="NO_SELECTION_IMPROVEMENT",
            winner_candidate_id=None,
            candidate_estimates=by_candidate,
        )
    winner = max(choices, key=lambda cid: (by_candidate[cid]["selection_mean"], -cid))
    d1 = by_candidate[winner]["selection_mean"] - baseline["selection_mean"]
    d2 = by_candidate[winner]["reference_mean"] - baseline["reference_mean"]
    return dict(
        status="LABEL" if d2 > 1e-12 else "REFERENCE_NOT_IMPROVED",
        winner_candidate_id=winner if d2 > 1e-12 else None,
        selected_candidate_id=winner,
        selection_delta=d1,
        reference_delta=d2,
        candidate_estimates=by_candidate,
    )


async def collect(args):
    out, model, source = args.output.resolve(), args.model.resolve(), args.source.resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / "collection_started.json").exists(), "fresh collection required")
    source_summary = json.loads((source / "collection_summary.json").read_text())
    require(
        source_summary["seed"] == 43 and source_summary["episode_ids"] == list(EIGHT_EPISODES),
        "fixed seed43 eight-window source",
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
        set(parent_map) == set(EIGHT_EPISODES)
        and all(r["status"] == "COMPLETE" for r in parent_map.values()),
        "eight complete replica0 parents",
    )
    write(
        out / "collection_started.json",
        dict(
            model_path=str(model),
            source=str(source),
            seed=args.seed,
            planned_anchors=32,
            maximum_branches=512,
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

    # Existing pure menu rule, with no historical inventory preparation.
    from m2_same_prefix_menu import menu_from_visible

    os.environ["USE_TORCH"] = "1"
    from vllm import LLM, SamplingParams

    from src.agents.screening_prompt import apply_screening_action_protocol
    from src.env.trading import TradingEnv

    prepared = {}
    for row in pq.read_table(PREPARED).to_pylist():
        eid = row["extra_info"]["episode_id"]
        if eid in EIGHT_EPISODES:
            require(row["extra_info"]["source_split"] == "train", "Train-only source")
            kwargs = json.loads(row["env_kwargs"])
            kwargs.update(capture_interactions=True, reward_mode="finance_trajectory")
            prepared[eid] = dict(kwargs=kwargs, prompt=row["prompt"])
    require(set(prepared) == set(EIGHT_EPISODES), "eight prepared episodes")
    _install_adapter(prepared, out)
    helper = _helper("counterfactual_native_worker", "m2_continuation_gpu_worker.py")
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
            "training_samples.jsonl",
            "replay_samples.jsonl",
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
        # Existing successful native parent requests provide inexpensive replay data.
        for parent in parents:
            if parent["role"] != "parent" or parent["status"] != "COMPLETE":
                continue
            for index, step in enumerate(parent["timesteps"]):
                if step["advancing"] and not step["invalid"]:
                    append(
                        "replay_samples.jsonl",
                        dict(
                            prompt_token_ids=step["prompt_token_ids"],
                            response_token_ids=step["response_token_ids"],
                            response_text=step["response_text"],
                            anchor_id=f"{parent['trajectory_key']}/request{index}",
                            kind="replay",
                            episode_id=parent["episode_id"],
                        ),
                    )
                    counts["replay_samples"] += 1
        active = []
        for eid in EIGHT_EPISODES:
            parent = parent_map[eid]
            for session in SESSIONS:
                index = next(
                    i
                    for i, s in enumerate(parent["timesteps"])
                    if s["decision_session_index"] == session
                )
                anchor = dict(
                    anchor_id=f"{parent['trajectory_key']}/j{session}/request{index}",
                    episode_id=eid,
                    session_index=session,
                    parent_trajectory_key=parent["trajectory_key"],
                    parent_request_index=index,
                )
                state = await replay(anchor, "snapshot")
                visible = json.loads(state["env"]._state_text())
                menu = menu_from_visible(visible)
                first = parent["timesteps"][index]
                _, calls = state["env"].format_wrapper.parse_response(first["response_text"])
                original = None
                if first["advancing"] and not first["invalid"] and len(calls) == 1:
                    call = calls[0]
                    key = order_key(call.name, call.arguments)
                    original = dict(
                        action_key=key,
                        family=call.name,
                        symbol=call.arguments.get("symbol"),
                        target_weight=call.arguments.get("target_weight"),
                        response_text=first["response_text"],
                        response_token_ids=first["response_token_ids"],
                    )
                candidates = select_candidates(menu, original, visible)
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
                    original_request_legal=original is not None,
                    original_candidate_id=0 if original is not None else None,
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
        require(len(anchors) == 32 and planned <= 512, "bounded planned branches")
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
                        "counterfactual-order-v1",
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
            rows = [r for r in results if r["anchor_id"] == anchor["anchor_id"]]
            selection = select_winner(anchor, rows)
            append(
                "anchor_readout.jsonl",
                dict(
                    anchor_id=anchor["anchor_id"],
                    episode_id=anchor["episode_id"],
                    session_index=anchor["session_index"],
                    **selection,
                ),
            )
            counts["selection_" + selection["status"]] += 1
            winner = selection.get("winner_candidate_id")
            if winner is not None:
                candidate = anchor["candidates"][winner]
                append(
                    "training_samples.jsonl",
                    dict(
                        prompt_token_ids=anchor["prompt_token_ids"],
                        response_token_ids=candidate["response_token_ids"],
                        response_text=candidate["response_text"],
                        anchor_id=anchor["anchor_id"],
                        kind="counterfactual_winner",
                        episode_id=anchor["episode_id"],
                        session_index=anchor["session_index"],
                        economic_order_key=candidate["action_key"],
                        selection_delta=selection["selection_delta"],
                        reference_delta=selection["reference_delta"],
                    ),
                )
                counts["training_samples"] += 1
        summary = dict(
            status="COMPLETE" if counts["branches_failed"] == 0 else "COMPLETE_WITH_MODEL_FAILURES",
            model_path=str(model),
            source_directory=str(source),
            seed=args.seed,
            episode_ids=list(EIGHT_EPISODES),
            sessions=list(SESSIONS),
            planned_anchors=32,
            captured_anchors=len(anchors),
            maximum_branches=512,
            planned_branches=planned,
            actual_branches=len(results),
            branches_complete=counts["branches_complete"],
            branches_failed=counts["branches_failed"],
            label_count=counts["training_samples"],
            training_samples=counts["training_samples"],
            replay_samples=counts["replay_samples"],
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
            original_legal_anchors=sum(a["original_request_legal"] for a in anchors),
            selection_counts={
                k.removeprefix("selection_"): v
                for k, v in counts.items()
                if k.startswith("selection_")
            },
            return_definition="terminal_financial_Y - 0.05 * remaining_invalid_count",
            selection_rule=(
                "first2 select highest Q above original; last2 must beat original; "
                "one winner per anchor; no resampling"
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
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    require(args.seed == 43, "fixed demo seed43")
    print(canonical(asyncio.run(collect(args))), flush=True)


if __name__ == "__main__":
    main()

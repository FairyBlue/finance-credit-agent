"""Small current-policy Train collector with same-prefix heads and EST tails.

GPU/model imports are lazy. Each call is intended to run in a fresh subprocess,
so the controller can alternate collection and training without resident engines.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
PREPARED = ROOT / "runs/m2-action-pooling-comparison-20261013-v6/seed42/prepared/train.parquet"
EPISODES = ("formal-train-0001", "formal-train-0009", "formal-train-0065", "formal-train-0073")
EIGHT_EPISODES = tuple(f"formal-train-{i:04d}" for i in (1, 9, 17, 25, 65, 73, 149, 157))
PROTOCOL = "single_tool_portfolio_v1"
MAX_PROMPT, MAX_RESPONSE, MAX_ATTEMPTS = 8192, 4096, 4
MAX_BATCH = 32


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, default=lambda x: x.isoformat())


def object_sha(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(condition, message) -> None:
    if not condition:
        raise ValueError(message)


def write(path, value) -> None:
    with Path(path).open("x") as stream:
        stream.write(canonical(value) + "\n")


def request_seed(seed: int, key: str, session: int, attempt: int, round_index: int) -> int:
    # Parent evaluation uses common random numbers across rounds/checkpoints.
    payload = ["same-prefix-online-demo-v1", seed, key, session, attempt]
    if not key.startswith("parent/"):
        payload.append(round_index)
    return int.from_bytes(hashlib.sha256(canonical(payload).encode()).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _helper(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/diagnostics" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _adapter_configuration(prepared):
    kwargs = next(iter(prepared.values()))["kwargs"]
    fields = (
        "manifest_root",
        "storage_root",
        "symbols",
        "data_version",
        "data_start",
        "data_end",
        "input_manifest_sha256",
        "input_file_count",
        "input_sessions_per_symbol",
        "exchange_calendar",
        "market_timezone",
        "market_close_time",
    )
    expected = {
        key: kwargs.get(key, "XNYS" if key == "exchange_calendar" else None) for key in fields
    }
    for key in ("manifest_root", "storage_root"):
        expected[key] = str(Path(expected[key]).resolve())
    expected["symbols"] = tuple(expected["symbols"])
    return expected


def _install_adapter(prepared, out):
    """Reuse a trusted local production object; any cache failure uses the loader."""
    import cloudpickle

    from src.data import StockBenchAdapter
    from src.env import trading

    out = Path(out)
    expected = _adapter_configuration(prepared)
    sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("src/data/stockbench.py", "scripts/diagnostics/m2_onpolicy_credit_driver.py")
    }
    location = os.environ.get("FCA_ONLINE_ADAPTER_CACHE")
    cache = Path(location).resolve() if location else None
    failure = None
    if cache is not None and cache.is_file():
        try:
            with cache.open("rb") as stream:
                payload = cloudpickle.load(stream)
            require(payload["configuration"] == expected, "adapter cache configuration mismatch")
            require(payload["loader_source_sha256"] == sources, "adapter cache loader changed")
            adapter = payload["adapter"]
            require(
                isinstance(adapter, StockBenchAdapter)
                and adapter._observations
                and adapter._data_version == expected["data_version"],
                "production adapter cache",
            )

            def bound_adapter(**actual):
                require(actual == expected, "cached adapter configuration differs")
                return adapter

            trading._cached_storage_adapter = bound_adapter
            write(
                out / "fast_preprocessing_receipt.json",
                dict(
                    status="READY",
                    cache_hit=True,
                    cache_path=str(cache),
                    configuration=expected,
                    observations=len(adapter._observations),
                    loader_source_sha256=sources,
                    implementation="trusted local cloudpickle of production StockBenchAdapter",
                    market_checksum_repeated=False,
                    sampling_rules_changed=False,
                ),
            )
            print("CPU production adapter cache hit", str(cache), flush=True)
            return
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            print("Adapter cache unavailable; using production loader:", failure, flush=True)
    helper = _helper("online_demo_fast_adapter", "m2_onpolicy_credit_driver.py")
    helper.install_fast_adapter(SimpleNamespace(data=prepared), out)
    if cache is None or cache.exists():
        return
    temporary = None
    try:
        adapter = trading._cached_storage_adapter(**expected)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=cache.parent, prefix=".adapter-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            cloudpickle.dump(
                dict(configuration=expected, loader_source_sha256=sources, adapter=adapter), stream
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, cache)  # Atomic publish, never replace an existing cache.
        write(
            out / "adapter_cache_write_receipt.json",
            dict(
                status="CACHED",
                cache_path=str(cache),
                bytes=cache.stat().st_size,
                observations=len(adapter._observations),
                loader_source_sha256=sources,
                prior_cache_failure=failure,
            ),
        )
        print("CPU production adapter cached", str(cache), flush=True)
    except Exception as error:
        print("Optional adapter cache write skipped:", type(error).__name__, str(error), flush=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def collect(
    model_path,
    out,
    episode_ids=EPISODES,
    seed=42,
    *,
    parents_only=False,
    round_index=1,
    fixed_prefix_source=None,
    continuations_per_anchor=4,
    parent_repetitions=1,
    local_credit=False,
    local_credit_source=None,
    experiment_config=None,
    split="train",
    prepared_path=None,
    decision_mode="native",
    ranker_path=None,
):
    """Collect actual parent/head/EST trajectories; no REF or optimizer update."""
    return asyncio.run(
        _collect(
            model_path,
            out,
            episode_ids,
            seed,
            parents_only,
            round_index,
            fixed_prefix_source,
            continuations_per_anchor,
            parent_repetitions,
            local_credit,
            local_credit_source,
            experiment_config,
            split,
            prepared_path,
            decision_mode,
            ranker_path,
        )
    )


async def _collect(
    model_path,
    out,
    episode_ids,
    seed,
    parents_only,
    round_index,
    fixed_prefix_source=None,
    continuations_per_anchor=4,
    parent_repetitions=1,
    local_credit=False,
    local_credit_source=None,
    experiment_config=None,
    split="train",
    prepared_path=None,
    decision_mode="native",
    ranker_path=None,
):
    out, model_path = Path(out).resolve(), Path(model_path).resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / "collection_started.json").exists(), "fresh collection directory required")
    episode_ids = list(episode_ids)
    broad = None
    if experiment_config is not None:
        from src.agents.broad_prepare import load_config

        broad = load_config(experiment_config)
        require(split in ("train", "valid"), "broad split is Train or Valid only")
        require(episode_ids == broad[f"{split}_episode_ids"], "declared broad episode list")
        require(seed == 43 and parents_only, "broad native/grouped collect-only profile")
        require(parent_repetitions == (4 if split == "train" else 2), "declared broad replicas")
        require(not local_credit and fixed_prefix_source is None, "broad parents only")
        require(split == "valid" or decision_mode == "native", "Train policy remains native")
        expected_prepared = Path(broad[f"prepared_{split}_path"]).resolve()
        require(
            prepared_path is None or Path(prepared_path).resolve() == expected_prepared,
            "prepared input must match broad experiment configuration",
        )
        prepared_file = expected_prepared
    else:
        require(len(episode_ids) in (4, 8), "online demo requires four or eight Train windows")
        require(
            split == "train" and prepared_path is None and decision_mode == "native",
            "extended split/decision modes require an explicit broad experiment config",
        )
        prepared_file = PREPARED
    require(len(episode_ids) == len(set(episode_ids)), "unique Train episodes")
    require(parent_repetitions in ((2, 4) if broad else (1, 4)), "declared parent repetitions")
    require(decision_mode in ("native", "menu", "ranker"), "known decision mode")
    require(decision_mode != "ranker" or ranker_path is not None, "ranker artifact required")
    grouped_parents = parent_repetitions in (2, 4)
    require(
        local_credit or local_credit_source is None, "local-credit source requires local credit"
    )
    if local_credit:
        require(
            grouped_parents
            and seed == 42
            and episode_ids == list(EPISODES)
            and fixed_prefix_source is None,
            "local credit requires original four windows, seed42, four parents, no fixed source",
        )
        parents_only = False
    if grouped_parents:
        original_profile = seed == 42 and set(episode_ids) == set(EPISODES)
        eight_window_profile = seed in (42, 43) and episode_ids == list(EIGHT_EPISODES)
        require(
            (parents_only or local_credit)
            and fixed_prefix_source is None
            and (original_profile or eight_window_profile or broad is not None),
            "grouped parents require original four seed42 or fixed eight seed42/43, parents only",
        )
    local_source, source_parents = None, None
    if local_credit_source is not None:
        local_source = Path(local_credit_source).resolve()
        source_summary = json.loads((local_source / "collection_summary.json").read_text())
        require(
            source_summary["seed"] == seed
            and source_summary["episode_ids"] == episode_ids
            and source_summary["parent_repetitions"] == 4
            and Path(source_summary["model_path"]).resolve() == model_path,
            "local parent source must match four-window common SFT model/profile",
        )
        source_parents = [
            json.loads(line)
            for line in (local_source / "trajectory_results.jsonl").read_text().splitlines()
            if json.loads(line)["role"] == "parent"
        ]
        expected_keys = {f"rloo-parent/{eid}/r{rep}" for eid in episode_ids for rep in range(4)}
        require(
            len(source_parents) == 16
            and {row["trajectory_key"] for row in source_parents} == expected_keys
            and all(row["status"] == "COMPLETE" for row in source_parents),
            "local parent source requires the exact sixteen complete parents",
        )
    fixed_file, fixed_sha, source_snapshots = None, None, {}
    if fixed_prefix_source is not None:
        require(
            not parents_only and seed == 42 and round_index == 0,
            "fixed-j15 probe uses seed42 round0 without parent generation",
        )
        require(
            len(episode_ids) == 8 and continuations_per_anchor == 4,
            "fixed-j15 probe requires eight anchors and four continuations",
        )
        fixed_file = Path(fixed_prefix_source).resolve()
        if fixed_file.is_dir():
            fixed_file = fixed_file / "anchor_snapshots.jsonl"
        payload = fixed_file.read_bytes()
        fixed_sha = hashlib.sha256(payload).hexdigest()
        for line in payload.splitlines():
            snap = json.loads(line)
            require(snap["episode_id"] not in source_snapshots, "duplicate fixed anchor")
            require(
                snap["anchor_id"] == f"{snap['episode_id']}/j15"
                and snap["captured_before_first_request_at_session"] is True,
                "original first-entry j15 anchor",
            )
            source_snapshots[snap["episode_id"]] = snap
        require(set(source_snapshots) == set(episode_ids), "fixed eight-anchor source identity")
    write(
        out / "collection_started.json",
        dict(
            model_path=str(model_path),
            seed=seed,
            round_index=round_index,
            episode_ids=episode_ids,
            parents_only=parents_only,
            parent_repetitions=parent_repetitions,
            **(
                dict(
                    local_credit=True,
                    local_credit_source=str(local_source) if local_source else None,
                    parent_generation_reused=source_parents is not None,
                )
                if local_credit
                else {}
            ),
            REF_used=False,
            optimizer_updates=0,
            fixed_prefix_source=str(fixed_file) if fixed_file else None,
            continuations_per_anchor=continuations_per_anchor if fixed_file else None,
            **(
                dict(
                    experiment_profile=broad["profile"],
                    experiment_config=str(Path(experiment_config).resolve()),
                    source_split=split,
                    decision_mode=decision_mode,
                    formal_valid_access=split == "valid",
                    formal_final_oos_access=False,
                    ranker_path=str(Path(ranker_path).resolve()) if ranker_path else None,
                )
                if broad
                else {}
            ),
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
    import pyarrow.parquet as pq
    from agent_r1.env.base import Action
    from vllm import LLM, SamplingParams

    from src.agents.screening_prompt import apply_screening_action_protocol
    from src.env.trading import TradingEnv

    prepared = {}
    for row in pq.read_table(prepared_file).to_pylist():
        eid = row["extra_info"]["episode_id"]
        if eid not in episode_ids:
            continue
        require(row["extra_info"]["source_split"] == split, "declared history split only")
        kwargs = json.loads(row["env_kwargs"])
        require(len(kwargs["episode_dates"]) == 21 and kwargs["screening"], "21-session screening")
        kwargs.update(capture_interactions=True, reward_mode="finance_trajectory")
        prepared[eid] = dict(kwargs=kwargs, prompt=row["prompt"])
    require(set(prepared) == set(episode_ids), "all requested episodes present in Train parquet")
    if broad and split == "valid":
        os.environ["FCA_ONLINE_ADAPTER_CACHE"] = str(prepared_file.parent / "valid_adapter.pkl")
    _install_adapter(prepared, out)

    def make_environment(eid):
        kwargs = prepared[eid]["kwargs"]
        if broad and split == "valid":
            from src.agents.broad_valid_env import make_valid_environment

            return make_valid_environment(
                kwargs, experiment_config=experiment_config, episode_id=eid
            )
        return TradingEnv(**kwargs)

    if broad:
        first_episode = episode_ids[0]
        probe = make_environment(first_episode)
        probe.reset(raw_prompt=prepared[first_episode]["prompt"])
        del probe
    gpu_helper = _helper("online_demo_native_generation", "m2_continuation_gpu_worker.py")
    compiler, _ = gpu_helper.resolve_compiler()
    os.environ["CC"] = compiler
    engine_settings = gpu_helper.engine_settings(str(model_path))
    engine_settings["seed"] = seed
    write(out / "engine_settings.json", engine_settings)
    print("Loading current-policy vLLM engine", str(model_path), flush=True)
    llm = LLM(**engine_settings)
    tokenizer = llm.get_tokenizer()
    results, head_results, snapshots = [], [], source_snapshots
    counts = Counter()
    logs = {
        name: (out / name).open("x")
        for name in (
            "requests.jsonl",
            "trajectory_results.jsonl",
            "head_results.jsonl",
            "anchor_snapshots.jsonl",
        )
    }
    if decision_mode != "native":
        logs["decisions.jsonl"] = (out / "decisions.jsonl").open("x")
    if local_credit:
        logs.update({name: (out / name).open("x") for name in ("anchors.jsonl", "heads.jsonl")})

    def append(name, value):
        logs[name].write(canonical(value) + "\n")
        logs[name].flush()

    def prompt_ids(state):
        return tokenizer.apply_chat_template(
            apply_screening_action_protocol(state["obs"].messages, PROTOCOL),
            tools=state["env"].tool_schemas,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def financial(state):
        env = state["env"]
        return json.loads(
            canonical(
                dict(
                    visible=json.loads(env._state_text()),
                    equity_curve=env._equity_curve,
                    net_returns=env._net_returns,
                    turnovers=env._turnovers,
                    high_water_mark=env._high_water_mark,
                    maximum_position_weight=env._maximum_position_weight,
                )
            )
        )

    def reset(eid, key, role, **extra):
        data = prepared[eid]
        env = make_environment(eid)
        return dict(
            episode_id=eid,
            trajectory_key=key,
            role=role,
            env=env,
            obs=env.reset(raw_prompt=data["prompt"]),
            timesteps=[],
            attempt=0,
            prefix_requests=0,
            **extra,
        )

    def snapshot(state):
        if local_credit:
            session = state["env"].current_session_index
            if state.get("replicate_index") != 0 or session not in (5, 15):
                return
            # Capture once, before the first real request, including an invalid first action.
            if any(
                s["parent_trajectory_key"] == state["trajectory_key"]
                and s["session_index"] == session
                for s in snapshots.values()
            ):
                return
            index = len(state["timesteps"])
            anchor_id = f"{state['trajectory_key']}/j{session}/request{index}"
            ids = prompt_ids(state)
            snap = dict(
                anchor_id=anchor_id,
                anchor_key=anchor_id,
                episode_id=state["episode_id"],
                trajectory_key=state["trajectory_key"],
                parent_trajectory_key=state["trajectory_key"],
                parent_request_index=index,
                replicate_index=0,
                session_index=session,
                prefix_timesteps=list(state["timesteps"]),
                prefix_requests=index,
                financial_state=financial(state),
                observation_messages=state["obs"].messages,
                prompt_token_ids=ids,
                prompt_sha256=object_sha(ids),
                captured_before_first_request_at_session=True,
            )
            snapshots[anchor_id] = snap
            append("anchor_snapshots.jsonl", snap)
            return
        if state["env"].current_session_index != 15 or state["episode_id"] in snapshots:
            return
        ids = prompt_ids(state)
        snap = dict(
            anchor_id=f"{state['episode_id']}/j15",
            episode_id=state["episode_id"],
            prefix_timesteps=list(state["timesteps"]),
            prefix_requests=len(state["timesteps"]),
            financial_state=financial(state),
            observation_messages=state["obs"].messages,
            prompt_token_ids=ids,
            prompt_sha256=object_sha(ids),
            captured_before_first_request_at_session=True,
        )
        snapshots[state["episode_id"]] = snap
        append("anchor_snapshots.jsonl", snap)

    async def replay(eid, key, role, *, snapshot_key=None, **extra):
        snap = snapshots[eid if snapshot_key is None else snapshot_key]
        state = reset(eid, key, role, **extra)
        for step in snap["prefix_timesteps"]:
            state["obs"], _, _, info = await state["env"].step(
                Action(text=step["response_text"], token_ids=step["response_token_ids"])
            )
            require(json.loads(canonical(info)) == step["info"], "prefix replay info identity")
        require(
            state["obs"].messages == snap["observation_messages"]
            and financial(state) == snap["financial_state"]
            and prompt_ids(state) == snap["prompt_token_ids"],
            "full-prefix replay identity",
        )
        state["prefix_requests"] = snap["prefix_requests"]
        return state

    def generate(states):
        pending, returned = [], []
        for state in states:
            ids = prompt_ids(state)
            if len(ids) > MAX_PROMPT:
                returned.append((state, dict(status="PROMPT_LIMIT", prompt_token_ids=ids)))
                continue
            session = state["env"].current_session_index
            rs = request_seed(seed, state["trajectory_key"], session, state["attempt"], round_index)
            pending.append((state, ids, rs))
        for start in range(0, len(pending), MAX_BATCH):
            chunk = pending[start : start + MAX_BATCH]
            params = [SamplingParams(**gpu_helper.SAMPLING, seed=rs) for _, _, rs in chunk]
            native = llm.generate(
                [{"prompt_token_ids": ids} for _, ids, _ in chunk], params, use_tqdm=False
            )
            require(len(native) == len(chunk), "native batch size")
            for (state, ids, rs), result in zip(chunk, native, strict=True):
                require(
                    list(result.prompt_token_ids) == ids and len(result.outputs) == 1,
                    "native prompt and single output",
                )
                answer = result.outputs[0]
                response = list(answer.token_ids)
                raw = dict(
                    response_token_ids=response,
                    returned_finished=result.finished,
                    finish_reason=answer.finish_reason,
                    stop_reason=answer.stop_reason,
                )
                status, reason = gpu_helper.generation_integrity(raw)
                record = dict(
                    **raw,
                    status=status,
                    failure_reason=reason,
                    request_seed=rs,
                    prompt_token_ids=ids,
                    response_text=tokenizer.decode(response, skip_special_tokens=True),
                )
                append(
                    "requests.jsonl",
                    dict(
                        trajectory_key=state["trajectory_key"],
                        role=state["role"],
                        decision_session_index=state["env"].current_session_index,
                        attempt=state["attempt"],
                        **record,
                    ),
                )
                counts["model_requests"] += 1
                counts["prompt_tokens"] += len(ids)
                counts["response_tokens"] += len(response)
                counts[f"{state['role']}_requests"] += 1
                returned.append((state, record))
        return returned

    async def apply(state, generated):
        if generated["status"] != "COMPLETE":
            return generated["status"]
        env = state["env"]
        before = env.current_session_index
        proposal = generated
        selection = None
        if decision_mode != "native":
            from src.agents.order_ranker import select_order

            selection = select_order(
                json.loads(env._state_text()),
                proposal["response_text"],
                mode=decision_mode,
                seed=seed,
                key=f"{state['trajectory_key']}/j{before}/a{state['attempt']}",
                ranker_path=ranker_path,
            )
            selected_text = selection["response_text"]
            selected_ids = (
                proposal["response_token_ids"]
                if selected_text == proposal["response_text"]
                else tokenizer.encode(selected_text + tokenizer.eos_token, add_special_tokens=False)
            )
            generated = dict(proposal, response_text=selected_text, response_token_ids=selected_ids)
            append(
                "decisions.jsonl",
                dict(
                    trajectory_key=state["trajectory_key"],
                    decision_session_index=before,
                    attempt=state["attempt"],
                    request_seed=proposal["request_seed"],
                    proposal_response_text=proposal["response_text"],
                    proposal_response_token_ids=proposal["response_token_ids"],
                    **selection,
                ),
            )
        obs, reward, _, info = await env.step(
            Action(text=generated["response_text"], token_ids=generated["response_token_ids"])
        )
        state["obs"] = obs
        after = env.current_session_index
        require(after in (before, before + 1), "single-session transition")
        invalid = bool(info.get("invalid_action"))
        require(not (invalid and after != before), "invalid head cannot advance")
        trace = asdict(env.credit_step_trace[-1])
        step = dict(
            prompt_token_ids=generated["prompt_token_ids"],
            response_token_ids=generated["response_token_ids"],
            response_text=generated["response_text"],
            request_seed=generated["request_seed"],
            action_family=trace["action"],
            advancing=after == before + 1,
            invalid=invalid,
            invalid_action=invalid,
            decision_session_index=before,
            result_session_index=after,
            done=env.episode_complete,
            reward=float(reward),
            info=json.loads(canonical(info)),
            trace=trace,
        )
        if selection is not None:
            step.update(
                decision_mode=decision_mode,
                proposal_response_text=proposal["response_text"],
                proposal_response_token_ids=proposal["response_token_ids"],
                selected_candidate_id=selection["selected_candidate_id"],
                selected_economic_order_key=selection["economic_order_key"],
                proposal_legal=selection["proposal_legal"],
                selection_mode=selection["selection_mode"],
            )
        state["timesteps"].append(step)
        counts["invalid_requests"] += invalid
        state["attempt"] = 0 if after != before else state["attempt"] + 1
        if env.episode_complete:
            return "COMPLETE"
        return "REQUEST_LIMIT" if state["attempt"] >= MAX_ATTEMPTS else None

    def finish(state, status):
        metrics = state["env"].screening_terminal_metrics() if status == "COMPLETE" else None
        y = metrics["terminal_score"] if metrics else None
        if metrics:
            require(
                math.isfinite(y) and state["env"].current_session_index == 20,
                "finite original terminal financial Y",
            )
        steps = state["timesteps"]
        prefix = state["prefix_requests"]
        total = prefix + len(steps)
        invalids = sum(step["invalid"] for step in steps)
        row = {k: state[k] for k in ("trajectory_key", "role", "episode_id", "prefix_requests")}
        row.update(
            {
                k: state[k]
                for k in (
                    "anchor_id",
                    "slot_id",
                    "estimate_replicate",
                    "continuation_replicate",
                    "replicate_index",
                    "anchor_key",
                    "head_index",
                    "draw_index",
                    "parent_trajectory_key",
                    "parent_request_index",
                )
                if k in state
            }
        )
        row.update(
            status=status,
            terminal_financial_Y=y,
            Y=y,
            terminal_metrics=metrics,
            timesteps=steps,
            head_requests=int(state["role"] in ("estimate", "est")),
            tail_requests=len(steps) - int(state["role"] in ("estimate", "est")),
            total_requests=total,
            remaining_requests=len(steps),
            remaining_invalid_count=invalids,
            financial_remaining_return=(len(steps) * y / total - 0.05 * invalids)
            if y is not None and total
            else None,
            last_session_index=state["env"].current_session_index,
        )
        if state["role"] == "est":
            row["terminal_Y_remaining_return"] = y - 0.05 * invalids if y is not None else None
        if state["role"] == "fixed_continuation":
            anchor = state["anchor_equity"]
            final = state["env"]._equity_curve[-1] if status == "COMPLETE" else None
            row.update(
                anchor_equity=anchor,
                final_equity=final,
                anchor_equity_basis="last booked equity after replaying the full j15 prefix",
                suffix_net_return=final / anchor - 1 if final is not None else None,
                first_action_invalid=steps[0]["invalid"] if steps else None,
                all_invalid_count=invalids,
                anchor_prompt_sha256=state["anchor_prompt_sha256"],
                anchor_financial_state_sha256=state["anchor_financial_state_sha256"],
            )
        results.append(row)
        append("trajectory_results.jsonl", row)
        print("FINISHED", row["trajectory_key"], status, y, flush=True)
        return row

    async def complete(states, capture=False):
        active = list(states)
        while active:
            if capture:
                for state in active:
                    snapshot(state)
            for state, generated in generate(active):
                status = await apply(state, generated)
                if status:
                    finish(state, status)
                    active.remove(state)

    async def collect_local_credit(parents):
        """Estimate sparse local credit; never train on or replace the branch actions."""
        parent_map = {row["trajectory_key"]: row for row in parents}
        anchors, native_states, legal = [], [], []

        def head_row(anchor, slot, state=None, generated=None, status=None):
            step = state["timesteps"][0] if state and state["timesteps"] else None
            is_legal = bool(step and step["advancing"] and not step["invalid"])
            key = None
            if is_legal:
                _, calls = state["env"].format_wrapper.parse_response(step["response_text"])
                require(len(calls) == 1, "single advancing local head call")
                call, family = calls[0], step["action_family"]
                key = (
                    [
                        family,
                        call.arguments["symbol"].strip().upper(),
                        float(call.arguments["target_weight"]),
                    ]
                    if family in ("buy", "sell")
                    else [family]
                )
            row = dict(
                anchor_id=anchor["anchor_id"],
                anchor_key=anchor["anchor_id"],
                episode_id=anchor["episode_id"],
                session_index=anchor["session_index"],
                parent_trajectory_key=anchor["parent_trajectory_key"],
                parent_request_index=anchor["parent_request_index"],
                group_id=anchor["anchor_id"],
                slot_id=slot,
                head_index=slot,
                head_key=state["trajectory_key"] if state else None,
                status="ADVANCING" if is_legal else (status or "INVALID"),
                legal_head=is_legal,
                economic_order_key=canonical(key) if key else None,
                prompt_token_ids=generated.get("prompt_token_ids", []) if generated else [],
                response_token_ids=generated.get("response_token_ids", []) if generated else [],
                response_text=generated.get("response_text") if generated else None,
                prefix_requests=state["prefix_requests"] if state else None,
                timesteps=state["timesteps"] if state else [],
                parent_actual_action=slot == 0,
            )
            if state:
                row.update(
                    post_first_financial_state_sha256=object_sha(financial(state)),
                    post_first_observation_messages_sha256=object_sha(state["obs"].messages),
                )
            head_results.append(row)
            append("head_results.jsonl", row)
            append("heads.jsonl", row)
            if is_legal:
                legal.append((anchor, state, row))

        for eid in episode_ids:
            parent_key = f"rloo-parent/{eid}/r0"
            parent = parent_map[parent_key]
            for session in (5, 15):
                snap = next(
                    (
                        s
                        for s in snapshots.values()
                        if s["parent_trajectory_key"] == parent_key
                        and s["session_index"] == session
                    ),
                    None,
                )
                index = snap["parent_request_index"] if snap else None
                anchor_id = snap["anchor_id"] if snap else f"{parent_key}/j{session}/requestmissing"
                first = (
                    parent["timesteps"][index]
                    if index is not None and index < len(parent["timesteps"])
                    else None
                )
                anchor = dict(
                    anchor_id=anchor_id,
                    anchor_key=anchor_id,
                    episode_id=eid,
                    trajectory_key=parent_key,
                    parent_trajectory_key=parent_key,
                    parent_request_index=index,
                    replicate_index=0,
                    session_index=session,
                    prompt_token_ids=snap["prompt_token_ids"] if snap else [],
                    captured_before_first_request_at_session=bool(snap),
                    status="CAPTURED" if first else "MISSING_PARENT_REQUEST",
                )
                anchors.append(anchor)
                append("anchors.jsonl", anchor)
                if not snap or not first:
                    for slot in range(4):
                        head_row(anchor, slot, status="MISSING_ANCHOR_OR_PARENT_REQUEST")
                    continue
                require(
                    first["prompt_token_ids"] == snap["prompt_token_ids"],
                    "actual parent first request matches local anchor prompt",
                )
                state = await replay(
                    eid,
                    f"local-head/{anchor_id}/s0",
                    "local_head",
                    snapshot_key=anchor_id,
                    anchor_id=anchor_id,
                    slot_id=0,
                )
                generated = dict(first, status="COMPLETE")
                prior_invalids = counts["invalid_requests"]
                status = await apply(state, generated)
                counts["invalid_requests"] = (
                    prior_invalids  # Forced original action is not new inference.
                )
                require(
                    state["timesteps"][0]["info"] == first["info"], "actual head replay identity"
                )
                head_row(anchor, 0, state, generated, status)
                for slot in range(1, 4):
                    native_states.append(
                        await replay(
                            eid,
                            f"local-head/{anchor_id}/s{slot}",
                            "local_head",
                            snapshot_key=anchor_id,
                            anchor_id=anchor_id,
                            anchor_key=anchor_id,
                            slot_id=slot,
                            head_index=slot,
                        )
                    )
        anchor_map = {row["anchor_id"]: row for row in anchors}
        for state, generated in generate(native_states):
            status = await apply(state, generated)
            head_row(anchor_map[state["anchor_id"]], state["slot_id"], state, generated, status)
        branches = []
        for anchor, original, head in legal:
            for rep in range(2):
                anchor_id, slot = anchor["anchor_id"], head["slot_id"]
                state = await replay(
                    anchor["episode_id"],
                    f"local-est/{anchor_id}/s{slot}/r{rep}",
                    "est",
                    snapshot_key=anchor_id,
                    anchor_id=anchor_id,
                    anchor_key=anchor_id,
                    slot_id=slot,
                    head_index=slot,
                    estimate_replicate=rep,
                    draw_index=rep,
                    parent_trajectory_key=anchor["parent_trajectory_key"],
                    parent_request_index=anchor["parent_request_index"],
                )
                first = original["timesteps"][0]
                state["obs"], _, _, info = await state["env"].step(
                    Action(text=first["response_text"], token_ids=first["response_token_ids"])
                )
                require(
                    json.loads(canonical(info)) == first["info"]
                    and object_sha(financial(state)) == head["post_first_financial_state_sha256"]
                    and object_sha(state["obs"].messages)
                    == head["post_first_observation_messages_sha256"],
                    "local head replay identity",
                )
                state["timesteps"] = [dict(first)]
                branches.append(state)
        await complete(branches)
        return anchors

    try:
        if fixed_file is not None:
            states = []
            for eid in episode_ids:
                snap = snapshots[eid]
                append("anchor_snapshots.jsonl", snap)
                for rep in range(continuations_per_anchor):
                    state = await replay(
                        eid,
                        f"fixed-j15/{eid}/r{rep}",
                        "fixed_continuation",
                        anchor_id=snap["anchor_id"],
                        continuation_replicate=rep,
                    )
                    require(state["env"].current_session_index == 15, "fixed replay reaches j15")
                    state.update(
                        anchor_equity=state["env"]._equity_curve[-1],
                        anchor_prompt_sha256=object_sha(snap["prompt_token_ids"]),
                        anchor_financial_state_sha256=object_sha(snap["financial_state"]),
                    )
                    states.append(state)
            await complete(states)
            summary = dict(
                status="COMPLETE",
                model_path=str(model_path),
                fixed_prefix_source=str(fixed_file),
                source_sha256=fixed_sha,
                episode_ids=episode_ids,
                seed=seed,
                round_index=round_index,
                continuations_per_anchor=continuations_per_anchor,
                planned_trajectories=len(episode_ids) * continuations_per_anchor,
                complete_trajectories=sum(r["status"] == "COMPLETE" for r in results),
                failed_trajectories=sum(r["status"] != "COMPLETE" for r in results),
                **counts,
                optimizer_updates=0,
                REF_used=False,
                request_cap_per_session=MAX_ATTEMPTS,
                terminal_session_index=20,
                anchor_equity_basis="last booked equity at original SFT j15 prefix",
                formal_valid_oos_access=False,
                trajectory_file="trajectory_results.jsonl",
            )
            write(out / "fixed_prefix_summary.json", summary)
            return summary
        print("Collecting fresh full-reset parents", episode_ids, flush=True)
        parent_states = (
            [
                reset(eid, f"rloo-parent/{eid}/r{rep}", "parent", replicate_index=rep)
                for eid in episode_ids
                for rep in range(parent_repetitions)
            ]
            if grouped_parents
            else [reset(eid, f"parent/{eid}", "parent") for eid in episode_ids]
        )
        if source_parents is not None:
            selected = {
                state["trajectory_key"]: state
                for state in parent_states
                if state["replicate_index"] == 0
            }
            for original in source_parents:
                if original["trajectory_key"] in selected:
                    state = selected[original["trajectory_key"]]
                    for step in original["timesteps"]:
                        snapshot(state)
                        state["obs"], _, _, info = await state["env"].step(
                            Action(text=step["response_text"], token_ids=step["response_token_ids"])
                        )
                        require(
                            json.loads(canonical(info)) == step["info"],
                            "source parent prefix replay info identity",
                        )
                        state["timesteps"].append(dict(step))
                row = dict(
                    original,
                    parent_generation_reused=True,
                    parent_source_directory=str(local_source),
                )
                results.append(row)
                append("trajectory_results.jsonl", row)
        else:
            await complete(parent_states, capture=not grouped_parents or local_credit)
        local_anchors = None
        if local_credit:
            local_anchors = await collect_local_credit(
                [r for r in results if r["role"] == "parent"]
            )
        if not parents_only and not local_credit:
            heads = []
            for eid in episode_ids:
                for slot in range(4):
                    if eid not in snapshots:
                        head = dict(
                            episode_id=eid,
                            anchor_id=f"{eid}/j15",
                            slot_id=slot,
                            status="MISSING_ANCHOR",
                            legal_head=False,
                            timesteps=[],
                        )
                        head_results.append(head)
                        append("head_results.jsonl", head)
                        continue
                    heads.append(
                        await replay(
                            eid,
                            f"head/{eid}/j15/s{slot}",
                            "head",
                            anchor_id=f"{eid}/j15",
                            slot_id=slot,
                        )
                    )
            legal = []
            for state, generated in generate(heads):
                status = await apply(state, generated)
                step = state["timesteps"][0] if state["timesteps"] else None
                is_legal = bool(step and step["advancing"] and not step["invalid"])
                key = None
                if is_legal:
                    _, calls = state["env"].format_wrapper.parse_response(step["response_text"])
                    require(len(calls) == 1, "single advancing head call")
                    call, family = calls[0], step["action_family"]
                    key = (
                        [
                            family,
                            call.arguments["symbol"].strip().upper(),
                            float(call.arguments["target_weight"]),
                        ]
                        if family in ("buy", "sell")
                        else [family]
                    )
                row = dict(
                    episode_id=state["episode_id"],
                    anchor_id=state["anchor_id"],
                    group_id=state["anchor_id"],
                    slot_id=state["slot_id"],
                    head_key=state["trajectory_key"],
                    status="ADVANCING"
                    if is_legal
                    else (status or ("INVALID" if step and step["invalid"] else "NONADVANCING")),
                    legal_head=is_legal,
                    economic_order_key=canonical(key) if key else None,
                    prompt_token_ids=generated.get("prompt_token_ids", []),
                    response_token_ids=generated.get("response_token_ids", []),
                    response_text=generated.get("response_text"),
                    prefix_requests=state["prefix_requests"],
                    timesteps=state["timesteps"],
                    post_first_financial_state_sha256=object_sha(financial(state)),
                    post_first_observation_messages_sha256=object_sha(state["obs"].messages),
                )
                head_results.append(row)
                append("head_results.jsonl", row)
                if is_legal:
                    legal.append((state, row))
            branches = []
            for original, head in legal:
                for rep in range(2):
                    eid, slot = head["episode_id"], head["slot_id"]
                    state = await replay(
                        eid,
                        f"est/{eid}/j15/s{slot}/r{rep}",
                        "estimate",
                        anchor_id=head["anchor_id"],
                        slot_id=slot,
                        estimate_replicate=rep,
                    )
                    first = original["timesteps"][0]
                    state["obs"], _, _, info = await state["env"].step(
                        Action(text=first["response_text"], token_ids=first["response_token_ids"])
                    )
                    require(
                        json.loads(canonical(info)) == first["info"]
                        and object_sha(financial(state))
                        == head["post_first_financial_state_sha256"]
                        and object_sha(state["obs"].messages)
                        == head["post_first_observation_messages_sha256"],
                        "head replay identity",
                    )
                    state["timesteps"] = [dict(first)]
                    branches.append(state)
            await complete(branches)
        parent_map = {row["trajectory_key"]: row for row in results if row["role"] == "parent"}
        parents = [parent_map[state["trajectory_key"]] for state in parent_states]
        groups = []
        if grouped_parents:
            for eid in episode_ids:
                members = [row for row in parents if row["episode_id"] == eid]
                complete_group = len(members) == parent_repetitions and all(
                    row["status"] == "COMPLETE" for row in members
                )
                groups.append(
                    dict(
                        episode_id=eid,
                        planned_trajectories=parent_repetitions,
                        complete_trajectories=sum(r["status"] == "COMPLETE" for r in members),
                        failed_trajectories=sum(r["status"] != "COMPLETE" for r in members),
                        status="COMPLETE" if complete_group else "NA",
                        trajectory_keys=[r["trajectory_key"] for r in members],
                        mean_terminal_financial_Y=math.fsum(r["Y"] for r in members)
                        / parent_repetitions
                        if complete_group
                        else None,
                        mean_raw_net_return=math.fsum(
                            r["terminal_metrics"]["raw_net_return"] for r in members
                        )
                        / parent_repetitions
                        if complete_group
                        else None,
                    )
                )
        summary = dict(
            status="COMPLETE",
            model_path=str(model_path),
            episode_ids=episode_ids,
            seed=seed,
            round_index=round_index,
            parents_only=parents_only,
            parent_repetitions=parent_repetitions,
            planned_parent_trajectories=len(episode_ids) * parent_repetitions,
            parent_identities=[
                dict(
                    episode_id=row["episode_id"],
                    trajectory_key=row["trajectory_key"],
                    replicate_index=row.get("replicate_index"),
                    status=row["status"],
                )
                for row in parents
            ],
            parent_groups=groups,
            **counts,
            parents_complete=sum(row["status"] == "COMPLETE" for row in parents),
            parents_failed=sum(row["status"] != "COMPLETE" for row in parents),
            parent_terminal_Y=[row["Y"] for row in parents],
            parent_terminal_metrics=[row["terminal_metrics"] for row in parents],
            captured_anchors=len(snapshots),
            head_slots=len(head_results),
            legal_heads=sum(row["legal_head"] for row in head_results),
            estimate_paths=sum(row["role"] == "estimate" for row in results),
            estimate_complete=sum(
                row["role"] == "estimate" and row["status"] == "COMPLETE" for row in results
            ),
            REF_used=False,
            optimizer_updates=0,
            request_cap_per_session=MAX_ATTEMPTS,
            prepared_train_parquet=str(prepared_file) if split == "train" else None,
            **(
                dict(
                    experiment_profile=broad["profile"],
                    experiment_config=str(Path(experiment_config).resolve()),
                    source_split=split,
                    prepared_parquet=str(prepared_file),
                    decision_mode=decision_mode,
                    formal_valid_access=split == "valid",
                    formal_final_oos_access=False,
                    ranker_path=str(Path(ranker_path).resolve()) if ranker_path else None,
                )
                if broad
                else {}
            ),
            formal_valid_oos_access=split == "valid",
            terminal_session_index=20,
            trajectory_file="trajectory_results.jsonl",
        )
        if local_credit:
            estimates = [row for row in results if row["role"] == "est"]
            summary.update(
                local_credit=True,
                parents_only=False,
                planned_local_anchors=8,
                planned_local_head_slots=32,
                local_anchor_groups=local_anchors,
                parent_generation_reused=source_parents is not None,
                local_credit_source=str(local_source) if local_source else None,
                parent_requests=counts["parent_requests"],
                local_head_requests=counts["local_head_requests"],
                local_est_requests=counts["est_requests"],
                local_requests=counts["local_head_requests"] + counts["est_requests"],
                estimate_paths=len(estimates),
                estimate_complete=sum(row["status"] == "COMPLETE" for row in estimates),
                local_return_definition="terminal_financial_Y - 0.05 * remaining_invalid_count",
                local_candidate_slots="actual parent request + three native heads; no replacement",
            )
        write(out / "collection_summary.json", summary)
        return summary
    finally:
        for stream in logs.values():
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode-ids", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round", "--round-index", dest="round_index", type=int, default=1)
    parser.add_argument("--parents-only", action="store_true")
    parser.add_argument("--parent-repetitions", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--decision-mode", choices=("native", "menu", "ranker"), default="native")
    parser.add_argument("--ranker-path", type=Path)
    parser.add_argument("--fixed-prefix-source", type=Path)
    parser.add_argument("--continuations-per-anchor", type=int, default=4)
    parser.add_argument("--local-credit", action="store_true")
    parser.add_argument("--local-credit-source", type=Path)
    args = parser.parse_args()
    episode_ids = args.episode_ids
    if episode_ids is None:
        if args.experiment_config:
            from src.agents.broad_prepare import load_config

            episode_ids = load_config(args.experiment_config)[f"{args.split}_episode_ids"]
        else:
            episode_ids = EPISODES
    result = collect(
        args.model,
        args.output,
        episode_ids,
        args.seed,
        parents_only=args.parents_only,
        round_index=args.round_index,
        fixed_prefix_source=args.fixed_prefix_source,
        continuations_per_anchor=args.continuations_per_anchor,
        parent_repetitions=args.parent_repetitions,
        local_credit=args.local_credit,
        local_credit_source=args.local_credit_source,
        experiment_config=args.experiment_config,
        split=args.split,
        prepared_path=args.prepared,
        decision_mode=args.decision_mode,
        ranker_path=args.ranker_path,
    )
    print(canonical(result), flush=True)


if __name__ == "__main__":
    main()

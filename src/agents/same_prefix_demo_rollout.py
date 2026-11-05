"""Two short post-update environment continuations for the cached-head demo."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "runs/m2-onpolicy-credit-replication-20261019-v1"
TRAIN_ROWS = ROOT / "runs/m2-action-pooling-comparison-20261013-v6/seed42/prepared/train.parquet"


def run_rollouts(model, tokenizer, output_directory, source_directory=DEFAULT_SOURCE, count=2):
    """Continue cached Train prefixes with the updated model; no performance claim."""
    import torch

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if count <= 0:
        return []
    model.eval()
    model.gradient_checkpointing_disable()
    original_use_cache = model.config.use_cache
    model.config.use_cache = True

    async def run():
        sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
        import pyarrow.parquet as pq
        from agent_r1.env.base import Action
        from m2_onpolicy_credit_driver import install_fast_adapter

        from src.agents.screening_prompt import apply_screening_action_protocol
        from src.env.trading import TradingEnv

        snapshots = [
            json.loads(line)
            for line in (Path(source_directory) / "anchor_snapshots.jsonl").read_text().splitlines()
        ]
        selected = sorted(
            (r for r in snapshots if r["decision_session_index"] == 15),
            key=lambda r: r["anchor_id"],
        )[:count]
        train_rows = {
            r["extra_info"]["episode_id"]: r for r in pq.read_table(TRAIN_ROWS).to_pylist()
        }
        first_kwargs = json.loads(train_rows[selected[0]["episode_id"]]["env_kwargs"])
        adapter_dir = output / "rollout_preparation"
        adapter_dir.mkdir(exist_ok=True)
        install_fast_adapter(SimpleNamespace(data={"demo": {"kwargs": first_kwargs}}), adapter_dir)
        results = []
        with (output / "post_update_trajectory_actions.jsonl").open("w") as actions:
            for index, snapshot in enumerate(selected):
                row = train_rows[snapshot["episode_id"]]
                kwargs = json.loads(row["env_kwargs"])
                kwargs.update(capture_interactions=True, reward_mode="finance_trajectory")
                env = TradingEnv(**kwargs)
                obs = env.reset(raw_prompt=row["prompt"])
                for item in snapshot["prefix"]:
                    obs, _, _, _ = await env.step(
                        Action(text=item["response_text"], token_ids=item["response_token_ids"])
                    )
                if obs.messages != snapshot["observation_messages"]:
                    raise ValueError("cached demo prefix did not reproduce its observation")
                calls, attempts, input_tokens, output_tokens = 0, 0, 0, 0
                reason = None
                while not env.episode_complete and calls < 24:
                    prompt = tokenizer.apply_chat_template(
                        apply_screening_action_protocol(obs.messages, "single_tool_portfolio_v1"),
                        tools=env.tool_schemas,
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    if len(prompt) > 8192:
                        reason = "prompt_limit"
                        break
                    torch.manual_seed(20261022 + index * 100 + calls)
                    inputs = torch.tensor([prompt], dtype=torch.long, device=model.device)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        generated = model.generate(
                            input_ids=inputs,
                            attention_mask=torch.ones_like(inputs),
                            max_new_tokens=128,
                            do_sample=True,
                            temperature=1.0,
                            top_p=1.0,
                            top_k=0,
                            repetition_penalty=1.0,
                            use_cache=True,
                            eos_token_id=[151645, 151643],
                            pad_token_id=151643,
                        )
                    tokens = generated[0, len(prompt) :].tolist()
                    input_tokens += len(prompt)
                    output_tokens += len(tokens)
                    calls += 1
                    if not tokens or tokens[-1] not in (151645, 151643):
                        reason = "response_incomplete"
                        break
                    text = tokenizer.decode(tokens[:-1], skip_special_tokens=False)
                    previous_session = env.current_session_index
                    obs, _, _, info = await env.step(Action(text=text, token_ids=tokens))
                    advanced = env.current_session_index > previous_session
                    actions.write(
                        json.dumps(
                            dict(
                                anchor_id=snapshot["anchor_id"],
                                request=calls,
                                session_before=previous_session,
                                session_after=env.current_session_index,
                                response_text=text,
                                response_token_ids=tokens,
                                advanced=advanced,
                                info=info,
                            ),
                            default=str,
                        )
                        + "\n"
                    )
                    actions.flush()
                    attempts = 0 if advanced else attempts + 1
                    if attempts >= 4:
                        reason = "session_request_limit"
                        break
                complete = bool(env.episode_complete)
                result = dict(
                    anchor_id=snapshot["anchor_id"],
                    episode_id=snapshot["episode_id"],
                    start_session=15,
                    last_session=env.current_session_index,
                    status="COMPLETE" if complete else "INCOMPLETE",
                    failure_reason=None if complete else (reason or "request_limit"),
                    model_calls=calls,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    terminal_metrics=env.screening_terminal_metrics() if complete else None,
                    purpose="post-update cached-prefix continuation demo, no paired gain claim",
                )
                results.append(result)
                print(
                    json.dumps({"event": "post_update_trajectory", **result}, default=str),
                    flush=True,
                )
        return results

    try:
        results = asyncio.run(run())
        (output / "post_update_trajectories.json").write_text(
            json.dumps(results, indent=2, default=str)
        )
        return results
    finally:
        model.config.use_cache = original_use_cache

"""One multistep S0 Monte Carlo/GAE update on complete Train parent trajectories.

Every actual request, including rejected and recovery requests, is one actor
and critic example. Gamma=lambda=1 gives U_t=G_t-V(s_t); this short demo does
not implement repeated PPO epochs or claim a complete StepPO reproduction.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from src.agents.same_prefix_online_train import (
    adam_update,
    finite,
    records,
    retire_loaded_input,
    retirement_inputs,
    scalar_value,
)
from src.agents.same_prefix_training_demo import (
    append,
    checkpoint_probes,
    probe_delta,
    probes,
    require,
    response_logprob,
    write,
)

PROFILE = "multistep_s0_monte_carlo_demo_v1"
PENALTY = 0.05


def trajectory_rows(trajectory: dict) -> tuple[list[dict], dict]:
    """Validate one complete timeline and preserve every real request."""
    steps = trajectory["timesteps"]
    score = trajectory.get("terminal_financial_Y", trajectory.get("Y"))
    require(finite(score), "complete parent needs finite terminal financial Y")
    require(isinstance(steps, list) and 1 < len(steps) <= 80, "actual parent timeline required")
    total = len(steps)
    require(
        trajectory.get("prefix_requests") == 0
        and trajectory.get("total_requests") == total
        and trajectory.get("last_session_index") == 20,
        "parent must be a complete full-reset original timeline",
    )
    # The final residual-corrected financial share conserves terminal Y.
    share = score / total
    financial_rewards = [share] * (total - 1) + [score - share * (total - 1)]
    require(
        math.isclose(math.fsum(financial_rewards), score, rel_tol=1e-12, abs_tol=1e-12),
        "M1 financial reward must conserve Y",
    )
    batch, counts, invalid_counts = [], Counter(), Counter()
    expected_session = 0
    for index, (step, financial_reward) in enumerate(zip(steps, financial_rewards, strict=True)):
        prompt, response = step["prompt_token_ids"], step["response_token_ids"]
        require(
            isinstance(prompt, list)
            and isinstance(response, list)
            and 0 < len(prompt) <= 8192
            and 0 < len(response) <= 4096
            and all(type(token) is int and token >= 0 for token in prompt + response),
            "original nonempty prompt/response token masks within fixed limits required",
        )
        before, after = step["decision_session_index"], step["result_session_index"]
        invalid, advancing, done = step["invalid"], step["advancing"], step["done"]
        require(
            type(before) is int
            and type(after) is int
            and before == expected_session
            and 0 <= before < 20
            and type(invalid) is bool
            and type(advancing) is bool
            and type(done) is bool
            and after == before + int(advancing)
            and (not invalid or not advancing)
            and done == (index == total - 1)
            and (not done or after == 20),
            "actual request/session timeline is inconsistent",
        )
        require(
            step.get("invalid_action", invalid) == invalid,
            "invalid request flags disagree",
        )
        counts[before] += 1
        invalid_counts[before] += int(invalid)
        expected_session = after
        batch.append(
            dict(
                episode_id=trajectory["episode_id"],
                trajectory_key=trajectory["trajectory_key"],
                request_index=index,
                decision_session_index=before,
                result_session_index=after,
                invalid=invalid,
                advancing=advancing,
                done=done,
                prompt_token_ids=prompt,
                response_token_ids=response,
                response_text=step["response_text"],
                terminal_financial_Y=score,
                trajectory_requests=total,
                financial_reward=financial_reward,
                training_reward=financial_reward - PENALTY * int(invalid),
            )
        )
    require(
        expected_session == 20 and set(counts) == set(range(20)) and max(counts.values()) <= 4,
        "full 20-session coverage within four requests per session required",
    )
    running = 0.0
    for row in reversed(batch):
        running = math.fsum((row["training_reward"], running))
        row["return_G"] = running
    invalid_total = sum(row["invalid"] for row in batch)
    require(
        math.isclose(
            batch[0]["return_G"], score - PENALTY * invalid_total, rel_tol=1e-12, abs_tol=1e-12
        ),
        "suffix return must retain every actual invalid penalty",
    )
    return batch, dict(
        episode_id=trajectory["episode_id"],
        status="COMPLETE",
        training_rows=total,
        invalid_rows=invalid_total,
        terminal_financial_Y=score,
        total_training_reward=batch[0]["return_G"],
        session_coverage=[
            dict(session=j, requests=counts[j], invalid_requests=invalid_counts[j])
            for j in range(20)
        ],
    )


def build_batch(directory: Path) -> tuple[list[dict], dict]:
    collection = json.loads((directory / "collection_summary.json").read_text())
    episodes = collection["episode_ids"]
    require(
        isinstance(episodes, list) and len(episodes) == len(set(episodes)) == 8,
        "eight predeclared unique Train windows required",
    )
    parents = {}
    # Head, EST and REF records are never labels or actor examples here.
    for trajectory in records(directory / "trajectory_results.jsonl"):
        if trajectory.get("role") != "parent":
            continue
        episode = trajectory["episode_id"]
        require(episode in episodes and episode not in parents, "unexpected/duplicate parent")
        parents[episode] = trajectory
    batch, coverage = [], []
    for episode in episodes:
        parent = parents.get(episode)
        if parent is None or parent.get("status") != "COMPLETE":
            coverage.append(
                dict(
                    episode_id=episode,
                    status="NA",
                    training_rows=0,
                    reason="MISSING" if parent is None else parent.get("status"),
                )
            )
            continue
        rows, detail = trajectory_rows(parent)
        batch.extend(rows)
        coverage.append(detail)
    complete = sum(row["status"] == "COMPLETE" for row in coverage)
    return batch, dict(
        episode_ids=episodes,
        planned_trajectories=8,
        complete_trajectories=complete,
        missing_or_incomplete_trajectories=8 - complete,
        training_rows=len(batch),
        invalid_rows=sum(row["invalid"] for row in batch),
        trajectories=coverage,
        source_collection_model_requests=collection.get("model_requests"),
        head_EST_REF_used=False,
    )


def run(args: argparse.Namespace) -> None:
    require(args.arm == "S0" and args.seed == 42, "fixed S0 seed42 demo")
    require(args.alpha == 0.5 and args.eta == 0.25, "inherited alpha/eta fixed; S is zero")
    require(
        args.actor_learning_rate == 2e-6 and args.critic_learning_rate == 1e-6,
        "fixed actor/critic learning rates",
    )
    retire_paths = retirement_inputs(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "training_summary.json").exists(), "fresh round; no retry/resume")
    require(
        not (output / "actor").exists() and not (output / "critic").exists(),
        "fresh actor/critic export required",
    )
    batch, coverage = build_batch(args.data.resolve())
    summary = dict(
        profile=PROFILE,
        status="STARTING",
        arm="S0",
        seed=args.seed,
        gamma=1.0,
        lam=1.0,
        alpha=args.alpha,
        eta=args.eta,
        residual_S=0.0,
        invalid_penalty=PENALTY,
        coverage=coverage,
        planned_trajectories=8,
        complete_trajectories=coverage["complete_trajectories"],
        missing_or_incomplete_trajectories=coverage["missing_or_incomplete_trajectories"],
        training_rows=len(batch),
        invalid_rows=coverage["invalid_rows"],
        actor_updates=0,
        critic_updates=0,
        REF_used=False,
        head_EST_used=False,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        actor_scope="every actual request of complete parent trajectories, including invalid",
        critic_scope="one pre-action scalar value loss per actual request",
        actor_value_timing="frozen critic values before the current critic update",
        optimizer_state_reset_each_round=True,
        performance_claim=False,
        objective="single accumulated Monte Carlo/GAE update, not repeated PPO epochs",
        retired_input_checkpoints=[],
    )
    write(output / "training_summary.json", summary)
    try:
        require(len(batch) > 1, "at least one complete actual parent required; no missing fill")
        os.environ.update(
            USE_TORCH="1",
            USE_TF="0",
            USE_FLAX="0",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
        )
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=False
        )
        common = dict(
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        critic = AutoModelForSequenceClassification.from_pretrained(
            args.critic or args.model, num_labels=1, **common
        )
        retire_loaded_input("critic", retire_paths, critic, torch, output, summary)
        initial_critic = probes(critic)
        critic.eval()
        with torch.no_grad():
            values = [float(scalar_value(critic, row, torch)) for row in batch]
        for row, value in zip(batch, values, strict=True):
            row["value_before_critic_update"] = value
            row["U"] = row["return_G"] - value
        unwhitened = torch.tensor([row["U"] for row in batch], dtype=torch.float32)
        whitened = (unwhitened - unwhitened.mean()) / (unwhitened.std(correction=1) + 1e-8)
        require(bool(torch.isfinite(whitened).all()), "finite whole-batch whitening required")
        for row, advantage in zip(batch, whitened.tolist(), strict=True):
            row["actor_advantage"] = advantage
        with (output / "request_advantages.jsonl").open("x") as stream:
            for row in batch:
                append(stream, row)
        summary.update(
            status="CRITIC_UPDATE",
            whitening_mean=float(unwhitened.mean()),
            whitening_std_sample=float(unwhitened.std(correction=1)),
        )
        write(output / "training_summary.json", summary)
        before_mse = math.fsum(
            (v - row["return_G"]) ** 2 for v, row in zip(values, batch, strict=True)
        ) / len(batch)
        critic_update = adam_update(
            critic,
            batch,
            lambda row, model=critic: (
                0.5 * (scalar_value(model, row, torch) - row["return_G"]) ** 2
            ),
            args.critic_learning_rate,
            torch,
        )
        with torch.no_grad():
            after_values = [float(scalar_value(critic, row, torch)) for row in batch]
        after_mse = math.fsum(
            (v - row["return_G"]) ** 2 for v, row in zip(after_values, batch, strict=True)
        ) / len(batch)
        critic_update.update(
            before_request_MSE=before_mse,
            after_request_MSE=after_mse,
            BF16_parameter_delta=probe_delta(initial_critic, probes(critic)),
        )
        critic.save_pretrained(output / "critic", safe_serialization=True, max_shard_size="4GB")
        critic_update["checkpoint_readback"] = checkpoint_probes(output / "critic", probes(critic))
        del critic
        gc.collect()
        torch.cuda.empty_cache()
        summary.update(
            status="ACTOR_UPDATE",
            critic_updates=1,
            critic=critic_update,
            critic_mse_before=before_mse,
            critic_mse_after=after_mse,
            critic_checkpoint=str(output / "critic"),
        )
        write(output / "training_summary.json", summary)
        actor = AutoModelForCausalLM.from_pretrained(args.model, **common)
        retire_loaded_input("actor", retire_paths, actor, torch, output, summary)
        initial_actor = probes(actor)
        actor_update = adam_update(
            actor,
            batch,
            lambda row: -row["actor_advantage"] * response_logprob(actor, row, torch),
            args.actor_learning_rate,
            torch,
        )
        actor_update["BF16_parameter_delta"] = probe_delta(initial_actor, probes(actor))
        actor.save_pretrained(output / "actor", safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(output / "actor")
        actor_update["checkpoint_readback"] = checkpoint_probes(output / "actor", probes(actor))
        summary.update(
            status="COMPLETE",
            actor_updates=1,
            actor=actor_update,
            actor_checkpoint=str(output / "actor"),
        )
        write(output / "training_summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    except BaseException as exc:
        summary.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        write(output / "training_summary.json", summary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--critic", type=Path)
    parser.add_argument("--arm", choices=("S0",), default="S0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--actor-learning-rate", type=float, default=2e-6)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-6)
    parser.add_argument("--retire-input-checkpoints", action="store_true")
    started = time.monotonic()
    run(parser.parse_args())
    print(f"training round elapsed_seconds={time.monotonic() - started:.3f}", flush=True)


if __name__ == "__main__":
    main()

"""Tokenizer-level context-budget checks for the frozen G1 smoke trajectory."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from agent_r1.env.base import Action, Observation

from src.env.trading import TradingEnv


class ChatTemplateTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, object]],
        *,
        tools: Sequence[Mapping[str, object]],
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> object: ...


def _messages(value: object) -> list[dict[str, object]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise RuntimeError("processed prompt must be a list of chat messages")
    return [dict(item) for item in value]


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise RuntimeError(f"processed {name} must be a mapping")
    return dict(value)


def _token_count(
    tokenizer: ChatTemplateTokenizer,
    observation: Observation,
    *,
    tools: Sequence[Mapping[str, object]],
) -> int:
    if observation.messages is None:
        raise RuntimeError("trading prompt-budget audit requires chat messages")
    encoded = tokenizer.apply_chat_template(
        observation.messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=True,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if not isinstance(encoded, Sequence):
        raise RuntimeError("tokenizer chat template did not return token ids")
    return len(encoded)


def _environment_from_row(
    row: pd.Series, *, repo_root: Path
) -> tuple[str, TradingEnv, list[dict[str, object]]]:
    env_kwargs = _mapping(row["env_kwargs"], name="env_kwargs")
    agent_name = str(row["agent_name"])
    env_type = env_kwargs.pop("env_type", agent_name)
    if agent_name != "trading" or env_type != "trading":
        raise RuntimeError(f"prompt-budget audit requires trading env, got {env_type!r}")
    for key in ("manifest_root", "storage_root"):
        path = Path(str(env_kwargs[key])).expanduser()
        if not path.is_absolute():
            env_kwargs[key] = str((repo_root / path).resolve())
    prompt = _messages(row["prompt"])
    extra_info = _mapping(row["extra_info"], name="extra_info")
    episode_id = str(extra_info.get("episode_id") or row.name)
    return episode_id, TradingEnv(raw_prompt=prompt, **env_kwargs), prompt


async def _measure_episode(
    *,
    tokenizer: ChatTemplateTokenizer,
    env: TradingEnv,
    raw_prompt: list[dict[str, object]],
) -> list[dict[str, int | str]]:
    """Measure two reachable paths: every-session hold and maximum distinct positions."""

    tools = env.tool_schemas
    retrieve = Action(
        text='<tool_call>{"name":"retrieve","arguments":{"query":":","limit":10}}</tool_call>'
    )
    hold = Action(text='<tool_call>{"name":"hold","arguments":{}}</tool_call>')
    measurements: list[dict[str, int | str]] = []

    async def measure_path(*, build_positions: bool) -> None:
        path = "positions" if build_positions else "hold"
        observation = env.reset(raw_prompt=raw_prompt)
        done = False
        session = 0
        symbols = tuple(env.symbols)
        target_weight = min(0.8, 0.8 / len(symbols))
        while not done:
            measurements.append(
                {
                    "stage": f"{path}_session_{session}_state",
                    "prompt_tokens": _token_count(tokenizer, observation, tools=tools),
                }
            )
            observation, _, retrieve_done, retrieve_info = await env.step(retrieve)
            if retrieve_done or retrieve_info.get("invalid_action"):
                raise RuntimeError("maximum retrieve unexpectedly failed or terminated the episode")
            measurements.append(
                {
                    "stage": f"{path}_session_{session}_after_max_retrieve",
                    "prompt_tokens": _token_count(tokenizer, observation, tools=tools),
                }
            )

            if build_positions and session < len(symbols):
                action = Action(
                    text=(
                        '<tool_call>{"name":"buy","arguments":'
                        f'{{"symbol":"{symbols[session]}","target_weight":{target_weight}}}'
                        "}</tool_call>"
                    )
                )
            else:
                action = hold
            observation, _, done, info = await env.step(action)
            if info.get("invalid_action"):
                raise RuntimeError(
                    f"prompt-budget {path} trajectory produced an invalid advance: {info}"
                )
            session += 1

    await measure_path(build_positions=False)
    await measure_path(build_positions=True)
    return measurements


def verify_smoke_prompt_budget(
    *,
    train_file: str | Path,
    repo_root: str | Path,
    tokenizer: ChatTemplateTokenizer,
    max_prompt_tokens: int,
    max_response_tokens: int,
    model_context_tokens: int,
) -> dict[str, object]:
    """Exercise every smoke train episode and fail before GPU training on overflow."""

    for name, value in (
        ("max_prompt_tokens", max_prompt_tokens),
        ("max_response_tokens", max_response_tokens),
        ("model_context_tokens", model_context_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    root = Path(repo_root).expanduser().resolve()
    path = Path(train_file).expanduser()
    if not path.is_absolute():
        path = root / path
    frame = pd.read_parquet(path)
    if frame.empty:
        raise RuntimeError("prompt-budget audit requires at least one train episode")

    episodes: dict[str, list[dict[str, int | str]]] = {}
    for _, row in frame.iterrows():
        episode_id, env, prompt = _environment_from_row(row, repo_root=root)
        if episode_id in episodes:
            raise RuntimeError(
                f"duplicate processed episode_id in prompt-budget audit: {episode_id}"
            )
        measurements = asyncio.run(
            _measure_episode(tokenizer=tokenizer, env=env, raw_prompt=prompt)
        )
        for measurement in measurements:
            count = measurement["prompt_tokens"]
            assert isinstance(count, int)
            stage = str(measurement["stage"])
            if count > max_prompt_tokens:
                raise RuntimeError(
                    f"prompt budget exceeded for {episode_id} at {stage}: "
                    f"{count} > {max_prompt_tokens}"
                )
            if count + max_response_tokens > model_context_tokens:
                raise RuntimeError(
                    f"model context exceeded for {episode_id} at {stage}: "
                    f"{count} + {max_response_tokens} > {model_context_tokens}"
                )
        episodes[episode_id] = measurements

    maximum = max(
        int(measurement["prompt_tokens"])
        for measurements in episodes.values()
        for measurement in measurements
    )
    return {
        "status": "PROMPT_BUDGET_VERIFIED",
        "episode_count": len(episodes),
        "max_observed_prompt_tokens": maximum,
        "max_prompt_tokens": max_prompt_tokens,
        "max_response_tokens": max_response_tokens,
        "model_context_tokens": model_context_tokens,
        "measurements": episodes,
    }

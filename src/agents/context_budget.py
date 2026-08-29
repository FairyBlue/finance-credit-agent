"""Tokenized prompt-budget audit for the bounded G1 trading trajectory."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


class ChatTemplateTokenizer(Protocol):
    """The tokenizer surface used by Agent-R1's non-multimodal flow."""

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> Sequence[int]: ...


@dataclass(frozen=True)
class PromptBudgetCase:
    """One exact environment observation as seen by the model."""

    label: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]


@dataclass(frozen=True)
class PromptBudgetScenario:
    """A legal retrieve pattern that the tokenizer gate must cover."""

    label: str
    retrieve_queries: tuple[str, ...]


@dataclass(frozen=True)
class PromptBudgetAudit:
    """Serializable evidence that every audited prompt fits both limits."""

    cases_checked: int
    max_case: str
    max_prompt_tokens: int
    model_context_tokens: int
    prompt_limit: int
    response_limit: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def verify_tokenized_prompt_budget(
    *,
    tokenizer: ChatTemplateTokenizer,
    cases: Sequence[PromptBudgetCase],
    prompt_limit: int,
    response_limit: int,
    model_context_tokens: int,
) -> PromptBudgetAudit:
    """Apply the real chat template and reject prompts that exceed either budget."""

    if prompt_limit <= 0 or response_limit <= 0 or model_context_tokens <= 0:
        raise ValueError("prompt, response, and model-context limits must be positive")
    if not cases:
        raise RuntimeError("prompt-budget audit received no cases")

    counts: list[tuple[str, int]] = []
    for case in cases:
        token_ids = tokenizer.apply_chat_template(
            case.messages,
            tools=case.tools,
            add_generation_prompt=True,
            tokenize=True,
        )
        count = len(token_ids)
        counts.append((case.label, count))
        if count > prompt_limit:
            raise RuntimeError(
                f"{case.label} uses {count} tokens, exceeding configured prompt limit "
                f"{prompt_limit}"
            )
        if count + response_limit > model_context_tokens:
            raise RuntimeError(
                f"{case.label} prompt plus reserved response uses "
                f"{count + response_limit} tokens, exceeding model context limit "
                f"{model_context_tokens}"
            )

    max_case, max_prompt_tokens = max(counts, key=lambda item: item[1])
    return PromptBudgetAudit(
        cases_checked=len(counts),
        max_case=max_case,
        max_prompt_tokens=max_prompt_tokens,
        model_context_tokens=model_context_tokens,
        prompt_limit=prompt_limit,
        response_limit=response_limit,
    )


def build_prompt_budget_scenarios(
    *,
    advancing_actions: int,
    max_query_chars: int,
    max_agent_steps: int,
    reserved_information_steps: int,
) -> tuple[PromptBudgetScenario, ...]:
    """Cover both the planned path and the legal three-retrieve Unicode boundary."""

    if (
        advancing_actions <= 0
        or max_query_chars <= 0
        or max_agent_steps <= 0
        or reserved_information_steps < 0
    ):
        raise ValueError("prompt-budget scenario limits are invalid")
    max_uniform_retrieves = max_agent_steps // advancing_actions - 1
    if max_uniform_retrieves < reserved_information_steps:
        raise RuntimeError("planned information steps exceed max_agent_steps")
    unicode_retrieves = min(3, max_uniform_retrieves)
    if unicode_retrieves < 3:
        raise RuntimeError("G1 no longer permits the required three-retrieve budget path")
    unicode_query = "💹" * max_query_chars
    return (
        PromptBudgetScenario(
            label="planned-ascii",
            retrieve_queries=("market",) * reserved_information_steps,
        ),
        PromptBudgetScenario(
            label="legal-unicode-max",
            retrieve_queries=(unicode_query,) * unicode_retrieves,
        ),
    )


def _required(config: dict[str, Any], *path: str) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing config value: {'.'.join(path)}")
        value = value[key]
    return value


def _normalize_messages(value: object, *, label: str) -> list[dict[str, Any]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{label} prompt is not a non-empty message list")
    messages: list[dict[str, Any]] = []
    for message in value:
        if not isinstance(message, dict):
            raise RuntimeError(f"{label} prompt contains a non-mapping message")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise RuntimeError(f"{label} prompt message lacks string role/content")
        messages.append({"role": role, "content": content})
    return messages


def _model_context_tokens(model_config: object) -> int:
    value = getattr(model_config, "max_position_embeddings", None)
    if value is None:
        text_config = getattr(model_config, "text_config", None)
        value = getattr(text_config, "max_position_embeddings", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError("cached model config does not expose a positive context limit")
    return value


def _tool_action(name: str, arguments: dict[str, object] | None = None) -> str:
    payload = {"arguments": arguments or {}, "name": name}
    return f"<tool_call>{json.dumps(payload, separators=(',', ':'), sort_keys=True)}</tool_call>"


async def _collect_episode_cases(
    *,
    label: str,
    raw_prompt: list[dict[str, Any]],
    env_kwargs: dict[str, Any],
    reward_mode: str,
    scenario: PromptBudgetScenario,
) -> list[PromptBudgetCase]:
    from agent_r1.env.base import Action

    from src.env.trading import TradingEnv

    env = TradingEnv(reward_mode=reward_mode, **env_kwargs)
    observation = env.reset(raw_prompt=raw_prompt)
    if observation.messages is None:
        raise RuntimeError(f"{label} reset did not produce chat messages")
    cases = [
        PromptBudgetCase(
            label=f"{label}/{scenario.label}/reset",
            messages=observation.messages,
            tools=env.tool_schemas,
        )
    ]

    advancing_actions = len(env_kwargs.get("episode_dates", [])) - 1
    if advancing_actions < 1:
        raise RuntimeError(f"{label} has no advancing actions")
    for session_index in range(advancing_actions):
        for retrieval_index, query in enumerate(scenario.retrieve_queries):
            observation, _, done, _ = await env.step(
                Action(
                    text=_tool_action(
                        "retrieve",
                        {
                            "limit": TradingEnv.MAX_RETRIEVE_RESULTS,
                            "query": query,
                        },
                    )
                )
            )
            if done or observation.messages is None:
                raise RuntimeError(f"{label} terminated during bounded retrieval audit")
            cases.append(
                PromptBudgetCase(
                    label=(
                        f"{label}/{scenario.label}/session-{session_index}/"
                        f"retrieve-{retrieval_index}"
                    ),
                    messages=observation.messages,
                    tools=env.tool_schemas,
                )
            )

        observation, _, done, _ = await env.step(Action(text=_tool_action("hold")))
        expected_done = session_index == advancing_actions - 1
        if done is not expected_done or observation.messages is None:
            raise RuntimeError(f"{label} terminal boundary changed during prompt audit")
        cases.append(
            PromptBudgetCase(
                label=f"{label}/{scenario.label}/session-{session_index}/hold",
                messages=observation.messages,
                tools=env.tool_schemas,
            )
        )
    return cases


def verify_smoke_context_budget(
    config: dict[str, Any],
    *,
    runtime: dict[str, object],
    repo_root: str | Path,
) -> dict[str, object]:
    """Audit every G1 train candidate using cached model files and the real tokenizer."""

    import pandas as pd
    from transformers import AutoConfig, AutoTokenizer  # type: ignore[import-not-found]

    root = Path(repo_root).expanduser().resolve()
    agent_path = root / str(_required(config, "framework", "agent_r1_path"))
    if str(agent_path) not in sys.path:
        sys.path.insert(0, str(agent_path))
    from src.env.trading import TradingEnv

    snapshot_value = runtime.get("model_snapshot_path")
    if not isinstance(snapshot_value, str) or not snapshot_value:
        raise RuntimeError("preflight runtime evidence lacks a resolved model snapshot path")
    snapshot = Path(snapshot_value).expanduser().resolve()
    if not snapshot.is_dir():
        raise RuntimeError(f"resolved model snapshot is unavailable: {snapshot}")

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model_config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    model_context_tokens = _model_context_tokens(model_config)

    train_path = root / str(_required(config, "data", "processed", "train_file"))
    frame = pd.read_parquet(train_path)
    expected_episodes = int(_required(config, "training", "episodes"))
    if len(frame) != expected_episodes:
        raise RuntimeError(
            f"prompt-budget train candidate count {len(frame)} does not match "
            f"training.episodes={expected_episodes}"
        )

    reward_modes = {
        "A1": "outcome_only",
        "A2": "generic_step",
        "A3": "finance_trajectory",
    }
    arm_id = str(_required(config, "arm", "id"))
    if arm_id not in reward_modes:
        raise RuntimeError(f"unsupported arm for prompt-budget audit: {arm_id}")
    information_steps = int(_required(config, "training", "reserved_information_steps_per_session"))
    max_agent_steps = int(_required(config, "training", "max_agent_steps"))

    cases: list[PromptBudgetCase] = []
    checked_scenarios: set[str] = set()
    for index, row in frame.iterrows():
        label = f"train-{index}"
        raw_prompt = _normalize_messages(row["prompt"], label=label)
        try:
            env_kwargs = json.loads(row["env_kwargs"])
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"{label} env_kwargs is not valid JSON") from error
        if not isinstance(env_kwargs, dict):
            raise RuntimeError(f"{label} env_kwargs is not a mapping")
        advancing_actions = len(env_kwargs.get("episode_dates", [])) - 1
        scenarios = build_prompt_budget_scenarios(
            advancing_actions=advancing_actions,
            max_query_chars=TradingEnv.MAX_RETRIEVE_QUERY_CHARS,
            max_agent_steps=max_agent_steps,
            reserved_information_steps=information_steps,
        )
        for scenario in scenarios:
            checked_scenarios.add(scenario.label)
            cases.extend(
                asyncio.run(
                    _collect_episode_cases(
                        label=label,
                        raw_prompt=raw_prompt,
                        env_kwargs=env_kwargs,
                        reward_mode=reward_modes[arm_id],
                        scenario=scenario,
                    )
                )
            )

    audit = verify_tokenized_prompt_budget(
        tokenizer=tokenizer,
        cases=cases,
        prompt_limit=int(_required(config, "training", "max_prompt_length")),
        response_limit=int(_required(config, "training", "max_response_length")),
        model_context_tokens=model_context_tokens,
    )
    return {
        **audit.as_dict(),
        "episodes_checked": len(frame),
        "model_snapshot_path": str(snapshot),
        "scenarios_checked": sorted(checked_scenarios),
        "tokenizer_class": type(tokenizer).__name__,
    }

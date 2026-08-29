"""Prompt budgeting must fail closed before a G1 rollout reaches the model."""

from __future__ import annotations

from typing import Any

import pytest

from src.agents.context_budget import PromptBudgetCase, verify_tokenized_prompt_budget


class _CountingTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> list[int]:
        assert add_generation_prompt is True
        assert tokenize is True
        characters = sum(len(str(message["content"])) for message in messages)
        return [1] * (characters + len(tools) * 10)


def test_tokenized_prompt_budget_passes_and_fails_closed_at_both_limits() -> None:
    tokenizer = _CountingTokenizer()
    case = PromptBudgetCase(
        label="smoke-train-001/reset",
        messages=[{"role": "user", "content": "x" * 80}],
        tools=[{"name": "hold"}],
    )

    audit = verify_tokenized_prompt_budget(
        tokenizer=tokenizer,
        cases=[case],
        prompt_limit=100,
        response_limit=20,
        model_context_tokens=120,
    )
    assert audit.max_prompt_tokens == 90
    assert audit.max_case == case.label

    with pytest.raises(RuntimeError, match="configured prompt limit"):
        verify_tokenized_prompt_budget(
            tokenizer=tokenizer,
            cases=[case],
            prompt_limit=89,
            response_limit=20,
            model_context_tokens=120,
        )

    with pytest.raises(RuntimeError, match="model context limit"):
        verify_tokenized_prompt_budget(
            tokenizer=tokenizer,
            cases=[case],
            prompt_limit=100,
            response_limit=31,
            model_context_tokens=120,
        )

"""External Agent-R1 flows; imported only inside the GPU training runtime."""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from agent_r1.agent_flow.agent_env_loop import AgentEnvLoop

# Importing TradingEnv registers `env_type: trading` without touching upstream files.
from src.env.observation_memory import validate_observation_profile
from src.env.trading import TradingEnv as _TradingEnv
from src.rewards import STEP_CREDIT_OUTPUT_SEMANTICS, StepContext, build_credit_strategy

from .credit_evidence import record_step_credit_evidence
from .rollout_seed import (
    ROLLOUT_IDENTITY_KWARG,
    ROLLOUT_SEED_SCHEME,
    build_seeded_server_manager,
    normalize_rollout_identity,
)
from .screening_prompt import (
    action_protocol_sha256,
    apply_screening_action_protocol,
    validate_action_protocol,
)

SCREENING_EVIDENCE_DIR_ENV = "FINANCE_SCREENING_EVIDENCE_DIR"
_MAX_SCREENING_FAILURE_TOKEN_IDS = 4096
_MAX_SCREENING_PROMPT_TOKEN_IDS = 8192


class ScreeningIncompleteError(RuntimeError):
    """A rollout cannot enter the screening training or evaluation population."""


class ScreeningResponseManager:
    """Enforce an explicit response cap before upstream silently slices tokens.

    Pinned verl folds vLLM's ``length`` into ``completed``. Consequently any
    response exhausting its cap is rejected, even when its last token is EOS.
    This conservative boundary rule is identical for all screening arms.
    """

    def __init__(
        self,
        manager: Any,
        *,
        prompt_limit: int,
        response_limit: int,
        context_limit: int,
        capture_token_ids: bool = False,
    ) -> None:
        if min(prompt_limit, response_limit, context_limit) <= 0:
            raise ValueError("screening generation limits must be positive")
        if not isinstance(capture_token_ids, bool):
            raise ValueError("capture_token_ids must be a boolean")
        self._manager = manager
        self.prompt_limit = prompt_limit
        self.response_limit = response_limit
        self.context_limit = context_limit
        self.capture_token_ids = capture_token_ids
        self.requests: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        record: dict[str, Any] = {
            "request_index": len(self.requests),
            "request_seed": sampling_params.get("seed"),
            "prompt_token_count": len(prompt_ids),
            "prompt_sha256": sha256(json.dumps(prompt_ids).encode("ascii")).hexdigest(),
            "response_token_count": 0,
            "response_cap": self.response_limit,
            "stop_reason": None,
            "finish_reason": None,
            "status": "STARTED",
        }
        self.requests.append(record)
        if self.capture_token_ids:
            record["manager_request_id"] = request_id
            record["prompt_token_ids"] = list(prompt_ids[:_MAX_SCREENING_PROMPT_TOKEN_IDS])
            record["prompt_token_ids_truncated"] = (
                len(prompt_ids) > _MAX_SCREENING_PROMPT_TOKEN_IDS
            )
        if len(prompt_ids) > self.prompt_limit:
            record["status"] = "CONTEXT_OVERFLOW"
            raise ScreeningIncompleteError("screening prompt exceeds its frozen limit")
        if len(prompt_ids) + self.response_limit > self.context_limit:
            record["status"] = "CONTEXT_OVERFLOW"
            raise ScreeningIncompleteError("screening context cannot fit the full response budget")
        params = dict(sampling_params)
        for key in ("max_tokens", "max_new_tokens"):
            if key in params and params.pop(key) != self.response_limit:
                raise RuntimeError("screening response cap differs from the resolved contract")
        params["max_tokens"] = self.response_limit
        if self.capture_token_ids:
            # Capture exactly what this boundary forwards. Downstream verl may
            # translate keys or supply defaults; this is not an engine receipt.
            record["forwarded_sampling_params"] = deepcopy(params)
        try:
            output = await self._manager.generate(
                request_id, prompt_ids=prompt_ids, sampling_params=params, **kwargs
            )
        except Exception:
            record["status"] = "GENERATION_ERROR"
            raise
        tokens = list(output.token_ids)
        stop_reason = getattr(output, "stop_reason", None)
        finish_reason = getattr(output, "finish_reason", None)
        record.update(
            {
                "response_token_count": len(tokens),
                "response_sha256": sha256(json.dumps(tokens).encode("ascii")).hexdigest(),
                "stop_reason": stop_reason,
                "finish_reason": finish_reason,
            }
        )
        if self.capture_token_ids:
            diagnostic_limit = min(self.response_limit, _MAX_SCREENING_FAILURE_TOKEN_IDS)
            record["response_token_ids"] = tokens[:diagnostic_limit]
            record["response_token_ids_truncated"] = len(tokens) > diagnostic_limit
            record["response_last_token_id"] = tokens[-1] if tokens else None
        if (
            len(tokens) >= self.response_limit
            or stop_reason in {"length", "abort", "aborted"}
            or finish_reason in {"length", "abort", "aborted"}
        ):
            record["status"] = "RESPONSE_INCOMPLETE"
            diagnostic_limit = min(self.response_limit, _MAX_SCREENING_FAILURE_TOKEN_IDS)
            record["response_token_ids"] = tokens[:diagnostic_limit]
            record["response_token_ids_truncated"] = len(tokens) > diagnostic_limit
            # Preserve the actual boundary even if an upstream overrun exceeds
            # the bounded diagnostic prefix. Successes are hash-only by default.
            record["response_last_token_id"] = tokens[-1] if tokens else None
            raise ScreeningIncompleteError("screening response truncated or aborted")
        if (
            not tokens
            or stop_reason not in {"completed", "stop"}
            or finish_reason not in {None, "stop"}
        ):
            record["status"] = "UNKNOWN_COMPLETION"
            raise ScreeningIncompleteError("screening response has no confirmed completion reason")
        record["status"] = "COMPLETE"
        return output


class TradingAgentEnvLoop(AgentEnvLoop):
    """Agent-R1's generic step-MDP loop with the external trading env registered."""

    server_manager: Any
    _active_trading_env: _TradingEnv

    def _create_env(self, **kwargs: Any) -> _TradingEnv:
        env = super()._create_env(**kwargs)
        if not isinstance(env, _TradingEnv):
            raise RuntimeError("trading flow resolved a non-trading environment")
        self._active_trading_env = env
        return env

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any):
        if ROLLOUT_IDENTITY_KWARG not in kwargs:
            raise RuntimeError(
                "deterministic rollout identity was not supplied by the runtime hook"
            )
        identity = normalize_rollout_identity(kwargs.pop(ROLLOUT_IDENTITY_KWARG))
        original_manager = self.server_manager
        seeded_manager = build_seeded_server_manager(
            original_manager,
            base_seed=self.config.data.seed,
            trajectory=identity,
        )
        self.server_manager = seeded_manager
        try:
            output = await super().run(sampling_params, **kwargs)
        finally:
            self.server_manager = original_manager
        if len(output.steps) != len(seeded_manager.request_seeds):
            raise RuntimeError(
                "rollout seed evidence count does not match generated environment steps"
            )
        for step, request_seed in zip(output.steps, seeded_manager.request_seeds, strict=True):
            step.extra_fields["finance_rollout_seed"] = request_seed
            step.extra_fields["finance_rollout_seed_scheme"] = ROLLOUT_SEED_SCHEME
        env = getattr(self, "_active_trading_env", None)
        if not isinstance(env, _TradingEnv):
            raise RuntimeError("trading rollout did not retain its environment evidence")
        trace = env.credit_step_trace
        if len(output.steps) != len(trace):
            raise RuntimeError("trading step trace count does not match generated steps")
        for step, record in zip(output.steps, trace, strict=True):
            step.extra_fields["trading_action"] = record.action
            step.extra_fields["trading_decision_session_index"] = record.decision_session_index
            step.extra_fields["trading_decision_at"] = record.decision_at
            step.extra_fields["trading_result_session_index"] = record.result_session_index
            step.extra_fields["trading_env_reward"] = record.reward
            step.extra_fields["trading_env_done"] = record.done
            step.extra_fields["trading_invalid_action"] = record.invalid_action
            step.extra_fields["trading_episode_complete"] = env.episode_complete
        return output

    async def _obs_to_prompt(
        self, obs: Any, tools: list[dict[str, Any]] | None = None
    ) -> list[int]:
        prompt_ids = await super()._obs_to_prompt(obs, tools=tools)
        if len(prompt_ids) > self.prompt_length:
            raise RuntimeError(
                f"tokenized trading prompt uses {len(prompt_ids)} tokens, exceeding "
                f"configured prompt limit {self.prompt_length}; rollout aborted"
            )
        return prompt_ids


class FinanceCreditAgentEnvLoop(TradingAgentEnvLoop):
    """A3 loop that retroactively allocates the final finance trajectory signal."""

    def __init__(
        self,
        *args: Any,
        credit_strategy: str = "m1_uniform",
        credit_semantics: str = STEP_CREDIT_OUTPUT_SEMANTICS,
        **kwargs: Any,
    ) -> None:
        self._credit_strategy = build_credit_strategy(credit_strategy)
        if credit_semantics != self._credit_strategy.output_semantics:
            raise ValueError(
                "configured credit semantics do not match the selected strategy: "
                f"{credit_semantics!r} != {self._credit_strategy.output_semantics!r}"
            )
        super().__init__(*args, **kwargs)

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any):
        output = await super().run(sampling_params, **kwargs)
        if not output.steps:
            raise RuntimeError("finance credit requires at least one completed environment step")
        env = self._active_trading_env
        if env.reward_mode != "finance_trajectory":
            raise RuntimeError("finance credit requires reward_mode=finance_trajectory")
        if not env.episode_complete:
            raise RuntimeError(
                "finance credit requires a completed trajectory; rollout stopped before A_fin"
            )
        trace = env.credit_step_trace
        if not trace[-1].done or any(record.done for record in trace[:-1]):
            raise RuntimeError("finance trajectory has an invalid terminal-step trace")
        observed_rewards = tuple(float(step.reward_score or 0.0) for step in output.steps)
        trace_rewards = tuple(record.reward for record in trace)
        if observed_rewards != trace_rewards:
            raise RuntimeError("Agent-R1 rewards do not match retained trading evidence")
        trajectory_advantage = math.fsum(observed_rewards)
        contexts = [
            StepContext(
                step_id=f"step-{index}",
                action=record.action,
                metadata={
                    "decision_at": record.decision_at,
                    "decision_session_index": record.decision_session_index,
                    "invalid_action": record.invalid_action,
                    "result_session_index": record.result_session_index,
                },
            )
            for index, record in enumerate(trace)
        ]
        credits = self._credit_strategy.allocate(
            trajectory_advantage=trajectory_advantage,
            steps=contexts,
        )
        if sum(credit.credit for credit in credits) != trajectory_advantage:
            raise RuntimeError("step-credit allocation did not conserve A_fin exactly")
        for step, credit in zip(output.steps, credits, strict=True):
            step.reward_score = credit.credit
            step.extra_fields["finance_credit"] = credit.credit
            step.extra_fields["finance_credit_mechanism"] = credit.mechanism
            step.extra_fields["finance_credit_output_semantics"] = credit.output_semantics
            step.extra_fields["finance_credit_step_count"] = len(credits)
            step.extra_fields["finance_trajectory_advantage"] = trajectory_advantage
            step.extra_fields["finance_trajectory_complete"] = True
        record_step_credit_evidence(
            identity=normalize_rollout_identity(kwargs[ROLLOUT_IDENTITY_KWARG]),
            base_seed=self.config.data.seed,
            mechanism=self._credit_strategy.name,
            output_semantics=self._credit_strategy.output_semantics,
            trajectory_advantage=trajectory_advantage,
            credits=[float(step.reward_score) for step in output.steps],
            actions=[context.action for context in contexts],
        )
        return output


class ScreeningTradingAgentEnvLoop(TradingAgentEnvLoop):
    """Four-arm Train-only screening with a shared scorer and completion gate."""

    def __init__(self, *args: Any, screening_arm: str, **kwargs: Any) -> None:
        if screening_arm not in {"A2", "M1", "M2", "M2-plain"}:
            raise ValueError("unknown screening arm")
        self._screening_arm = screening_arm
        self._credit_strategy = build_credit_strategy("m1_uniform")
        kwargs["screening"] = True
        kwargs["finance_scorer_id"] = "benchmark_scale_v2"
        kwargs["reward_mode"] = "outcome_only" if screening_arm == "A2" else "finance_trajectory"
        super().__init__(*args, **kwargs)
        self._action_protocol = validate_action_protocol(
            self.config.data.get("screening_action_protocol", "legacy")
        )
        self._capture_interactions = self.config.data.get("screening_capture_interactions", False)
        if not isinstance(self._capture_interactions, bool):
            raise ValueError("screening_capture_interactions must be a boolean")
        self._observation_profile = validate_observation_profile(
            self.config.data.get("screening_observation_profile", "legacy")
        )

    def _create_env(self, **kwargs: Any) -> _TradingEnv:
        supplied = kwargs.get("env_kwargs", {})
        if isinstance(supplied, str):
            supplied = json.loads(supplied)
        row = dict(supplied)
        for key in (
            "screening",
            "finance_scorer_id",
            "finance_volatility_floor",
            "risk_lambda",
            "turnover_limit",
            "position_limit",
            "screening_calibration_sha256",
        ):
            if key in self.env_kwargs and key in row and self.env_kwargs[key] != row[key]:
                raise RuntimeError(f"screening row overrides frozen {key}")
        # Reward placement belongs to the arm, while all sampled rows share the
        # same source/scorer configuration and model-visible prompt schema.
        row["reward_mode"] = self.env_kwargs["reward_mode"]
        row["capture_interactions"] = self._capture_interactions
        if (
            "observation_profile" in row
            and row["observation_profile"] != self._observation_profile
        ):
            raise RuntimeError("screening row overrides frozen observation_profile")
        row["observation_profile"] = self._observation_profile
        forwarded = dict(kwargs)
        forwarded["env_kwargs"] = row
        env = super()._create_env(**forwarded)
        if not env.screening or env.episode_session_count != 21:
            raise RuntimeError("screening requires a Train-only 21-session environment")
        return env

    async def _obs_to_prompt(
        self, obs: Any, tools: list[dict[str, Any]] | None = None
    ) -> list[int]:
        if self._action_protocol != "legacy":
            if obs.messages is None or obs.text is not None or obs.token_ids is not None:
                raise RuntimeError("screening action protocol requires message observations")
            obs = replace(
                obs,
                messages=apply_screening_action_protocol(obs.messages, self._action_protocol),
            )
        try:
            return await super()._obs_to_prompt(obs, tools=tools)
        except RuntimeError as error:
            if "exceeding configured prompt limit" in str(error):
                raise ScreeningIncompleteError(str(error)) from error
            raise

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any):
        identity = normalize_rollout_identity(kwargs[ROLLOUT_IDENTITY_KWARG])
        root_value = os.environ.get(SCREENING_EVIDENCE_DIR_ENV, "")
        root = Path(root_value).expanduser()
        if not root_value or not root.is_absolute():
            raise RuntimeError("screening requires an absolute FINANCE_SCREENING_EVIDENCE_DIR")
        root.mkdir(parents=True, exist_ok=True)
        group_uid = kwargs.get("uid")
        if not isinstance(group_uid, str) or not group_uid:
            raise RuntimeError("screening requires the upstream sibling group UID")
        extra_info = kwargs.get("extra_info", {})
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        if (
            extra_info.get("source_split") != "train"
            or extra_info.get("scientific_use") != "screening"
        ):
            raise RuntimeError("screening row must declare historical Train-only scientific use")
        episode_id = kwargs.get("episode_id") or extra_info.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise RuntimeError("screening requires the frozen episode ID")
        rollout_config = self.config.actor_rollout_ref.rollout
        rollout_n = int(rollout_config.val_kwargs.n if identity.validate else rollout_config.n)
        if (not identity.validate and rollout_n != 4) or not 0 <= identity.rollout_n < rollout_n:
            raise RuntimeError("screening rollout identity does not match the frozen sibling count")
        configured_context = rollout_config.get("max_model_len")
        if configured_context is None:
            raise RuntimeError("screening requires an explicit rollout.max_model_len")
        context_limit = int(configured_context)
        original_manager = self.server_manager
        manager = ScreeningResponseManager(
            original_manager,
            prompt_limit=self.prompt_length,
            response_limit=self.response_length,
            context_limit=context_limit,
            capture_token_ids=self._capture_interactions,
        )
        self.server_manager = manager
        record: dict[str, Any] = {
            "schema_version": 1,
            "arm": self._screening_arm,
            "base_seed": int(self.config.data.seed),
            "episode_id": episode_id,
            "group_uid": group_uid,
            "identity": identity.as_dict(),
            "rollout_count": rollout_n,
            "chat_template_kwargs": dict(
                self.config.data.get("apply_chat_template_kwargs", {})
            ),
            "action_protocol": self._action_protocol,
            "observation_profile": self._observation_profile,
            "action_protocol_sha256": action_protocol_sha256(self._action_protocol),
            "pid": os.getpid(),
            "complete": False,
        }
        # A reused flow must not attach the previous trajectory's evidence if
        # construction of this trajectory's environment fails.
        if hasattr(self, "_active_trading_env"):
            del self._active_trading_env
        try:
            output = await super().run(sampling_params, **kwargs)
            env = self._active_trading_env
            trace = env.credit_step_trace
            if (
                not output.steps
                or not env.episode_complete
                or env.current_session_index != env.episode_session_count - 1
                or not trace[-1].done
                or any(step.done for step in trace[:-1])
            ):
                raise ScreeningIncompleteError(
                    "screening rollout stopped before its exact terminal"
                )
            advancing = [
                step for step in trace if step.result_session_index > step.decision_session_index
            ]
            if [step.decision_session_index for step in advancing] != list(range(20)) or any(
                step.result_session_index != step.decision_session_index + 1
                or step.action not in {"buy", "sell", "hold", "abstain"}
                or step.invalid_action
                for step in advancing
            ):
                raise RuntimeError("screening trajectory lacks exactly one advance per session")
            if len(manager.requests) != len(output.steps):
                raise RuntimeError("screening request/step evidence is misaligned")
            observed = tuple(float(step.reward_score) for step in output.steps)
            if observed != tuple(step.reward for step in trace):
                raise RuntimeError("screening upstream rewards differ from environment evidence")
            metrics = env.screening_terminal_metrics()
            terminal_score = float(metrics["terminal_score"])
            if identity.validate:
                # The common evaluator must never compare A2's R_net with Y.
                credits = [0.0] * len(trace)
                credits[-1] = terminal_score
                credit_semantics = "terminal_common_evaluation_score"
            elif self._screening_arm == "A2":
                credits = list(observed)
                if (
                    any(value != 0.0 for value in credits[:-1])
                    or credits[-1] != metrics["raw_net_return"]
                ):
                    raise RuntimeError("screening A2 is not terminal-only net return")
                credit_semantics = "terminal_net_return"
            else:
                allocated = self._credit_strategy.allocate(
                    trajectory_advantage=terminal_score,
                    steps=[
                        StepContext(step_id=f"step-{index}", action=step.action)
                        for index, step in enumerate(trace)
                    ],
                )
                credits = [credit.credit for credit in allocated]
                credit_semantics = STEP_CREDIT_OUTPUT_SEMANTICS
                if sum(credits) != terminal_score:
                    raise RuntimeError(
                        "screening uniform credits do not conserve the terminal score"
                    )
            for step, credit in zip(output.steps, credits, strict=True):
                step.reward_score = credit
                step.extra_fields.update(
                    {
                        "finance_screening_arm": self._screening_arm,
                        "finance_screening_group_uid": group_uid,
                        "finance_screening_policy_step": identity.step,
                        "finance_screening_rollout_id": identity.rollout_n,
                        "finance_screening_rollout_n": rollout_n,
                        "finance_screening_terminal_score": terminal_score,
                        "finance_screening_net_return": float(metrics["raw_net_return"]),
                        "finance_screening_trajectory_complete": True,
                        "finance_screening_step_count": len(trace),
                        "finance_screening_episode_id": episode_id,
                        "finance_screening_credit_semantics": credit_semantics,
                        "finance_screening_calibration_sha256": metrics["calibration_sha256"],
                    }
                )
            record.update(
                {
                    "complete": True,
                    "status": "COMPLETE",
                    "metrics": metrics,
                    "terminal_score": terminal_score,
                    "credits": credits,
                    "credit_semantics": credit_semantics,
                }
            )
            return output
        except Exception as error:
            record["status"] = (
                "INCOMPLETE" if isinstance(error, ScreeningIncompleteError) else "ERROR"
            )
            record["error_type"] = type(error).__name__
            record["error"] = str(error)
            raise
        finally:
            self.server_manager = original_manager
            record["requests"] = manager.requests
            record["generated_response_tokens"] = sum(
                request["response_token_count"] for request in manager.requests
            )
            record["prompt_tokens"] = sum(
                request["prompt_token_count"] for request in manager.requests
            )
            evidence_env = getattr(self, "_active_trading_env", None)
            record["steps"] = (
                [asdict(step) for step in evidence_env.credit_step_trace]
                if evidence_env is not None
                else []
            )
            if self._capture_interactions:
                record["interaction_capture_version"] = 1
                record["interactions"] = (
                    evidence_env.interaction_trace if evidence_env is not None else []
                )
            with (root / f"screening-{os.getpid()}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")

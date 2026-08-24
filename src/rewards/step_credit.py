"""Strategy interface for finance-native step-level credit assignment."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class StepContext:
    step_id: str
    action: str
    portfolio_state: str | None = None
    regime: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.step_id or not self.action:
            raise ValueError("step_id and action must not be empty")


@dataclass(frozen=True, slots=True)
class StepCredit:
    step_id: str
    credit: float
    mechanism: str


@dataclass(frozen=True, slots=True)
class CreditEvidence:
    """Optional sampled evidence consumed by M2/M3 implementations.

    `group_step_outcomes` maps a `(portfolio_state, regime)` anchor to sampled
    outcomes by step id. `counterfactual_rollouts` maps a masked information-step
    id to rollout summaries containing decision-change and outcome-quality terms.
    Concrete schemas stay open until those estimators are pre-registered.
    """

    group_step_outcomes: Mapping[tuple[str, str], Mapping[str, Sequence[float]]] = field(
        default_factory=dict
    )
    counterfactual_rollouts: Mapping[str, Sequence[Mapping[str, float]]] = field(
        default_factory=dict
    )


class StepCreditStrategy(ABC):
    name: str

    @abstractmethod
    def allocate(
        self,
        *,
        trajectory_advantage: float,
        steps: Sequence[StepContext],
        evidence: CreditEvidence | None = None,
    ) -> tuple[StepCredit, ...]:
        """Allocate one finite trajectory advantage across observable steps."""


class UniformBroadcastCredit(StepCreditStrategy):
    """M1 ablation: divide `A_fin` uniformly and conserve its total exactly."""

    name = "m1_uniform"

    def allocate(
        self,
        *,
        trajectory_advantage: float,
        steps: Sequence[StepContext],
        evidence: CreditEvidence | None = None,
    ) -> tuple[StepCredit, ...]:
        del evidence
        value = float(trajectory_advantage)
        if not math.isfinite(value):
            raise ValueError("trajectory_advantage must be finite")
        if not steps:
            raise ValueError("cannot allocate credit to an empty trajectory")
        share = value / len(steps)
        credits = [share] * len(steps)
        credits[-1] = value - sum(credits[:-1])
        return tuple(
            StepCredit(step_id=step.step_id, credit=credit, mechanism=self.name)
            for step, credit in zip(steps, credits, strict=True)
        )


class StateRegimeGroupCredit(StepCreditStrategy):
    """M2 contract: within-anchor relative step advantage.

    Future implementation must group sampled alternatives by
    `(portfolio_state, market_regime)`, then center outcome estimates within each
    anchor before assigning credit. It must reject missing anchor coverage rather
    than falling back to trajectory broadcasting.

    TODO: freeze portfolio-state discretization, regime labels, minimum group
    sample size, and centering estimator before implementing.
    """

    name = "m2_state_regime_group"

    def allocate(
        self,
        *,
        trajectory_advantage: float,
        steps: Sequence[StepContext],
        evidence: CreditEvidence | None = None,
    ) -> tuple[StepCredit, ...]:
        del trajectory_advantage, steps, evidence
        raise NotImplementedError(
            "M2 is an explicit research TODO; do not substitute uniform broadcast"
        )


class CounterfactualInformationCredit(StepCreditStrategy):
    """M3 contract: sampled contribution of retrieve/compute information.

    Future implementation must mask a selected tool response, re-roll the
    downstream decision, and estimate `decision_change * outcome_quality`. It
    must log mask sampling probabilities and rollout seeds so the estimator is
    auditable.

    TODO: freeze the decision-distance metric, outcome-quality target, mask
    sampler, and number of counterfactual rollouts before implementing.
    """

    name = "m3_counterfactual_information"

    def allocate(
        self,
        *,
        trajectory_advantage: float,
        steps: Sequence[StepContext],
        evidence: CreditEvidence | None = None,
    ) -> tuple[StepCredit, ...]:
        del trajectory_advantage, steps, evidence
        raise NotImplementedError(
            "M3 is an explicit research TODO; do not substitute uniform broadcast"
        )


def build_credit_strategy(name: str) -> StepCreditStrategy:
    strategies: dict[str, type[StepCreditStrategy]] = {
        UniformBroadcastCredit.name: UniformBroadcastCredit,
        StateRegimeGroupCredit.name: StateRegimeGroupCredit,
        CounterfactualInformationCredit.name: CounterfactualInformationCredit,
    }
    if name not in strategies:
        raise ValueError(f"unknown credit strategy {name!r}; available: {sorted(strategies)}")
    return strategies[name]()

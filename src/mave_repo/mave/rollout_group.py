from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from mave_repro.core.types import (
    MAVEAdvantageRecord,
    RewardCostObservation,
    RolloutGroup,
)

from mave_repro.mave.advantage import (
    MAVEAdvantageBatch,
    compute_mave_advantage,
)
from mave_repro.mave.curve_fit import (
    CurveFitConfig,
    CurveFitResult,
    fit_reward_cost_curve,
)
from mave_repro.mave.derivatives import (
    DerivativeResult,
    evaluate_derivatives,
)


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True, slots=True)
class MAVEGroupConfig:
    """
    Paper Table 7 defaults:

        K = 4
        lambda = 0.5
        beta1 = 1.0
        beta2 = 1.0
        eta = 0.001
        mapping = tanh
    """

    expected_group_size: int = 4

    strict_group_size: bool = True

    lambda_cost: float = 0.5

    beta1: float = 1.0
    beta2: float = 1.0

    normalization_eps: float = 1e-8

    curve_fit: CurveFitConfig = field(
        default_factory=CurveFitConfig
    )


# ============================================================
# Processed group
# ============================================================


@dataclass(slots=True)
class ProcessedRolloutGroup:
    """
    Complete MAVE processing result for one rollout group.
    """

    context_id: str

    observations: tuple[
        RewardCostObservation,
        ...
    ]

    curve_fit: CurveFitResult

    derivatives: DerivativeResult

    advantage_batch: MAVEAdvantageBatch

    records: tuple[
        MAVEAdvantageRecord,
        ...
    ]

    @property
    def advantages(
        self,
    ) -> Tensor:
        return (
            self
            .advantage_batch
            .advantage
        )

    @property
    def costs(
        self,
    ) -> Tensor:
        return (
            self
            .advantage_batch
            .costs
        )

    @property
    def rewards(
        self,
    ) -> Tensor:
        return (
            self
            .advantage_batch
            .rewards
        )


# ============================================================
# Validation
# ============================================================


def validate_rollout_group(
    group: RolloutGroup,
    *,
    config: MAVEGroupConfig,
) -> None:

    if group.size == 0:
        raise ValueError(
            "Rollout group is empty."
        )

    if (
        config.strict_group_size
        and group.size
        != config.expected_group_size
    ):
        raise ValueError(
            "Unexpected rollout group size. "
            f"Expected K="
            f"{config.expected_group_size}, "
            f"received K={group.size}."
        )

    for rollout in group.rollouts:

        if (
            rollout.context_id
            != group.context_id
        ):
            raise ValueError(
                "All rollouts must originate "
                "from the same planning context."
            )

        if rollout.cost < 0:
            raise ValueError(
                "Rollout cost must be "
                "non-negative."
            )


# ============================================================
# Conversion
# ============================================================


def observations_from_group(
    group: RolloutGroup,
) -> tuple[
    RewardCostObservation,
    ...
]:
    """
    Extract:

        {(c_k, r_k)}_{k=1}^K
    """

    return tuple(
        RewardCostObservation(
            cost=float(
                rollout.cost
            ),
            reward=float(
                rollout.downstream_return
            ),
            rollout_index=index,
        )
        for index, rollout
        in enumerate(
            group.rollouts
        )
    )


def tensors_from_group(
    group: RolloutGroup,
) -> tuple[
    Tensor,
    Tensor,
]:

    costs = torch.tensor(
        [
            rollout.cost
            for rollout
            in group.rollouts
        ],
        dtype=torch.float64,
    )

    rewards = torch.tensor(
        [
            rollout.downstream_return
            for rollout
            in group.rollouts
        ],
        dtype=torch.float64,
    )

    return (
        costs,
        rewards,
    )


# ============================================================
# Full MAVE group processing
# ============================================================


def process_rollout_group(
    group: RolloutGroup,
    *,
    config: MAVEGroupConfig | None = None,
) -> ProcessedRolloutGroup:
    """
    Complete paper pipeline for one escalation rollout group.

    1. Extract (c_k, r_k).
    2. Fit local reward-cost curve phi(c).
    3. Evaluate D1 = phi'(c_k).
    4. Evaluate D2 = phi''(c_k).
    5. Compute J_k = r_k - lambda c_k.
    6. Normalize J, D1, D2 within the group.
    7. Compute A_MAVE.
    """

    config = (
        config
        or MAVEGroupConfig()
    )

    validate_rollout_group(
        group,
        config=config,
    )

    observations = (
        observations_from_group(
            group
        )
    )

    costs, rewards = (
        tensors_from_group(
            group
        )
    )

    # --------------------------------------------------------
    # Eq. (5): local curve fitting
    # --------------------------------------------------------

    fitted = (
        fit_reward_cost_curve(
            costs,
            rewards,
            config=(
                config.curve_fit
            ),
        )
    )

    if not fitted.is_finite:
        raise FloatingPointError(
            "Reward-cost curve fitting "
            "produced a non-finite loss."
        )

    # --------------------------------------------------------
    # Eq. (6): marginal-value signals
    # --------------------------------------------------------

    derivative_result = (
        evaluate_derivatives(
            fitted.mapping,
            costs,
        )
    )

    # --------------------------------------------------------
    # Eqs. (7)-(9): MAVE advantage
    # --------------------------------------------------------

    advantage_batch = (
        compute_mave_advantage(
            costs=costs,
            rewards=rewards,
            d1=(
                derivative_result.d1
            ),
            d2=(
                derivative_result.d2
            ),
            lambda_cost=(
                config.lambda_cost
            ),
            beta1=config.beta1,
            beta2=config.beta2,
            eps=(
                config
                .normalization_eps
            ),
        )
    )

    return ProcessedRolloutGroup(
        context_id=(
            group.context_id
        ),

        observations=(
            observations
        ),

        curve_fit=fitted,

        derivatives=(
            derivative_result
        ),

        advantage_batch=(
            advantage_batch
        ),

        records=(
            advantage_batch
            .records()
        ),
    )


# ============================================================
# Convenience processor
# ============================================================


class MAVEGroupProcessor:
    """
    Small state-free wrapper useful inside the training loop.

    Example
    -------
        processor = MAVEGroupProcessor()

        processed = processor(group)

        advantages = processed.advantages
    """

    def __init__(
        self,
        config: MAVEGroupConfig | None = None,
    ) -> None:

        self.config = (
            config
            or MAVEGroupConfig()
        )

    def __call__(
        self,
        group: RolloutGroup,
    ) -> ProcessedRolloutGroup:

        return process_rollout_group(
            group,
            config=self.config,
        )
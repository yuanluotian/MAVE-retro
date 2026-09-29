from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor

from mave.mave.rollout_group import (
    MAVEGroupProcessor,
)

from mave.models.escalation_policy import (
    EscalationPolicy,
)

from mave.training.collector import (
    CollectedEscalationGroup,
)

from mave.training.grpo import (
    compute_grpo_loss,
)


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True, slots=True)
class EscalationTrainConfig:
    learning_rate: float = 3e-5

    clip_epsilon: float = 0.2

    kl_coefficient: float = 0.001

    max_grad_norm: (
        float | None
    ) = None


# ============================================================
# Statistics
# ============================================================


@dataclass(frozen=True, slots=True)
class EscalationTrainStats:
    loss: float

    policy_loss: float

    kl_loss: float

    mean_kl: float

    mean_ratio: float

    clip_fraction: float

    num_groups: int

    num_rollouts: int

    num_decisions: int

    mean_advantage: float

    mean_rollout_cost: float

    mean_rollout_return: float


# ============================================================
# Trainer
# ============================================================


class EscalationTrainer:
    """
    Optimize escalation policy mu using MAVE-GRPO.

    Fixed reaction-policy behavior must be guaranteed by the
    caller / alternating trainer while groups are collected.
    """

    def __init__(
        self,
        *,
        policy: EscalationPolicy,

        # A frozen reference policy used for the KL term.
        reference_policy: (
            EscalationPolicy | None
        ),

        mave_processor: (
            MAVEGroupProcessor
        ),

        config: (
            EscalationTrainConfig
        ) = EscalationTrainConfig(),

        optimizer: (
            torch.optim.Optimizer
            | None
        ) = None,

        trainable_parameters: (
            Iterable[
                torch.nn.Parameter
            ]
            | None
        ) = None,
    ) -> None:

        self.policy = policy

        self.reference_policy = (
            reference_policy
        )

        self.mave_processor = (
            mave_processor
        )

        self.config = config

        if (
            config.kl_coefficient > 0
            and reference_policy is None
        ):
            raise ValueError(
                "A frozen reference_policy is required "
                "because KL coefficient > 0."
            )

        if optimizer is None:

            if (
                trainable_parameters
                is None
            ):
                trainable_parameters = (
                    policy
                    .backbone
                    .parameters()
                )

            optimizer = (
                torch.optim.AdamW(
                    trainable_parameters,
                    lr=(
                        config
                        .learning_rate
                    ),
                )
            )

        self.optimizer = optimizer

    # ========================================================
    # One optimization batch
    # ========================================================

    def train_batch(
        self,
        groups: Sequence[
            CollectedEscalationGroup
        ],
    ) -> EscalationTrainStats:

        if not groups:
            raise ValueError(
                "Escalation training batch "
                "is empty."
            )

        new_logprobs: list[
            Tensor
        ] = []

        old_logprobs: list[
            float
        ] = []

        reference_logprobs: list[
            Tensor
        ] = []

        advantages: list[
            float
        ] = []

        rollout_costs: list[
            float
        ] = []

        rollout_returns: list[
            float
        ] = []

        num_rollouts = 0

        # ====================================================
        # Process each MAVE rollout group
        # ====================================================

        for collected_group in groups:

            processed = (
                self.mave_processor(
                    collected_group
                    .rollout_group
                )
            )

            group_advantages = (
                processed
                .advantages
                .detach()
                .cpu()
            )

            traces = (
                collected_group
                .traces
            )

            if (
                len(traces)
                != len(
                    group_advantages
                )
            ):
                raise RuntimeError(
                    "MAVE advantages / rollout "
                    "trace mismatch."
                )

            for rollout_index, trace in enumerate(
                traces
            ):

                num_rollouts += 1

                advantage = float(
                    group_advantages[
                        rollout_index
                    ]
                )

                rollout_costs.append(
                    trace.rollout.cost
                )

                rollout_returns.append(
                    trace.rollout
                    .downstream_return
                )

                # --------------------------------------------
                # Paper:
                # rollout-level A_MAVE is assigned to every
                # escalation decision in the rollout.
                # --------------------------------------------

                for step in (
                    trace
                    .escalation_steps
                ):

                    current_logprob = (
                        self.policy
                        .action_logprob(
                            step.context,
                            step.available_queries,
                            step.action_index,
                        )
                    )

                    new_logprobs.append(
                        current_logprob
                    )

                    old_logprobs.append(
                        step.old_logprob
                    )

                    advantages.append(
                        advantage
                    )

                    if (
                        self.reference_policy
                        is not None
                    ):
                        with torch.no_grad():

                            reference = (
                                self
                                .reference_policy
                                .action_logprob(
                                    step.context,
                                    step
                                    .available_queries,
                                    step
                                    .action_index,
                                )
                            )

                        reference_logprobs.append(
                            reference.detach()
                        )

        if not new_logprobs:
            raise RuntimeError(
                "No escalation decisions were "
                "collected in this batch."
            )

        # ====================================================
        # Flatten into GRPO tensors
        # ====================================================

        new_tensor = torch.stack(
            new_logprobs
        )

        device = (
            new_tensor.device
        )

        dtype = (
            new_tensor.dtype
        )

        old_tensor = torch.tensor(
            old_logprobs,
            dtype=dtype,
            device=device,
        )

        advantage_tensor = torch.tensor(
            advantages,
            dtype=dtype,
            device=device,
        )

        if reference_logprobs:

            reference_tensor = (
                torch.stack(
                    reference_logprobs
                )
                .to(
                    device=device,
                    dtype=dtype,
                )
            )

        else:
            reference_tensor = None

        # ====================================================
        # GRPO
        # ====================================================

        result = compute_grpo_loss(
            new_logprobs=(
                new_tensor
            ),
            old_logprobs=(
                old_tensor
            ),
            advantages=(
                advantage_tensor
            ),
            reference_logprobs=(
                reference_tensor
            ),
            clip_epsilon=(
                self.config
                .clip_epsilon
            ),
            kl_coefficient=(
                self.config
                .kl_coefficient
            ),
        )

        # ====================================================
        # Update
        # ====================================================

        self.optimizer.zero_grad(
            set_to_none=True
        )

        result.loss.backward()

        if (
            self.config
            .max_grad_norm
            is not None
        ):
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for group
                    in self.optimizer
                    .param_groups
                    for parameter
                    in group[
                        "params"
                    ]
                    if parameter.grad
                    is not None
                ],
                max_norm=(
                    self.config
                    .max_grad_norm
                ),
            )

        self.optimizer.step()

        return EscalationTrainStats(
            loss=float(
                result.loss.detach()
            ),
            policy_loss=float(
                result
                .policy_loss
                .detach()
            ),
            kl_loss=float(
                result
                .kl_loss
                .detach()
            ),
            mean_kl=float(
                result
                .mean_kl
                .detach()
            ),
            mean_ratio=float(
                result
                .mean_ratio
                .detach()
            ),
            clip_fraction=float(
                result
                .clip_fraction
                .detach()
            ),
            num_groups=len(
                groups
            ),
            num_rollouts=(
                num_rollouts
            ),
            num_decisions=len(
                new_logprobs
            ),
            mean_advantage=(
                sum(advantages)
                / len(advantages)
            ),
            mean_rollout_cost=(
                sum(rollout_costs)
                / len(rollout_costs)
            ),
            mean_rollout_return=(
                sum(rollout_returns)
                / len(rollout_returns)
            ),
        )
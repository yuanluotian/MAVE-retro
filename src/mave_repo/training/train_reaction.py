from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor

from mave_repro.core.utils import (
    standardize_tensor,
)

from mave_repro.models.reaction_policy import (
    ReactionPolicy,
)

from mave_repro.training.collector import (
    ReactionTrainingGroup,
)

from mave_repro.training.grpo import (
    compute_grpo_loss,
)


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True, slots=True)
class ReactionTrainConfig:
    learning_rate: float = 3e-5

    clip_epsilon: float = 0.2

    kl_coefficient: float = 0.001

    normalization_eps: float = 1e-8

    max_grad_norm: (
        float | None
    ) = None


# ============================================================
# Statistics
# ============================================================


@dataclass(frozen=True, slots=True)
class ReactionTrainStats:
    loss: float

    policy_loss: float

    kl_loss: float

    mean_kl: float

    mean_ratio: float

    clip_fraction: float

    num_groups: int

    num_actions: int

    mean_return: float

    mean_advantage: float


# ============================================================
# Trainer
# ============================================================


class ReactionTrainer:
    """
    Standard GRPO training for reaction policy pi.

    Escalation policy mu should remain fixed while reaction
    groups are collected and pi is updated.
    """

    def __init__(
        self,
        *,
        policy: ReactionPolicy,

        reference_policy: (
            ReactionPolicy | None
        ),

        config: (
            ReactionTrainConfig
        ) = ReactionTrainConfig(),

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
    # Batch update
    # ========================================================

    def train_batch(
        self,
        groups: Sequence[
            ReactionTrainingGroup
        ],
    ) -> ReactionTrainStats:

        if not groups:
            raise ValueError(
                "Reaction training batch is empty."
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

        returns_all: list[
            float
        ] = []

        # ====================================================
        # Standard group-relative return advantage
        # ====================================================

        for group in groups:

            if not group.steps:
                continue

            returns = torch.tensor(
                [
                    step.downstream_return
                    for step
                    in group.steps
                ],
                dtype=torch.float64,
            )

            normalized = (
                standardize_tensor(
                    returns,
                    eps=(
                        self.config
                        .normalization_eps
                    ),
                )
            )

            for step, advantage in zip(
                group.steps,
                normalized,
                strict=True,
            ):

                new_logprob = (
                    self.policy
                    .action_logprob(
                        target_smiles=(
                            step.target_smiles
                        ),
                        unsolved_molecules=(
                            step
                            .unsolved_molecules
                        ),
                        reactions=(
                            step.reactions
                        ),
                        feedback=(
                            step.feedback
                        ),
                        action_index=(
                            step.action_index
                        ),
                    )
                )

                new_logprobs.append(
                    new_logprob
                )

                old_logprobs.append(
                    step.old_logprob
                )

                advantages.append(
                    float(
                        advantage
                    )
                )

                returns_all.append(
                    step.downstream_return
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
                                target_smiles=(
                                    step
                                    .target_smiles
                                ),
                                unsolved_molecules=(
                                    step
                                    .unsolved_molecules
                                ),
                                reactions=(
                                    step
                                    .reactions
                                ),
                                feedback=(
                                    step.feedback
                                ),
                                action_index=(
                                    step
                                    .action_index
                                ),
                            )
                        )

                    reference_logprobs.append(
                        reference.detach()
                    )

        if not new_logprobs:
            raise RuntimeError(
                "No reaction actions were "
                "collected."
            )

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

        return ReactionTrainStats(
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
            num_actions=len(
                new_logprobs
            ),
            mean_return=(
                sum(returns_all)
                / len(returns_all)
            ),
            mean_advantage=(
                sum(advantages)
                / len(advantages)
            ),
        )
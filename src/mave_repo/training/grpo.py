from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# ============================================================
# Result
# ============================================================


@dataclass(frozen=True, slots=True)
class GRPOLossResult:
    loss: Tensor

    policy_loss: Tensor
    kl_loss: Tensor

    mean_ratio: Tensor
    mean_kl: Tensor

    clip_fraction: Tensor


# ============================================================
# KL estimators
# ============================================================


def kl_k3(
    *,
    logprob: Tensor,
    reference_logprob: Tensor,
) -> Tensor:
    """
    Non-negative Monte-Carlo KL estimator:

        r = log pi_ref - log pi

        exp(r) - r - 1

    This estimator choice is not specified by the paper.
    """

    log_ratio = (
        reference_logprob
        - logprob
    )

    return (
        torch.exp(
            log_ratio
        )
        - log_ratio
        - 1.0
    )


# ============================================================
# GRPO objective
# ============================================================


def compute_grpo_loss(
    *,
    new_logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    reference_logprobs: (
        Tensor | None
    ) = None,
    clip_epsilon: float = 0.2,
    kl_coefficient: float = 0.001,
) -> GRPOLossResult:
    """
    Clipped GRPO/PPO-style policy objective.

        ratio =
            exp(log pi_new - log pi_old)

        L_clip =
            min(
                ratio * A,
                clip(ratio) * A
            )

    KL regularization is added when a reference policy is
    supplied.
    """

    if new_logprobs.ndim != 1:
        raise ValueError(
            "new_logprobs must be 1D."
        )

    if (
        old_logprobs.shape
        != new_logprobs.shape
        or advantages.shape
        != new_logprobs.shape
    ):
        raise ValueError(
            "GRPO tensors must have identical shapes."
        )

    if clip_epsilon < 0:
        raise ValueError(
            "clip_epsilon must be non-negative."
        )

    if kl_coefficient < 0:
        raise ValueError(
            "kl_coefficient must be non-negative."
        )

    if (
        kl_coefficient > 0
        and reference_logprobs is None
    ):
        raise ValueError(
            "reference_logprobs are required when "
            "kl_coefficient > 0."
        )

    ratio = torch.exp(
        new_logprobs
        - old_logprobs
    )

    lower = (
        1.0
        - float(
            clip_epsilon
        )
    )

    upper = (
        1.0
        + float(
            clip_epsilon
        )
    )

    clipped_ratio = torch.clamp(
        ratio,
        min=lower,
        max=upper,
    )

    unclipped = (
        ratio
        * advantages
    )

    clipped = (
        clipped_ratio
        * advantages
    )

    surrogate = torch.minimum(
        unclipped,
        clipped,
    )

    policy_loss = (
        -surrogate.mean()
    )

    clip_fraction = (
        (
            torch.abs(
                ratio - 1.0
            )
            > float(
                clip_epsilon
            )
        )
        .to(
            dtype=torch.float32
        )
        .mean()
    )

    # --------------------------------------------------------
    # KL
    # --------------------------------------------------------

    if reference_logprobs is None:

        kl_values = (
            torch.zeros_like(
                new_logprobs
            )
        )

        kl_loss = (
            new_logprobs.sum()
            * 0.0
        )

    else:

        if (
            reference_logprobs.shape
            != new_logprobs.shape
        ):
            raise ValueError(
                "reference_logprobs shape mismatch."
            )

        kl_values = kl_k3(
            logprob=new_logprobs,
            reference_logprob=(
                reference_logprobs
            ),
        )

        kl_loss = (
            float(
                kl_coefficient
            )
            * kl_values.mean()
        )

    total = (
        policy_loss
        + kl_loss
    )

    return GRPOLossResult(
        loss=total,
        policy_loss=(
            policy_loss
        ),
        kl_loss=kl_loss,
        mean_ratio=(
            ratio.mean()
        ),
        mean_kl=(
            kl_values.mean()
        ),
        clip_fraction=(
            clip_fraction
        ),
    )
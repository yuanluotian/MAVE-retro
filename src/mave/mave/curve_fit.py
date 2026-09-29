from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from mave.mave.mappings import (
    PolynomialMapping,
    RewardCostMapping,
    SigmoidMapping,
    TanhMapping,
    TinyNeuralMapping,
)


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True, slots=True)
class CurveFitConfig:
    """
    Local reward-cost curve fitting configuration.

    Paper-specified default:
        mapping = tanh
        eta = 0.001

    Optimizer details are reproduction choices.
    """

    mapping: str = "tanh"

    eta: float = 0.001

    optimizer: str = "lbfgs"

    learning_rate: float = 1.0

    max_steps: int = 100

    tolerance_grad: float = 1e-9

    tolerance_change: float = 1e-12

    dtype: torch.dtype = torch.float64


# ============================================================
# Result
# ============================================================


@dataclass(slots=True)
class CurveFitResult:
    mapping: RewardCostMapping

    total_loss: float
    data_loss: float
    regularization_loss: float

    num_optimizer_steps: int

    unique_costs: int

    is_finite: bool

    parameters: dict[str, float]


# ============================================================
# Utilities
# ============================================================


def _vector(
    values: Sequence[float] | Tensor,
    *,
    dtype: torch.dtype,
) -> Tensor:

    if isinstance(
        values,
        Tensor,
    ):
        tensor = (
            values
            .detach()
            .to(
                dtype=dtype,
                device="cpu",
            )
            .reshape(-1)
        )

    else:
        tensor = torch.tensor(
            list(values),
            dtype=dtype,
        ).reshape(-1)

    if tensor.numel() == 0:
        raise ValueError(
            "Cannot fit an empty rollout group."
        )

    if not torch.isfinite(
        tensor
    ).all():
        raise ValueError(
            "Curve-fitting data contains "
            "non-finite values."
        )

    return tensor


def _initial_statistics(
    costs: Tensor,
    rewards: Tensor,
) -> dict[str, float]:

    c_min = float(
        costs.min()
    )

    c_max = float(
        costs.max()
    )

    r_min = float(
        rewards.min()
    )

    r_max = float(
        rewards.max()
    )

    c_range = max(
        c_max - c_min,
        1e-6,
    )

    r_range = max(
        r_max - r_min,
        1e-3,
    )

    return {
        "c_min": c_min,
        "c_max": c_max,
        "c_mean": float(
            costs.mean()
        ),
        "c_range": c_range,

        "r_min": r_min,
        "r_max": r_max,
        "r_mean": float(
            rewards.mean()
        ),
        "r_range": r_range,
    }


def initialize_mapping(
    mapping_name: str,
    costs: Tensor,
    rewards: Tensor,
) -> RewardCostMapping:
    """
    Deterministic initialization for local fitting.

    Initialization is not specified by the paper and therefore
    constitutes a reproduction implementation choice.
    """

    stats = _initial_statistics(
        costs,
        rewards,
    )

    name = (
        mapping_name
        .strip()
        .lower()
    )

    if name == "tanh":

        # tanh output roughly spans a-b to a+b.
        return TanhMapping(
            a=(
                stats["r_min"]
                + stats["r_max"]
            ) / 2.0,
            b=(
                stats["r_range"]
                / 2.0
            ),
            kappa=(
                1.0
                / stats["c_range"]
            ),
            tau=stats[
                "c_mean"
            ],
        )

    if name == "sigmoid":

        # sigmoid output roughly spans a to a+b.
        return SigmoidMapping(
            a=stats[
                "r_min"
            ],
            b=stats[
                "r_range"
            ],
            kappa=(
                1.0
                / stats["c_range"]
            ),
            tau=stats[
                "c_mean"
            ],
        )

    if name in {
        "polynomial",
        "quadratic",
    }:
        return PolynomialMapping(
            a0=stats[
                "r_mean"
            ],
            a1=0.0,
            a2=0.0,
        )

    if name in {
        "tiny_neural",
        "tiny-neural",
        "neural",
    }:
        return TinyNeuralMapping(
            c_min=stats[
                "c_min"
            ],
            c_max=stats[
                "c_max"
            ],
            a=stats[
                "r_min"
            ],
            weight_init=(
                stats[
                    "r_range"
                ] / 2.0
            ),
        )

    raise ValueError(
        f"Unknown reward-cost mapping: "
        f"{mapping_name!r}"
    )


# ============================================================
# Objective
# ============================================================


def fitting_objective(
    mapping: RewardCostMapping,
    costs: Tensor,
    rewards: Tensor,
    *,
    eta: float,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
]:
    """
    Paper Eq. (5):

        sum_k (r_k - phi(c_k))^2
        + eta ||theta||_2^2
    """

    predictions = mapping(
        costs
    )

    data_loss = (
        (
            rewards
            - predictions
        ).square()
        .sum()
    )

    theta = (
        mapping
        .regularization_vector()
    )

    regularization = (
        float(eta)
        * theta.square().sum()
    )

    total = (
        data_loss
        + regularization
    )

    return (
        total,
        data_loss,
        regularization,
    )


# ============================================================
# Fit
# ============================================================


def fit_reward_cost_curve(
    costs: Sequence[float] | Tensor,
    rewards: Sequence[float] | Tensor,
    *,
    config: CurveFitConfig | None = None,
) -> CurveFitResult:

    config = (
        config
        or CurveFitConfig()
    )

    if config.eta < 0:
        raise ValueError(
            "eta must be non-negative."
        )

    costs_tensor = _vector(
        costs,
        dtype=config.dtype,
    )

    rewards_tensor = _vector(
        rewards,
        dtype=config.dtype,
    )

    if (
        costs_tensor.shape
        != rewards_tensor.shape
    ):
        raise ValueError(
            "costs and rewards must have "
            "the same length."
        )

    mapping = initialize_mapping(
        config.mapping,
        costs_tensor,
        rewards_tensor,
    )

    mapping = mapping.to(
        dtype=config.dtype,
        device="cpu",
    )

    optimizer_name = (
        config.optimizer
        .strip()
        .lower()
    )

    step_counter = 0

    # ========================================================
    # LBFGS
    # ========================================================

    if optimizer_name == "lbfgs":

        optimizer = torch.optim.LBFGS(
            mapping.parameters(),
            lr=config.learning_rate,
            max_iter=config.max_steps,
            tolerance_grad=(
                config.tolerance_grad
            ),
            tolerance_change=(
                config.tolerance_change
            ),
            line_search_fn=(
                "strong_wolfe"
            ),
        )

        def closure() -> Tensor:
            nonlocal step_counter

            optimizer.zero_grad(
                set_to_none=True
            )

            total, _, _ = (
                fitting_objective(
                    mapping,
                    costs_tensor,
                    rewards_tensor,
                    eta=config.eta,
                )
            )

            total.backward()

            step_counter += 1

            return total

        optimizer.step(
            closure
        )

    # ========================================================
    # Adam
    # ========================================================

    elif optimizer_name == "adam":

        optimizer = torch.optim.Adam(
            mapping.parameters(),
            lr=config.learning_rate,
        )

        previous_loss: float | None = (
            None
        )

        for _ in range(
            config.max_steps
        ):
            optimizer.zero_grad(
                set_to_none=True
            )

            total, _, _ = (
                fitting_objective(
                    mapping,
                    costs_tensor,
                    rewards_tensor,
                    eta=config.eta,
                )
            )

            total.backward()

            optimizer.step()

            step_counter += 1

            current = float(
                total.detach()
            )

            if (
                previous_loss
                is not None
                and abs(
                    previous_loss
                    - current
                )
                < config.tolerance_change
            ):
                break

            previous_loss = current

    else:
        raise ValueError(
            f"Unsupported optimizer: "
            f"{config.optimizer!r}"
        )

    # ========================================================
    # Final diagnostics
    # ========================================================

    with torch.no_grad():

        (
            total,
            data_loss,
            regularization,
        ) = fitting_objective(
            mapping,
            costs_tensor,
            rewards_tensor,
            eta=config.eta,
        )

    return CurveFitResult(
        mapping=mapping,

        total_loss=float(
            total.detach()
        ),

        data_loss=float(
            data_loss.detach()
        ),

        regularization_loss=float(
            regularization.detach()
        ),

        num_optimizer_steps=(
            step_counter
        ),

        unique_costs=int(
            torch.unique(
                costs_tensor
            ).numel()
        ),

        is_finite=bool(
            torch.isfinite(
                total
            ).item()
        ),

        parameters=(
            mapping
            .parameter_dict()
        ),
    )
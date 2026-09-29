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


@dataclass(frozen=True, slots=True)
class DerivativeResult:
    """
    Marginal-value signals evaluated at observed costs.

        D1 = phi'(c)
        D2 = phi''(c)
    """

    costs: Tensor
    d1: Tensor
    d2: Tensor


def _cost_tensor(
    costs: Sequence[float] | Tensor,
) -> Tensor:

    if isinstance(
        costs,
        Tensor,
    ):
        return (
            costs
            .to(
                dtype=torch.float64
            )
            .reshape(-1)
        )

    return torch.tensor(
        list(costs),
        dtype=torch.float64,
    ).reshape(-1)


# ============================================================
# First derivative
# ============================================================


def first_derivative(
    mapping: RewardCostMapping,
    costs: Sequence[float] | Tensor,
) -> Tensor:

    c = _cost_tensor(
        costs
    )

    # --------------------------------------------------------
    # Tanh
    #
    # phi'(c)
    # = b kappa sech^2(z)
    # --------------------------------------------------------

    if isinstance(
        mapping,
        TanhMapping,
    ):

        z = (
            mapping.kappa
            * (
                c
                - mapping.tau
            )
        )

        tanh_z = torch.tanh(
            z
        )

        sech_squared = (
            1.0
            - tanh_z.square()
        )

        return (
            mapping.b
            * mapping.kappa
            * sech_squared
        )

    # --------------------------------------------------------
    # Sigmoid
    #
    # phi'(c)
    # = b kappa p(1-p)
    # --------------------------------------------------------

    if isinstance(
        mapping,
        SigmoidMapping,
    ):

        z = (
            mapping.kappa
            * (
                c
                - mapping.tau
            )
        )

        p = torch.sigmoid(z)

        return (
            mapping.b
            * mapping.kappa
            * p
            * (
                1.0 - p
            )
        )

    # --------------------------------------------------------
    # Polynomial
    #
    # phi'(c)
    # = a1 + 2 a2 c
    # --------------------------------------------------------

    if isinstance(
        mapping,
        PolynomialMapping,
    ):

        return (
            mapping.a1
            + 2.0
            * mapping.a2
            * c
        )

    # --------------------------------------------------------
    # Tiny neural
    #
    # alpha / Delta_c
    # sum_j w_j h_j(1-h_j)
    # --------------------------------------------------------

    if isinstance(
        mapping,
        TinyNeuralMapping,
    ):

        h = mapping.hidden(c)

        term = (
            mapping.weights
            * h
            * (
                1.0 - h
            )
        )

        return (
            mapping.alpha
            / mapping.delta_cost
            * term.sum(
                dim=-1
            )
        )

    # --------------------------------------------------------
    # Generic autograd fallback
    # --------------------------------------------------------

    return _autograd_derivatives(
        mapping,
        c,
    )[0]


# ============================================================
# Second derivative
# ============================================================


def second_derivative(
    mapping: RewardCostMapping,
    costs: Sequence[float] | Tensor,
) -> Tensor:

    c = _cost_tensor(
        costs
    )

    # --------------------------------------------------------
    # Tanh
    #
    # phi''(c)
    # = -2 b kappa^2 sech^2(z) tanh(z)
    # --------------------------------------------------------

    if isinstance(
        mapping,
        TanhMapping,
    ):

        z = (
            mapping.kappa
            * (
                c
                - mapping.tau
            )
        )

        tanh_z = torch.tanh(
            z
        )

        sech_squared = (
            1.0
            - tanh_z.square()
        )

        return (
            -2.0
            * mapping.b
            * mapping.kappa.square()
            * sech_squared
            * tanh_z
        )

    # --------------------------------------------------------
    # Sigmoid
    #
    # phi''(c)
    # = b kappa^2 p(1-p)(1-2p)
    # --------------------------------------------------------

    if isinstance(
        mapping,
        SigmoidMapping,
    ):

        z = (
            mapping.kappa
            * (
                c
                - mapping.tau
            )
        )

        p = torch.sigmoid(z)

        return (
            mapping.b
            * mapping.kappa.square()
            * p
            * (
                1.0 - p
            )
            * (
                1.0 - 2.0 * p
            )
        )

    # --------------------------------------------------------
    # Polynomial
    #
    # phi''(c) = 2 a2
    # --------------------------------------------------------

    if isinstance(
        mapping,
        PolynomialMapping,
    ):

        return torch.ones_like(
            c
        ) * (
            2.0
            * mapping.a2
        )

    # --------------------------------------------------------
    # Tiny neural
    #
    # alpha^2 / Delta_c^2
    # sum_j [
    #   w_j h_j(1-h_j)(1-2h_j)
    # ]
    # --------------------------------------------------------

    if isinstance(
        mapping,
        TinyNeuralMapping,
    ):

        h = mapping.hidden(c)

        term = (
            mapping.weights
            * h
            * (
                1.0 - h
            )
            * (
                1.0 - 2.0 * h
            )
        )

        return (
            mapping.alpha.square()
            / mapping.delta_cost.square()
            * term.sum(
                dim=-1
            )
        )

    return _autograd_derivatives(
        mapping,
        c,
    )[1]


# ============================================================
# Joint evaluation
# ============================================================


def evaluate_derivatives(
    mapping: RewardCostMapping,
    costs: Sequence[float] | Tensor,
) -> DerivativeResult:

    c = _cost_tensor(
        costs
    )

    with torch.no_grad():

        d1 = first_derivative(
            mapping,
            c,
        )

        d2 = second_derivative(
            mapping,
            c,
        )

    return DerivativeResult(
        costs=c.detach(),
        d1=d1.detach(),
        d2=d2.detach(),
    )


# ============================================================
# Generic fallback
# ============================================================


def _autograd_derivatives(
    mapping: RewardCostMapping,
    costs: Tensor,
) -> tuple[
    Tensor,
    Tensor,
]:
    """
    Generic fallback for future differentiable mapping classes.

    The paper mappings above use their explicit analytic
    derivatives instead.
    """

    c = (
        costs
        .detach()
        .clone()
        .requires_grad_(True)
    )

    values = mapping(c)

    d1 = torch.autograd.grad(
        outputs=values.sum(),
        inputs=c,
        create_graph=True,
    )[0]

    d2 = torch.autograd.grad(
        outputs=d1.sum(),
        inputs=c,
        create_graph=False,
    )[0]

    return (
        d1,
        d2,
    )
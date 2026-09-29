from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mave.core.registry import MAPPING_REGISTRY


# ============================================================
# Helpers
# ============================================================


def _scalar_tensor(
    value: float,
) -> Tensor:
    return torch.tensor(
        float(value),
        dtype=torch.float64,
    )


def inverse_softplus(
    value: float,
    *,
    eps: float = 1e-8,
) -> float:
    """
    Inverse of softplus for positive initialization.

    softplus(x) = log(1 + exp(x))
    """
    value = max(
        float(value),
        eps,
    )

    tensor = torch.tensor(
        value,
        dtype=torch.float64,
    )

    # Stable inverse:
    # x = y + log(1 - exp(-y))
    result = (
        tensor
        + torch.log(
            -torch.expm1(-tensor)
        )
    )

    return float(
        result.item()
    )


# ============================================================
# Base mapping
# ============================================================


class RewardCostMapping(
    nn.Module,
    ABC,
):
    """
    Base class for local reward-cost mappings phi(c; theta).

    Each mapping is fitted independently for one rollout group.
    """

    @abstractmethod
    def forward(
        self,
        cost: Tensor,
    ) -> Tensor:
        ...

    @abstractmethod
    def regularization_vector(
        self,
    ) -> Tensor:
        """
        Parameter vector used in

            eta ||theta||_2^2
        """

    @abstractmethod
    def parameter_dict(
        self,
    ) -> dict[str, float]:
        """
        Human-readable fitted parameters for logging.
        """


# ============================================================
# Tanh mapping
# ============================================================


@MAPPING_REGISTRY.register("tanh")
class TanhMapping(
    RewardCostMapping
):
    """
    Paper Eq. (4)/(11):

        phi(c) = a + b tanh[kappa(c - tau)]

        b > 0
        kappa > 0
    """

    def __init__(
        self,
        *,
        a: float = 0.0,
        b: float = 1.0,
        kappa: float = 1.0,
        tau: float = 0.0,
        positivity_eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.positivity_eps = (
            float(positivity_eps)
        )

        self.a = nn.Parameter(
            _scalar_tensor(a)
        )

        self.raw_b = nn.Parameter(
            _scalar_tensor(
                inverse_softplus(
                    b
                )
            )
        )

        self.raw_kappa = nn.Parameter(
            _scalar_tensor(
                inverse_softplus(
                    kappa
                )
            )
        )

        self.tau = nn.Parameter(
            _scalar_tensor(tau)
        )

    @property
    def b(self) -> Tensor:
        return (
            F.softplus(
                self.raw_b
            )
            + self.positivity_eps
        )

    @property
    def kappa(self) -> Tensor:
        return (
            F.softplus(
                self.raw_kappa
            )
            + self.positivity_eps
        )

    def forward(
        self,
        cost: Tensor,
    ) -> Tensor:

        z = (
            self.kappa
            * (
                cost
                - self.tau
            )
        )

        return (
            self.a
            + self.b
            * torch.tanh(z)
        )

    def regularization_vector(
        self,
    ) -> Tensor:
        # Eq. (5) regularizes theta itself.
        return torch.stack(
            [
                self.a,
                self.b,
                self.kappa,
                self.tau,
            ]
        )

    def parameter_dict(
        self,
    ) -> dict[str, float]:
        return {
            "a": float(
                self.a.detach()
            ),
            "b": float(
                self.b.detach()
            ),
            "kappa": float(
                self.kappa.detach()
            ),
            "tau": float(
                self.tau.detach()
            ),
        }


# ============================================================
# Sigmoid mapping
# ============================================================


@MAPPING_REGISTRY.register("sigmoid")
class SigmoidMapping(
    RewardCostMapping
):
    """
    Appendix Eq. (14):

        phi(c) = a + b sigma[kappa(c - tau)]

    with positive b and kappa.
    """

    def __init__(
        self,
        *,
        a: float = 0.0,
        b: float = 1.0,
        kappa: float = 1.0,
        tau: float = 0.0,
        positivity_eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.positivity_eps = (
            float(positivity_eps)
        )

        self.a = nn.Parameter(
            _scalar_tensor(a)
        )

        self.raw_b = nn.Parameter(
            _scalar_tensor(
                inverse_softplus(b)
            )
        )

        self.raw_kappa = nn.Parameter(
            _scalar_tensor(
                inverse_softplus(
                    kappa
                )
            )
        )

        self.tau = nn.Parameter(
            _scalar_tensor(tau)
        )

    @property
    def b(self) -> Tensor:
        return (
            F.softplus(
                self.raw_b
            )
            + self.positivity_eps
        )

    @property
    def kappa(self) -> Tensor:
        return (
            F.softplus(
                self.raw_kappa
            )
            + self.positivity_eps
        )

    def forward(
        self,
        cost: Tensor,
    ) -> Tensor:

        z = (
            self.kappa
            * (
                cost
                - self.tau
            )
        )

        return (
            self.a
            + self.b
            * torch.sigmoid(z)
        )

    def regularization_vector(
        self,
    ) -> Tensor:
        return torch.stack(
            [
                self.a,
                self.b,
                self.kappa,
                self.tau,
            ]
        )

    def parameter_dict(
        self,
    ) -> dict[str, float]:
        return {
            "a": float(
                self.a.detach()
            ),
            "b": float(
                self.b.detach()
            ),
            "kappa": float(
                self.kappa.detach()
            ),
            "tau": float(
                self.tau.detach()
            ),
        }


# ============================================================
# Quadratic / polynomial mapping
# ============================================================


@MAPPING_REGISTRY.register("polynomial")
class PolynomialMapping(
    RewardCostMapping
):
    """
    Appendix Eq. (17):

        phi(c) = a0 + a1 c + a2 c^2

    The appendix does not impose positivity or monotonicity
    constraints on this quadratic alternative.
    """

    def __init__(
        self,
        *,
        a0: float = 0.0,
        a1: float = 0.0,
        a2: float = 0.0,
    ) -> None:
        super().__init__()

        self.a0 = nn.Parameter(
            _scalar_tensor(a0)
        )

        self.a1 = nn.Parameter(
            _scalar_tensor(a1)
        )

        self.a2 = nn.Parameter(
            _scalar_tensor(a2)
        )

    def forward(
        self,
        cost: Tensor,
    ) -> Tensor:

        return (
            self.a0
            + self.a1 * cost
            + self.a2 * cost.square()
        )

    def regularization_vector(
        self,
    ) -> Tensor:
        return torch.stack(
            [
                self.a0,
                self.a1,
                self.a2,
            ]
        )

    def parameter_dict(
        self,
    ) -> dict[str, float]:
        return {
            "a0": float(
                self.a0.detach()
            ),
            "a1": float(
                self.a1.detach()
            ),
            "a2": float(
                self.a2.detach()
            ),
        }


# ============================================================
# Tiny neural mapping
# ============================================================


@MAPPING_REGISTRY.register("tiny_neural")
class TinyNeuralMapping(
    RewardCostMapping
):
    """
    Appendix Eqs. (20)-(22).

    Normalize cost:

        c_tilde =
            (c - c_min)
            / (c_max - c_min + eps)

    Fixed basis functions:

        h_j = sigmoid[
            alpha(c_tilde - tau_j)
        ]

    with:

        tau_1 = 1/3
        tau_2 = 2/3
        alpha = 6

    and:

        phi(c) =
            a + sum_j w_j h_j

        w_j = softplus(raw_w_j)

    Only:
        a, raw_w1, raw_w2

    are trainable.
    """

    def __init__(
        self,
        *,
        c_min: float,
        c_max: float,
        a: float = 0.0,
        weight_init: float = 0.1,
        alpha: float = 6.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if c_max < c_min:
            raise ValueError(
                "c_max must be >= c_min."
            )

        self.eps = float(eps)

        self.register_buffer(
            "c_min",
            _scalar_tensor(
                c_min
            ),
        )

        self.register_buffer(
            "c_max",
            _scalar_tensor(
                c_max
            ),
        )

        self.register_buffer(
            "alpha",
            _scalar_tensor(
                alpha
            ),
        )

        self.register_buffer(
            "centers",
            torch.tensor(
                [
                    1.0 / 3.0,
                    2.0 / 3.0,
                ],
                dtype=torch.float64,
            ),
        )

        self.a = nn.Parameter(
            _scalar_tensor(a)
        )

        raw_init = (
            inverse_softplus(
                weight_init
            )
        )

        self.raw_weights = nn.Parameter(
            torch.tensor(
                [
                    raw_init,
                    raw_init,
                ],
                dtype=torch.float64,
            )
        )

    @property
    def delta_cost(self) -> Tensor:
        return (
            self.c_max
            - self.c_min
            + self.eps
        )

    @property
    def weights(self) -> Tensor:
        return F.softplus(
            self.raw_weights
        )

    def normalized_cost(
        self,
        cost: Tensor,
    ) -> Tensor:
        return (
            cost
            - self.c_min
        ) / self.delta_cost

    def hidden(
        self,
        cost: Tensor,
    ) -> Tensor:
        """
        Returns shape [..., 2].
        """

        normalized = (
            self.normalized_cost(
                cost
            )
        )

        z = (
            self.alpha
            * (
                normalized.unsqueeze(-1)
                - self.centers
            )
        )

        return torch.sigmoid(z)

    def forward(
        self,
        cost: Tensor,
    ) -> Tensor:

        hidden = self.hidden(
            cost
        )

        return (
            self.a
            + (
                hidden
                * self.weights
            ).sum(
                dim=-1
            )
        )

    def regularization_vector(
        self,
    ) -> Tensor:
        # The appendix states that a, raw_w1 and raw_w2 are the
        # three fitted parameters.
        return torch.cat(
            [
                self.a.reshape(1),
                self.raw_weights,
            ]
        )

    def parameter_dict(
        self,
    ) -> dict[str, float]:

        weights = (
            self.weights
            .detach()
            .cpu()
        )

        return {
            "a": float(
                self.a.detach()
            ),
            "w1": float(
                weights[0]
            ),
            "w2": float(
                weights[1]
            ),
            "raw_w1": float(
                self.raw_weights[
                    0
                ].detach()
            ),
            "raw_w2": float(
                self.raw_weights[
                    1
                ].detach()
            ),
            "c_min": float(
                self.c_min
            ),
            "c_max": float(
                self.c_max
            ),
            "alpha": float(
                self.alpha
            ),
        }
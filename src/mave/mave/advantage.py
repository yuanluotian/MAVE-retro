from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from mave.core.types import (
    MAVEAdvantageRecord,
)
from mave.core.utils import (
    standardize_tensor,
)


@dataclass(frozen=True, slots=True)
class MAVEAdvantageBatch:
    """
    All quantities entering Eq. (7)-(9).
    """

    costs: Tensor
    rewards: Tensor

    j: Tensor
    d1: Tensor
    d2: Tensor

    j_normalized: Tensor
    d1_normalized: Tensor
    d2_normalized: Tensor

    advantage: Tensor

    lambda_cost: float
    beta1: float
    beta2: float

    def __len__(self) -> int:
        return int(
            self.costs.numel()
        )

    def records(
        self,
    ) -> tuple[
        MAVEAdvantageRecord,
        ...
    ]:

        output: list[
            MAVEAdvantageRecord
        ] = []

        for index in range(
            len(self)
        ):
            output.append(
                MAVEAdvantageRecord(
                    rollout_index=index,

                    cost=float(
                        self.costs[
                            index
                        ]
                    ),

                    reward=float(
                        self.rewards[
                            index
                        ]
                    ),

                    j=float(
                        self.j[
                            index
                        ]
                    ),

                    d1=float(
                        self.d1[
                            index
                        ]
                    ),

                    d2=float(
                        self.d2[
                            index
                        ]
                    ),

                    j_normalized=float(
                        self
                        .j_normalized[
                            index
                        ]
                    ),

                    d1_normalized=float(
                        self
                        .d1_normalized[
                            index
                        ]
                    ),

                    d2_normalized=float(
                        self
                        .d2_normalized[
                            index
                        ]
                    ),

                    advantage=float(
                        self.advantage[
                            index
                        ]
                    ),
                )
            )

        return tuple(output)


# ============================================================
# Helpers
# ============================================================


def _vector(
    values: Sequence[float] | Tensor,
) -> Tensor:

    if isinstance(
        values,
        Tensor,
    ):
        tensor = (
            values
            .detach()
            .to(
                dtype=torch.float64,
                device="cpu",
            )
            .reshape(-1)
        )

    else:
        tensor = torch.tensor(
            list(values),
            dtype=torch.float64,
        ).reshape(-1)

    if tensor.numel() == 0:
        raise ValueError(
            "MAVE advantage requires "
            "a non-empty rollout group."
        )

    if not torch.isfinite(
        tensor
    ).all():
        raise ValueError(
            "MAVE advantage input contains "
            "non-finite values."
        )

    return tensor


# ============================================================
# MAVE advantage
# ============================================================


def compute_mave_advantage(
    *,
    costs: Sequence[float] | Tensor,
    rewards: Sequence[float] | Tensor,
    d1: Sequence[float] | Tensor,
    d2: Sequence[float] | Tensor,
    lambda_cost: float = 0.5,
    beta1: float = 1.0,
    beta2: float = 1.0,
    eps: float = 1e-8,
) -> MAVEAdvantageBatch:
    """
    Implements paper Eqs. (7)-(9).

    Eq. (7):
        J_k = r_k - lambda c_k

    Eq. (8):
        X_hat =
            (X - mean(X))
            / (std(X) + eps)

    Eq. (9):
        A_MAVE =
            J_hat
            + beta1 D1_hat
            + beta2 D2_hat
    """

    c = _vector(costs)
    r = _vector(rewards)

    first = _vector(d1)
    second = _vector(d2)

    shapes = {
        c.shape,
        r.shape,
        first.shape,
        second.shape,
    }

    if len(shapes) != 1:
        raise ValueError(
            "costs, rewards, d1, and d2 "
            "must have identical shapes."
        )

    # --------------------------------------------------------
    # Eq. (7)
    # --------------------------------------------------------

    j = (
        r
        - float(
            lambda_cost
        )
        * c
    )

    # --------------------------------------------------------
    # Eq. (8)
    # --------------------------------------------------------

    j_hat = standardize_tensor(
        j,
        eps=eps,
    )

    d1_hat = standardize_tensor(
        first,
        eps=eps,
    )

    d2_hat = standardize_tensor(
        second,
        eps=eps,
    )

    # --------------------------------------------------------
    # Eq. (9)
    # --------------------------------------------------------

    advantage = (
        j_hat
        + float(beta1)
        * d1_hat
        + float(beta2)
        * d2_hat
    )

    return MAVEAdvantageBatch(
        costs=c,
        rewards=r,

        j=j,

        d1=first,
        d2=second,

        j_normalized=j_hat,
        d1_normalized=d1_hat,
        d2_normalized=d2_hat,

        advantage=advantage,

        lambda_cost=float(
            lambda_cost
        ),

        beta1=float(
            beta1
        ),

        beta2=float(
            beta2
        ),
    )
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mave.oracle.base import (
    FeedbackType,
)


# ============================================================
# Cost specification
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class CostSpec:

    base: float

    minimum: float

    maximum: float


# ============================================================
# Paper cost schedule
# ============================================================


COST_SPECS: dict[
    FeedbackType,
    CostSpec,
] = {

    # --------------------------------------------------------
    # L0
    # --------------------------------------------------------

    FeedbackType.NONE:
        CostSpec(
            base=0.00,
            minimum=0.00,
            maximum=0.00,
        ),

    # --------------------------------------------------------
    # L1 Structural
    # --------------------------------------------------------

    FeedbackType.ACTIVITY_ASSESSMENT:
        CostSpec(
            base=0.10,
            minimum=0.08,
            maximum=0.12,
        ),

    FeedbackType.REACTION_CLASS:
        CostSpec(
            base=0.25,
            minimum=0.20,
            maximum=0.30,
        ),

    FeedbackType.REACTION_CENTER:
        CostSpec(
            base=0.30,
            minimum=0.24,
            maximum=0.36,
        ),

    FeedbackType.BOND_DISCONNECTION:
        CostSpec(
            base=0.35,
            minimum=0.28,
            maximum=0.42,
        ),

    # --------------------------------------------------------
    # L2 Evaluative
    # --------------------------------------------------------

    FeedbackType.REACTION_FEASIBILITY:
        CostSpec(
            base=0.80,
            minimum=0.64,
            maximum=0.96,
        ),

    FeedbackType.REACTION_YIELD:
        CostSpec(
            base=1.00,
            minimum=0.80,
            maximum=1.20,
        ),

    FeedbackType.SELECTIVITY_ASSESSMENT:
        CostSpec(
            base=1.20,
            minimum=0.96,
            maximum=1.44,
        ),

    # --------------------------------------------------------
    # L3 Comparative
    # --------------------------------------------------------

    FeedbackType.REACTION_COMPARISON:
        CostSpec(
            base=2.00,
            minimum=1.60,
            maximum=2.40,
        ),

    FeedbackType.ROUTE_COMPARISON:
        CostSpec(
            base=3.50,
            minimum=2.80,
            maximum=4.20,
        ),

    # --------------------------------------------------------
    # L4 Strategic
    # --------------------------------------------------------

    FeedbackType.GOAL_SUGGESTION:
        CostSpec(
            base=5.00,
            minimum=4.00,
            maximum=6.00,
        ),

    FeedbackType.MULTI_STEP_STRATEGY:
        CostSpec(
            base=10.00,
            minimum=8.00,
            maximum=12.00,
        ),
}


# ============================================================
# Cost model
# ============================================================


class AcquisitionCostModel:
    """
    Compute query-specific acquisition cost.

    The paper specifies:
        - base cost
        - ±20% range
        - adjustment according to query complexity

    It does not specify the exact mathematical mapping from
    complexity to cost.

    Default reproduction choice:

        complexity = 0.0 -> minimum
        complexity = 0.5 -> base
        complexity = 1.0 -> maximum

    No random sampling is used.
    """

    def __init__(
        self,
        *,
        complexity_to_cost: (
            Callable[
                [
                    CostSpec,
                    float,
                ],
                float,
            ]
            | None
        ) = None,
    ) -> None:

        self.complexity_to_cost = (
            complexity_to_cost
        )

    def spec(
        self,
        feedback_type: FeedbackType,
    ) -> CostSpec:

        if feedback_type not in COST_SPECS:
            raise KeyError(
                "No acquisition-cost specification for "
                f"{feedback_type.value!r}."
            )

        return COST_SPECS[
            feedback_type
        ]

    def compute(
        self,
        feedback_type: FeedbackType,
        complexity_score: (
            float
            | None
        ) = None,
    ) -> float:

        spec = self.spec(
            feedback_type
        )

        if (
            feedback_type
            == FeedbackType.NONE
        ):
            return 0.0

        # ----------------------------------------------------
        # No complexity estimate:
        # use the paper's base cost.
        # ----------------------------------------------------

        if complexity_score is None:
            return float(
                spec.base
            )

        complexity = float(
            complexity_score
        )

        if not 0.0 <= complexity <= 1.0:
            raise ValueError(
                "complexity_score must lie in [0, 1]."
            )

        # ----------------------------------------------------
        # Optional custom mapping
        # ----------------------------------------------------

        if (
            self.complexity_to_cost
            is not None
        ):

            value = float(
                self.complexity_to_cost(
                    spec,
                    complexity,
                )
            )

        # ----------------------------------------------------
        # Default reproduction mapping
        #
        # [0, 1] -> [minimum, maximum]
        #
        # Because the paper's ranges are symmetric around the
        # base cost, complexity=0.5 gives exactly the base cost.
        # ----------------------------------------------------

        else:

            value = (
                spec.minimum
                + complexity
                * (
                    spec.maximum
                    - spec.minimum
                )
            )

        # Numerical safety.
        value = max(
            spec.minimum,
            min(
                spec.maximum,
                value,
            ),
        )

        return float(
            value
        )
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class CostSpec:
    """
    Relative acquisition-cost specification.
    """

    base: float
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if self.base < 0:
            raise ValueError(
                "base cost must be non-negative."
            )

        if not (
            0 <= self.minimum
            <= self.base
            <= self.maximum
        ):
            raise ValueError(
                "Expected minimum <= base <= maximum."
            )


# ============================================================
# Paper-specified Table 6
# ============================================================


DEFAULT_COST_SPECS: dict[
    str,
    CostSpec,
] = {

    # L0
    "none": CostSpec(
        base=0.00,
        minimum=0.00,
        maximum=0.00,
    ),

    # L1
    "activity_assessment": CostSpec(
        base=0.10,
        minimum=0.08,
        maximum=0.12,
    ),

    "reaction_class": CostSpec(
        base=0.25,
        minimum=0.20,
        maximum=0.30,
    ),

    "reaction_center": CostSpec(
        base=0.30,
        minimum=0.24,
        maximum=0.36,
    ),

    "bond_disconnection": CostSpec(
        base=0.35,
        minimum=0.28,
        maximum=0.42,
    ),

    # L2
    "reaction_feasibility": CostSpec(
        base=0.80,
        minimum=0.64,
        maximum=0.96,
    ),

    "reaction_yield": CostSpec(
        base=1.00,
        minimum=0.80,
        maximum=1.20,
    ),

    "selectivity_assessment": CostSpec(
        base=1.20,
        minimum=0.96,
        maximum=1.44,
    ),

    # L3
    "reaction_comparison": CostSpec(
        base=2.00,
        minimum=1.60,
        maximum=2.40,
    ),

    "route_comparison": CostSpec(
        base=3.50,
        minimum=2.80,
        maximum=4.20,
    ),

    # L4
    "goal_suggestion": CostSpec(
        base=5.00,
        minimum=4.00,
        maximum=6.00,
    ),

    "multi_step_strategy": CostSpec(
        base=10.00,
        minimum=8.00,
        maximum=12.00,
    ),
}


class RelativeCostModel:
    """
    Controlled relative-cost model.

    Paper-specified:
        query cost varies within ±20% of the base cost according
        to query-specific acquisition complexity.

    Paper-unspecified:
        the exact mathematical mapping from complexity to cost.

    Reproduction choice:
        complexity_score in [0, 1] is linearly mapped from the
        minimum to maximum cost. Therefore complexity=0.5
        reproduces the base cost.

        If no complexity score is available, use the base cost.
    """

    def __init__(
        self,
        specs: Mapping[
            str,
            CostSpec,
        ] | None = None,
    ) -> None:

        self.specs = dict(
            specs
            or DEFAULT_COST_SPECS
        )

    def spec(
        self,
        feedback_type: str,
    ) -> CostSpec:

        try:
            return self.specs[
                feedback_type
            ]

        except KeyError as exc:
            raise KeyError(
                f"No cost specification for "
                f"{feedback_type!r}."
            ) from exc

    def compute(
        self,
        feedback_type: str,
        *,
        complexity_score: (
            float | None
        ) = None,
    ) -> float:

        spec = self.spec(
            feedback_type
        )

        if complexity_score is None:
            return float(
                spec.base
            )

        score = float(
            complexity_score
        )

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                "complexity_score must lie in [0, 1]."
            )

        value = (
            spec.minimum
            + score
            * (
                spec.maximum
                - spec.minimum
            )
        )

        return float(value)
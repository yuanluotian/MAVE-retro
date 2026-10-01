from __future__ import annotations

from typing import Any, Mapping, Sequence

from mave.chemistry.reaction import Reaction
from mave.chemistry.route import SynthesisRoute
from mave.oracle.base import (
    FeedbackOracle,
    FeedbackQuery,
    FeedbackType,
    OracleContext,
    OracleResult,
)
from mave.oracle.backends.round_trip import (
    RoundTripBackend,
)
from mave.oracle.backends.yield_model import (
    YieldModelBackend,
)
from mave.oracle.registry import (
    register_oracle,
)


@register_oracle("comparative")
class ComparativeOracle(FeedbackOracle):
    """
    L3 comparative feedback.

    Reaction comparison:
        compare predicted yields.

    Route comparison:
        compare round-trip route scores.
    """

    supported_types = frozenset({
        FeedbackType.REACTION_COMPARISON,
        FeedbackType.ROUTE_COMPARISON,
    })

    def __init__(
        self,
        *,
        yield_backend: YieldModelBackend,
        round_trip_backend: RoundTripBackend,
    ) -> None:

        self.yield_backend = yield_backend
        self.round_trip_backend = round_trip_backend

    def query(
        self,
        query: FeedbackQuery,
        context: OracleContext,
    ) -> OracleResult:

        if query.feedback_type == FeedbackType.REACTION_COMPARISON:
            return self._compare_reactions(
                query
            )

        if query.feedback_type == FeedbackType.ROUTE_COMPARISON:
            return self._compare_routes(
                query
            )

        raise ValueError(
            f"Unsupported comparative feedback type: "
            f"{query.feedback_type.value}"
        )

    # ========================================================
    # Reaction comparison
    # ========================================================

    def _compare_reactions(
        self,
        query: FeedbackQuery,
    ) -> OracleResult:

        reaction_a, reaction_b = (
            self._require_pair(
                query.alternatives,
                Reaction,
                "reaction",
            )
        )

        result_a = (
            self.yield_backend.predict(
                reaction_a
            )
        )

        result_b = (
            self.yield_backend.predict(
                reaction_b
            )
        )

        score_a = float(
            result_a.content
        )

        score_b = float(
            result_b.content
        )

        if score_a > score_b:

            conclusion = (
                "Candidate reaction 1 receives stronger "
                "yield-based support."
            )

        elif score_b > score_a:

            conclusion = (
                "Candidate reaction 2 receives stronger "
                "yield-based support."
            )

        else:

            conclusion = (
                "The two candidate reactions receive "
                "equal yield-based support."
            )

        text = (
            "Candidate reaction 1 has a predicted normalized "
            f"yield of {score_a:.3f}, while candidate reaction 2 "
            f"has a predicted normalized yield of {score_b:.3f}. "
            f"{conclusion}"
        )

        return OracleResult(
            content=text,

            complexity_score=(
                self._mean_complexity(
                    result_a.complexity_score,
                    result_b.complexity_score,
                )
            ),

            metadata={
                "feedback_type":
                    FeedbackType
                    .REACTION_COMPARISON
                    .value,

                "backend":
                    "yield_model",
            },
        )

    # ========================================================
    # Route comparison
    # ========================================================

    def _compare_routes(
        self,
        query: FeedbackQuery,
    ) -> OracleResult:

        route_a, route_b = (
            self._require_pair(
                query.alternatives,
                SynthesisRoute,
                "route",
            )
        )

        result_a = (
            self.round_trip_backend
            .route_score(
                route_a
            )
        )

        result_b = (
            self.round_trip_backend
            .route_score(
                route_b
            )
        )

        score_a = self._extract_score(
            result_a.content
        )

        score_b = self._extract_score(
            result_b.content
        )

        if score_a > score_b:

            conclusion = (
                "Partial route 1 receives stronger "
                "round-trip support."
            )

        elif score_b > score_a:

            conclusion = (
                "Partial route 2 receives stronger "
                "round-trip support."
            )

        else:

            conclusion = (
                "The two partial routes receive equal "
                "round-trip support."
            )

        text = (
            "Partial route 1 has a round-trip score of "
            f"{score_a:.3f}, while partial route 2 has a "
            f"score of {score_b:.3f}. {conclusion}"
        )

        return OracleResult(
            content=text,

            complexity_score=(
                self._mean_complexity(
                    result_a.complexity_score,
                    result_b.complexity_score,
                )
            ),

            metadata={
                "feedback_type":
                    FeedbackType
                    .ROUTE_COMPARISON
                    .value,

                "backend":
                    "round_trip",
            },
        )

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _require_pair(
        values: Sequence[Any],
        expected_type: type,
        name: str,
    ) -> tuple[Any, Any]:

        if len(values) != 2:

            raise ValueError(
                f"{name.capitalize()} comparison "
                f"requires exactly two alternatives."
            )

        a, b = values

        if not isinstance(
            a,
            expected_type,
        ) or not isinstance(
            b,
            expected_type,
        ):
            raise TypeError(
                f"Both alternatives must be "
                f"{expected_type.__name__} objects."
            )

        return a, b

    @staticmethod
    def _extract_score(
        content: Any,
    ) -> float:

        if isinstance(
            content,
            (int, float),
        ):
            return float(content)

        if isinstance(
            content,
            Mapping,
        ):

            for key in (
                "score",
                "route_score",
                "value",
            ):

                if key in content:
                    return float(
                        content[key]
                    )

        raise TypeError(
            "Route backend output must contain "
            "a scalar score."
        )

    @staticmethod
    def _mean_complexity(
        *values: float | None,
    ) -> float | None:

        available = [
            float(value)
            for value in values
            if value is not None
        ]

        if not available:
            return None

        return sum(
            available
        ) / len(
            available
        )
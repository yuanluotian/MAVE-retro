from __future__ import annotations

from itertools import combinations

from mave_repro.core.types import (
    FeedbackLevel,
    FeedbackQuery,
)
from mave_repro.oracle.base import (
    BackendResult,
    FeedbackOracle,
    OracleContext,
)
from mave_repro.oracle.backends.round_trip import (
    RoundTripBackend,
)
from mave_repro.oracle.backends.yield_model import (
    YieldModelBackend,
)
from mave_repro.oracle.registry import (
    register_oracle,
)


@register_oracle(
    "comparative"
)
class ComparativeOracle(
    FeedbackOracle
):
    """
    L3 comparative feedback.

    reaction_comparison:
        compare two candidate reactions using predicted yield.

    route_comparison:
        compare two partial synthesis routes using the common
        round-trip backend.
    """

    level = FeedbackLevel.L3

    def __init__(
        self,
        *,
        yield_backend: YieldModelBackend,
        round_trip_backend: (
            RoundTripBackend
        ),
        comparison_pool_size: (
            int | None
        ) = 5,
    ) -> None:

        self.yield_backend = (
            yield_backend
        )

        self.round_trip_backend = (
            round_trip_backend
        )

        # Paper does not specify how many candidate pairs are
        # exposed to the escalation policy.
        self.comparison_pool_size = (
            comparison_pool_size
        )

    @property
    def feedback_types(
        self,
    ) -> tuple[str, ...]:
        return (
            "reaction_comparison",
            "route_comparison",
        )

    def available_queries(
        self,
        context: OracleContext,
    ) -> tuple[
        FeedbackQuery,
        ...
    ]:

        queries: list[
            FeedbackQuery
        ] = []

        # ----------------------------------------------------
        # Reaction comparisons
        # ----------------------------------------------------

        num_candidates = len(
            context.candidates
        )

        if (
            self.comparison_pool_size
            is None
        ):
            pool_size = (
                num_candidates
            )

        else:
            pool_size = min(
                num_candidates,
                self.comparison_pool_size,
            )

        for i, j in combinations(
            range(pool_size),
            2,
        ):
            queries.append(
                FeedbackQuery(
                    level=self.level,
                    feedback_type=(
                        "reaction_comparison"
                    ),
                    payload={
                        "reaction_indices":
                            [i, j],
                    },
                )
            )

        # ----------------------------------------------------
        # Route comparisons
        # ----------------------------------------------------

        routes = (
            context.state.route,
            *context.route_alternatives,
        )

        for i, j in combinations(
            range(len(routes)),
            2,
        ):
            queries.append(
                FeedbackQuery(
                    level=self.level,
                    feedback_type=(
                        "route_comparison"
                    ),
                    payload={
                        "route_indices":
                            [i, j],
                    },
                )
            )

        return tuple(queries)

    # ========================================================
    # Evaluation
    # ========================================================

    def evaluate(
        self,
        context: OracleContext,
        query: FeedbackQuery,
    ) -> BackendResult:

        self.validate_query(
            query
        )

        if (
            query.feedback_type
            == "reaction_comparison"
        ):
            return (
                self._compare_reactions(
                    context,
                    query,
                )
            )

        if (
            query.feedback_type
            == "route_comparison"
        ):
            return (
                self._compare_routes(
                    context,
                    query,
                )
            )

        raise RuntimeError(
            "Unreachable feedback type."
        )

    # ========================================================
    # Reaction comparison
    # ========================================================

    def _compare_reactions(
        self,
        context: OracleContext,
        query: FeedbackQuery,
    ) -> BackendResult:

        i, j = [
            int(value)
            for value
            in query.payload[
                "reaction_indices"
            ]
        ]

        reaction_i = (
            context.candidates[i]
        )

        reaction_j = (
            context.candidates[j]
        )

        result_i = (
            self.yield_backend.predict(
                reaction_i
            )
        )

        result_j = (
            self.yield_backend.predict(
                reaction_j
            )
        )

        yield_i = float(
            result_i.content
        )

        yield_j = float(
            result_j.content
        )

        if yield_i > yield_j:
            preferred = i

        elif yield_j > yield_i:
            preferred = j

        else:
            preferred = None

        complexity_values = [
            value
            for value in (
                result_i.complexity_score,
                result_j.complexity_score,
            )
            if value is not None
        ]

        complexity = (
            None
            if not complexity_values
            else sum(
                complexity_values
            ) / len(
                complexity_values
            )
        )

        return BackendResult(
            content={
                "reaction_a": i,
                "reaction_b": j,
                "yield_a": yield_i,
                "yield_b": yield_j,
                "preferred_reaction": (
                    preferred
                ),
            },
            complexity_score=(
                complexity
            ),
            metadata={
                "backend": (
                    "yield_model"
                ),
            },
        )

    # ========================================================
    # Route comparison
    # ========================================================

    def _compare_routes(
        self,
        context: OracleContext,
        query: FeedbackQuery,
    ) -> BackendResult:

        routes = (
            context.state.route,
            *context.route_alternatives,
        )

        i, j = [
            int(value)
            for value
            in query.payload[
                "route_indices"
            ]
        ]

        result_i = (
            self.round_trip_backend
            .route_score(
                routes[i]
            )
        )

        result_j = (
            self.round_trip_backend
            .route_score(
                routes[j]
            )
        )

        score_i = float(
            result_i.content
        )

        score_j = float(
            result_j.content
        )

        if score_i > score_j:
            preferred = i

        elif score_j > score_i:
            preferred = j

        else:
            preferred = None

        complexity_values = [
            value
            for value in (
                result_i.complexity_score,
                result_j.complexity_score,
            )
            if value is not None
        ]

        complexity = (
            None
            if not complexity_values
            else sum(
                complexity_values
            ) / len(
                complexity_values
            )
        )

        return BackendResult(
            content={
                "route_a": i,
                "route_b": j,
                "score_a": score_i,
                "score_b": score_j,
                "preferred_route": (
                    preferred
                ),
            },
            complexity_score=(
                complexity
            ),
            metadata={
                "backend": (
                    "round_trip"
                ),
            },
        )
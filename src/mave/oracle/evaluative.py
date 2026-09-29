from __future__ import annotations

from mave.core.types import (
    FeedbackLevel,
    FeedbackQuery,
)
from mave.oracle.base import (
    BackendResult,
    FeedbackOracle,
    OracleContext,
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


@register_oracle(
    "evaluative"
)
class EvaluativeOracle(
    FeedbackOracle
):
    """
    L2 candidate-level feedback.

      reaction feasibility
      reaction yield
      selectivity assessment
    """

    level = FeedbackLevel.L2

    def __init__(
        self,
        *,
        yield_backend: YieldModelBackend,
        round_trip_backend: (
            RoundTripBackend
        ),
    ) -> None:

        self.yield_backend = (
            yield_backend
        )

        self.round_trip_backend = (
            round_trip_backend
        )

    @property
    def feedback_types(
        self,
    ) -> tuple[str, ...]:
        return (
            "reaction_feasibility",
            "reaction_yield",
            "selectivity_assessment",
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

        for index, reaction in enumerate(
            context.candidates
        ):

            payload = {
                "reaction_index": index,
                "reaction_smiles": (
                    reaction.reaction_smiles
                ),
            }

            for feedback_type in (
                self.feedback_types
            ):
                queries.append(
                    FeedbackQuery(
                        level=self.level,
                        feedback_type=(
                            feedback_type
                        ),
                        payload=payload,
                    )
                )

        return tuple(queries)

    def _reaction(
        self,
        context: OracleContext,
        query: FeedbackQuery,
    ):

        index = int(
            query.payload[
                "reaction_index"
            ]
        )

        if not (
            0 <= index
            < len(
                context.candidates
            )
        ):
            raise IndexError(
                "Invalid reaction_index."
            )

        return (
            context.candidates[index]
        )

    def evaluate(
        self,
        context: OracleContext,
        query: FeedbackQuery,
    ) -> BackendResult:

        self.validate_query(
            query
        )

        reaction = self._reaction(
            context,
            query,
        )

        if (
            query.feedback_type
            == "reaction_yield"
        ):
            return (
                self.yield_backend.predict(
                    reaction
                )
            )

        if (
            query.feedback_type
            == "reaction_feasibility"
        ):
            return (
                self.round_trip_backend
                .reaction_feasibility(
                    reaction
                )
            )

        if (
            query.feedback_type
            == "selectivity_assessment"
        ):
            return (
                self.round_trip_backend
                .selectivity_assessment(
                    reaction
                )
            )

        raise RuntimeError(
            "Unreachable feedback type."
        )
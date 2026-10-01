from __future__ import annotations

from typing import Any, Mapping

from mave.chemistry.reaction import Reaction
from mave.oracle.base import (
    BackendResult,
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


@register_oracle("evaluative")
class EvaluativeOracle(FeedbackOracle):
    """
    L2 candidate-level evaluative feedback.
    """

    supported_types = frozenset({
        FeedbackType.REACTION_FEASIBILITY,
        FeedbackType.REACTION_YIELD,
        FeedbackType.SELECTIVITY_ASSESSMENT,
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

        if not self.supports(query.feedback_type):
            raise ValueError(
                f"Unsupported evaluative feedback type: "
                f"{query.feedback_type.value}"
            )

        reaction = self._require_reaction(
            query
        )

        if query.feedback_type == FeedbackType.REACTION_YIELD:

            result = self.yield_backend.predict(
                reaction
            )

            value = float(
                result.content
            )

            text = (
                "The predicted normalized reaction yield "
                f"is {value:.3f}."
            )

        elif query.feedback_type == FeedbackType.REACTION_FEASIBILITY:

            result = (
                self.round_trip_backend
                .reaction_feasibility(
                    reaction
                )
            )

            text = self._format_feasibility(
                result.content
            )

        else:

            result = (
                self.round_trip_backend
                .selectivity_assessment(
                    reaction
                )
            )

            text = self._format_selectivity(
                result.content
            )

        return self._make_result(
            text=text,
            backend_result=result,
            feedback_type=query.feedback_type,
        )

    @staticmethod
    def _require_reaction(
        query: FeedbackQuery,
    ) -> Reaction:

        if not isinstance(
            query.target,
            Reaction,
        ):
            raise TypeError(
                "L2 feedback requires "
                "query.target to be a Reaction."
            )

        return query.target

    @staticmethod
    def _format_feasibility(
        data: Any,
    ) -> str:

        if isinstance(data, str):
            return data

        if isinstance(data, bool):

            if data:
                return (
                    "The candidate reaction is predicted "
                    "to be feasible."
                )

            return (
                "The candidate reaction is predicted "
                "to be infeasible."
            )

        if isinstance(data, Mapping):

            feasible = data.get(
                "feasible"
            )

            score = data.get(
                "score"
            )

            if feasible is not None:

                text = (
                    "The candidate reaction is predicted "
                    f"to be {'feasible' if feasible else 'infeasible'}."
                )

                if score is not None:
                    text += (
                        f" The round-trip feasibility score "
                        f"is {float(score):.3f}."
                    )

                return text

            if score is not None:

                return (
                    "The candidate reaction receives a "
                    f"feasibility score of {float(score):.3f}."
                )

        if isinstance(data, (int, float)):

            return (
                "The candidate reaction receives a "
                f"feasibility score of {float(data):.3f}."
            )

        return (
            "The feasibility backend returned the following "
            f"evidence: {data}."
        )

    @staticmethod
    def _format_selectivity(
        data: Any,
    ) -> str:

        if isinstance(data, str):
            return data

        if isinstance(data, Mapping):

            assessment = (
                data.get("selectivity")
                or data.get("assessment")
                or data.get("label")
            )

            preferred_site = data.get(
                "preferred_site"
            )

            score = data.get(
                "score"
            )

            if preferred_site is not None:

                text = (
                    f"The preferred reactive site is "
                    f"{preferred_site}."
                )

                if score is not None:
                    text += (
                        f" The selectivity score is "
                        f"{float(score):.3f}."
                    )

                return text

            if assessment is not None:

                text = (
                    "The expected selectivity of the "
                    f"candidate reaction is {assessment}."
                )

                if score is not None:
                    text += (
                        f" The selectivity score is "
                        f"{float(score):.3f}."
                    )

                return text

            if score is not None:

                return (
                    "The candidate reaction receives a "
                    f"selectivity score of {float(score):.3f}."
                )

        if isinstance(data, (int, float)):

            return (
                "The candidate reaction receives a "
                f"selectivity score of {float(data):.3f}."
            )

        return (
            "The selectivity backend returned the following "
            f"evidence: {data}."
        )

    @staticmethod
    def _make_result(
        *,
        text: str,
        backend_result: BackendResult,
        feedback_type: FeedbackType,
    ) -> OracleResult:

        metadata = dict(
            backend_result.metadata
        )

        metadata["feedback_type"] = (
            feedback_type.value
        )

        return OracleResult(
            content=text,
            complexity_score=(
                backend_result.complexity_score
            ),
            metadata=metadata,
        )
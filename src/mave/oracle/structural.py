from __future__ import annotations

from typing import Any, Mapping

from mave.oracle.base import (
    BackendResult,
    FeedbackOracle,
    FeedbackQuery,
    FeedbackType,
    OracleContext,
    OracleResult,
    coerce_feedback_type,
)
from mave.oracle.backends.round_trip import (
    RoundTripBackend,
)
from mave.oracle.context import (
    native_candidates,
    native_selected_molecule,
)
from mave.oracle.registry import (
    register_oracle,
)


@register_oracle("structural")
class StructuralOracle(FeedbackOracle):
    """
    L1 structural feedback.

    The round-trip backend returns structured evidence.
    This oracle converts that evidence into natural language.
    """

    supported_types = frozenset({
        FeedbackType.ACTIVITY_ASSESSMENT,
        FeedbackType.REACTION_CLASS,
        FeedbackType.REACTION_CENTER,
        FeedbackType.BOND_DISCONNECTION,
    })

    def __init__(
        self,
        *,
        backend: RoundTripBackend,
    ) -> None:
        self.backend = backend

    def query(
        self,
        query: FeedbackQuery,
        context: OracleContext,
    ) -> OracleResult:

        if not self.supports(query.feedback_type):
            raise ValueError(
                f"Unsupported structural feedback type: "
                f"{query.feedback_type.value}"
            )

        product = native_selected_molecule(context)
        candidates = native_candidates(context)

        if query.feedback_type == FeedbackType.ACTIVITY_ASSESSMENT:
            result = self.backend.activity_assessment(
                product,
                candidates,
            )
            text = self._format_activity(
                result.content
            )

        elif query.feedback_type == FeedbackType.REACTION_CLASS:
            result = self.backend.reaction_class(
                product,
                candidates,
            )
            text = self._format_reaction_class(
                result.content
            )

        elif query.feedback_type == FeedbackType.REACTION_CENTER:
            result = self.backend.reaction_center(
                product,
                candidates,
            )
            text = self._format_reaction_center(
                result.content
            )

        else:
            result = self.backend.bond_disconnection(
                product,
                candidates,
            )
            text = self._format_bond_disconnection(
                result.content
            )

        return self._make_result(
            text=text,
            backend_result=result,
            feedback_type=query.feedback_type,
        )

    @staticmethod
    def _format_activity(
        data: Any,
    ) -> str:

        if isinstance(data, str):
            return data

        if isinstance(data, Mapping):

            assessment = (
                data.get("activity")
                or data.get("assessment")
                or data.get("label")
            )

            score = data.get("score")

            if assessment is not None:
                text = (
                    f"The local functional-group reactivity "
                    f"is assessed as {assessment}."
                )

                if score is not None:
                    text += (
                        f" The associated round-trip score "
                        f"is {float(score):.3f}."
                    )

                return text

            if score is not None:
                return (
                    "The current product receives a local "
                    f"reactivity score of {float(score):.3f}."
                )

        if isinstance(data, (int, float)):
            return (
                "The current product receives a local "
                f"reactivity score of {float(data):.3f}."
            )

        return (
            "The structural backend identified plausible "
            f"local reactivity evidence: {data}."
        )

    @staticmethod
    def _format_reaction_class(
        data: Any,
    ) -> str:

        if isinstance(data, str):
            return (
                f"A plausible reaction class for the current "
                f"product is {data}."
            )

        if isinstance(data, Mapping):

            label = (
                data.get("reaction_class")
                or data.get("class")
                or data.get("label")
            )

            score = data.get("score")

            if label is not None:
                text = (
                    f"A plausible reaction class for the "
                    f"current product is {label}."
                )

                if score is not None:
                    text += (
                        f" Its round-trip support score "
                        f"is {float(score):.3f}."
                    )

                return text

        if isinstance(data, (int, float)):
            return (
                "The round-trip backend returned a reaction-class "
                f"support score of {float(data):.3f}."
            )

        return (
            "The structural backend returned reaction-class "
            f"evidence: {data}."
        )

    @staticmethod
    def _format_reaction_center(
        data: Any,
    ) -> str:

        if isinstance(data, str):
            return data

        if isinstance(data, Mapping):

            center = (
                data.get("reaction_center")
                or data.get("center")
                or data.get("atom_indices")
                or data.get("atoms")
            )

            score = data.get("score")

            if center is not None:
                text = (
                    "A plausible reaction center involves "
                    f"{center}."
                )

                if score is not None:
                    text += (
                        f" Its round-trip support score "
                        f"is {float(score):.3f}."
                    )

                return text

        if isinstance(data, (int, float)):
            return (
                "The round-trip backend returned a "
                f"reaction-center score of {float(data):.3f}."
            )

        return (
            "The structural backend identified the following "
            f"reaction-center evidence: {data}."
        )

    @staticmethod
    def _format_bond_disconnection(
        data: Any,
    ) -> str:

        if isinstance(data, str):
            return data

        if isinstance(data, Mapping):

            bond = (
                data.get("bond")
                or data.get("bond_indices")
                or data.get("disconnection")
            )

            score = data.get("score")

            if bond is not None:
                text = (
                    "A plausible retrosynthetic disconnection "
                    f"is associated with bond {bond}."
                )

                if score is not None:
                    text += (
                        f" Its round-trip support score "
                        f"is {float(score):.3f}."
                    )

                return text

        if isinstance(data, (int, float)):
            return (
                "The round-trip backend returned a bond-"
                f"disconnection score of {float(data):.3f}."
            )

        return (
            "The structural backend returned plausible "
            f"bond-disconnection evidence: {data}."
        )

    @staticmethod
    def _make_result(
        *,
        text: str,
        backend_result: BackendResult,
        feedback_type: FeedbackType | str,
    ) -> OracleResult:

        metadata = dict(
            backend_result.metadata
        )

        metadata["feedback_type"] = (
            coerce_feedback_type(feedback_type).value
        )

        return OracleResult(
            content=text,
            complexity_score=(
                backend_result.complexity_score
            ),
            metadata=metadata,
        )

from __future__ import annotations

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
from mave_repro.oracle.registry import (
    register_oracle,
)


@register_oracle(
    "structural"
)
class StructuralOracle(
    FeedbackOracle
):
    """
    L1 structural feedback.

    Paper-defined feedback types:
      - activity assessment
      - reaction class
      - reaction center
      - bond disconnection
    """

    level = FeedbackLevel.L1

    def __init__(
        self,
        backend: RoundTripBackend,
    ) -> None:
        self.backend = backend

    @property
    def feedback_types(
        self,
    ) -> tuple[str, ...]:
        return (
            "activity_assessment",
            "reaction_class",
            "reaction_center",
            "bond_disconnection",
        )

    def available_queries(
        self,
        context: OracleContext,
    ) -> tuple[
        FeedbackQuery,
        ...
    ]:

        common_payload = {
            "selected_index": (
                context.selected_index
            ),
            "product_smiles": (
                context
                .selected_molecule
                .canonical_smiles
            ),
        }

        return tuple(
            FeedbackQuery(
                level=self.level,
                feedback_type=(
                    feedback_type
                ),
                payload=(
                    common_payload
                ),
            )
            for feedback_type
            in self.feedback_types
        )

    def evaluate(
        self,
        context: OracleContext,
        query: FeedbackQuery,
    ) -> BackendResult:

        self.validate_query(
            query
        )

        product = (
            context.selected_molecule
        )

        candidates = (
            context.candidates
        )

        operation = {
            "activity_assessment":
                self.backend
                .activity_assessment,

            "reaction_class":
                self.backend
                .reaction_class,

            "reaction_center":
                self.backend
                .reaction_center,

            "bond_disconnection":
                self.backend
                .bond_disconnection,
        }[
            query.feedback_type
        ]

        return operation(
            product,
            candidates,
        )
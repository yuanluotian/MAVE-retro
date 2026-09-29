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
from mave.oracle.backends.depth_traversal import (
    DepthTraversalBackend,
)
from mave.oracle.registry import (
    register_oracle,
)


@register_oracle(
    "strategic"
)
class StrategicOracle(
    FeedbackOracle
):
    """
    L4 search-level strategic feedback.

    Paper-defined query types:
      - goal suggestion
      - multi-step strategy
    """

    level = FeedbackLevel.L4

    def __init__(
        self,
        backend: DepthTraversalBackend,
    ) -> None:
        self.backend = backend

    @property
    def feedback_types(
        self,
    ) -> tuple[str, ...]:
        return (
            "goal_suggestion",
            "multi_step_strategy",
        )

    def available_queries(
        self,
        context: OracleContext,
    ) -> tuple[
        FeedbackQuery,
        ...
    ]:

        payload = {
            "state_id": (
                context.state_id
            ),
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
                payload=payload,
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

        if (
            query.feedback_type
            == "goal_suggestion"
        ):
            return (
                self.backend
                .goal_suggestion(
                    context
                )
            )

        if (
            query.feedback_type
            == "multi_step_strategy"
        ):
            return (
                self.backend
                .multi_step_strategy(
                    context
                )
            )

        raise RuntimeError(
            "Unreachable feedback type."
        )
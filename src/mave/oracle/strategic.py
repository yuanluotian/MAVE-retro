from __future__ import annotations

from typing import Any, Mapping, Sequence

from mave.oracle.base import (
    BackendResult,
    FeedbackOracle,
    FeedbackQuery,
    FeedbackType,
    OracleContext,
    OracleResult,
)
from mave.oracle.backends.depth_traversal import (
    DepthTraversalBackend,
)
from mave.oracle.registry import (
    register_oracle,
)


@register_oracle("strategic")
class StrategicOracle(FeedbackOracle):
    """
    L4 strategic feedback.

    depth_traversal.py:
        AND-OR search -> structured evidence

    strategic.py:
        structured evidence -> natural-language guidance
    """

    supported_types = frozenset({
        FeedbackType.GOAL_SUGGESTION,
        FeedbackType.MULTI_STEP_STRATEGY,
    })

    def __init__(
        self,
        *,
        backend: DepthTraversalBackend,
    ) -> None:

        self.backend = backend

    def query(
        self,
        query: FeedbackQuery,
        context: OracleContext,
    ) -> OracleResult:

        if query.feedback_type == FeedbackType.GOAL_SUGGESTION:

            result = (
                self.backend
                .goal_suggestion(
                    context
                )
            )

            text = (
                self._format_goal_suggestion(
                    result.content
                )
            )

        elif query.feedback_type == FeedbackType.MULTI_STEP_STRATEGY:

            result = (
                self.backend
                .multi_step_strategy(
                    context
                )
            )

            text = (
                self._format_multi_step_strategy(
                    result.content
                )
            )

        else:

            raise ValueError(
                f"Unsupported strategic feedback type: "
                f"{query.feedback_type.value}"
            )

        return self._make_result(
            text=text,
            backend_result=result,
            feedback_type=query.feedback_type,
        )

    # ========================================================
    # Goal suggestion
    # ========================================================

    def _format_goal_suggestion(
        self,
        data: Any,
    ) -> str:

        if not isinstance(
            data,
            Mapping,
        ):

            return (
                "The strategic search returned the following "
                f"subgoal evidence: {data}."
            )

        subgoals = data.get(
            "subgoals",
            [],
        )

        if not subgoals:

            return (
                "The bounded local search did not identify a "
                "clear intermediate subgoal. Continue exploring "
                "the current reaction candidates without "
                "committing to a specific intermediate."
            )

        lines = [
            "Potential intermediate subgoals:"
        ]

        for index, item in enumerate(
            subgoals,
            start=1,
        ):

            smiles = str(
                item.get(
                    "smiles",
                    "unknown intermediate",
                )
            )

            support = (
                self._subgoal_support(
                    item
                )
            )

            lines.append(
                f"{index}. Consider {smiles}. "
                f"{support}"
            )

        if len(subgoals) > 1:

            lines.append(
                "These are alternative subgoals rather than "
                "mandatory intermediates; earlier suggestions "
                "receive stronger support from the bounded "
                "local search."
            )

        return "\n".join(
            lines
        )

    @staticmethod
    def _subgoal_support(
        item: Mapping[str, Any],
    ) -> str:

        solved = int(
            item.get(
                "solved_support",
                0,
            )
        )

        frontier = int(
            item.get(
                "frontier_support",
                0,
            )
        )

        occurrence = int(
            item.get(
                "occurrence_count",
                0,
            )
        )

        depth = int(
            item.get(
                "minimum_depth",
                0,
            )
        )

        if solved >= 2:

            return (
                "This intermediate is repeatedly supported by "
                "locally viable downstream decompositions and "
                "is a strong candidate for further planning."
            )

        if solved >= 1:

            return (
                "This intermediate is supported by at least one "
                "locally viable downstream decomposition and "
                "may provide a useful planning subgoal."
            )

        if frontier >= 2 or occurrence >= 2:

            return (
                "This intermediate appears across multiple "
                "plausible partial decompositions, although "
                "its downstream synthesis remains unresolved "
                "within the bounded search."
            )

        if depth <= 1:

            return (
                "This intermediate appears as a nearby "
                "alternative subgoal, but currently has "
                "limited downstream support."
            )

        return (
            "This is a plausible but weakly supported "
            "alternative intermediate."
        )

    # ========================================================
    # Multi-step strategy
    # ========================================================

    def _format_multi_step_strategy(
        self,
        data: Any,
    ) -> str:

        if not isinstance(
            data,
            Mapping,
        ):

            return (
                "The strategic search returned the following "
                f"multi-step evidence: {data}."
            )

        strategies = data.get(
            "strategies",
            [],
        )

        if not strategies:

            return (
                "The bounded local search did not reveal a clear "
                "multi-step transformation pattern. Continue "
                "planning without committing to a specific "
                "transformation ordering."
            )

        lines = [
            "Possible multi-step strategies:"
        ]

        for index, item in enumerate(
            strategies,
            start=1,
        ):

            transformations = (
                item.get(
                    "transformations",
                    [],
                )
            )

            labels = [
                self._transformation_label(
                    step
                )
                for step
                in transformations
            ]

            labels = [
                label
                for label in labels
                if label
            ]

            if not labels:
                continue

            direction = (
                self._strategy_direction(
                    labels
                )
            )

            support = (
                self._strategy_support(
                    item
                )
            )

            lines.append(
                f"{index}. {direction} "
                f"{support}"
            )

        if len(lines) == 1:

            return (
                "The bounded local search did not reveal a "
                "clear interpretable multi-step transformation "
                "pattern."
            )

        lines.append(
            "These suggestions are high-level strategic "
            "directions rather than complete synthesis routes, "
            "and alternative transformation orderings may "
            "remain viable."
        )

        return "\n".join(
            lines
        )

    @staticmethod
    def _transformation_label(
        item: Any,
    ) -> str | None:

        if isinstance(
            item,
            str,
        ):
            return item

        if not isinstance(
            item,
            Mapping,
        ):
            return str(item)

        reaction_class = (
            item.get(
                "reaction_class"
            )
        )

        if reaction_class:
            return str(
                reaction_class
            )

        template_id = (
            item.get(
                "template_id"
            )
        )

        if template_id is not None:
            return (
                f"reaction template "
                f"{template_id}"
            )

        key = item.get(
            "key"
        )

        if key:
            return str(key)

        return None

    @staticmethod
    def _strategy_direction(
        labels: Sequence[str],
    ) -> str:

        if len(labels) == 1:

            return (
                f"Consider {labels[0]} as the next "
                f"high-level transformation."
            )

        if len(labels) == 2:

            return (
                f"First consider {labels[0]}, followed by "
                f"{labels[1]}."
            )

        return (
            "Consider the transformation sequence "
            + " -> ".join(
                labels
            )
            + "."
        )

    @staticmethod
    def _strategy_support(
        item: Mapping[str, Any],
    ) -> str:

        solved = int(
            item.get(
                "solved_support",
                0,
            )
        )

        frontier = int(
            item.get(
                "frontier_support",
                0,
            )
        )

        occurrence = int(
            item.get(
                "occurrence_count",
                0,
            )
        )

        if solved >= 2:

            return (
                "This ordering is repeatedly supported by "
                "locally viable retrosynthetic branches."
            )

        if solved >= 1:

            return (
                "This ordering is supported by a locally viable "
                "branch and is a plausible direction for "
                "continued planning."
            )

        if frontier >= 2 or occurrence >= 2:

            return (
                "This ordering appears in multiple plausible "
                "partial branches, but remains unresolved "
                "within the bounded search."
            )

        return (
            "This is a plausible but weakly supported "
            "strategic direction."
        )

    # ========================================================
    # Result
    # ========================================================

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
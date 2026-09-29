from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from mave.oracle.base import (
    BackendResult,
    OracleContext,
)
from mave.oracle.registry import (
    register_backend,
)


class DepthTraversalBackend(ABC):
    """
    Backend for L4 strategic feedback.

    The backend may internally perform a depth-limited traversal
    of the environment-defined AND-OR search space.

    Only compact feedback is returned to the planner. The
    internal search tree must remain private.
    """

    @abstractmethod
    def goal_suggestion(
        self,
        context: OracleContext,
    ) -> BackendResult:
        ...

    @abstractmethod
    def multi_step_strategy(
        self,
        context: OracleContext,
    ) -> BackendResult:
        ...


@register_backend(
    "callable_depth_traversal"
)
class CallableDepthTraversalBackend(
    DepthTraversalBackend
):
    """
    Adapter around an externally implemented depth-limited
    search routine.

    The paper does not specify the traversal depth, so max_depth
    is intentionally optional.
    """

    def __init__(
        self,
        *,
        goal_suggestion_fn: Callable[
            [OracleContext],
            Any,
        ],
        multi_step_strategy_fn: Callable[
            [OracleContext],
            Any,
        ],
        max_depth: int | None = None,
    ) -> None:

        if (
            max_depth is not None
            and max_depth <= 0
        ):
            raise ValueError(
                "max_depth must be positive."
            )

        self.goal_suggestion_fn = (
            goal_suggestion_fn
        )

        self.multi_step_strategy_fn = (
            multi_step_strategy_fn
        )

        self.max_depth = max_depth

    @staticmethod
    def _wrap(
        value: Any,
        operation: str,
    ) -> BackendResult:

        if isinstance(
            value,
            BackendResult,
        ):
            return value

        return BackendResult(
            content=value,
            metadata={
                "backend": (
                    "depth_traversal"
                ),
                "operation": operation,
            },
        )

    def goal_suggestion(
        self,
        context: OracleContext,
    ) -> BackendResult:

        value = (
            self.goal_suggestion_fn(
                context
            )
        )

        return self._wrap(
            value,
            "goal_suggestion",
        )

    def multi_step_strategy(
        self,
        context: OracleContext,
    ) -> BackendResult:

        value = (
            self.multi_step_strategy_fn(
                context
            )
        )

        return self._wrap(
            value,
            "multi_step_strategy",
        )
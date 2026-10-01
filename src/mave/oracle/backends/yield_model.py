from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from mave.chemistry.reaction import (
    Reaction,
)
from mave.oracle.base import (
    BackendResult,
)
from mave.oracle.registry import (
    register_backend,
)


class YieldModelBackend(ABC):
    """
    Interface for surrogate reaction-yield prediction.
    """

    @abstractmethod
    def predict(
        self,
        reaction: Reaction,
    ) -> BackendResult:
        """
        content must contain a scalar yield score in [0, 1].
        """

    def score(
        self,
        reaction: Reaction,
    ) -> float:

        result = self.predict(
            reaction
        )

        score = float(
            result.content
        )

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                "Yield score must lie in [0, 1]."
            )

        return score


@register_backend(
    "callable_yield_model"
)
class CallableYieldModelBackend(
    YieldModelBackend
):
    """
    Adapter around an externally supplied yield model.

    Parameters
    ----------
    predict_fn:
        reaction -> normalized yield score

    complexity_fn:
        optional reaction -> normalized complexity score in
        [0, 1].
    """

    def __init__(
        self,
        predict_fn: Callable[
            [Reaction],
            float,
        ],
        *,
        complexity_fn: (
            Callable[
                [Reaction],
                float,
            ]
            | None
        ) = None,
    ) -> None:

        self.predict_fn = (
            predict_fn
        )

        self.complexity_fn = (
            complexity_fn
        )

    def predict(
        self,
        reaction: Reaction,
    ) -> BackendResult:

        value = float(
            self.predict_fn(
                reaction
            )
        )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "Yield predictor must return "
                "a normalized score in [0, 1]."
            )

        complexity = None

        if (
            self.complexity_fn
            is not None
        ):
            complexity = float(
                self.complexity_fn(
                    reaction
                )
            )

        return BackendResult(
            content=value,
            complexity_score=(
                complexity
            ),
            metadata={
                "backend": (
                    "yield_model"
                ),
            },
        )
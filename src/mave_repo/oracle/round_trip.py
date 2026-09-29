from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Sequence

from mave_repro.chemistry.molecule import (
    Molecule,
)
from mave_repro.chemistry.reaction import (
    Reaction,
)
from mave_repro.chemistry.route import (
    SynthesisRoute,
)
from mave_repro.oracle.base import (
    BackendResult,
)
from mave_repro.oracle.registry import (
    register_backend,
)


class RoundTripBackend(ABC):
    """
    Common surrogate backend for multiple structural,
    evaluative, and comparative feedback types.

    The paper describes the common round-trip model but does not
    uniquely specify its public API. We therefore expose the
    required semantic operations explicitly.
    """

    # ========================================================
    # L1 Structural
    # ========================================================

    @abstractmethod
    def activity_assessment(
        self,
        product: Molecule,
        candidates: Sequence[
            Reaction
        ],
    ) -> BackendResult:
        ...

    @abstractmethod
    def reaction_class(
        self,
        product: Molecule,
        candidates: Sequence[
            Reaction
        ],
    ) -> BackendResult:
        ...

    @abstractmethod
    def reaction_center(
        self,
        product: Molecule,
        candidates: Sequence[
            Reaction
        ],
    ) -> BackendResult:
        ...

    @abstractmethod
    def bond_disconnection(
        self,
        product: Molecule,
        candidates: Sequence[
            Reaction
        ],
    ) -> BackendResult:
        ...

    # ========================================================
    # L2 Evaluative
    # ========================================================

    @abstractmethod
    def reaction_feasibility(
        self,
        reaction: Reaction,
    ) -> BackendResult:
        ...

    @abstractmethod
    def selectivity_assessment(
        self,
        reaction: Reaction,
    ) -> BackendResult:
        ...

    # ========================================================
    # L3 Comparative
    # ========================================================

    @abstractmethod
    def route_score(
        self,
        route: SynthesisRoute,
    ) -> BackendResult:
        ...


@register_backend(
    "callable_round_trip"
)
class CallableRoundTripBackend(
    RoundTripBackend
):
    """
    Flexible adapter around an existing round-trip model.

    Each operation is supplied as a named callable. A callable may
    return either BackendResult directly or raw content, in which
    case it is wrapped automatically.
    """

    def __init__(
        self,
        *,
        functions: Mapping[
            str,
            Callable[..., Any],
        ],
    ) -> None:

        self.functions = dict(
            functions
        )

    def _invoke(
        self,
        name: str,
        *args: Any,
    ) -> BackendResult:

        fn = self.functions.get(
            name
        )

        if fn is None:
            raise NotImplementedError(
                f"Round-trip operation "
                f"{name!r} was not supplied."
            )

        result = fn(
            *args
        )

        if isinstance(
            result,
            BackendResult,
        ):
            return result

        return BackendResult(
            content=result,
            metadata={
                "backend": (
                    "round_trip"
                ),
                "operation": name,
            },
        )

    def activity_assessment(
        self,
        product: Molecule,
        candidates: Sequence[
            Reaction
        ],
    ) -> BackendResult:
        return self._invoke(
            "activity_assessment",
            product,
            candidates,
        )

    def reaction_class(
        self,
        product: Molecule,
        candidates: Sequence[
            Reaction
        ],
    ) -> BackendResult:
        return self._invoke(
            "reaction_class",
            product,
            candidates,
        )

    def reaction_center(
        self,
        product: Molecule,
        candidates: Sequence[
            Reaction
        ],
    ) -> BackendResult:
        return self._invoke(
            "reaction_center",
            product,
            candidates,
        )

    def bond_disconnection(
        self,
        product: Molecule,
        candidates: Sequence[
            Reaction
        ],
    ) -> BackendResult:
        return self._invoke(
            "bond_disconnection",
            product,
            candidates,
        )

    def reaction_feasibility(
        self,
        reaction: Reaction,
    ) -> BackendResult:
        return self._invoke(
            "reaction_feasibility",
            reaction,
        )

    def selectivity_assessment(
        self,
        reaction: Reaction,
    ) -> BackendResult:
        return self._invoke(
            "selectivity_assessment",
            reaction,
        )

    def route_score(
        self,
        route: SynthesisRoute,
    ) -> BackendResult:
        return self._invoke(
            "route_score",
            route,
        )
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
from typing import (
    Any,
    Mapping,
    Protocol,
    Sequence,
)

from mave.chemistry.molecule import Molecule
from mave.chemistry.reaction import Reaction
from mave.chemistry.route import SynthesisRoute


# ============================================================
# Feedback hierarchy
# ============================================================


class FeedbackLevel(IntEnum):
    L0 = 0
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4


class FeedbackType(str, Enum):

    # L0
    NONE = "none"

    # L1 Structural
    ACTIVITY_ASSESSMENT = "activity_assessment"
    REACTION_CLASS = "reaction_class"
    REACTION_CENTER = "reaction_center"
    BOND_DISCONNECTION = "bond_disconnection"

    # L2 Evaluative
    REACTION_FEASIBILITY = "reaction_feasibility"
    REACTION_YIELD = "reaction_yield"
    SELECTIVITY_ASSESSMENT = "selectivity_assessment"

    # L3 Comparative
    REACTION_COMPARISON = "reaction_comparison"
    ROUTE_COMPARISON = "route_comparison"

    # L4 Strategic
    GOAL_SUGGESTION = "goal_suggestion"
    MULTI_STEP_STRATEGY = "multi_step_strategy"

    @property
    def level(
        self,
    ) -> FeedbackLevel:
        return FEEDBACK_LEVELS[self]


FEEDBACK_LEVELS = {

    FeedbackType.NONE:
        FeedbackLevel.L0,

    FeedbackType.ACTIVITY_ASSESSMENT:
        FeedbackLevel.L1,

    FeedbackType.REACTION_CLASS:
        FeedbackLevel.L1,

    FeedbackType.REACTION_CENTER:
        FeedbackLevel.L1,

    FeedbackType.BOND_DISCONNECTION:
        FeedbackLevel.L1,

    FeedbackType.REACTION_FEASIBILITY:
        FeedbackLevel.L2,

    FeedbackType.REACTION_YIELD:
        FeedbackLevel.L2,

    FeedbackType.SELECTIVITY_ASSESSMENT:
        FeedbackLevel.L2,

    FeedbackType.REACTION_COMPARISON:
        FeedbackLevel.L3,

    FeedbackType.ROUTE_COMPARISON:
        FeedbackLevel.L3,

    FeedbackType.GOAL_SUGGESTION:
        FeedbackLevel.L4,

    FeedbackType.MULTI_STEP_STRATEGY:
        FeedbackLevel.L4,
}


# ============================================================
# Backend result
# ============================================================


@dataclass(frozen=True, slots=True)
class BackendResult:
    """
    Output of a surrogate backend.

    content:
        Structured evidence. It may be a float, bool, dict,
        list, etc.

    complexity_score:
        Optional normalized acquisition complexity in [0, 1].

    BackendResult is NOT directly exposed to the planner.
    """

    content: Any

    complexity_score: (
        float
        | None
    ) = None

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:

        if self.complexity_score is not None:

            value = float(
                self.complexity_score
            )

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "complexity_score must lie in [0, 1]."
                )


# ============================================================
# Oracle result
# ============================================================


@dataclass(frozen=True, slots=True)
class OracleResult:
    """
    Output of the oracle layer.

    Unlike BackendResult, content MUST already be natural
    language because this object is planner-facing.
    """

    content: str

    complexity_score: (
        float
        | None
    ) = None

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:

        if not isinstance(
            self.content,
            str,
        ):
            raise TypeError(
                "OracleResult.content must be a string."
            )

        if self.complexity_score is not None:

            value = float(
                self.complexity_score
            )

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "complexity_score must lie in [0, 1]."
                )


# ============================================================
# Query
# ============================================================


@dataclass(frozen=True, slots=True)
class FeedbackQuery:
    """
    Request for one feedback source.

    Typical usage:

    L1:
        target=None

    L2:
        target=Reaction

    L3 reaction comparison:
        alternatives=(reaction_a, reaction_b)

    L3 route comparison:
        alternatives=(route_a, route_b)

    L4:
        target=None
    """

    feedback_type: FeedbackType

    target: Any = None

    alternatives: tuple[
        Any,
        ...
    ] = ()

    params: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    # Optional explicit cache identifier.
    cache_key: str | None = None

    @property
    def level(
        self,
    ) -> FeedbackLevel:
        return self.feedback_type.level


# ============================================================
# Context
# ============================================================


@dataclass(frozen=True, slots=True)
class OracleContext:
    """
    Planning information available when an oracle is queried.
    """

    state: Any

    selected_index: int

    selected_molecule: Molecule

    candidates: tuple[
        Reaction,
        ...
    ] = ()

    routes: tuple[
        SynthesisRoute,
        ...
    ] = ()

    feedback_history: tuple[
        "Feedback",
        ...
    ] = ()

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )


# ============================================================
# Final planner-visible feedback
# ============================================================


@dataclass(frozen=True, slots=True)
class Feedback:
    """
    Final feedback passed to the planner.

    content is ALWAYS natural language.
    """

    level: FeedbackLevel

    feedback_type: FeedbackType

    content: str

    cost: float

    cached: bool = False

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def as_cached(
        self,
    ) -> "Feedback":
        """
        Cached feedback is reused with zero additional
        acquisition cost.
        """

        metadata = dict(
            self.metadata
        )

        metadata[
            "original_acquisition_cost"
        ] = self.cost

        return replace(
            self,
            cost=0.0,
            cached=True,
            metadata=metadata,
        )


# ============================================================
# Oracle interface
# ============================================================


class FeedbackOracle(ABC):
    """
    Converts structured backend evidence into natural-language
    feedback.
    """

    supported_types: frozenset[
        FeedbackType
    ] = frozenset()

    def supports(
        self,
        feedback_type: FeedbackType,
    ) -> bool:

        return (
            feedback_type
            in self.supported_types
        )

    @property
    def level(
        self,
    ) -> FeedbackLevel:

        if not self.supported_types:
            raise ValueError(
                "Oracle has no supported feedback types."
            )

        levels = {
            item.level
            for item
            in self.supported_types
        }

        if len(levels) != 1:
            raise ValueError(
                "One oracle should operate at one "
                "feedback level."
            )

        return next(
            iter(levels)
        )

    @abstractmethod
    def query(
        self,
        query: FeedbackQuery,
        context: OracleContext,
    ) -> OracleResult:
        ...


# ============================================================
# Infrastructure protocols
# ============================================================


class CostModelProtocol(Protocol):

    def compute(
        self,
        feedback_type: FeedbackType,
        complexity_score: (
            float
            | None
        ) = None,
    ) -> float:
        ...


class FeedbackCacheProtocol(Protocol):

    def get(
        self,
        query: FeedbackQuery,
        context: OracleContext,
    ) -> Feedback | None:
        ...

    def set(
        self,
        query: FeedbackQuery,
        context: OracleContext,
        feedback: Feedback,
    ) -> None:
        ...

    def clear(
        self,
    ) -> None:
        ...


# ============================================================
# Multi-fidelity oracle system
# ============================================================


class MultiFidelityOracleSystem:
    """
    Central interface used by the planner.

    Pipeline:

        FeedbackQuery
             ↓
        FeedbackOracle
             ↓
        natural language
             ↓
        acquisition cost
             ↓
        Feedback
             ↓
        planner
    """

    def __init__(
        self,
        *,
        oracles: Sequence[
            FeedbackOracle
        ],
        cost_model: CostModelProtocol,
        cache: (
            FeedbackCacheProtocol
            | None
        ) = None,
    ) -> None:

        self.cost_model = (
            cost_model
        )

        self.cache = cache

        self._oracles: dict[
            FeedbackType,
            FeedbackOracle,
        ] = {}

        for oracle in oracles:

            for feedback_type in (
                oracle.supported_types
            ):

                if (
                    feedback_type
                    in self._oracles
                ):
                    raise ValueError(
                        "Multiple oracles registered for "
                        f"{feedback_type.value!r}."
                    )

                self._oracles[
                    feedback_type
                ] = oracle

    def query(
        self,
        query: FeedbackQuery,
        context: OracleContext,
    ) -> Feedback:

        # ----------------------------------------------------
        # L0: no external feedback
        # ----------------------------------------------------

        if (
            query.feedback_type
            == FeedbackType.NONE
        ):

            return Feedback(
                level=FeedbackLevel.L0,
                feedback_type=(
                    FeedbackType.NONE
                ),
                content="",
                cost=0.0,
                cached=False,
                metadata={
                    "queried": False,
                },
            )

        # ----------------------------------------------------
        # Cached feedback
        # ----------------------------------------------------

        if self.cache is not None:

            cached = self.cache.get(
                query,
                context,
            )

            if cached is not None:

                return cached.as_cached()

        # ----------------------------------------------------
        # Resolve oracle
        # ----------------------------------------------------

        oracle = self._oracles.get(
            query.feedback_type
        )

        if oracle is None:

            raise KeyError(
                "No oracle registered for "
                f"{query.feedback_type.value!r}."
            )

        # ----------------------------------------------------
        # Structured backend -> natural language
        # ----------------------------------------------------

        result = oracle.query(
            query,
            context,
        )

        if not isinstance(
            result.content,
            str,
        ):
            raise TypeError(
                "Oracle output must be natural language."
            )

        # ----------------------------------------------------
        # Acquisition cost
        # ----------------------------------------------------

        cost = self.cost_model.compute(
            query.feedback_type,
            result.complexity_score,
        )

        metadata = dict(
            result.metadata
        )

        metadata[
            "complexity_score"
        ] = result.complexity_score

        feedback = Feedback(
            level=query.level,
            feedback_type=(
                query.feedback_type
            ),
            content=result.content,
            cost=float(cost),
            cached=False,
            metadata=metadata,
        )

        # ----------------------------------------------------
        # Cache acquired feedback
        # ----------------------------------------------------

        if self.cache is not None:

            self.cache.set(
                query,
                context,
                feedback,
            )

        return feedback

    def available_feedback_types(
        self,
    ) -> tuple[
        FeedbackType,
        ...
    ]:

        return tuple(
            sorted(
                self._oracles,
                key=lambda item: (
                    item.level,
                    item.value,
                ),
            )
        )

    def clear_cache(
        self,
    ) -> None:

        if self.cache is not None:
            self.cache.clear()
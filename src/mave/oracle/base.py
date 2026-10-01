from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from mave.core.types import (
    Feedback,
    FeedbackLevel,
    FeedbackQuery,
    OracleContext,
)


# ============================================================
# Feedback taxonomy
# ============================================================


class FeedbackType(str, Enum):
    """
    Feedback sources used by the multi-fidelity oracle hierarchy.

    FeedbackQuery stores the feedback type as its string value so that
    core types remain independent of the oracle package. This enum is the
    oracle-side taxonomy and provides the level mapping used for validation,
    cost lookup, query generation, and oracle dispatch.
    """

    # L0: no external feedback. In the new interaction protocol L0 is
    # represented by the planner's STOP/no-query behavior; it is retained
    # here because the acquisition-cost table includes it.
    NONE = "none"

    # L1: Structural
    ACTIVITY_ASSESSMENT = "activity_assessment"
    REACTION_CLASS = "reaction_class"
    REACTION_CENTER = "reaction_center"
    BOND_DISCONNECTION = "bond_disconnection"

    # L2: Evaluative
    REACTION_FEASIBILITY = "reaction_feasibility"
    REACTION_YIELD = "reaction_yield"
    SELECTIVITY_ASSESSMENT = "selectivity_assessment"

    # L3: Comparative
    REACTION_COMPARISON = "reaction_comparison"
    ROUTE_COMPARISON = "route_comparison"

    # L4: Strategic
    GOAL_SUGGESTION = "goal_suggestion"
    MULTI_STEP_STRATEGY = "multi_step_strategy"

    @property
    def level(self) -> FeedbackLevel:
        return FEEDBACK_LEVELS[self]


FEEDBACK_LEVELS: dict[FeedbackType, FeedbackLevel] = {
    FeedbackType.NONE: FeedbackLevel.L0,

    FeedbackType.ACTIVITY_ASSESSMENT: FeedbackLevel.L1,
    FeedbackType.REACTION_CLASS: FeedbackLevel.L1,
    FeedbackType.REACTION_CENTER: FeedbackLevel.L1,
    FeedbackType.BOND_DISCONNECTION: FeedbackLevel.L1,

    FeedbackType.REACTION_FEASIBILITY: FeedbackLevel.L2,
    FeedbackType.REACTION_YIELD: FeedbackLevel.L2,
    FeedbackType.SELECTIVITY_ASSESSMENT: FeedbackLevel.L2,

    FeedbackType.REACTION_COMPARISON: FeedbackLevel.L3,
    FeedbackType.ROUTE_COMPARISON: FeedbackLevel.L3,

    FeedbackType.GOAL_SUGGESTION: FeedbackLevel.L4,
    FeedbackType.MULTI_STEP_STRATEGY: FeedbackLevel.L4,
}


FEEDBACK_TYPES_BY_LEVEL: dict[
    FeedbackLevel,
    tuple[FeedbackType, ...],
] = {
    level: tuple(
        feedback_type
        for feedback_type in FeedbackType
        if feedback_type.level == level
    )
    for level in FeedbackLevel
}


def coerce_feedback_type(
    value: FeedbackType | str,
) -> FeedbackType:
    """
    Convert the core string representation to the oracle enum.

    Keeping this conversion at the oracle boundary prevents core/types.py
    from importing mave.oracle and therefore avoids a circular dependency.
    """

    if isinstance(value, FeedbackType):
        return value

    try:
        return FeedbackType(str(value))
    except ValueError as exc:
        valid = ", ".join(item.value for item in FeedbackType)
        raise ValueError(
            f"Unknown feedback type {value!r}. Valid values: {valid}."
        ) from exc


def feedback_type_value(
    value: FeedbackType | str,
) -> str:
    """Return the canonical string representation of a feedback type."""

    return coerce_feedback_type(value).value


# ============================================================
# Backend result
# ============================================================


@dataclass(frozen=True, slots=True)
class BackendResult:
    """
    Output of a surrogate backend.

    `content` is structured backend evidence and is NOT directly exposed to
    the planner. It may be a float, bool, mapping, list, or another backend
    representation.

    `complexity_score`, when available, is normalized to [0, 1] and is used
    by the acquisition-cost model to obtain a query-specific cost.
    """

    content: Any

    complexity_score: float | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.complexity_score is not None:
            value = float(self.complexity_score)

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
    Natural-language result produced by one feedback oracle.

    The backend may internally produce structured evidence, but the oracle
    layer must convert that evidence into a faithful natural-language answer
    before returning it here. Only this answer is exposed to the planner.
    """

    content: str

    complexity_score: float | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError(
                "OracleResult.content must be a string."
            )

        if not self.content.strip():
            raise ValueError(
                "OracleResult.content must not be empty."
            )

        if self.complexity_score is not None:
            value = float(self.complexity_score)

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "complexity_score must lie in [0, 1]."
                )


# ============================================================
# Oracle interface
# ============================================================


class FeedbackOracle(ABC):
    """
    Interface implemented by structural/evaluative/comparative/strategic
    oracle modules.

    A FeedbackOracle receives an already selected FeedbackQuery, executes the
    corresponding surrogate backend using the query's structured payload and
    the OracleContext, and returns a natural-language OracleResult.

    Important:
        The oracle must not infer execution targets by parsing query.question.
        Structured execution objects belong in query.payload.
    """

    supported_types: frozenset[FeedbackType] = frozenset()

    def supports(
        self,
        feedback_type: FeedbackType | str,
    ) -> bool:
        try:
            normalized = coerce_feedback_type(feedback_type)
        except ValueError:
            return False

        return normalized in self.supported_types

    @property
    def level(self) -> FeedbackLevel:
        if not self.supported_types:
            raise ValueError(
                "Oracle has no supported feedback types."
            )

        levels = {
            item.level
            for item in self.supported_types
        }

        if len(levels) != 1:
            raise ValueError(
                "One oracle should operate at exactly one feedback level."
            )

        return next(iter(levels))

    @abstractmethod
    def query(
        self,
        query: FeedbackQuery,
        context: OracleContext,
    ) -> OracleResult:
        """
        Execute one selected query and return a natural-language answer.
        """
        ...


# ============================================================
# Infrastructure protocols
# ============================================================


class CostModelProtocol(Protocol):
    """
    Minimal interface required from the acquisition-cost model.
    """

    def compute(
        self,
        feedback_type: FeedbackType,
        complexity_score: float | None = None,
    ) -> float:
        ...


class FeedbackCacheProtocol(Protocol):
    """
    Cache interface.

    Cache identity should be based on query semantics and structured targets,
    not on the exact wording of query.question.
    """

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

    def clear(self) -> None:
        ...


class FeedbackQueryFactoryProtocol(Protocol):
    """
    Query-generation interface used by MultiFidelityOracleSystem.

    The query factory is responsible for enumerating legal next-level
    questions from the current planning context. Every returned query must
    already contain:

        - level
        - feedback_type
        - query_id
        - natural-language question
        - planner-visible target_ids
        - structured execution payload

    The language model chooses among these legal queries; it does not freely
    generate an unconstrained oracle call.
    """

    def build_next_level_queries(
        self,
        *,
        context: OracleContext,
        current_level: FeedbackLevel,
    ) -> Sequence[FeedbackQuery]:
        ...


# ============================================================
# Multi-fidelity oracle system
# ============================================================


class MultiFidelityOracleSystem:
    """
    Central planner-facing interface for multi-fidelity feedback.

    Interaction protocol
    --------------------

        planning context
              |
              v
        available_next_queries(...)
              |
              v
        legal FeedbackQuery objects
        (structured payload + natural-language question)
              |
              v
        planner prompt / structured ESCALATE decision
              |
              v
        query(query, context)
              |
              v
        FeedbackOracle
              |
              v
        surrogate backend evidence
              |
              v
        natural-language OracleResult
              |
              v
        acquisition cost + cache
              |
              v
        Feedback(question, answer, ...)
              |
              v
        next planner prompt

    The planner therefore interacts with the oracle system through explicit
    natural-language questions and answers, while oracle execution remains
    deterministic and structured underneath the language interface.
    """

    def __init__(
        self,
        *,
        oracles: Sequence[FeedbackOracle],
        cost_model: CostModelProtocol,
        query_factory: FeedbackQueryFactoryProtocol,
        cache: FeedbackCacheProtocol | None = None,
    ) -> None:
        self.cost_model = cost_model
        self.query_factory = query_factory
        self.cache = cache

        self._oracles: dict[
            FeedbackType,
            FeedbackOracle,
        ] = {}

        for oracle in oracles:
            if not oracle.supported_types:
                raise ValueError(
                    f"{oracle.__class__.__name__} declares no supported "
                    "feedback types."
                )

            # Trigger the one-level invariant early.
            _ = oracle.level

            for feedback_type in oracle.supported_types:
                if feedback_type == FeedbackType.NONE:
                    raise ValueError(
                        "FeedbackType.NONE must not be registered to an "
                        "external oracle. L0 is represented by STOP/no query."
                    )

                if feedback_type in self._oracles:
                    existing = self._oracles[feedback_type]
                    raise ValueError(
                        "Multiple oracles registered for "
                        f"{feedback_type.value!r}: "
                        f"{existing.__class__.__name__} and "
                        f"{oracle.__class__.__name__}."
                    )

                self._oracles[feedback_type] = oracle

    # ========================================================
    # Query enumeration
    # ========================================================

    def available_next_queries(
        self,
        context: OracleContext,
        current_level: FeedbackLevel | int | None = None,
    ) -> tuple[FeedbackQuery, ...]:
        """
        Enumerate legal questions at exactly the next feedback level.

        Query construction is delegated to query_factory so that the same
        question/action space is used by training rollouts and MCTS inference.
        """

        if current_level is None:
            level = context.current_level
        else:
            level = FeedbackLevel(int(current_level))

        if level == FeedbackLevel.L4:
            return ()

        raw_queries = self.query_factory.build_next_level_queries(
            context=context,
            current_level=level,
        )

        queries = tuple(raw_queries)

        expected_level = FeedbackLevel(int(level) + 1)

        seen_query_ids: set[str] = set()

        for query in queries:
            self._validate_query(
                query,
                expected_level=expected_level,
            )

            if query.query_id in seen_query_ids:
                raise ValueError(
                    "Query factory produced duplicate query_id "
                    f"{query.query_id!r} within one planning context."
                )

            seen_query_ids.add(query.query_id)

        return queries

    # ========================================================
    # Query execution
    # ========================================================

    def query(
        self,
        query: FeedbackQuery,
        context: OracleContext,
    ) -> Feedback:
        """
        Execute one planner-selected feedback query.

        This is the canonical argument order used by the new codebase:

            oracle.query(query, context)

        The planner-visible question is preserved verbatim in the returned
        Feedback record, and the oracle's natural-language response becomes
        Feedback.answer.
        """

        self._validate_query(query)

        feedback_type = coerce_feedback_type(
            query.feedback_type
        )

        if feedback_type == FeedbackType.NONE:
            raise ValueError(
                "L0/NONE is not an executable oracle query. "
                "Use the planner's STOP action instead."
            )

        # ----------------------------------------------------
        # Cache lookup
        # ----------------------------------------------------

        if self.cache is not None:
            cached = self.cache.get(
                query,
                context,
            )

            if cached is not None:
                self._validate_cached_feedback(
                    cached,
                    query,
                )

                metadata = dict(cached.metadata)
                metadata.setdefault(
                    "original_acquisition_cost",
                    float(cached.cost),
                )
                metadata["cache_hit"] = True

                return replace(
                    cached,
                    cost=0.0,
                    cached=True,
                    metadata=metadata,
                )

        # ----------------------------------------------------
        # Resolve the oracle for this feedback type
        # ----------------------------------------------------

        oracle = self._oracles.get(
            feedback_type
        )

        if oracle is None:
            raise KeyError(
                "No oracle registered for "
                f"{feedback_type.value!r}."
            )

        # ----------------------------------------------------
        # Structured backend evidence -> natural-language answer
        # ----------------------------------------------------

        result = oracle.query(
            query,
            context,
        )

        if not isinstance(result, OracleResult):
            raise TypeError(
                f"{oracle.__class__.__name__}.query() must return "
                "OracleResult."
            )

        answer = result.content.strip()

        if not answer:
            raise ValueError(
                "Oracle returned an empty natural-language answer."
            )

        # ----------------------------------------------------
        # Acquisition cost
        # ----------------------------------------------------

        cost = float(
            self.cost_model.compute(
                feedback_type,
                result.complexity_score,
            )
        )

        if cost < 0:
            raise ValueError(
                "Acquisition cost must be non-negative."
            )

        # ----------------------------------------------------
        # Build complete Q/A record
        # ----------------------------------------------------

        metadata = dict(query.metadata)
        metadata.update(result.metadata)

        metadata.update({
            "complexity_score": result.complexity_score,
            "cache_hit": False,
            "oracle_class": oracle.__class__.__name__,
            "feedback_type": feedback_type.value,
        })

        feedback = Feedback.from_query(
            query,
            answer=answer,
            cost=cost,
            cached=False,
            metadata=metadata,
        )

        # ----------------------------------------------------
        # Cache newly acquired feedback
        # ----------------------------------------------------

        if self.cache is not None:
            self.cache.set(
                query,
                context,
                feedback,
            )

        return feedback

    # ========================================================
    # Validation
    # ========================================================

    def _validate_query(
        self,
        query: FeedbackQuery,
        *,
        expected_level: FeedbackLevel | None = None,
    ) -> None:
        if not isinstance(query, FeedbackQuery):
            raise TypeError(
                "Expected mave.core.types.FeedbackQuery."
            )

        feedback_type = coerce_feedback_type(
            query.feedback_type
        )

        if feedback_type == FeedbackType.NONE:
            raise ValueError(
                "FeedbackType.NONE is not a legal external query."
            )

        if query.level != feedback_type.level:
            raise ValueError(
                "FeedbackQuery level/type mismatch: "
                f"query.level={query.level.name}, "
                f"feedback_type={feedback_type.value!r} belongs to "
                f"{feedback_type.level.name}."
            )

        if expected_level is not None and query.level != expected_level:
            raise ValueError(
                "Query factory must return only the next feedback level: "
                f"expected {expected_level.name}, received "
                f"{query.level.name} for {query.query_id!r}."
            )

        if not query.query_id.strip():
            raise ValueError(
                "FeedbackQuery.query_id must not be empty."
            )

        if not query.question.strip():
            raise ValueError(
                "FeedbackQuery.question must not be empty."
            )

        if feedback_type not in self._oracles:
            raise KeyError(
                "No registered oracle can execute "
                f"{feedback_type.value!r}."
            )

    @staticmethod
    def _validate_cached_feedback(
        feedback: Feedback,
        query: FeedbackQuery,
    ) -> None:
        """
        Guard against cache-key collisions or stale incompatible entries.
        """

        if feedback.query_id != query.query_id:
            raise ValueError(
                "Cached feedback/query mismatch: query_id differs."
            )

        if feedback.level != query.level:
            raise ValueError(
                "Cached feedback/query mismatch: feedback level differs."
            )

        if (
            feedback_type_value(feedback.feedback_type)
            != feedback_type_value(query.feedback_type)
        ):
            raise ValueError(
                "Cached feedback/query mismatch: feedback type differs."
            )

        if tuple(feedback.target_ids) != tuple(query.target_ids):
            raise ValueError(
                "Cached feedback/query mismatch: target_ids differ."
            )

        if feedback.question != query.question:
            # The cache should be semantic rather than wording-based. A query
            # may therefore use different wording for the same semantic key.
            # We do not reject that case; the answer remains valid, but the
            # returned record should preserve the question actually asked.
            pass

    # ========================================================
    # Introspection
    # ========================================================

    def available_feedback_types(
        self,
    ) -> tuple[FeedbackType, ...]:
        """
        Feedback types that have executable registered oracles.
        """

        return tuple(
            sorted(
                self._oracles,
                key=lambda item: (
                    int(item.level),
                    item.value,
                ),
            )
        )

    def feedback_types_at_level(
        self,
        level: FeedbackLevel | int,
    ) -> tuple[FeedbackType, ...]:
        """
        Registered feedback types at one hierarchy level.
        """

        normalized_level = FeedbackLevel(
            int(level)
        )

        return tuple(
            feedback_type
            for feedback_type
            in self.available_feedback_types()
            if feedback_type.level == normalized_level
        )

    def get_oracle(
        self,
        feedback_type: FeedbackType | str,
    ) -> FeedbackOracle:
        """
        Return the registered oracle for one feedback type.
        """

        normalized = coerce_feedback_type(
            feedback_type
        )

        oracle = self._oracles.get(
            normalized
        )

        if oracle is None:
            raise KeyError(
                "No oracle registered for "
                f"{normalized.value!r}."
            )

        return oracle

    def clear_cache(self) -> None:
        if self.cache is not None:
            self.cache.clear()

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

from mave.core.types import (
    FeedbackLevel,
    FeedbackQuery,
    OracleContext,
)
from mave.oracle.base import (
    FeedbackType,
    coerce_feedback_type,
)
from mave.oracle.context import (
    native_candidates,
    native_selected_molecule,
    native_state,
)


# ============================================================
# Query-cost hints
# ============================================================

# These values mirror the hierarchy configuration / paper-level
# relative acquisition-cost schedule. They are prompt-facing hints only;
# the actual charged cost is still computed by the oracle cost model.
DEFAULT_BASE_COSTS: dict[FeedbackType, float] = {
    FeedbackType.ACTIVITY_ASSESSMENT: 0.10,
    FeedbackType.REACTION_CLASS: 0.25,
    FeedbackType.REACTION_CENTER: 0.30,
    FeedbackType.BOND_DISCONNECTION: 0.35,
    FeedbackType.REACTION_FEASIBILITY: 0.80,
    FeedbackType.REACTION_YIELD: 1.00,
    FeedbackType.SELECTIVITY_ASSESSMENT: 1.20,
    FeedbackType.REACTION_COMPARISON: 2.00,
    FeedbackType.ROUTE_COMPARISON: 3.50,
    FeedbackType.GOAL_SUGGESTION: 5.00,
    FeedbackType.MULTI_STEP_STRATEGY: 10.00,
}


DEFAULT_COST_RANGES: dict[
    FeedbackType,
    tuple[float, float],
] = {
    FeedbackType.ACTIVITY_ASSESSMENT: (0.08, 0.12),
    FeedbackType.REACTION_CLASS: (0.20, 0.30),
    FeedbackType.REACTION_CENTER: (0.24, 0.36),
    FeedbackType.BOND_DISCONNECTION: (0.28, 0.42),
    FeedbackType.REACTION_FEASIBILITY: (0.64, 0.96),
    FeedbackType.REACTION_YIELD: (0.80, 1.20),
    FeedbackType.SELECTIVITY_ASSESSMENT: (0.96, 1.44),
    FeedbackType.REACTION_COMPARISON: (1.60, 2.40),
    FeedbackType.ROUTE_COMPARISON: (2.80, 4.20),
    FeedbackType.GOAL_SUGGESTION: (4.00, 6.00),
    FeedbackType.MULTI_STEP_STRATEGY: (8.00, 12.00),
}


# ============================================================
# Query-factory configuration
# ============================================================


@dataclass(frozen=True, slots=True)
class FeedbackQueryFactoryConfig:
    """
    Configuration for deterministic legal-query enumeration.

    The factory never asks the language model to freely generate an oracle
    question. Instead, it deterministically constructs the legal natural-
    language questions for the next feedback level, and the planner chooses
    among those questions through structured prediction.

    Notes
    -----
    `max_candidates` and the pair limits are optional deterministic caps.
    They default to None so that this module does not silently introduce a
    candidate-selection rule that is not specified elsewhere.

    If a cap is used, candidates/routes are taken in their existing order.
    Therefore the upstream planner must provide a deterministic ordering
    (normally the single-step model ranking for reactions).
    """

    enabled_feedback_types: frozenset[FeedbackType] = field(
        default_factory=lambda: frozenset(
            feedback_type
            for feedback_type in FeedbackType
            if feedback_type != FeedbackType.NONE
        )
    )

    max_candidates: int | None = None
    max_reaction_pairs: int | None = None
    max_route_pairs: int | None = None

    include_cost_hints: bool = True

    base_costs: Mapping[FeedbackType, float] = field(
        default_factory=lambda: dict(DEFAULT_BASE_COSTS)
    )

    cost_ranges: Mapping[
        FeedbackType,
        tuple[float, float],
    ] = field(
        default_factory=lambda: dict(DEFAULT_COST_RANGES)
    )

    # OracleContext.metadata keys inspected for partial routes at L3.
    route_metadata_keys: tuple[str, ...] = (
        "routes",
        "partial_routes",
        "candidate_routes",
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("max_candidates", self.max_candidates),
            ("max_reaction_pairs", self.max_reaction_pairs),
            ("max_route_pairs", self.max_route_pairs),
        ):
            if value is not None and value <= 0:
                raise ValueError(
                    f"{name} must be positive when provided."
                )

        if FeedbackType.NONE in self.enabled_feedback_types:
            raise ValueError(
                "FeedbackType.NONE is not an executable query type."
            )


# ============================================================
# Query factory
# ============================================================


class FeedbackQueryFactory:
    """
    Construct legal natural-language feedback queries.

    Hierarchy
    ---------
    L1 Structural
        One query per structural feedback type for the current molecule.

    L2 Evaluative
        One query per (candidate reaction, evaluative feedback type).

    L3 Comparative
        Reaction comparison is instantiated over candidate-reaction pairs.
        Route comparison is instantiated over partial-route pairs when routes
        are supplied through OracleContext.metadata.

    L4 Strategic
        One goal-suggestion query and one multi-step-strategy query for the
        current planning state.

    Every FeedbackQuery contains both:
        - planner-visible language fields:
          query_id, question, target_ids
        - oracle-execution fields:
          payload

    The oracle must execute payload directly and must never recover execution
    targets by parsing the natural-language question.
    """

    def __init__(
        self,
        config: FeedbackQueryFactoryConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else FeedbackQueryFactoryConfig()
        )

    # ========================================================
    # Public API
    # ========================================================

    def build_next_level_queries(
        self,
        *,
        context: OracleContext,
        current_level: FeedbackLevel,
    ) -> tuple[FeedbackQuery, ...]:
        """
        Build all legal queries at exactly current_level + 1.

        This method is intentionally deterministic. Given the same context,
        configuration, and candidate ordering, it returns queries in the same
        order with the same IDs and wording.
        """

        level = FeedbackLevel(int(current_level))

        if level == FeedbackLevel.L4:
            return ()

        next_level = FeedbackLevel(int(level) + 1)

        if next_level == FeedbackLevel.L1:
            queries = self._build_l1_queries(context)

        elif next_level == FeedbackLevel.L2:
            queries = self._build_l2_queries(context)

        elif next_level == FeedbackLevel.L3:
            queries = self._build_l3_queries(context)

        elif next_level == FeedbackLevel.L4:
            queries = self._build_l4_queries(context)

        else:
            queries = ()

        self._validate_unique_query_ids(queries)

        return tuple(queries)

    # ========================================================
    # L1: structural
    # ========================================================

    def _build_l1_queries(
        self,
        context: OracleContext,
    ) -> list[FeedbackQuery]:
        selected = native_selected_molecule(context)
        target_ids = ("M_CURRENT",)

        specs = (
            (
                FeedbackType.ACTIVITY_ASSESSMENT,
                "ACTIVITY",
                (
                    "What is the local functional-group reactivity and "
                    "compatibility of the current molecule?"
                ),
            ),
            (
                FeedbackType.REACTION_CLASS,
                "CLASS",
                (
                    "What reaction class is plausible for the current "
                    "molecule?"
                ),
            ),
            (
                FeedbackType.REACTION_CENTER,
                "CENTER",
                (
                    "Which atoms or bonds form a plausible reaction center "
                    "in the current molecule?"
                ),
            ),
            (
                FeedbackType.BOND_DISCONNECTION,
                "BOND",
                (
                    "Which bond is a plausible retrosynthetic disconnection "
                    "in the current molecule?"
                ),
            ),
        )

        queries: list[FeedbackQuery] = []

        for feedback_type, short_name, question in specs:
            if not self._enabled(feedback_type):
                continue

            query_id = f"Q_L1_{short_name}"

            payload = {
                "selected_molecule": selected,
                "candidates": native_candidates(context),
            }

            queries.append(
                self._make_query(
                    level=FeedbackLevel.L1,
                    feedback_type=feedback_type,
                    query_id=query_id,
                    question=question,
                    target_ids=target_ids,
                    payload=payload,
                    semantic_key=(
                        feedback_type.value,
                        self._molecule_semantic_key(selected),
                    ),
                    extra_metadata={
                        "scope": "current_molecule",
                    },
                )
            )

        return queries

    # ========================================================
    # L2: evaluative
    # ========================================================

    def _build_l2_queries(
        self,
        context: OracleContext,
    ) -> list[FeedbackQuery]:
        candidates = self._candidate_items(context)

        specs = (
            (
                FeedbackType.REACTION_FEASIBILITY,
                "FEASIBILITY",
                lambda reaction_id: (
                    f"Is candidate reaction {reaction_id} predicted to be "
                    "feasible?"
                ),
            ),
            (
                FeedbackType.REACTION_YIELD,
                "YIELD",
                lambda reaction_id: (
                    "What is the predicted normalized yield of candidate "
                    f"reaction {reaction_id}?"
                ),
            ),
            (
                FeedbackType.SELECTIVITY_ASSESSMENT,
                "SELECTIVITY",
                lambda reaction_id: (
                    "What is the expected selectivity of candidate reaction "
                    f"{reaction_id}?"
                ),
            ),
        )

        queries: list[FeedbackQuery] = []

        for index, candidate in candidates:
            reaction_id = self._reaction_id(index)

            for feedback_type, short_name, question_fn in specs:
                if not self._enabled(feedback_type):
                    continue

                query_id = (
                    f"Q_L2_{short_name}_{reaction_id}"
                )

                question = question_fn(reaction_id)

                payload = {
                    "reaction": candidate,
                    "target": candidate,
                    "candidate_index": index,
                    "reaction_index": index,
                    "reaction_id": reaction_id,
                    "target_ids": (reaction_id,),
                }

                queries.append(
                    self._make_query(
                        level=FeedbackLevel.L2,
                        feedback_type=feedback_type,
                        query_id=query_id,
                        question=question,
                        target_ids=(reaction_id,),
                        payload=payload,
                        semantic_key=(
                            feedback_type.value,
                            self._reaction_semantic_key(
                                candidate,
                                fallback_id=reaction_id,
                            ),
                        ),
                        extra_metadata={
                            "scope": "candidate_reaction",
                            "candidate_index": index,
                            "reaction_id": reaction_id,
                        },
                    )
                )

        return queries

    # ========================================================
    # L3: comparative
    # ========================================================

    def _build_l3_queries(
        self,
        context: OracleContext,
    ) -> list[FeedbackQuery]:
        queries: list[FeedbackQuery] = []

        if self._enabled(
            FeedbackType.REACTION_COMPARISON
        ):
            queries.extend(
                self._build_reaction_comparison_queries(
                    context
                )
            )

        if self._enabled(
            FeedbackType.ROUTE_COMPARISON
        ):
            queries.extend(
                self._build_route_comparison_queries(
                    context
                )
            )

        return queries

    def _build_reaction_comparison_queries(
        self,
        context: OracleContext,
    ) -> list[FeedbackQuery]:
        candidate_items = self._candidate_items(context)

        pair_items = list(
            combinations(candidate_items, 2)
        )

        pair_items = self._limit(
            pair_items,
            self.config.max_reaction_pairs,
        )

        queries: list[FeedbackQuery] = []

        for (
            (index_a, reaction_a),
            (index_b, reaction_b),
        ) in pair_items:
            reaction_id_a = self._reaction_id(index_a)
            reaction_id_b = self._reaction_id(index_b)

            target_ids = (
                reaction_id_a,
                reaction_id_b,
            )

            query_id = (
                "Q_L3_REACTION_COMPARE_"
                f"{reaction_id_a}_{reaction_id_b}"
            )

            question = (
                "Which candidate reaction is preferable between "
                f"{reaction_id_a} and {reaction_id_b} based on predicted "
                "reaction yield?"
            )

            alternatives = (
                reaction_a,
                reaction_b,
            )

            payload = {
                # New execution representation.
                "reactions": alternatives,

                # Compatibility with the previous comparative oracle shape.
                "alternatives": alternatives,

                "reaction_indices": (
                    index_a,
                    index_b,
                ),
                "reaction_ids": target_ids,
                "target_ids": target_ids,
            }

            queries.append(
                self._make_query(
                    level=FeedbackLevel.L3,
                    feedback_type=(
                        FeedbackType.REACTION_COMPARISON
                    ),
                    query_id=query_id,
                    question=question,
                    target_ids=target_ids,
                    payload=payload,
                    semantic_key=(
                        FeedbackType.REACTION_COMPARISON.value,
                        self._reaction_semantic_key(
                            reaction_a,
                            fallback_id=reaction_id_a,
                        ),
                        self._reaction_semantic_key(
                            reaction_b,
                            fallback_id=reaction_id_b,
                        ),
                    ),
                    extra_metadata={
                        "scope": "reaction_pair",
                        "reaction_indices": (
                            index_a,
                            index_b,
                        ),
                        "reaction_ids": target_ids,
                    },
                )
            )

        return queries

    def _build_route_comparison_queries(
        self,
        context: OracleContext,
    ) -> list[FeedbackQuery]:
        routes = self._extract_routes(context)

        if len(routes) < 2:
            return []

        route_items = [
            (index, route)
            for index, route in enumerate(routes)
        ]

        pair_items = list(
            combinations(route_items, 2)
        )

        pair_items = self._limit(
            pair_items,
            self.config.max_route_pairs,
        )

        queries: list[FeedbackQuery] = []

        for (
            (index_a, route_a),
            (index_b, route_b),
        ) in pair_items:
            route_id_a = self._route_id(
                route_a,
                index_a,
            )
            route_id_b = self._route_id(
                route_b,
                index_b,
            )

            target_ids = (
                route_id_a,
                route_id_b,
            )

            query_id = (
                "Q_L3_ROUTE_COMPARE_"
                f"{route_id_a}_{route_id_b}"
            )

            question = (
                "Which partial route is preferable between "
                f"{route_id_a} and {route_id_b} based on round-trip "
                "support?"
            )

            alternatives = (
                route_a,
                route_b,
            )

            payload = {
                "routes": alternatives,
                "alternatives": alternatives,
                "route_indices": (
                    index_a,
                    index_b,
                ),
                "route_ids": target_ids,
                "target_ids": target_ids,
            }

            queries.append(
                self._make_query(
                    level=FeedbackLevel.L3,
                    feedback_type=(
                        FeedbackType.ROUTE_COMPARISON
                    ),
                    query_id=query_id,
                    question=question,
                    target_ids=target_ids,
                    payload=payload,
                    semantic_key=(
                        FeedbackType.ROUTE_COMPARISON.value,
                        self._route_semantic_key(
                            route_a,
                            fallback_id=route_id_a,
                        ),
                        self._route_semantic_key(
                            route_b,
                            fallback_id=route_id_b,
                        ),
                    ),
                    extra_metadata={
                        "scope": "route_pair",
                        "route_indices": (
                            index_a,
                            index_b,
                        ),
                        "route_ids": target_ids,
                    },
                )
            )

        return queries

    # ========================================================
    # L4: strategic
    # ========================================================

    def _build_l4_queries(
        self,
        context: OracleContext,
    ) -> list[FeedbackQuery]:
        state_id = (
            context.effective_context_id
            or context.state_id
            or "CURRENT_STATE"
        )

        target_ids = ("PLAN",)

        specs = (
            (
                FeedbackType.GOAL_SUGGESTION,
                "GOAL",
                (
                    "What intermediate subgoals or purchasable building "
                    "blocks should be considered for the current planning "
                    "state?"
                ),
            ),
            (
                FeedbackType.MULTI_STEP_STRATEGY,
                "STRATEGY",
                (
                    "What high-level multi-step synthesis strategy should "
                    "guide the current planning state?"
                ),
            ),
        )

        queries: list[FeedbackQuery] = []

        for feedback_type, short_name, question in specs:
            if not self._enabled(feedback_type):
                continue

            query_id = f"Q_L4_{short_name}"

            payload = {
                "state": native_state(context),
                "selected_molecule": native_selected_molecule(context),
                "candidates": native_candidates(context),
                "context_id": context.effective_context_id,
            }

            queries.append(
                self._make_query(
                    level=FeedbackLevel.L4,
                    feedback_type=feedback_type,
                    query_id=query_id,
                    question=question,
                    target_ids=target_ids,
                    payload=payload,
                    semantic_key=(
                        feedback_type.value,
                        str(state_id),
                        self._molecule_semantic_key(
                            context.selected_molecule
                        ),
                    ),
                    extra_metadata={
                        "scope": "planning_state",
                    },
                )
            )

        return queries

    # ========================================================
    # Query construction
    # ========================================================

    def _make_query(
        self,
        *,
        level: FeedbackLevel,
        feedback_type: FeedbackType,
        query_id: str,
        question: str,
        target_ids: tuple[str, ...],
        payload: Mapping[str, Any],
        semantic_key: tuple[Any, ...],
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> FeedbackQuery:
        if feedback_type.level != level:
            raise ValueError(
                f"{feedback_type.value!r} belongs to "
                f"{feedback_type.level.name}, not {level.name}."
            )

        metadata: dict[str, Any] = {
            # Duplicated intentionally for compatibility with prompt/rendering
            # code that historically read these fields from metadata.
            "query_id": query_id,
            "question": question,
            "target_ids": target_ids,

            # Cache/query identity must not depend on question wording.
            "semantic_key": semantic_key,
        }

        if self.config.include_cost_hints:
            base_cost = self.config.base_costs.get(
                feedback_type
            )
            cost_range = self.config.cost_ranges.get(
                feedback_type
            )

            if base_cost is not None:
                metadata["base_cost"] = float(
                    base_cost
                )

            if cost_range is not None:
                metadata["cost_range"] = (
                    float(cost_range[0]),
                    float(cost_range[1]),
                )

        if extra_metadata is not None:
            metadata.update(
                dict(extra_metadata)
            )

        return FeedbackQuery(
            level=level,
            feedback_type=feedback_type.value,
            query_id=query_id,
            question=question,
            target_ids=target_ids,
            payload=dict(payload),
            metadata=metadata,
        )

    # ========================================================
    # Candidate / route enumeration
    # ========================================================

    def _candidate_items(
        self,
        context: OracleContext,
    ) -> list[tuple[int, Any]]:
        items = [
            (index, candidate)
            for index, candidate
            in enumerate(native_candidates(context))
        ]

        if self.config.max_candidates is not None:
            items = items[
                : self.config.max_candidates
            ]

        return items

    def _extract_routes(
        self,
        context: OracleContext,
    ) -> tuple[Any, ...]:
        for key in self.config.route_metadata_keys:
            value = context.metadata.get(key)

            if value is None:
                continue

            if isinstance(
                value,
                Sequence,
            ) and not isinstance(
                value,
                (str, bytes),
            ):
                return tuple(value)

            raise TypeError(
                f"OracleContext.metadata[{key!r}] must be a sequence "
                "of partial routes."
            )

        return ()

    # ========================================================
    # Stable planner-visible IDs
    # ========================================================

    @staticmethod
    def _reaction_id(
        zero_based_index: int,
    ) -> str:
        """
        Local planner-visible candidate ID.

        IDs are deliberately derived from candidate order rather than an
        optional external reaction_id so that the prompt and query factory
        always agree on R1, R2, ... within one planning context.
        """
        return f"R{zero_based_index + 1}"

    @staticmethod
    def _route_id(
        route: Any,
        zero_based_index: int,
    ) -> str:
        # Prefer an explicitly supplied route ID when available.
        explicit = getattr(
            route,
            "route_id",
            None,
        )

        if explicit is None:
            metadata = getattr(
                route,
                "metadata",
                None,
            )

            if isinstance(
                metadata,
                Mapping,
            ):
                explicit = (
                    metadata.get("route_id")
                    or metadata.get("id")
                )

        if explicit is not None:
            text = str(explicit).strip()
            if text:
                return text

        return f"T{zero_based_index + 1}"

    # ========================================================
    # Semantic identity helpers
    # ========================================================

    @classmethod
    def _reaction_semantic_key(
        cls,
        reaction: Any,
        *,
        fallback_id: str,
    ) -> tuple[Any, ...]:
        """
        Stable chemistry-oriented identity when the reaction object exposes
        the repository's usual Reaction/ReactionCandidate attributes.
        """

        key = getattr(
            reaction,
            "key",
            None,
        )

        if key is not None:
            try:
                return (
                    "reaction_key",
                    cls._freeze(key),
                )
            except Exception:
                pass

        product = getattr(
            reaction,
            "product_smiles",
            None,
        )

        reactants = getattr(
            reaction,
            "reactant_smiles",
            None,
        )

        if product is not None and reactants is not None:
            return (
                "reaction_smiles",
                str(product),
                tuple(
                    sorted(
                        str(item)
                        for item in reactants
                    )
                ),
            )

        return (
            "local_reaction_id",
            fallback_id,
        )

    @classmethod
    def _route_semantic_key(
        cls,
        route: Any,
        *,
        fallback_id: str,
    ) -> tuple[Any, ...]:
        explicit = getattr(
            route,
            "route_id",
            None,
        )

        if explicit is not None:
            return (
                "route_id",
                str(explicit),
            )

        metadata = getattr(
            route,
            "metadata",
            None,
        )

        if isinstance(
            metadata,
            Mapping,
        ):
            explicit = (
                metadata.get("route_id")
                or metadata.get("id")
            )

            if explicit is not None:
                return (
                    "route_id",
                    str(explicit),
                )

        return (
            "local_route_id",
            fallback_id,
        )

    @staticmethod
    def _molecule_semantic_key(
        molecule: Any,
    ) -> str:
        canonical = getattr(
            molecule,
            "canonical_smiles",
            None,
        )

        if canonical is not None:
            return str(canonical)

        return str(molecule)

    @classmethod
    def _freeze(
        cls,
        value: Any,
    ) -> Any:
        """
        Convert common nested structures to hashable deterministic tuples.
        """

        if isinstance(
            value,
            Mapping,
        ):
            return tuple(
                sorted(
                    (
                        str(key),
                        cls._freeze(item),
                    )
                    for key, item
                    in value.items()
                )
            )

        if isinstance(
            value,
            Sequence,
        ) and not isinstance(
            value,
            (str, bytes),
        ):
            return tuple(
                cls._freeze(item)
                for item in value
            )

        return value

    # ========================================================
    # Generic helpers
    # ========================================================

    def _enabled(
        self,
        feedback_type: FeedbackType | str,
    ) -> bool:
        normalized = coerce_feedback_type(
            feedback_type
        )

        return (
            normalized
            in self.config.enabled_feedback_types
        )

    @staticmethod
    def _limit(
        values: Sequence[Any],
        limit: int | None,
    ) -> list[Any]:
        if limit is None:
            return list(values)

        return list(values[:limit])

    @staticmethod
    def _validate_unique_query_ids(
        queries: Iterable[FeedbackQuery],
    ) -> None:
        seen: set[str] = set()

        for query in queries:
            if query.query_id in seen:
                raise ValueError(
                    "FeedbackQueryFactory produced duplicate query_id "
                    f"{query.query_id!r}."
                )

            seen.add(
                query.query_id
            )


# ============================================================
# Convenience constructors
# ============================================================


def build_default_query_factory(
    *,
    max_candidates: int | None = None,
    max_reaction_pairs: int | None = None,
    max_route_pairs: int | None = None,
    include_cost_hints: bool = True,
) -> FeedbackQueryFactory:
    """
    Build the full L1-L4 hierarchy query factory.
    """

    return FeedbackQueryFactory(
        FeedbackQueryFactoryConfig(
            max_candidates=max_candidates,
            max_reaction_pairs=max_reaction_pairs,
            max_route_pairs=max_route_pairs,
            include_cost_hints=include_cost_hints,
        )
    )


def build_query_factory_for_types(
    feedback_types: Iterable[
        FeedbackType | str
    ],
    *,
    max_candidates: int | None = None,
    max_reaction_pairs: int | None = None,
    max_route_pairs: int | None = None,
    include_cost_hints: bool = True,
) -> FeedbackQueryFactory:
    """
    Build a factory restricted to a subset of the global hierarchy.

    This is useful for controlled ablations. The feedback types keep their
    global hierarchy levels; this function does not remap reaction_yield from
    L2 to L1 or otherwise change the paper-level taxonomy.
    """

    normalized = frozenset(
        coerce_feedback_type(item)
        for item in feedback_types
    )

    return FeedbackQueryFactory(
        FeedbackQueryFactoryConfig(
            enabled_feedback_types=normalized,
            max_candidates=max_candidates,
            max_reaction_pairs=max_reaction_pairs,
            max_route_pairs=max_route_pairs,
            include_cost_hints=include_cost_hints,
        )
    )

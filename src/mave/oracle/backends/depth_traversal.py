from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import Callable

from mave.chemistry.molecule import Molecule
from mave.chemistry.reaction import Reaction
from mave.environment.purchasable import PurchasableDatabase
from mave.models.single_step import SingleStepModel
from mave.oracle.base import BackendResult, OracleContext
from mave.oracle.context import native_candidates, native_selected_molecule
from mave.oracle.registry import register_backend


# ============================================================
# Interface
# ============================================================


class DepthTraversalBackend(ABC):
    """
    Structured backend for L4 strategic feedback.

    This layer does NOT generate natural language.
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


# ============================================================
# Search state
# ============================================================


class SearchStatus(str, Enum):
    SOLVED = "solved"
    FRONTIER = "frontier"
    DEAD_END = "dead_end"
    CYCLE = "cycle"


@dataclass(frozen=True, slots=True)
class TraversalConfig:
    """
    max_depth and max_expansions bound the internal L4 search.

    The exact values are implementation choices because the
    paper only specifies a depth-limited traversal.
    """

    max_depth: int = 3
    max_expansions: int = 50

    top_k_per_node: int = 50

    goal_top_n: int = 3
    strategy_top_n: int = 3

    max_strategy_length: int = 3
    min_strategy_support: int = 2

    def __post_init__(self) -> None:
        if self.max_depth <= 0:
            raise ValueError(
                "max_depth must be positive."
            )

        if self.max_expansions <= 0:
            raise ValueError(
                "max_expansions must be positive."
            )

        if self.top_k_per_node <= 0:
            raise ValueError(
                "top_k_per_node must be positive."
            )


# ============================================================
# Workload statistics
# ============================================================


@dataclass(slots=True)
class TraversalStats:
    wall_clock_seconds: float = 0.0

    expanded_molecules: int = 0
    expansion_cache_hits: int = 0

    single_step_calls: int = 0

    generated_reactions: int = 0
    generated_precursors: int = 0

    purchasable_hits: int = 0

    cycle_prunes: int = 0
    depth_limit_hits: int = 0
    expansion_limit_hits: int = 0
    dead_end_hits: int = 0

    max_depth_reached: int = 0


# ============================================================
# Structured evidence
# ============================================================


@dataclass(frozen=True, slots=True)
class TransformationEvidence:
    """
    Semantic identity of one retrosynthetic transformation.

    Prefer reaction class when available, otherwise template ID.
    """

    key: str

    reaction_class: str | None = None

    template_id: str | None = None

    source: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "reaction_class":
                self.reaction_class,
            "template_id":
                self.template_id,
            "source":
                self.source,
        }


@dataclass(frozen=True, slots=True)
class SubgoalEvidence:
    smiles: str

    occurrence_count: int

    solved_support: int
    frontier_support: int

    minimum_depth: int

    rank: int

    def to_dict(self) -> dict:
        return {
            "smiles":
                self.smiles,

            "occurrence_count":
                self.occurrence_count,

            "solved_support":
                self.solved_support,

            "frontier_support":
                self.frontier_support,

            "minimum_depth":
                self.minimum_depth,

            "rank":
                self.rank,
        }


@dataclass(frozen=True, slots=True)
class StrategyEvidence:
    transformations: tuple[
        TransformationEvidence,
        ...
    ]

    solved_support: int
    frontier_support: int

    occurrence_count: int

    rank: int

    def to_dict(self) -> dict:
        return {
            "transformations": [
                item.to_dict()
                for item
                in self.transformations
            ],

            "solved_support":
                self.solved_support,

            "frontier_support":
                self.frontier_support,

            "occurrence_count":
                self.occurrence_count,

            "rank":
                self.rank,
        }


@dataclass(frozen=True, slots=True)
class MoleculeOutcome:
    """
    Internal result of one molecule OR-node search.
    """

    status: SearchStatus

    solved_paths: tuple[
        tuple[
            TransformationEvidence,
            ...
        ],
        ...
    ] = ()

    frontier_paths: tuple[
        tuple[
            TransformationEvidence,
            ...
        ],
        ...
    ] = ()


@dataclass(frozen=True, slots=True)
class TraversalEvidence:
    root_status: SearchStatus

    subgoals: tuple[
        SubgoalEvidence,
        ...
    ]

    strategies: tuple[
        StrategyEvidence,
        ...
    ]

    stats: TraversalStats


@dataclass(frozen=True, slots=True)
class _ExpansionResult:
    reactions: tuple[
        Reaction,
        ...
    ]

    budget_limited: bool = False


# ============================================================
# Subgoal evidence collector
# ============================================================


class _SubgoalCollector:

    def __init__(self) -> None:
        self.occurrence: Counter[str] = Counter()
        self.solved: Counter[str] = Counter()
        self.frontier: Counter[str] = Counter()

        self.minimum_depth: dict[
            str,
            int,
        ] = {}

    def add(
        self,
        *,
        molecule: Molecule,
        depth: int,
        status: SearchStatus,
    ) -> None:

        smiles = (
            molecule.canonical_smiles
        )

        self.occurrence[
            smiles
        ] += 1

        if status == SearchStatus.SOLVED:
            self.solved[
                smiles
            ] += 1

        elif status == SearchStatus.FRONTIER:
            self.frontier[
                smiles
            ] += 1

        if smiles not in self.minimum_depth:
            self.minimum_depth[
                smiles
            ] = depth

        else:
            self.minimum_depth[
                smiles
            ] = min(
                self.minimum_depth[
                    smiles
                ],
                depth,
            )


# ============================================================
# Depth-limited AND-OR backend
# ============================================================


@register_backend(
    "depth_limited_traversal"
)
class DepthLimitedTraversalBackend(
    DepthTraversalBackend
):

    def __init__(
        self,
        *,
        single_step_model: SingleStepModel,
        purchasable_db: PurchasableDatabase,
        config: TraversalConfig = TraversalConfig(),
        complexity_fn: (
            Callable[
                [TraversalStats],
                float,
            ]
            | None
        ) = None,
    ) -> None:

        self.single_step_model = (
            single_step_model
        )

        self.purchasable_db = (
            purchasable_db
        )

        self.config = config

        self.complexity_fn = (
            complexity_fn
        )

        # Same intermediate reached through different routes:
        # reuse single-step predictions.
        self._reaction_cache: dict[
            str,
            tuple[Reaction, ...],
        ] = {}

        # Goal suggestion and multi-step strategy can share the
        # same expensive traversal.
        self._evidence_cache: dict[
            tuple,
            TraversalEvidence,
        ] = {}

    # ========================================================
    # Public API
    # ========================================================

    def goal_suggestion(
        self,
        context: OracleContext,
    ) -> BackendResult:
        """
        Structured output:

        {
            "root_status": ...,
            "subgoals": [...]
        }
        """

        evidence = (
            self._get_evidence(
                context
            )
        )

        return BackendResult(
            content={
                "root_status":
                    evidence
                    .root_status
                    .value,

                "subgoals": [
                    item.to_dict()
                    for item
                    in evidence.subgoals
                ],
            },

            complexity_score=(
                self._complexity(
                    evidence.stats
                )
            ),

            metadata=(
                self._stats_metadata(
                    evidence.stats
                )
            ),
        )

    def multi_step_strategy(
        self,
        context: OracleContext,
    ) -> BackendResult:
        """
        Structured output:

        {
            "root_status": ...,
            "strategies": [...]
        }
        """

        evidence = (
            self._get_evidence(
                context
            )
        )

        return BackendResult(
            content={
                "root_status":
                    evidence
                    .root_status
                    .value,

                "strategies": [
                    item.to_dict()
                    for item
                    in evidence.strategies
                ],
            },

            complexity_score=(
                self._complexity(
                    evidence.stats
                )
            ),

            metadata=(
                self._stats_metadata(
                    evidence.stats
                )
            ),
        )

    # ========================================================
    # Traversal cache
    # ========================================================

    def _get_evidence(
        self,
        context: OracleContext,
    ) -> TraversalEvidence:

        key = self._context_key(
            context
        )

        if key not in self._evidence_cache:

            self._evidence_cache[
                key
            ] = self._traverse(
                context
            )

        return self._evidence_cache[
            key
        ]

    def _context_key(
        self,
        context: OracleContext,
    ) -> tuple:

        return (
            context.state.state_id,

            context.selected_molecule,

            tuple(
                str(
                    reaction.key
                )
                for reaction
                in native_candidates(context)
            ),

            self.config.max_depth,
            self.config.max_expansions,
            self.config.top_k_per_node,
        )

    def clear_cache(
        self,
    ) -> None:

        self._reaction_cache.clear()

        self._evidence_cache.clear()

    # ========================================================
    # Main traversal
    # ========================================================

    def _traverse(
        self,
        context: OracleContext,
    ) -> TraversalEvidence:

        start_time = perf_counter()

        stats = TraversalStats()

        collector = (
            _SubgoalCollector()
        )

        root = (
            self.purchasable_db
            .annotate(
                native_selected_molecule(context)
            )
        )

        outcome = self._search_molecule(
            molecule=root,
            depth=0,
            ancestors=frozenset(),
            context=context,
            stats=stats,
            collector=collector,
            use_root_candidates=True,
        )

        stats.wall_clock_seconds = (
            perf_counter()
            - start_time
        )

        subgoals = (
            self._build_subgoals(
                collector
            )
        )

        strategies = (
            self._build_strategies(
                solved_paths=(
                    outcome.solved_paths
                ),
                frontier_paths=(
                    outcome.frontier_paths
                ),
            )
        )

        return TraversalEvidence(
            root_status=(
                outcome.status
            ),

            subgoals=subgoals,

            strategies=strategies,

            stats=stats,
        )

    # ========================================================
    # Molecule OR node
    # ========================================================

    def _search_molecule(
        self,
        *,
        molecule: Molecule,
        depth: int,
        ancestors: frozenset[str],
        context: OracleContext,
        stats: TraversalStats,
        collector: _SubgoalCollector,
        use_root_candidates: bool,
    ) -> MoleculeOutcome:

        molecule = (
            self.purchasable_db
            .annotate(
                molecule
            )
        )

        smiles = (
            molecule.canonical_smiles
        )

        stats.max_depth_reached = max(
            stats.max_depth_reached,
            depth,
        )

        # ----------------------------------------------------
        # Purchasable leaf
        # ----------------------------------------------------

        if molecule.is_purchasable:

            stats.purchasable_hits += 1

            return MoleculeOutcome(
                status=SearchStatus.SOLVED
            )

        # ----------------------------------------------------
        # Cycle
        # ----------------------------------------------------

        if smiles in ancestors:

            stats.cycle_prunes += 1

            return MoleculeOutcome(
                status=SearchStatus.CYCLE
            )

        # ----------------------------------------------------
        # Depth limit != success
        # ----------------------------------------------------

        if depth >= self.config.max_depth:

            stats.depth_limit_hits += 1

            return MoleculeOutcome(
                status=SearchStatus.FRONTIER
            )

        expansion = self._get_reactions(
            molecule=molecule,
            context=context,
            stats=stats,
            use_root_candidates=(
                use_root_candidates
            ),
        )

        if expansion.budget_limited:

            stats.expansion_limit_hits += 1

            return MoleculeOutcome(
                status=SearchStatus.FRONTIER
            )

        reactions = (
            expansion.reactions
        )

        if not reactions:

            stats.dead_end_hits += 1

            return MoleculeOutcome(
                status=SearchStatus.DEAD_END
            )

        next_ancestors = (
            ancestors
            | frozenset(
                [smiles]
            )
        )

        outcomes: list[
            MoleculeOutcome
        ] = []

        # ----------------------------------------------------
        # OR node:
        # any viable reaction can solve the molecule.
        # ----------------------------------------------------

        for reaction in reactions:

            result = self._search_reaction(
                reaction=reaction,
                depth=depth,
                ancestors=(
                    next_ancestors
                ),
                context=context,
                stats=stats,
                collector=collector,
            )

            outcomes.append(
                result
            )

        solved_outcomes = [
            item
            for item in outcomes
            if item.status
            == SearchStatus.SOLVED
        ]

        frontier_outcomes = [
            item
            for item in outcomes
            if item.status
            == SearchStatus.FRONTIER
        ]

        solved_paths = tuple(
            path
            for item in outcomes
            for path
            in item.solved_paths
        )

        frontier_paths = tuple(
            path
            for item in outcomes
            for path
            in item.frontier_paths
        )

        if solved_outcomes:

            return MoleculeOutcome(
                status=SearchStatus.SOLVED,

                solved_paths=(
                    solved_paths
                ),

                frontier_paths=(
                    frontier_paths
                ),
            )

        if frontier_outcomes:

            return MoleculeOutcome(
                status=SearchStatus.FRONTIER,

                solved_paths=(
                    solved_paths
                ),

                frontier_paths=(
                    frontier_paths
                ),
            )

        return MoleculeOutcome(
            status=SearchStatus.DEAD_END
        )

    # ========================================================
    # Reaction AND node
    # ========================================================

    def _search_reaction(
        self,
        *,
        reaction: Reaction,
        depth: int,
        ancestors: frozenset[str],
        context: OracleContext,
        stats: TraversalStats,
        collector: _SubgoalCollector,
    ) -> MoleculeOutcome:

        stats.generated_reactions += 1

        children: list[
            tuple[
                Molecule,
                MoleculeOutcome,
            ]
        ] = []

        # ----------------------------------------------------
        # AND node:
        # every precursor must remain viable.
        # ----------------------------------------------------

        for reactant in reaction.reactants:

            stats.generated_precursors += 1

            reactant = (
                self.purchasable_db
                .annotate(
                    reactant
                )
            )

            outcome = self._search_molecule(
                molecule=reactant,
                depth=depth + 1,
                ancestors=ancestors,
                context=context,
                stats=stats,
                collector=collector,
                use_root_candidates=False,
            )

            children.append(
                (
                    reactant,
                    outcome,
                )
            )

        if not children:

            return MoleculeOutcome(
                status=SearchStatus.DEAD_END
            )

        statuses = [
            outcome.status
            for _, outcome
            in children
        ]

        # ----------------------------------------------------
        # AND semantics
        # ----------------------------------------------------

        if all(
            status
            == SearchStatus.SOLVED
            for status
            in statuses
        ):

            reaction_status = (
                SearchStatus.SOLVED
            )

        elif any(
            status
            in {
                SearchStatus.DEAD_END,
                SearchStatus.CYCLE,
            }
            for status
            in statuses
        ):

            reaction_status = (
                SearchStatus.DEAD_END
            )

        else:

            reaction_status = (
                SearchStatus.FRONTIER
            )

        # ----------------------------------------------------
        # A reaction with a failed precursor is not used as
        # strategic evidence.
        # ----------------------------------------------------

        if (
            reaction_status
            == SearchStatus.DEAD_END
        ):

            return MoleculeOutcome(
                status=SearchStatus.DEAD_END
            )

        # ----------------------------------------------------
        # Record subgoals only from viable AND reactions.
        # ----------------------------------------------------

        for reactant, outcome in children:

            if reactant.is_purchasable:
                continue

            collector.add(
                molecule=reactant,
                depth=depth + 1,
                status=outcome.status,
            )

        transformation = (
            self._transformation(
                reaction
            )
        )

        if (
            reaction_status
            == SearchStatus.SOLVED
        ):

            paths = (
                self._compose_paths(
                    transformation=(
                        transformation
                    ),
                    children=children,
                    solved=True,
                )
            )

            return MoleculeOutcome(
                status=(
                    SearchStatus.SOLVED
                ),

                solved_paths=paths,
            )

        paths = self._compose_paths(
            transformation=(
                transformation
            ),
            children=children,
            solved=False,
        )

        return MoleculeOutcome(
            status=(
                SearchStatus.FRONTIER
            ),

            frontier_paths=paths,
        )

    # ========================================================
    # Build root-to-leaf transformation evidence
    # ========================================================

    @staticmethod
    def _compose_paths(
        *,
        transformation: (
            TransformationEvidence
            | None
        ),
        children: list[
            tuple[
                Molecule,
                MoleculeOutcome,
            ]
        ],
        solved: bool,
    ) -> tuple[
        tuple[
            TransformationEvidence,
            ...
        ],
        ...
    ]:

        child_paths: list[
            tuple[
                TransformationEvidence,
                ...
            ]
        ] = []

        for _, outcome in children:

            if solved:
                paths = (
                    outcome.solved_paths
                )
            else:
                paths = (
                    outcome.solved_paths
                    + outcome.frontier_paths
                )

            child_paths.extend(
                paths
            )

        # All immediate precursors are terminal building blocks.
        if not child_paths:

            if transformation is None:
                return ()

            return (
                (
                    transformation,
                ),
            )

        output = []

        for path in child_paths:

            if transformation is None:
                output.append(
                    path
                )

            else:
                output.append(
                    (
                        transformation,
                    )
                    + path
                )

        return tuple(
            output
        )

    # ========================================================
    # Single-step expansion
    # ========================================================

    def _get_reactions(
        self,
        *,
        molecule: Molecule,
        context: OracleContext,
        stats: TraversalStats,
        use_root_candidates: bool,
    ) -> _ExpansionResult:

        smiles = (
            molecule.canonical_smiles
        )

        # ----------------------------------------------------
        # Current root candidates already exist in the planner.
        # Reuse them without another single-step call.
        # ----------------------------------------------------

        root_candidates = native_candidates(context)

        if use_root_candidates and root_candidates:

            stats.expanded_molecules += 1

            return _ExpansionResult(
                reactions=(
                    self._deduplicate(
                        tuple(
                            root_candidates[
                                :
                                self.config
                                .top_k_per_node
                            ]
                        )
                    )
                )
            )

        # ----------------------------------------------------
        # Transposition cache
        # ----------------------------------------------------

        if smiles in self._reaction_cache:

            stats.expansion_cache_hits += 1

            return _ExpansionResult(
                reactions=(
                    self._reaction_cache[
                        smiles
                    ]
                )
            )

        # ----------------------------------------------------
        # Expansion budget
        # ----------------------------------------------------

        if (
            stats.expanded_molecules
            >= self.config.max_expansions
        ):

            return _ExpansionResult(
                reactions=(),
                budget_limited=True,
            )

        reactions = (
            self.single_step_model
            .predict(
                smiles,
                top_k=(
                    self.config
                    .top_k_per_node
                ),
            )
        )

        stats.expanded_molecules += 1
        stats.single_step_calls += 1

        reactions = (
            self._deduplicate(
                tuple(
                    reactions
                )
            )
        )

        self._reaction_cache[
            smiles
        ] = reactions

        return _ExpansionResult(
            reactions=reactions
        )

    @staticmethod
    def _deduplicate(
        reactions: tuple[
            Reaction,
            ...
        ],
    ) -> tuple[
        Reaction,
        ...
    ]:

        output: list[
            Reaction
        ] = []

        seen = set()

        for reaction in reactions:

            key = str(
                reaction.key
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            output.append(
                reaction
            )

        return tuple(
            output
        )

    # ========================================================
    # Structured transformation identity
    # ========================================================

    @staticmethod
    def _transformation(
        reaction: Reaction,
    ) -> TransformationEvidence | None:

        metadata = (
            reaction.metadata
            or {}
        )

        reaction_class = (
            metadata.get(
                "reaction_class"
            )
            or metadata.get(
                "reaction_name"
            )
            or metadata.get(
                "transformation"
            )
        )

        if reaction_class:

            reaction_class = str(
                reaction_class
            )

            return TransformationEvidence(
                key=(
                    "class:"
                    + reaction_class
                ),

                reaction_class=(
                    reaction_class
                ),

                template_id=(
                    None
                    if reaction.template_id
                    is None
                    else str(
                        reaction.template_id
                    )
                ),

                source=(
                    "reaction_class"
                ),
            )

        if reaction.template_id is not None:

            template_id = str(
                reaction.template_id
            )

            return TransformationEvidence(
                key=(
                    "template:"
                    + template_id
                ),

                reaction_class=None,

                template_id=(
                    template_id
                ),

                source="template",
            )

        # No semantic transformation identity available.
        return None

    # ========================================================
    # Subgoal evidence
    # ========================================================

    def _build_subgoals(
        self,
        collector: _SubgoalCollector,
    ) -> tuple[
        SubgoalEvidence,
        ...
    ]:

        raw = []

        for (
            smiles,
            occurrence,
        ) in (
            collector
            .occurrence
            .items()
        ):

            raw.append(
                (
                    smiles,
                    occurrence,
                    collector.solved[
                        smiles
                    ],
                    collector.frontier[
                        smiles
                    ],
                    collector.minimum_depth[
                        smiles
                    ],
                )
            )

        # ----------------------------------------------------
        # Ranking heuristic:
        #
        # solved support
        #   > frontier support
        #   > occurrence
        #   > shallower depth
        #
        # No arbitrary weighted score is exposed.
        # ----------------------------------------------------

        raw.sort(
            key=lambda item: (
                -item[2],
                -item[3],
                -item[1],
                item[4],
                item[0],
            )
        )

        output = []

        for rank, (
            smiles,
            occurrence,
            solved,
            frontier,
            minimum_depth,
        ) in enumerate(
            raw[
                :
                self.config.goal_top_n
            ],
            start=1,
        ):

            output.append(
                SubgoalEvidence(
                    smiles=smiles,

                    occurrence_count=(
                        occurrence
                    ),

                    solved_support=(
                        solved
                    ),

                    frontier_support=(
                        frontier
                    ),

                    minimum_depth=(
                        minimum_depth
                    ),

                    rank=rank,
                )
            )

        return tuple(
            output
        )

    # ========================================================
    # Multi-step strategy evidence
    # ========================================================

    def _build_strategies(
        self,
        *,
        solved_paths: tuple[
            tuple[
                TransformationEvidence,
                ...
            ],
            ...
        ],
        frontier_paths: tuple[
            tuple[
                TransformationEvidence,
                ...
            ],
            ...
        ],
    ) -> tuple[
        StrategyEvidence,
        ...
    ]:

        solved_counter: Counter[
            tuple[
                TransformationEvidence,
                ...
            ]
        ] = Counter()

        frontier_counter: Counter[
            tuple[
                TransformationEvidence,
                ...
            ]
        ] = Counter()

        self._count_patterns(
            paths=solved_paths,
            counter=solved_counter,
        )

        self._count_patterns(
            paths=frontier_paths,
            counter=frontier_counter,
        )

        patterns = (
            set(
                solved_counter
            )
            | set(
                frontier_counter
            )
        )

        raw = []

        for pattern in patterns:

            solved = (
                solved_counter[
                    pattern
                ]
            )

            frontier = (
                frontier_counter[
                    pattern
                ]
            )

            occurrence = (
                solved
                + frontier
            )

            raw.append(
                (
                    pattern,
                    solved,
                    frontier,
                    occurrence,
                )
            )

        # Prefer recurrent patterns. If none reaches the support
        # threshold, keep weak alternatives rather than
        # inventing a confident strategy.
        supported = [
            item
            for item in raw
            if item[3]
            >= self.config
            .min_strategy_support
        ]

        if supported:
            raw = supported

        raw.sort(
            key=lambda item: (
                -item[1],
                -item[2],
                -item[3],
                -len(
                    item[0]
                ),
                tuple(
                    step.key
                    for step
                    in item[0]
                ),
            )
        )

        output = []

        for rank, (
            pattern,
            solved,
            frontier,
            occurrence,
        ) in enumerate(
            raw[
                :
                self.config
                .strategy_top_n
            ],
            start=1,
        ):

            output.append(
                StrategyEvidence(
                    transformations=(
                        pattern
                    ),

                    solved_support=(
                        solved
                    ),

                    frontier_support=(
                        frontier
                    ),

                    occurrence_count=(
                        occurrence
                    ),

                    rank=rank,
                )
            )

        return tuple(
            output
        )

    def _count_patterns(
        self,
        *,
        paths: tuple[
            tuple[
                TransformationEvidence,
                ...
            ],
            ...
        ],
        counter: Counter,
    ) -> None:
        """
        Count recurring contiguous transformation sequences.

        A pattern contributes at most once per branch, so support
        corresponds more closely to branch-level evidence than
        raw repeated n-gram frequency.
        """

        for path in paths:

            if len(path) < 2:
                continue

            patterns_in_path = set()

            max_length = min(
                len(path),
                self.config
                .max_strategy_length,
            )

            for length in range(
                2,
                max_length + 1,
            ):

                for start in range(
                    len(path)
                    - length
                    + 1
                ):

                    pattern = path[
                        start:
                        start + length
                    ]

                    patterns_in_path.add(
                        pattern
                    )

            for pattern in (
                patterns_in_path
            ):

                counter[
                    pattern
                ] += 1

    # ========================================================
    # Acquisition complexity
    # ========================================================

    def _complexity(
        self,
        stats: TraversalStats,
    ) -> float | None:
        """
        L4 cost is workload-dependent.

        The paper does not specify the exact normalization from
        traversal workload to [0, 1], so an external function can
        be provided. Without one, return None and let the cost
        layer use the base L4 cost.
        """

        if self.complexity_fn is None:
            return None

        value = float(
            self.complexity_fn(
                stats
            )
        )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "complexity_fn must return "
                "a value in [0, 1]."
            )

        return value

    # ========================================================
    # Structured workload metadata
    # ========================================================

    @staticmethod
    def _stats_metadata(
        stats: TraversalStats,
    ) -> dict:

        return {
            "backend":
                "depth_limited_traversal",

            "wall_clock_seconds":
                stats.wall_clock_seconds,

            "expanded_molecules":
                stats.expanded_molecules,

            "expansion_cache_hits":
                stats.expansion_cache_hits,

            "single_step_calls":
                stats.single_step_calls,

            "generated_reactions":
                stats.generated_reactions,

            "generated_precursors":
                stats.generated_precursors,

            "purchasable_hits":
                stats.purchasable_hits,

            "cycle_prunes":
                stats.cycle_prunes,

            "depth_limit_hits":
                stats.depth_limit_hits,

            "expansion_limit_hits":
                stats.expansion_limit_hits,

            "dead_end_hits":
                stats.dead_end_hits,

            "max_depth_reached":
                stats.max_depth_reached,
        }


# ============================================================
# Callable adapter
# ============================================================


@register_backend(
    "callable_depth_traversal"
)
class CallableDepthTraversalBackend(
    DepthTraversalBackend
):
    """
    Adapter for an external/original structured L4 backend.
    """

    def __init__(
        self,
        *,
        goal_suggestion_fn: Callable,
        multi_step_strategy_fn: Callable,
    ) -> None:

        self.goal_suggestion_fn = (
            goal_suggestion_fn
        )

        self.multi_step_strategy_fn = (
            multi_step_strategy_fn
        )

    @staticmethod
    def _wrap(
        value,
        feedback_type: str,
    ) -> BackendResult:

        if isinstance(
            value,
            BackendResult,
        ):
            return value

        return BackendResult(
            content=value,
            complexity_score=None,
            metadata={
                "backend":
                    "callable_depth_traversal",

                "feedback_type":
                    feedback_type,
            },
        )

    def goal_suggestion(
        self,
        context: OracleContext,
    ) -> BackendResult:

        return self._wrap(
            self.goal_suggestion_fn(
                context
            ),
            "goal_suggestion",
        )

    def multi_step_strategy(
        self,
        context: OracleContext,
    ) -> BackendResult:

        return self._wrap(
            self.multi_step_strategy_fn(
                context
            ),
            "multi_step_strategy",
        )

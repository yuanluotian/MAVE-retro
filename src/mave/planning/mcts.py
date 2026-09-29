from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from mave.chemistry.reaction import Reaction
from mave.core.types import (
    Feedback,
    FeedbackLevel,
    PlannerResult,
)
from mave.core.utils import (
    make_query_id,
)

from mave.environment.retro_env import (
    RetroEnvironment,
)

from mave.models.escalation_policy import (
    EscalationPolicy,
)
from mave.models.reaction_policy import (
    ReactionPolicy,
)

from mave.oracle.base import (
    MultiFidelityOracleSystem,
    OracleContext,
)

from mave.planning.and_or_tree import (
    AndOrTree,
    NodeStatus,
    TreePath,
)

from mave.planning.puct import (
    backup,
    select_frontier,
)


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    """
    Paper-specified defaults:
        top_k = 50
        single-step call budget B = 100
        maximum feedback level L = 4
        gamma = 0.95

    Paper-unspecified:
        cpuct
        feedback budget B_f
    """

    max_single_step_calls: int = 100

    top_k: int = 50

    max_feedback_level: int = 4

    gamma: float = 0.95

    # Explicitly required because the paper does not report it.
    cpuct: float | None = None

    # B_f is present in Algorithm 2 but no numerical value is
    # given in the paper.
    feedback_budget: float | None = None

    # Reproduction choice:
    # use argmax policies at inference.
    deterministic_policies: bool = True


# ============================================================
# Planner
# ============================================================


class MCTSPlanner:
    """
    Cost-aware escalation-guided MCTS inference.

    This follows Algorithm 2:

        Selection
        -> Expansion
        -> sequential feedback escalation
        -> reaction-policy scoring
        -> tree expansion
        -> backup

    Reward-cost curve fitting and MAVE derivatives are NOT used
    during inference.
    """

    def __init__(
        self,
        *,
        environment: RetroEnvironment,
        escalation_policy: EscalationPolicy,
        reaction_policy: ReactionPolicy,
        oracle: MultiFidelityOracleSystem,
        config: MCTSConfig,
    ) -> None:

        if config.cpuct is None:
            raise ValueError(
                "The paper does not specify c_puct. "
                "Set MCTSConfig.cpuct explicitly as a "
                "reproduction assumption."
            )

        if config.cpuct < 0:
            raise ValueError(
                "cpuct must be non-negative."
            )

        if (
            config.max_single_step_calls
            <= 0
        ):
            raise ValueError(
                "max_single_step_calls must "
                "be positive."
            )

        if not (
            0
            <= config.max_feedback_level
            <= 4
        ):
            raise ValueError(
                "max_feedback_level must "
                "lie in [0, 4]."
            )

        self.environment = (
            environment
        )

        self.escalation_policy = (
            escalation_policy
        )

        self.reaction_policy = (
            reaction_policy
        )

        self.oracle = oracle

        self.config = config

    # ========================================================
    # Public planning interface
    # ========================================================

    def plan(
        self,
        target_smiles: str,
    ) -> PlannerResult:

        initial_state = (
            self.environment.reset(
                target_smiles
            )
        )

        tree = AndOrTree(
            initial_state.target
        )

        single_step_calls = 0

        feedback_queries = 0
        feedback_cost = 0.0

        # ----------------------------------------------------
        # Target itself may already be purchasable.
        # ----------------------------------------------------

        if tree.root.is_solved:
            return PlannerResult(
                target_smiles=(
                    target_smiles
                ),
                success=True,
                route=(
                    tree.solved_route()
                ),
                single_step_calls=0,
                feedback_queries=0,
                feedback_cost=0.0,
            )

        # ====================================================
        # Main Algorithm 2 loop
        # ====================================================

        while (
            single_step_calls
            < self.config
            .max_single_step_calls
        ):

            # ------------------------------------------------
            # Complete route found.
            # ------------------------------------------------

            if tree.root.is_solved:

                return PlannerResult(
                    target_smiles=(
                        target_smiles
                    ),
                    success=True,
                    route=(
                        tree.solved_route()
                    ),
                    single_step_calls=(
                        single_step_calls
                    ),
                    feedback_queries=(
                        feedback_queries
                    ),
                    feedback_cost=(
                        feedback_cost
                    ),
                )

            if tree.root.is_dead:
                break

            # ------------------------------------------------
            # Selection — Eq. (27)
            # ------------------------------------------------

            path = select_frontier(
                tree,
                cpuct=float(
                    self.config.cpuct
                ),
            )

            frontier = (
                path.frontier
            )

            # Terminal selected frontier.
            if frontier.is_solved:

                backup(
                    path,
                    leaf_value=1.0,
                    gamma=(
                        self.config.gamma
                    ),
                )

                tree.refresh_upwards(
                    frontier
                )

                continue

            if frontier.is_dead:

                backup(
                    path,
                    leaf_value=0.0,
                    gamma=(
                        self.config.gamma
                    ),
                )

                tree.refresh_upwards(
                    frontier
                )

                continue

            # ------------------------------------------------
            # Expansion:
            # single-step model generates A(s)
            # ------------------------------------------------

            candidates = (
                self._generate_candidates(
                    frontier.molecule
                    .canonical_smiles,
                    frontier.molecule,
                )
            )

            single_step_calls += 1

            if not candidates:

                frontier.expanded = True
                frontier.status = (
                    NodeStatus.DEAD
                )

                tree.refresh_upwards(
                    frontier
                )

                backup(
                    path,
                    leaf_value=0.0,
                    gamma=(
                        self.config.gamma
                    ),
                )

                continue

            # ------------------------------------------------
            # Build planning/oracle state for this frontier.
            # ------------------------------------------------

            (
                state,
                selected_index,
            ) = tree.state_for_path(
                path
            )

            # ------------------------------------------------
            # Sequential escalation with mu
            # ------------------------------------------------

            feedback: list[
                Feedback
            ] = []

            current_level = (
                FeedbackLevel.L0
            )

            while (
                int(current_level)
                < self.config
                .max_feedback_level
            ):

                oracle_context = (
                    OracleContext(
                        state=state,
                        selected_index=(
                            selected_index
                        ),
                        candidates=(
                            candidates
                        ),
                        feedback_history=tuple(
                            feedback
                        ),
                    )
                )

                available_queries = (
                    self.oracle
                    .available_next_queries(
                        oracle_context,
                        current_level,
                    )
                )

                if not available_queries:
                    break

                policy_context = (
                    oracle_context
                    .to_escalation_context()
                )

                if (
                    self.config
                    .deterministic_policies
                ):
                    decision = (
                        self.escalation_policy
                        .greedy(
                            policy_context,
                            available_queries,
                        )
                    )

                else:
                    decision = (
                        self.escalation_policy
                        .sample(
                            policy_context,
                            available_queries,
                        )
                    )

                if decision.action.is_stop:
                    break

                query = (
                    decision.action.query
                )

                if query is None:
                    raise RuntimeError(
                        "ESCALATE decision missing "
                        "feedback query."
                    )

                # --------------------------------------------
                # B_f constraint.
                # --------------------------------------------

                if not self._query_fits_budget(
                    state_id=(
                        oracle_context
                        .state_id
                    ),
                    query=query,
                    current_cost=(
                        feedback_cost
                    ),
                ):
                    break

                result = self.oracle.query(
                    oracle_context,
                    query,
                )

                feedback_queries += 1

                feedback_cost += float(
                    result.cost
                )

                feedback.append(
                    result
                )

                current_level = (
                    query.level
                )

            # ------------------------------------------------
            # Reaction policy:
            #
            # P(s,a) = pi(a | s,f)
            # ------------------------------------------------

            priors = (
                self.reaction_policy
                .priors(
                    target_smiles=(
                        state.target
                        .canonical_smiles
                    ),
                    unsolved_molecules=[
                        molecule
                        .canonical_smiles
                        for molecule
                        in state
                        .unsolved_molecules
                    ],
                    reactions=candidates,
                    feedback=feedback,
                )
            )

            if len(priors) != len(
                candidates
            ):
                raise RuntimeError(
                    "Reaction policy returned "
                    "an invalid prior vector."
                )

            # ------------------------------------------------
            # Expand AND-OR tree.
            #
            # Q0(s,a) = P(s,a)
            # ------------------------------------------------

            tree.expand(
                frontier,
                candidates,
                priors,
                feedback=feedback,
            )

            # ------------------------------------------------
            # Eq. (29) / terminal value.
            # ------------------------------------------------

            if frontier.is_solved:
                leaf_value = 1.0

            elif frontier.is_dead:
                leaf_value = 0.0

            else:
                leaf_value = max(
                    priors
                )

            # ------------------------------------------------
            # Eq. (30) backup.
            # ------------------------------------------------

            backup(
                path,
                leaf_value=(
                    leaf_value
                ),
                gamma=(
                    self.config.gamma
                ),
            )

            tree.refresh_upwards(
                frontier
            )

            # Immediate route completion after expansion.
            if tree.root.is_solved:

                return PlannerResult(
                    target_smiles=(
                        target_smiles
                    ),
                    success=True,
                    route=(
                        tree.solved_route()
                    ),
                    single_step_calls=(
                        single_step_calls
                    ),
                    feedback_queries=(
                        feedback_queries
                    ),
                    feedback_cost=(
                        feedback_cost
                    ),
                )

        # ====================================================
        # Search budget exhausted / dead tree
        # ====================================================

        return PlannerResult(
            target_smiles=(
                target_smiles
            ),
            success=False,
            route=None,
            single_step_calls=(
                single_step_calls
            ),
            feedback_queries=(
                feedback_queries
            ),
            feedback_cost=(
                feedback_cost
            ),
            metadata={
                "root_status":
                    tree.root.status.value,
            },
        )

    # ========================================================
    # Candidate generation
    # ========================================================

    def _generate_candidates(
        self,
        product_smiles: str,
        product_molecule,
    ) -> tuple[
        Reaction,
        ...
    ]:
        """
        Generate top-k reactions and annotate precursor
        purchasability before inserting them into the tree.
        """

        raw = (
            self.environment
            .single_step_model
            .predict(
                product_smiles,
                top_k=(
                    self.config.top_k
                ),
            )
        )

        output: list[
            Reaction
        ] = []

        seen = set()

        for reaction in raw:

            annotated_reactants = tuple(
                self.environment
                .purchasable_db
                .annotate(
                    reactant
                )
                for reactant
                in reaction.reactants
            )

            annotated = Reaction(
                product=product_molecule,
                reactants=(
                    annotated_reactants
                ),
                score=reaction.score,
                template_id=(
                    reaction.template_id
                ),
                reaction_id=(
                    reaction.reaction_id
                ),
                metadata=(
                    reaction.metadata
                ),
            )

            if annotated.key in seen:
                continue

            seen.add(
                annotated.key
            )

            output.append(
                annotated
            )

            if (
                len(output)
                >= self.config.top_k
            ):
                break

        return tuple(output)

    # ========================================================
    # Feedback budget
    # ========================================================

    def _query_fits_budget(
        self,
        *,
        state_id: str,
        query,
        current_cost: float,
    ) -> bool:
        """
        Algorithm 2 checks:

            C + c(q) <= B_f

        before acquiring uncached feedback.

        Exact pre-query complexity-dependent c(q) is not exposed
        by the current oracle interface. Therefore, when B_f is
        configured, we conservatively check the maximum possible
        cost for that query type.

        Cached queries always cost zero.
        """

        budget = (
            self.config
            .feedback_budget
        )

        if budget is None:
            return True

        query_id = (
            query.query_id
            or make_query_id(
                state_id=state_id,
                level=int(
                    query.level
                ),
                feedback_type=(
                    query.feedback_type
                ),
                payload=query.payload,
            )
        )

        if self.oracle.cache.contains(
            query_id
        ):
            return True

        spec = (
            self.oracle
            .cost_model
            .spec(
                query.feedback_type
            )
        )

        return (
            current_cost
            + spec.maximum
            <= budget
        )
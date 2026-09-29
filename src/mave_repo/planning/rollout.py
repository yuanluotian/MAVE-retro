from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from mave_repro.chemistry.reaction import (
    Reaction,
)

from mave_repro.core.types import (
    EscalationDecision,
    Feedback,
    FeedbackLevel,
)

from mave_repro.environment.retro_env import (
    RetroEnvironment,
)
from mave_repro.environment.reward import (
    discounted_return,
)
from mave_repro.environment.state import (
    RetroState,
)

from mave_repro.models.escalation_policy import (
    EscalationPolicy,
)
from mave_repro.models.reaction_policy import (
    ReactionPolicy,
)

from mave_repro.oracle.base import (
    MultiFidelityOracleSystem,
    OracleContext,
)


# ============================================================
# Trajectory data
# ============================================================


@dataclass(frozen=True, slots=True)
class PlanningRolloutStep:
    state: RetroState

    selected_index: int

    candidates: tuple[
        Reaction,
        ...
    ]

    escalation_decisions: tuple[
        EscalationDecision,
        ...
    ]

    feedback: tuple[
        Feedback,
        ...
    ]

    reaction: Reaction

    reaction_index: int
    reaction_logprob: float

    reward: float

    next_state: RetroState


@dataclass(frozen=True, slots=True)
class PlanningTrajectory:
    initial_state: RetroState

    steps: tuple[
        PlanningRolloutStep,
        ...
    ]

    final_state: RetroState

    success: bool

    single_step_calls: int

    feedback_queries: int

    feedback_cost: float

    discounted_return: float

    metadata: dict = field(
        default_factory=dict
    )

    @property
    def rewards(
        self,
    ) -> tuple[float, ...]:
        return tuple(
            step.reward
            for step
            in self.steps
        )

    @property
    def num_steps(self) -> int:
        return len(
            self.steps
        )


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True, slots=True)
class PlanningRolloutConfig:
    """
    Training/trajectory rollout configuration.
    """

    max_steps: int = 50

    max_single_step_calls: int = 100

    top_k: int = 50

    max_feedback_level: int = 4

    gamma: float = 0.95

    feedback_budget: (
        float | None
    ) = None


# ============================================================
# Rollout runner
# ============================================================


class PlanningRolloutRunner:
    """
    Sample planning trajectories using mu and pi.

    Unlike MCTS inference, this class samples policy actions and
    is intended for rollout collection during training.

    The rollout is intentionally planner-light; MAVE itself only
    needs the resulting downstream return and feedback cost.
    """

    def __init__(
        self,
        *,
        environment: RetroEnvironment,
        escalation_policy: EscalationPolicy,
        reaction_policy: ReactionPolicy,
        oracle: MultiFidelityOracleSystem,
        config: (
            PlanningRolloutConfig
        ) = PlanningRolloutConfig(),
        molecule_selector: (
            Callable[
                [RetroState],
                int,
            ]
            | None
        ) = None,
    ) -> None:

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

        # Paper does not specify how a non-search rollout chooses
        # among multiple unsolved molecules.
        #
        # Reproduction choice:
        # use the first unresolved leaf in route order.
        self.molecule_selector = (
            molecule_selector
            or self._first_unsolved
        )

    # ========================================================
    # Rollout
    # ========================================================

    def rollout(
        self,
        initial_state: RetroState,
        *,
        forced_first_reaction: (
            Reaction | None
        ) = None,
        forced_first_feedback: Sequence[
            Feedback
        ] = (),
    ) -> PlanningTrajectory:
        """
        Continue planning until success or rollout budget
        exhaustion.

        `forced_first_reaction` is useful for Algorithm 1:
        the caller can sample the current-context escalation and
        reaction action externally, then use this runner only for
        the subsequent "continue planning to termination" part.
        """

        state = initial_state

        steps: list[
            PlanningRolloutStep
        ] = []

        single_step_calls = 0

        feedback_queries = 0
        feedback_cost = 0.0

        first_step = True

        while (
            not state.is_terminal
            and len(steps)
            < self.config.max_steps
            and single_step_calls
            < self.config
            .max_single_step_calls
        ):

            selected_index = (
                self.molecule_selector(
                    state
                )
            )

            # ------------------------------------------------
            # Candidate generation.
            # ------------------------------------------------

            candidates = (
                self.environment
                .candidates(
                    state,
                    selected_index=(
                        selected_index
                    ),
                    top_k=(
                        self.config.top_k
                    ),
                )
            )

            single_step_calls += 1

            if not candidates:
                break

            # =================================================
            # Forced first decision from Algorithm-1 collector.
            # =================================================

            if (
                first_step
                and forced_first_reaction
                is not None
            ):

                try:
                    reaction_index = (
                        candidates.index(
                            forced_first_reaction
                        )
                    )

                except ValueError as exc:
                    raise ValueError(
                        "forced_first_reaction is not "
                        "present in current candidate set."
                    ) from exc

                reaction = (
                    forced_first_reaction
                )

                feedback = list(
                    forced_first_feedback
                )

                escalation_decisions: list[
                    EscalationDecision
                ] = []

                reaction_logprob = 0.0

            # =================================================
            # Normal sampled planning step.
            # =================================================

            else:

                (
                    feedback,
                    escalation_decisions,
                    query_count,
                    query_cost,
                ) = self._sample_feedback(
                    state=state,
                    selected_index=(
                        selected_index
                    ),
                    candidates=candidates,
                    current_total_cost=(
                        feedback_cost
                    ),
                )

                feedback_queries += (
                    query_count
                )

                feedback_cost += (
                    query_cost
                )

                reaction_output = (
                    self.reaction_policy
                    .sample(
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

                reaction = (
                    reaction_output.reaction
                )

                reaction_index = (
                    reaction_output.index
                )

                reaction_logprob = (
                    reaction_output.logprob
                )

            # ------------------------------------------------
            # Environment transition.
            # ------------------------------------------------

            transition = (
                self.environment.step(
                    state,
                    reaction,
                    selected_index=(
                        selected_index
                    ),
                )
            )

            steps.append(
                PlanningRolloutStep(
                    state=state,
                    selected_index=(
                        selected_index
                    ),
                    candidates=(
                        candidates
                    ),
                    escalation_decisions=tuple(
                        escalation_decisions
                    ),
                    feedback=tuple(
                        feedback
                    ),
                    reaction=reaction,
                    reaction_index=(
                        reaction_index
                    ),
                    reaction_logprob=(
                        reaction_logprob
                    ),
                    reward=(
                        transition.reward
                    ),
                    next_state=(
                        transition.next_state
                    ),
                )
            )

            state = (
                transition.next_state
            )

            first_step = False

        rewards = tuple(
            step.reward
            for step
            in steps
        )

        return_value = (
            discounted_return(
                rewards,
                gamma=(
                    self.config.gamma
                ),
            )
        )

        return PlanningTrajectory(
            initial_state=(
                initial_state
            ),
            steps=tuple(
                steps
            ),
            final_state=state,
            success=(
                state.is_terminal
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
            discounted_return=(
                return_value
            ),
        )

    # ========================================================
    # Sequential feedback acquisition
    # ========================================================

    def _sample_feedback(
        self,
        *,
        state: RetroState,
        selected_index: int,
        candidates: Sequence[
            Reaction
        ],
        current_total_cost: float,
    ) -> tuple[
        list[Feedback],
        list[EscalationDecision],
        int,
        float,
    ]:

        feedback: list[
            Feedback
        ] = []

        decisions: list[
            EscalationDecision
        ] = []

        query_count = 0
        acquired_cost = 0.0

        level = FeedbackLevel.L0

        while (
            int(level)
            < self.config
            .max_feedback_level
        ):

            context = OracleContext(
                state=state,
                selected_index=(
                    selected_index
                ),
                candidates=tuple(
                    candidates
                ),
                feedback_history=tuple(
                    feedback
                ),
            )

            queries = (
                self.oracle
                .available_next_queries(
                    context,
                    level,
                )
            )

            if not queries:
                break

            policy_context = (
                context
                .to_escalation_context()
            )

            decision = (
                self.escalation_policy
                .sample(
                    policy_context,
                    queries,
                )
            )

            decisions.append(
                decision
            )

            if decision.action.is_stop:
                break

            query = (
                decision.action.query
            )

            if query is None:
                raise RuntimeError(
                    "ESCALATE action missing query."
                )

            # Training rollout budget handling:
            # use base cost as a pre-check because the exact
            # complexity-adjusted cost is only known after the
            # backend call in the current oracle API.
            if (
                self.config.feedback_budget
                is not None
            ):

                spec = (
                    self.oracle
                    .cost_model
                    .spec(
                        query.feedback_type
                    )
                )

                if (
                    current_total_cost
                    + acquired_cost
                    + spec.maximum
                    > self.config
                    .feedback_budget
                ):
                    break

            result = (
                self.oracle.query(
                    context,
                    query,
                )
            )

            feedback.append(
                result
            )

            query_count += 1

            acquired_cost += float(
                result.cost
            )

            level = (
                query.level
            )

        return (
            feedback,
            decisions,
            query_count,
            acquired_cost,
        )

    # ========================================================
    # Default frontier selection
    # ========================================================

    @staticmethod
    def _first_unsolved(
        state: RetroState,
    ) -> int:

        if not (
            state.unsolved_molecules
        ):
            raise RuntimeError(
                "State contains no unsolved "
                "molecules."
            )

        return 0
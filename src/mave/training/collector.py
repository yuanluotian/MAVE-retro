from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from mave.chemistry.reaction import Reaction

from mave.core.types import (
    EscalationContext,
    EscalationDecision,
    EscalationRollout,
    Feedback,
    FeedbackLevel,
    FeedbackQuery,
    ReactionCandidate,
    ReactionDecision,
    RolloutGroup,
)

from mave.environment.state import RetroState

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

from mave.planning.rollout import (
    PlanningRolloutRunner,
)


# ============================================================
# Policy-step records for GRPO
# ============================================================


@dataclass(frozen=True, slots=True)
class EscalationPolicyStep:
    """
    Everything required to recompute

        log mu(e | s, f)

    during the GRPO update.
    """

    context: EscalationContext

    available_queries: tuple[
        FeedbackQuery,
        ...
    ]

    action_index: int

    old_logprob: float


@dataclass(frozen=True, slots=True)
class ReactionPolicyStep:
    """
    Everything required to recompute

        log pi(a | s, f)
    """

    target_smiles: str

    unsolved_molecules: tuple[
        str,
        ...
    ]

    reactions: tuple[
        Reaction,
        ...
    ]

    feedback: tuple[
        Feedback,
        ...
    ]

    action_index: int

    old_logprob: float

    downstream_return: float


# ============================================================
# Escalation-group collection
# ============================================================


@dataclass(frozen=True, slots=True)
class CollectedEscalationRollout:
    rollout: EscalationRollout

    escalation_steps: tuple[
        EscalationPolicyStep,
        ...
    ]


@dataclass(frozen=True, slots=True)
class CollectedEscalationGroup:
    """
    K complete escalation rollouts from one planning context.
    """

    rollout_group: RolloutGroup

    traces: tuple[
        CollectedEscalationRollout,
        ...
    ]

    def __post_init__(self) -> None:
        if (
            len(self.rollout_group.rollouts)
            != len(self.traces)
        ):
            raise ValueError(
                "RolloutGroup/traces size mismatch."
            )


# ============================================================
# Reaction-policy group
# ============================================================


@dataclass(frozen=True, slots=True)
class ReactionTrainingGroup:
    """
    Group of reaction-policy actions sampled from the same
    planning state and the same acquired-feedback context.

    IMPORTANT:
    The paper states that pi uses standard GRPO with downstream
    returns but does not fully specify how its GRPO groups are
    constructed.

    Grouping multiple reaction samples from the same context is
    therefore a reproduction choice.
    """

    context_id: str

    steps: tuple[
        ReactionPolicyStep,
        ...
    ]


# ============================================================
# Helpers
# ============================================================


def reaction_to_candidate(
    reaction: Reaction,
) -> ReactionCandidate:

    return ReactionCandidate(
        product_smiles=(
            reaction.product_smiles
        ),
        reactant_smiles=(
            reaction.reactant_smiles
        ),
        single_step_score=(
            0.0
            if reaction.score is None
            else float(
                reaction.score
            )
        ),
        template_id=(
            reaction.template_id
        ),
        reaction_id=(
            reaction.reaction_id
        ),
        metadata=dict(
            reaction.metadata
        ),
    )


# ============================================================
# MAVE escalation-group collector
# ============================================================


class EscalationGroupCollector:
    """
    Implements Algorithm 1 at one fixed planning context.

    For each k = 1 ... K:

        1. start with l=0, f=empty, c=0
        2. sample STOP / ESCALATE with mu
        3. accumulate feedback and current escalation cost
        4. sample reaction action with pi
        5. continue planning to termination
        6. obtain downstream return r_k

    The returned RolloutGroup contains {(c_k, r_k)}.
    """

    def __init__(
        self,
        *,
        escalation_policy: EscalationPolicy,
        reaction_policy: ReactionPolicy,
        oracle: MultiFidelityOracleSystem,
        continuation_runner: PlanningRolloutRunner,
        group_size: int = 4,
        max_feedback_level: int = 4,

        # Algorithm 1 does not specify cache behavior across the
        # K independent branches.
        #
        # Reproduction choice:
        # isolate each rollout so that an identical query in
        # rollout k+1 does not become artificially free merely
        # because rollout k happened to query it.
        reset_cache_per_rollout: bool = True,
    ) -> None:

        if group_size <= 0:
            raise ValueError(
                "group_size must be positive."
            )

        self.escalation_policy = (
            escalation_policy
        )

        self.reaction_policy = (
            reaction_policy
        )

        self.oracle = oracle

        self.continuation_runner = (
            continuation_runner
        )

        self.group_size = int(
            group_size
        )

        self.max_feedback_level = int(
            max_feedback_level
        )

        self.reset_cache_per_rollout = (
            reset_cache_per_rollout
        )

    # ========================================================
    # Collection
    # ========================================================

    def collect(
        self,
        *,
        state: RetroState,
        selected_index: int,
        candidates: Sequence[
            Reaction
        ],
    ) -> CollectedEscalationGroup:

        if not candidates:
            raise ValueError(
                "Escalation group requires "
                "reaction candidates."
            )

        context_id = state.state_id

        collected: list[
            CollectedEscalationRollout
        ] = []

        for _ in range(
            self.group_size
        ):

            if (
                self.reset_cache_per_rollout
            ):
                self.oracle.clear_cache()

            collected.append(
                self._collect_one(
                    state=state,
                    selected_index=(
                        selected_index
                    ),
                    candidates=tuple(
                        candidates
                    ),
                    context_id=(
                        context_id
                    ),
                )
            )

        rollout_group = RolloutGroup(
            context_id=context_id,
            rollouts=tuple(
                item.rollout
                for item
                in collected
            ),
        )

        return CollectedEscalationGroup(
            rollout_group=(
                rollout_group
            ),
            traces=tuple(
                collected
            ),
        )

    # ========================================================
    # One rollout branch
    # ========================================================

    def _collect_one(
        self,
        *,
        state: RetroState,
        selected_index: int,
        candidates: tuple[
            Reaction,
            ...
        ],
        context_id: str,
    ) -> CollectedEscalationRollout:

        feedback: list[
            Feedback
        ] = []

        decisions: list[
            EscalationDecision
        ] = []

        policy_steps: list[
            EscalationPolicyStep
        ] = []

        current_level = (
            FeedbackLevel.L0
        )

        current_cost = 0.0

        # ====================================================
        # Sequential escalation
        # ====================================================

        while (
            int(current_level)
            < self.max_feedback_level
        ):

            oracle_context = (
                OracleContext(
                    state=state,
                    selected_index=(
                        selected_index
                    ),
                    candidates=candidates,
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

            decision = (
                self.escalation_policy
                .sample(
                    policy_context,
                    available_queries,
                )
            )

            action_index = int(
                decision.metadata[
                    "choice_index"
                ]
            )

            policy_steps.append(
                EscalationPolicyStep(
                    context=policy_context,
                    available_queries=tuple(
                        available_queries
                    ),
                    action_index=(
                        action_index
                    ),
                    old_logprob=(
                        decision.logprob
                    ),
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
                    "ESCALATE action has no query."
                )

            result = self.oracle.query(
                oracle_context,
                query,
            )

            feedback.append(
                result
            )

            current_cost += float(
                result.cost
            )

            current_level = (
                query.level
            )

        # ====================================================
        # Sample reaction under accumulated feedback
        # ====================================================

        reaction_output = (
            self.reaction_policy.sample(
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

        selected_reaction = (
            reaction_output.reaction
        )

        # ====================================================
        # Continue planning to termination
        # ====================================================

        trajectory = (
            self.continuation_runner
            .rollout(
                state,
                forced_first_reaction=(
                    selected_reaction
                ),
                forced_first_feedback=(
                    feedback
                ),
            )
        )

        # IMPORTANT:
        # c_k is the cost incurred during THIS escalation
        # process only. Do not add future feedback cost from the
        # downstream trajectory.
        c_k = current_cost

        r_k = (
            trajectory
            .discounted_return
        )

        reaction_decision = (
            ReactionDecision(
                reaction=(
                    reaction_to_candidate(
                        selected_reaction
                    )
                ),
                logprob=(
                    reaction_output
                    .logprob
                ),
                probability=(
                    reaction_output
                    .probability
                ),
                metadata={
                    "reaction_index":
                        reaction_output.index,
                },
            )
        )

        rollout = EscalationRollout(
            context_id=context_id,
            escalation_decisions=tuple(
                decisions
            ),
            feedback=tuple(
                feedback
            ),
            reaction_decision=(
                reaction_decision
            ),
            cost=float(
                c_k
            ),
            downstream_return=float(
                r_k
            ),
            success=(
                trajectory.success
            ),
            terminal_depth=(
                trajectory
                .final_state
                .depth
            ),
        )

        return CollectedEscalationRollout(
            rollout=rollout,
            escalation_steps=tuple(
                policy_steps
            ),
        )


# ============================================================
# Reaction-policy group collector
# ============================================================


class ReactionGroupCollector:
    """
    Collect K reaction actions from the same state/feedback
    context and continue each sampled action to termination.

    This provides ordinary group-relative downstream-return
    advantages for reaction-policy GRPO.

    This precise grouping is not fully specified by the paper
    and is therefore a reproduction choice.
    """

    def __init__(
        self,
        *,
        reaction_policy: ReactionPolicy,
        continuation_runner: PlanningRolloutRunner,
        group_size: int = 4,
    ) -> None:

        if group_size <= 0:
            raise ValueError(
                "group_size must be positive."
            )

        self.reaction_policy = (
            reaction_policy
        )

        self.continuation_runner = (
            continuation_runner
        )

        self.group_size = (
            int(group_size)
        )

    def collect(
        self,
        *,
        state: RetroState,
        candidates: Sequence[
            Reaction
        ],
        feedback: Sequence[
            Feedback
        ],
    ) -> ReactionTrainingGroup:

        candidates = tuple(
            candidates
        )

        feedback = tuple(
            feedback
        )

        if not candidates:
            raise ValueError(
                "Reaction group requires "
                "candidate reactions."
            )

        steps: list[
            ReactionPolicyStep
        ] = []

        for _ in range(
            self.group_size
        ):

            output = (
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

            trajectory = (
                self.continuation_runner
                .rollout(
                    state,
                    forced_first_reaction=(
                        output.reaction
                    ),
                    forced_first_feedback=(
                        feedback
                    ),
                )
            )

            steps.append(
                ReactionPolicyStep(
                    target_smiles=(
                        state.target
                        .canonical_smiles
                    ),
                    unsolved_molecules=tuple(
                        molecule
                        .canonical_smiles
                        for molecule
                        in state
                        .unsolved_molecules
                    ),
                    reactions=candidates,
                    feedback=feedback,
                    action_index=(
                        output.index
                    ),
                    old_logprob=(
                        output.logprob
                    ),
                    downstream_return=(
                        trajectory
                        .discounted_return
                    ),
                )
            )

        return ReactionTrainingGroup(
            context_id=(
                state.state_id
            ),
            steps=tuple(
                steps
            ),
        )
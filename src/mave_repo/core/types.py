from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Mapping, Sequence


# ============================================================
# Generic aliases
# ============================================================

Metadata = dict[str, Any]
Payload = dict[str, Any]


# ============================================================
# Feedback hierarchy
# ============================================================


class FeedbackLevel(IntEnum):
    """
    Ordered feedback levels used in the paper.

    L0: no feedback
    L1: structural
    L2: evaluative
    L3: comparative
    L4: strategic
    """

    L0 = 0
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4


class EscalationActionType(str, Enum):
    """
    Escalation action defined in Eq. (3):

        STOP
        ESCALATE(q)
    """

    STOP = "STOP"
    ESCALATE = "ESCALATE"


# ============================================================
# Reaction / planning state
# ============================================================


@dataclass(frozen=True, slots=True)
class ReactionCandidate:
    """
    Candidate retrosynthetic reaction proposed by the
    single-step model.

    The paper retains top-50 single-step predictions.
    """

    product_smiles: str
    reactant_smiles: tuple[str, ...]

    # Score returned by the single-step retrosynthesis model.
    single_step_score: float

    template_id: str | None = None

    # Optional stable identifier.
    reaction_id: str | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.product_smiles:
            raise ValueError("product_smiles must not be empty.")

        if len(self.reactant_smiles) == 0:
            raise ValueError(
                "reactant_smiles must contain at least one reactant."
            )

    @property
    def num_reactants(self) -> int:
        return len(self.reactant_smiles)


@dataclass(frozen=True, slots=True)
class PlanningState:
    """
    Retrosynthetic planning state.

    Following the paper, the state contains the current set of
    unsolved molecules. Applying a reaction replaces one selected
    product with its predicted reactants.
    """

    target_smiles: str

    unsolved_molecules: tuple[str, ...]

    # Number of reaction decisions already applied.
    depth: int = 0

    # Optional state identifier, useful for caching.
    state_id: str | None = None

    # Keep the core representation planner-agnostic.
    # Tree-specific information should live in planning/.
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target_smiles:
            raise ValueError("target_smiles must not be empty.")

        if self.depth < 0:
            raise ValueError("depth must be non-negative.")

    @property
    def is_empty(self) -> bool:
        return len(self.unsolved_molecules) == 0


# ============================================================
# Feedback
# ============================================================


@dataclass(frozen=True, slots=True)
class FeedbackQuery:
    """
    A query q belonging to one feedback level Q^(l).

    The exact payload depends on the oracle:
      - reaction candidate
      - reaction pair
      - partial route
      - current molecule
      - planning frontier
      etc.
    """

    level: FeedbackLevel

    feedback_type: str

    payload: Mapping[str, Any] = field(default_factory=dict)

    query_id: str | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level == FeedbackLevel.L0:
            raise ValueError(
                "L0 represents no feedback and should not contain a query."
            )

        if not self.feedback_type:
            raise ValueError("feedback_type must not be empty.")


@dataclass(frozen=True, slots=True)
class Feedback:
    """
    Feedback returned by oracle O(q).

    cost corresponds to C(q).
    """

    level: FeedbackLevel

    feedback_type: str

    content: Any

    cost: float

    query_id: str | None = None

    # True when retrieved from cache rather than newly acquired.
    cached: bool = False

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError("Feedback cost must be non-negative.")

        if not self.feedback_type:
            raise ValueError("feedback_type must not be empty.")


# ============================================================
# Escalation policy actions
# ============================================================


@dataclass(frozen=True, slots=True)
class EscalationAction:
    """
    Action sampled from escalation policy mu.

    Either:

        STOP

    or:

        ESCALATE(query)
    """

    action_type: EscalationActionType

    query: FeedbackQuery | None = None

    @classmethod
    def stop(cls) -> "EscalationAction":
        return cls(
            action_type=EscalationActionType.STOP,
            query=None,
        )

    @classmethod
    def escalate(
        cls,
        query: FeedbackQuery,
    ) -> "EscalationAction":
        return cls(
            action_type=EscalationActionType.ESCALATE,
            query=query,
        )

    def __post_init__(self) -> None:
        if (
            self.action_type == EscalationActionType.STOP
            and self.query is not None
        ):
            raise ValueError(
                "STOP action must not contain a feedback query."
            )

        if (
            self.action_type == EscalationActionType.ESCALATE
            and self.query is None
        ):
            raise ValueError(
                "ESCALATE action must contain a feedback query."
            )

    @property
    def is_stop(self) -> bool:
        return self.action_type == EscalationActionType.STOP

    @property
    def is_escalate(self) -> bool:
        return self.action_type == EscalationActionType.ESCALATE


# ============================================================
# Escalation context
# ============================================================


@dataclass(frozen=True, slots=True)
class EscalationContext:
    """
    Input context for escalation policy mu.

    Corresponds conceptually to:

        mu(. | s_t, f_t^(<=l))

    Candidate reactions are included because practical policy
    prompts may need access to the current reaction space.
    """

    state: PlanningState

    candidates: tuple[ReactionCandidate, ...]

    current_level: FeedbackLevel = FeedbackLevel.L0

    feedback_history: tuple[Feedback, ...] = ()

    context_id: str | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.candidates) == 0:
            raise ValueError(
                "EscalationContext requires at least one reaction candidate."
            )

    @property
    def cumulative_cost(self) -> float:
        return float(
            sum(feedback.cost for feedback in self.feedback_history)
        )

    @property
    def has_feedback(self) -> bool:
        return len(self.feedback_history) > 0


# ============================================================
# Policy trajectory records
# ============================================================


@dataclass(frozen=True, slots=True)
class EscalationDecision:
    """
    One decision made by escalation policy mu.

    logprob is needed later for GRPO.
    """

    level: FeedbackLevel

    action: EscalationAction

    logprob: float

    # Optional probability / entropy information for diagnostics.
    probability: float | None = None
    entropy: float | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReactionDecision:
    """
    Reaction action sampled or selected from policy pi.
    """

    reaction: ReactionCandidate

    logprob: float

    probability: float | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)


# ============================================================
# Complete escalation rollout
# ============================================================


@dataclass(frozen=True, slots=True)
class EscalationRollout:
    """
    One complete rollout k from a shared planning context.

    Paper-level quantities:

        c_k = accumulated feedback acquisition cost
        r_k = downstream planning return

    A single rollout-level MAVE advantage will later be assigned
    to all escalation decisions in this rollout.
    """

    context_id: str

    escalation_decisions: tuple[EscalationDecision, ...]

    feedback: tuple[Feedback, ...]

    reaction_decision: ReactionDecision

    cost: float

    downstream_return: float

    # Optional information about full downstream planning.
    success: bool | None = None

    terminal_depth: int | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError("Rollout cost must be non-negative.")

    @property
    def stopping_level(self) -> FeedbackLevel:
        """
        Highest acquired feedback level.

        If no feedback was acquired, return L0.
        """

        if not self.feedback:
            return FeedbackLevel.L0

        return max(
            feedback.level
            for feedback in self.feedback
        )

    @property
    def num_queries(self) -> int:
        return len(self.feedback)

    @property
    def num_escalation_decisions(self) -> int:
        return len(self.escalation_decisions)


# ============================================================
# Rollout group
# ============================================================


@dataclass(frozen=True, slots=True)
class RolloutGroup:
    """
    K complete escalation rollouts sampled from the same
    planning context.

    This is the direct input to reward-cost fitting.
    """

    context_id: str

    rollouts: tuple[EscalationRollout, ...]

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.rollouts) == 0:
            raise ValueError(
                "RolloutGroup must contain at least one rollout."
            )

        mismatched = [
            rollout.context_id
            for rollout in self.rollouts
            if rollout.context_id != self.context_id
        ]

        if mismatched:
            raise ValueError(
                "All rollouts in a group must share the same context_id."
            )

    def costs(self) -> tuple[float, ...]:
        return tuple(
            rollout.cost
            for rollout in self.rollouts
        )

    def returns(self) -> tuple[float, ...]:
        return tuple(
            rollout.downstream_return
            for rollout in self.rollouts
        )

    def successes(self) -> tuple[bool | None, ...]:
        return tuple(
            rollout.success
            for rollout in self.rollouts
        )

    @property
    def size(self) -> int:
        return len(self.rollouts)


# ============================================================
# MAVE observations / outputs
# ============================================================


@dataclass(frozen=True, slots=True)
class RewardCostObservation:
    """
    Single (c_k, r_k) observation used for fitting phi(c).
    """

    cost: float
    reward: float

    rollout_index: int | None = None


@dataclass(frozen=True, slots=True)
class MarginalValueSignals:
    """
    First- and second-order signals:

        D1 = phi'(c)
        D2 = phi''(c)
    """

    d1: float
    d2: float


@dataclass(frozen=True, slots=True)
class MAVEAdvantageRecord:
    """
    Diagnostics for one rollout's MAVE advantage.
    """

    rollout_index: int

    cost: float
    reward: float

    j: float
    d1: float
    d2: float

    j_normalized: float
    d1_normalized: float
    d2_normalized: float

    advantage: float


# ============================================================
# Planner result
# ============================================================


@dataclass(frozen=True, slots=True)
class PlannerResult:
    """
    Generic result returned by retrosynthetic planner.
    """

    target_smiles: str

    success: bool

    route: Any | None

    single_step_calls: int

    feedback_queries: int = 0
    feedback_cost: float = 0.0

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.single_step_calls < 0:
            raise ValueError(
                "single_step_calls must be non-negative."
            )

        if self.feedback_queries < 0:
            raise ValueError(
                "feedback_queries must be non-negative."
            )

        if self.feedback_cost < 0:
            raise ValueError(
                "feedback_cost must be non-negative."
            )
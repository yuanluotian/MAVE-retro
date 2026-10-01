from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    Candidate retrosynthetic reaction proposed by the single-step model.

    `reaction_id` is the stable planner-visible identifier used in prompts,
    e.g. R1, R2, ..., while the chemistry is always executed using the
    underlying structured fields rather than by parsing model-generated text.
    """

    product_smiles: str
    reactant_smiles: tuple[str, ...]
    single_step_score: float

    template_id: str | None = None
    reaction_id: str | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.product_smiles:
            raise ValueError("product_smiles must not be empty.")

        if len(self.reactant_smiles) == 0:
            raise ValueError(
                "reactant_smiles must contain at least one reactant."
            )

        if self.reaction_id is not None and not self.reaction_id:
            raise ValueError("reaction_id must be non-empty when provided.")

    @property
    def num_reactants(self) -> int:
        return len(self.reactant_smiles)

    @property
    def display_id(self) -> str:
        """
        Planner-visible identifier.

        Reaction IDs should normally be assigned before constructing prompts.
        This fallback is useful for diagnostics only.
        """
        return self.reaction_id or "<unassigned-reaction>"


@dataclass(frozen=True, slots=True)
class PlanningState:
    """
    Retrosynthetic planning state.

    The state contains the target and the current set of unsolved molecules.
    Applying a reaction replaces one selected product with its predicted
    reactants.
    """

    target_smiles: str
    unsolved_molecules: tuple[str, ...]

    # Number of reaction decisions already applied.
    depth: int = 0

    # Stable identifier useful for oracle caching and interaction records.
    state_id: str | None = None

    # Tree-specific information should remain in planning/.
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
# Natural-language feedback protocol
# ============================================================


@dataclass(frozen=True, slots=True)
class FeedbackQuery:
    """
    Executable feedback query selected by the escalation policy.

    A query deliberately has two representations:

    1. Planner-visible representation
       - query_id
       - question
       - target_ids
       - feedback_type

       These fields are rendered in the language-model prompt.

    2. Oracle-execution representation
       - payload

       `payload` stores the structured Python-side objects/arguments needed by
       the oracle backend. The oracle must not recover execution targets by
       parsing the natural-language question.

    Example
    -------
    FeedbackQuery(
        level=FeedbackLevel.L2,
        feedback_type="reaction_yield",
        query_id="Q_L2_YIELD_R3",
        question=(
            "What is the predicted normalized yield of candidate reaction R3?"
        ),
        target_ids=("R3",),
        payload={"reaction": reaction_candidate},
    )
    """

    level: FeedbackLevel
    feedback_type: str

    # Stable identifier for the semantic query within the planning context.
    query_id: str

    # Natural-language question shown to the planner/oracle interface.
    question: str

    # Planner-visible IDs of the objects being queried, e.g. ("R3",) or
    # ("R1", "R3"). These are descriptive references, not execution payloads.
    target_ids: tuple[str, ...] = ()

    # Structured backend inputs. Never reconstruct these by parsing `question`.
    payload: Mapping[str, Any] = field(default_factory=dict)

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level == FeedbackLevel.L0:
            raise ValueError(
                "L0 represents no feedback and should not contain a query."
            )

        if not self.feedback_type:
            raise ValueError("feedback_type must not be empty.")

        if not self.query_id:
            raise ValueError("query_id must not be empty.")

        if not self.question or not self.question.strip():
            raise ValueError("question must not be empty.")

        if any(not target_id for target_id in self.target_ids):
            raise ValueError("target_ids must not contain empty identifiers.")

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        """
        Query identity independent of natural-language wording.

        Oracle caches should normally combine this with a state/context ID.
        Query factories may override/extend semantic identity through
        metadata if additional parameters affect the oracle result.
        """
        explicit_key = self.metadata.get("semantic_key")
        if explicit_key is not None:
            if isinstance(explicit_key, tuple):
                return explicit_key
            if isinstance(explicit_key, Sequence) and not isinstance(
                explicit_key, (str, bytes)
            ):
                return tuple(explicit_key)
            return (explicit_key,)

        return (
            int(self.level),
            self.feedback_type,
            self.target_ids,
        )


@dataclass(frozen=True, slots=True)
class Feedback:
    """
    Natural-language answer returned by oracle O(q).

    Each feedback item preserves the complete question-answer association.
    This is important because accumulated feedback is rendered back into the
    planner prompt at the next escalation/reaction decision.

    `answer` is the canonical planner-visible field. `content` is provided as
    a read-only compatibility alias for older code that accessed
    feedback.content.
    """

    level: FeedbackLevel
    feedback_type: str

    query_id: str
    question: str
    answer: str

    target_ids: tuple[str, ...] = ()

    # Acquisition cost C(q). Cached feedback should normally have zero newly
    # incurred cost at the execution layer if that is the repository policy;
    # the `cached` flag records how the answer was obtained.
    cost: float = 0.0
    cached: bool = False

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level == FeedbackLevel.L0:
            raise ValueError("L0 should not produce oracle feedback.")

        if self.cost < 0:
            raise ValueError("Feedback cost must be non-negative.")

        if not self.feedback_type:
            raise ValueError("feedback_type must not be empty.")

        if not self.query_id:
            raise ValueError("query_id must not be empty.")

        if not self.question or not self.question.strip():
            raise ValueError("question must not be empty.")

        if not self.answer or not self.answer.strip():
            raise ValueError("answer must not be empty.")

        if any(not target_id for target_id in self.target_ids):
            raise ValueError("target_ids must not contain empty identifiers.")

    @property
    def content(self) -> str:
        """
        Backward-compatible alias for the natural-language oracle answer.
        """
        return self.answer

    @classmethod
    def from_query(
        cls,
        query: FeedbackQuery,
        *,
        answer: str,
        cost: float,
        cached: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Feedback":
        """
        Construct feedback while preserving the exact selected query.
        """
        merged_metadata: dict[str, Any] = dict(query.metadata)
        if metadata is not None:
            merged_metadata.update(metadata)

        return cls(
            level=query.level,
            feedback_type=query.feedback_type,
            query_id=query.query_id,
            question=query.question,
            answer=answer,
            target_ids=query.target_ids,
            cost=float(cost),
            cached=bool(cached),
            metadata=merged_metadata,
        )

    def as_cached(self, *, cost: float = 0.0) -> "Feedback":
        """
        Return a copy marked as retrieved from cache.

        The default newly incurred cost is zero. If the experimental protocol
        charges for cache retrieval, pass that cost explicitly.
        """
        return replace(self, cached=True, cost=float(cost))


# ============================================================
# Oracle context
# ============================================================


@dataclass(frozen=True, slots=True)
class OracleContext:
    """
    Structured context supplied to query generation and oracle execution.

    The natural-language question is carried by FeedbackQuery. OracleContext
    contains the underlying planning objects required to construct legal
    queries and execute them deterministically.

    `selected_molecule` is the frontier/current molecule being expanded.
    """

    state: PlanningState
    selected_molecule: str
    candidates: tuple[ReactionCandidate, ...]

    current_level: FeedbackLevel = FeedbackLevel.L0
    feedback_history: tuple[Feedback, ...] = ()

    # Optional explicit context identifier. If omitted, state.state_id is used.
    context_id: str | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.selected_molecule:
            raise ValueError("selected_molecule must not be empty.")

        if len(self.candidates) == 0:
            raise ValueError(
                "OracleContext requires at least one reaction candidate."
            )

    @property
    def state_id(self) -> str | None:
        """
        Compatibility/access helper used by planner and cache code.
        """
        return self.state.state_id

    @property
    def effective_context_id(self) -> str | None:
        return self.context_id or self.state.state_id

    @property
    def cumulative_cost(self) -> float:
        return float(
            sum(feedback.cost for feedback in self.feedback_history)
        )

    @property
    def has_feedback(self) -> bool:
        return len(self.feedback_history) > 0

    def to_escalation_context(
        self,
        *,
        available_queries: Sequence[FeedbackQuery] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "EscalationContext":
        """
        Convert the oracle/planning context into the policy-facing context.

        Keeping this conversion here prevents MCTS, rollout collection, and
        training code from constructing subtly different policy inputs.
        """
        merged_metadata: dict[str, Any] = dict(self.metadata)
        if metadata is not None:
            merged_metadata.update(metadata)

        return EscalationContext(
            state=self.state,
            candidates=self.candidates,
            current_level=self.current_level,
            feedback_history=self.feedback_history,
            available_queries=tuple(available_queries),
            selected_molecule=self.selected_molecule,
            context_id=self.effective_context_id,
            metadata=merged_metadata,
        )


# ============================================================
# Escalation policy actions
# ============================================================


@dataclass(frozen=True, slots=True)
class EscalationAction:
    """
    Structured action sampled from escalation policy mu.

    Either:

        STOP

    or:

        ESCALATE(query)

    The language model predicts among serialized legal structured actions;
    execution always uses the attached FeedbackQuery object.
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
    Input context for escalation policy mu:

        mu(. | s_t, f_t^(<=l))

    In the language-interface implementation the policy also receives the
    legal next-level queries. Each query already contains the natural-language
    question and structured execution payload.

    `selected_molecule` identifies the molecule currently being expanded.
    """

    state: PlanningState
    candidates: tuple[ReactionCandidate, ...]

    current_level: FeedbackLevel = FeedbackLevel.L0
    feedback_history: tuple[Feedback, ...] = ()

    # Legal queries available if the policy chooses ESCALATE.
    available_queries: tuple[FeedbackQuery, ...] = ()

    selected_molecule: str | None = None
    context_id: str | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.candidates) == 0:
            raise ValueError(
                "EscalationContext requires at least one reaction candidate."
            )

        for query in self.available_queries:
            expected_level = int(self.current_level) + 1
            if int(query.level) != expected_level:
                raise ValueError(
                    "available_queries must belong to the next feedback "
                    f"level L{expected_level}; received {query.level.name}."
                )

    @property
    def cumulative_cost(self) -> float:
        return float(
            sum(feedback.cost for feedback in self.feedback_history)
        )

    @property
    def has_feedback(self) -> bool:
        return len(self.feedback_history) > 0

    @property
    def effective_context_id(self) -> str | None:
        return self.context_id or self.state.state_id


# ============================================================
# Policy trajectory records
# ============================================================


@dataclass(frozen=True, slots=True)
class EscalationDecision:
    """
    One structured decision made by escalation policy mu.

    Besides the sampled action and log-probability, optional prompt/completion
    fields preserve the exact language-model interaction used for training
    diagnostics and policy-probability recomputation.
    """

    level: FeedbackLevel
    action: EscalationAction
    logprob: float

    probability: float | None = None
    entropy: float | None = None

    # Exact model-facing artifacts when available.
    prompt: str | None = None
    completion: str | None = None
    legal_completions: tuple[str, ...] = ()

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action.is_escalate and self.action.query is not None:
            expected_level = int(self.level) + 1
            if int(self.action.query.level) != expected_level:
                raise ValueError(
                    "An ESCALATE action must query the next feedback level."
                )


@dataclass(frozen=True, slots=True)
class ReactionDecision:
    """
    Reaction action sampled or selected from policy pi.

    `reaction_id` is available through `reaction.reaction_id`; the underlying
    ReactionCandidate remains the executable object.
    """

    reaction: ReactionCandidate
    logprob: float

    probability: float | None = None

    prompt: str | None = None
    completion: str | None = None
    legal_completions: tuple[str, ...] = ()

    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def reaction_id(self) -> str | None:
        return self.reaction.reaction_id


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

    A single rollout-level MAVE advantage can later be assigned to all
    escalation decisions in this rollout.
    """

    context_id: str

    escalation_decisions: tuple[EscalationDecision, ...]
    feedback: tuple[Feedback, ...]

    reaction_decision: ReactionDecision

    cost: float
    downstream_return: float

    success: bool | None = None
    terminal_depth: int | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.context_id:
            raise ValueError("context_id must not be empty.")

        if self.cost < 0:
            raise ValueError("Rollout cost must be non-negative.")

        if self.terminal_depth is not None and self.terminal_depth < 0:
            raise ValueError("terminal_depth must be non-negative.")

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
    K complete escalation rollouts sampled from the same planning context.

    This is the direct input to reward-cost fitting.
    """

    context_id: str
    rollouts: tuple[EscalationRollout, ...]

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.context_id:
            raise ValueError("context_id must not be empty.")

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

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError("cost must be non-negative.")


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
    Generic result returned by the retrosynthetic planner.
    """

    target_smiles: str
    success: bool
    route: Any | None
    single_step_calls: int

    feedback_queries: int = 0
    feedback_cost: float = 0.0

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target_smiles:
            raise ValueError("target_smiles must not be empty.")

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

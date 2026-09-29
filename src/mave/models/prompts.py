from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Mapping,
    Sequence,
)

from mave.chemistry.reaction import (
    Reaction,
)
from mave.core.types import (
    EscalationContext,
    Feedback,
    FeedbackQuery,
)


# ============================================================
# Prompt container
# ============================================================


@dataclass(frozen=True, slots=True)
class ChoicePrompt:
    """
    Structured multiple-choice policy prompt.

    labels[i] corresponds to payloads[i].
    """

    messages: tuple[
        Mapping[str, str],
        ...
    ]

    labels: tuple[str, ...]

    descriptions: tuple[str, ...]

    payloads: tuple[Any, ...]

    def __post_init__(self) -> None:

        lengths = {
            len(self.labels),
            len(self.descriptions),
            len(self.payloads),
        }

        if len(lengths) != 1:
            raise ValueError(
                "labels, descriptions, and payloads "
                "must have the same length."
            )

        if not self.labels:
            raise ValueError(
                "ChoicePrompt requires at least "
                "one choice."
            )


# ============================================================
# Generic formatting
# ============================================================


def choice_label(
    index: int,
) -> str:
    """
    Fixed structured label.

    Deliberately avoids natural-language action generation.
    """
    return f"CHOICE_{index}"


def choice_completion(
    label: str,
) -> str:
    """
    Completion scored by the language model.
    """
    return f"{label}"


def format_feedback(
    feedback: Feedback,
) -> str:

    content = feedback.content

    if isinstance(
        content,
        Mapping,
    ):
        content_string = "; ".join(
            f"{key}={value}"
            for key, value
            in content.items()
        )
    else:
        content_string = str(
            content
        )

    return (
        f"L{int(feedback.level)} "
        f"{feedback.feedback_type}: "
        f"{content_string}"
    )


def format_feedback_history(
    feedback: Sequence[Feedback],
) -> str:

    if not feedback:
        return "None"

    return "\n".join(
        f"- {format_feedback(item)}"
        for item in feedback
    )


def format_reaction(
    reaction: Reaction,
    *,
    index: int | None = None,
) -> str:

    prefix = (
        ""
        if index is None
        else f"[R{index}] "
    )

    score = (
        ""
        if reaction.score is None
        else (
            f" | single-step score="
            f"{reaction.score:.6f}"
        )
    )

    return (
        f"{prefix}"
        f"{reaction.product_smiles}"
        f" -> "
        f"{'.'.join(reaction.reactant_smiles)}"
        f"{score}"
    )


# ============================================================
# Escalation prompt
# ============================================================


def build_escalation_prompt(
    context: EscalationContext,
    available_queries: Sequence[
        FeedbackQuery
    ],
) -> ChoicePrompt:
    """
    Construct a constrained-choice prompt for mu.

    Legal choices:
        STOP
        ESCALATE(q) for q in next-level query set.

    IMPORTANT:
    This exact wording is a reproduction choice because the
    paper does not disclose the policy prompt.
    """

    labels: list[str] = []
    descriptions: list[str] = []
    payloads: list[
        FeedbackQuery | None
    ] = []

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    labels.append(
        choice_label(0)
    )

    descriptions.append(
        "STOP: use the currently accumulated evidence "
        "and do not acquire additional feedback."
    )

    payloads.append(None)

    # --------------------------------------------------------
    # ESCALATE(query)
    # --------------------------------------------------------

    for index, query in enumerate(
        available_queries,
        start=1,
    ):
        label = choice_label(
            index
        )

        labels.append(label)

        descriptions.append(
            f"ESCALATE: acquire "
            f"L{int(query.level)} "
            f"{query.feedback_type} feedback."
        )

        payloads.append(query)

    target = (
        context.state.target_smiles
    )

    unsolved = (
        ", ".join(
            context.state
            .unsolved_molecules
        )
        or "None"
    )

    reaction_candidates = "\n".join(
        (
            f"- R{index}: "
            f"{candidate.product_smiles}"
            f" -> "
            f"{'.'.join(candidate.reactant_smiles)}"
        )
        for index, candidate
        in enumerate(
            context.candidates
        )
    )

    option_text = "\n".join(
        f"{label}: {description}"
        for label, description
        in zip(
            labels,
            descriptions,
            strict=True,
        )
    )

    user_content = f"""
Target molecule:
{target}

Current unsolved molecules:
{unsolved}

Candidate retrosynthetic reactions:
{reaction_candidates}

Current feedback level:
L{int(context.current_level)}

Accumulated feedback:
{format_feedback_history(context.feedback_history)}

Available decisions:
{option_text}

Choose exactly one available decision.
Return only its choice label.
""".strip()

    messages = (
        {
            "role": "system",
            "content": (
                "You are the feedback-escalation policy "
                "for a retrosynthetic planner. "
                "Choose whether the currently accumulated "
                "evidence is sufficient or whether feedback "
                "from the next level should be acquired. "
                "Only choose from the supplied options."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    )

    return ChoicePrompt(
        messages=messages,
        labels=tuple(labels),
        descriptions=tuple(
            descriptions
        ),
        payloads=tuple(payloads),
    )


# ============================================================
# Reaction-selection prompt
# ============================================================


def build_reaction_prompt(
    *,
    target_smiles: str,
    unsolved_molecules: Sequence[str],
    reactions: Sequence[Reaction],
    feedback: Sequence[Feedback],
) -> ChoicePrompt:
    """
    Construct constrained candidate-selection prompt for pi.

    pi never generates a new retrosynthetic reaction here;
    it ranks/selects reactions supplied by the single-step model.

    This is a reproduction implementation choice.
    """

    if not reactions:
        raise ValueError(
            "Reaction prompt requires at least "
            "one candidate reaction."
        )

    labels = tuple(
        choice_label(index)
        for index
        in range(len(reactions))
    )

    descriptions = tuple(
        format_reaction(
            reaction,
            index=index,
        )
        for index, reaction
        in enumerate(reactions)
    )

    options = "\n".join(
        f"{label}: {description}"
        for label, description
        in zip(
            labels,
            descriptions,
            strict=True,
        )
    )

    unsolved = (
        ", ".join(
            unsolved_molecules
        )
        or "None"
    )

    user_content = f"""
Target molecule:
{target_smiles}

Current unsolved molecules:
{unsolved}

Accumulated external feedback:
{format_feedback_history(feedback)}

Candidate reactions:
{options}

Choose exactly one candidate reaction.
Return only its choice label.
""".strip()

    messages = (
        {
            "role": "system",
            "content": (
                "You are the reaction-selection policy "
                "for a retrosynthetic planner. "
                "Select one reaction from the supplied "
                "candidate set using the current planning "
                "state and accumulated feedback. "
                "Do not propose reactions outside the "
                "candidate set."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    )

    return ChoicePrompt(
        messages=messages,
        labels=labels,
        descriptions=descriptions,
        payloads=tuple(reactions),
    )
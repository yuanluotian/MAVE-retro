from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.distributions import Categorical

from mave_repro.core.registry import (
    POLICY_REGISTRY,
)
from mave_repro.core.types import (
    EscalationAction,
    EscalationContext,
    EscalationDecision,
    FeedbackQuery,
)

from mave_repro.models.backbone import (
    LLMBackbone,
)
from mave_repro.models.prompts import (
    ChoicePrompt,
    build_escalation_prompt,
    choice_completion,
)


@dataclass(frozen=True, slots=True)
class EscalationDistribution:
    """
    Policy distribution over legal escalation actions.
    """

    prompt: ChoicePrompt

    logits: Tensor
    probabilities: Tensor

    def __post_init__(self) -> None:
        if self.logits.ndim != 1:
            raise ValueError(
                "Escalation logits must be 1D."
            )

        if (
            self.logits.shape
            != self.probabilities.shape
        ):
            raise ValueError(
                "logits/probabilities shape mismatch."
            )


@POLICY_REGISTRY.register(
    "escalation"
)
class EscalationPolicy:
    """
    Sequential stop-or-escalate policy mu.

    For current level l, legal actions are:

        STOP

    or

        ESCALATE(q), q in Q^(l+1)

    The policy only scores legal actions supplied by the oracle
    layer. It does not construct oracle queries itself.
    """

    def __init__(
        self,
        backbone: LLMBackbone,
        *,
        temperature: float = 1.0,
        normalize_completion_logprob: bool = True,
    ) -> None:

        if temperature <= 0:
            raise ValueError(
                "temperature must be positive."
            )

        self.backbone = backbone

        self.temperature = float(
            temperature
        )

        self.normalize_completion_logprob = (
            normalize_completion_logprob
        )

    # ========================================================
    # Distribution
    # ========================================================

    def distribution(
        self,
        context: EscalationContext,
        available_queries: Sequence[
            FeedbackQuery
        ],
    ) -> EscalationDistribution:

        # Queries must belong to the immediately next level.
        expected_level = (
            int(context.current_level)
            + 1
        )

        for query in available_queries:
            if int(query.level) != expected_level:
                raise ValueError(
                    "EscalationPolicy received a query "
                    "outside the immediately next feedback "
                    "level. "
                    f"Current=L{int(context.current_level)}, "
                    f"query=L{int(query.level)}."
                )

        choice_prompt = (
            build_escalation_prompt(
                context,
                available_queries,
            )
        )

        rendered = (
            self.backbone.render_chat(
                choice_prompt.messages
            )
        )

        completions = tuple(
            choice_completion(label)
            for label
            in choice_prompt.labels
        )

        raw_scores = (
            self.backbone.score_completions(
                rendered,
                completions,
                normalize_by_length=(
                    self
                    .normalize_completion_logprob
                ),
            )
        )

        logits = (
            raw_scores
            / self.temperature
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=-1,
            )
        )

        return EscalationDistribution(
            prompt=choice_prompt,
            logits=logits,
            probabilities=probabilities,
        )

    # ========================================================
    # Sampling
    # ========================================================

    def sample(
        self,
        context: EscalationContext,
        available_queries: Sequence[
            FeedbackQuery
        ],
    ) -> EscalationDecision:

        result = self.distribution(
            context,
            available_queries,
        )

        categorical = Categorical(
            probs=result.probabilities
        )

        index = categorical.sample()

        logprob = categorical.log_prob(
            index
        )

        entropy = (
            categorical.entropy()
        )

        selected_index = int(
            index.item()
        )

        payload = (
            result.prompt
            .payloads[selected_index]
        )

        if payload is None:
            action = (
                EscalationAction.stop()
            )

        else:
            if not isinstance(
                payload,
                FeedbackQuery,
            ):
                raise TypeError(
                    "Escalation prompt payload must be "
                    "FeedbackQuery or None."
                )

            action = (
                EscalationAction.escalate(
                    payload
                )
            )

        return EscalationDecision(
            level=context.current_level,
            action=action,
            logprob=float(
                logprob.detach().cpu()
            ),
            probability=float(
                result
                .probabilities[selected_index]
                .detach()
                .cpu()
            ),
            entropy=float(
                entropy.detach().cpu()
            ),
            metadata={
                "choice_index": (
                    selected_index
                ),
                "choice_label": (
                    result
                    .prompt
                    .labels[selected_index]
                ),
            },
        )

    # ========================================================
    # Greedy decision
    # ========================================================

    def greedy(
        self,
        context: EscalationContext,
        available_queries: Sequence[
            FeedbackQuery
        ],
    ) -> EscalationDecision:

        result = self.distribution(
            context,
            available_queries,
        )

        index = torch.argmax(
            result.logits
        )

        selected_index = int(
            index.item()
        )

        probabilities = (
            result.probabilities
        )

        distribution = Categorical(
            probs=probabilities
        )

        payload = (
            result.prompt
            .payloads[selected_index]
        )

        if payload is None:
            action = (
                EscalationAction.stop()
            )
        else:
            action = (
                EscalationAction.escalate(
                    payload
                )
            )

        return EscalationDecision(
            level=context.current_level,
            action=action,
            logprob=float(
                torch.log(
                    probabilities[
                        selected_index
                    ]
                    + 1e-12
                )
                .detach()
                .cpu()
            ),
            probability=float(
                probabilities[
                    selected_index
                ]
                .detach()
                .cpu()
            ),
            entropy=float(
                distribution
                .entropy()
                .detach()
                .cpu()
            ),
            metadata={
                "choice_index": (
                    selected_index
                ),
                "choice_label": (
                    result.prompt.labels[
                        selected_index
                    ]
                ),
                "greedy": True,
            },
        )

    # ========================================================
    # GRPO support
    # ========================================================

    def action_logprob(
        self,
        context: EscalationContext,
        available_queries: Sequence[
            FeedbackQuery
        ],
        action_index: int,
    ) -> Tensor:
        """
        Recompute differentiable log pi_mu(action | context).

        Used during GRPO optimization.
        """

        result = self.distribution(
            context,
            available_queries,
        )

        if not (
            0
            <= action_index
            < len(result.prompt.labels)
        ):
            raise IndexError(
                "action_index outside policy "
                "distribution."
            )

        return torch.log_softmax(
            result.logits,
            dim=-1,
        )[action_index]
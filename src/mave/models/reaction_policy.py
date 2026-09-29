from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.distributions import Categorical

from mave.chemistry.reaction import (
    Reaction,
)
from mave.core.registry import (
    POLICY_REGISTRY,
)
from mave.core.types import (
    Feedback,
)

from mave.models.backbone import (
    LLMBackbone,
)
from mave.models.prompts import (
    ChoicePrompt,
    build_reaction_prompt,
    choice_completion,
)


@dataclass(frozen=True, slots=True)
class ReactionDistribution:
    """
    pi(a | s, f) over single-step reaction candidates.
    """

    reactions: tuple[
        Reaction,
        ...
    ]

    prompt: ChoicePrompt

    logits: Tensor
    probabilities: Tensor

    def __post_init__(self) -> None:

        n = len(self.reactions)

        if n == 0:
            raise ValueError(
                "ReactionDistribution cannot "
                "be empty."
            )

        if self.logits.shape != (n,):
            raise ValueError(
                "Invalid reaction-logit shape."
            )

        if (
            self.probabilities.shape
            != (n,)
        ):
            raise ValueError(
                "Invalid probability shape."
            )


@dataclass(frozen=True, slots=True)
class ReactionPolicyOutput:
    """
    Selected reaction and policy statistics.
    """

    reaction: Reaction

    index: int

    logprob: float
    probability: float
    entropy: float


@POLICY_REGISTRY.register(
    "reaction"
)
class ReactionPolicy:
    """
    Reaction policy pi.

    The single-step retrosynthesis model defines the action
    space A_t. The LLM policy scores/selects among those
    candidates conditioned on planning state and accumulated
    feedback.

    At MCTS inference, the resulting probabilities can be used
    directly as P(s,a).
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
        *,
        target_smiles: str,
        unsolved_molecules: Sequence[str],
        reactions: Sequence[Reaction],
        feedback: Sequence[Feedback],
    ) -> ReactionDistribution:

        if not reactions:
            raise ValueError(
                "Reaction policy requires at least "
                "one candidate."
            )

        choice_prompt = (
            build_reaction_prompt(
                target_smiles=(
                    target_smiles
                ),
                unsolved_molecules=(
                    unsolved_molecules
                ),
                reactions=reactions,
                feedback=feedback,
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

        return ReactionDistribution(
            reactions=tuple(
                reactions
            ),
            prompt=choice_prompt,
            logits=logits,
            probabilities=probabilities,
        )

    # ========================================================
    # Sampling
    # ========================================================

    def sample(
        self,
        *,
        target_smiles: str,
        unsolved_molecules: Sequence[str],
        reactions: Sequence[Reaction],
        feedback: Sequence[Feedback],
    ) -> ReactionPolicyOutput:

        result = self.distribution(
            target_smiles=(
                target_smiles
            ),
            unsolved_molecules=(
                unsolved_molecules
            ),
            reactions=reactions,
            feedback=feedback,
        )

        categorical = Categorical(
            probs=result.probabilities
        )

        index_tensor = (
            categorical.sample()
        )

        logprob = (
            categorical.log_prob(
                index_tensor
            )
        )

        entropy = (
            categorical.entropy()
        )

        index = int(
            index_tensor.item()
        )

        return ReactionPolicyOutput(
            reaction=(
                result.reactions[index]
            ),
            index=index,
            logprob=float(
                logprob.detach().cpu()
            ),
            probability=float(
                result
                .probabilities[index]
                .detach()
                .cpu()
            ),
            entropy=float(
                entropy.detach().cpu()
            ),
        )

    # ========================================================
    # Greedy selection
    # ========================================================

    def greedy(
        self,
        *,
        target_smiles: str,
        unsolved_molecules: Sequence[str],
        reactions: Sequence[Reaction],
        feedback: Sequence[Feedback],
    ) -> ReactionPolicyOutput:

        result = self.distribution(
            target_smiles=(
                target_smiles
            ),
            unsolved_molecules=(
                unsolved_molecules
            ),
            reactions=reactions,
            feedback=feedback,
        )

        index_tensor = torch.argmax(
            result.logits
        )

        index = int(
            index_tensor.item()
        )

        distribution = Categorical(
            probs=result.probabilities
        )

        probability = (
            result.probabilities[
                index
            ]
        )

        return ReactionPolicyOutput(
            reaction=(
                result.reactions[index]
            ),
            index=index,
            logprob=float(
                torch.log(
                    probability
                    + 1e-12
                )
                .detach()
                .cpu()
            ),
            probability=float(
                probability
                .detach()
                .cpu()
            ),
            entropy=float(
                distribution
                .entropy()
                .detach()
                .cpu()
            ),
        )

    # ========================================================
    # Planner interface
    # ========================================================

    @torch.no_grad()
    def priors(
        self,
        *,
        target_smiles: str,
        unsolved_molecules: Sequence[str],
        reactions: Sequence[Reaction],
        feedback: Sequence[Feedback],
    ) -> tuple[float, ...]:
        """
        Return MCTS priors:

            P(s,a) = pi(a | s,f)
        """

        result = self.distribution(
            target_smiles=(
                target_smiles
            ),
            unsolved_molecules=(
                unsolved_molecules
            ),
            reactions=reactions,
            feedback=feedback,
        )

        return tuple(
            float(value)
            for value in (
                result
                .probabilities
                .detach()
                .cpu()
                .tolist()
            )
        )

    # ========================================================
    # GRPO support
    # ========================================================

    def action_logprob(
        self,
        *,
        target_smiles: str,
        unsolved_molecules: Sequence[str],
        reactions: Sequence[Reaction],
        feedback: Sequence[Feedback],
        action_index: int,
    ) -> Tensor:
        """
        Differentiable log pi(a | s,f).
        """

        result = self.distribution(
            target_smiles=(
                target_smiles
            ),
            unsolved_molecules=(
                unsolved_molecules
            ),
            reactions=reactions,
            feedback=feedback,
        )

        if not (
            0
            <= action_index
            < len(result.reactions)
        ):
            raise IndexError(
                "action_index outside reaction "
                "distribution."
            )

        return torch.log_softmax(
            result.logits,
            dim=-1,
        )[action_index]
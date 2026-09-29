from __future__ import annotations

from typing import Protocol, Sequence

from mave_repro.chemistry.molecule import (
    Molecule,
    ensure_molecule,
)
from mave_repro.chemistry.reaction import Reaction

from mave_repro.environment.purchasable import (
    PurchasableDatabase,
)
from mave_repro.environment.reward import (
    RewardFunction,
)
from mave_repro.environment.state import (
    RetroState,
    StepResult,
)
from mave_repro.environment.transition import (
    apply_reaction,
)


class SingleStepPredictor(Protocol):
    """
    Minimal interface required from the single-step
    retrosynthesis model.

    The concrete implementation will live under models/.
    """

    def predict(
        self,
        product_smiles: str,
        *,
        top_k: int,
    ) -> Sequence[Reaction]:
        ...


class RetroEnvironment:
    """
    Retrosynthetic planning environment.

    Responsibilities
    ----------------
    - initialize target state
    - expose unsolved molecules
    - query single-step retrosynthesis candidates
    - execute deterministic reaction transitions
    - evaluate task reward
    - determine successful terminal states

    It intentionally does NOT implement:
    - MCTS
    - feedback acquisition
    - escalation policy
    - MAVE
    - GRPO
    """

    def __init__(
        self,
        *,
        single_step_model: SingleStepPredictor,
        purchasable_db: PurchasableDatabase,
        reward_fn: RewardFunction,
        top_k: int = 50,
    ) -> None:

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        self.single_step_model = (
            single_step_model
        )

        self.purchasable_db = (
            purchasable_db
        )

        self.reward_fn = reward_fn

        # Paper main setting: top-50 reactions.
        self.top_k = int(top_k)

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(
        self,
        target: str | Molecule,
    ) -> RetroState:
        """
        Create initial state for a target molecule.
        """

        target_molecule = (
            ensure_molecule(
                target
            )
        )

        target_molecule = (
            self.purchasable_db
            .annotate(
                target_molecule
            )
        )

        return RetroState.initial(
            target_molecule
        )

    # --------------------------------------------------------
    # State inspection
    # --------------------------------------------------------

    def is_terminal(
        self,
        state: RetroState,
    ) -> bool:
        return state.is_terminal

    def unsolved_molecules(
        self,
        state: RetroState,
    ) -> tuple[Molecule, ...]:
        return state.unsolved_molecules

    # --------------------------------------------------------
    # Single-step reaction generation
    # --------------------------------------------------------

    def candidates(
        self,
        state: RetroState,
        *,
        selected_index: int,
        top_k: int | None = None,
    ) -> tuple[Reaction, ...]:
        """
        Generate candidate retrosynthetic reactions for one
        unsolved molecule.

        The planner decides which frontier molecule to expand;
        the environment only generates candidates for it.
        """

        if state.is_terminal:
            return ()

        product = state.molecule_at(
            selected_index
        )

        k = (
            self.top_k
            if top_k is None
            else int(top_k)
        )

        if k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        raw_candidates = (
            self.single_step_model.predict(
                product.canonical_smiles,
                top_k=k,
            )
        )

        # ----------------------------------------------------
        # Validate + deduplicate while retaining model order.
        # ----------------------------------------------------

        unique: list[Reaction] = []
        seen: set[
            tuple[
                str,
                tuple[str, ...],
            ]
        ] = set()

        for reaction in raw_candidates:

            if not isinstance(
                reaction,
                Reaction,
            ):
                raise TypeError(
                    "Single-step model must return "
                    "Reaction objects."
                )

            if (
                reaction.product
                != product
            ):
                raise ValueError(
                    "Single-step model returned a reaction "
                    "whose product does not match the "
                    "requested molecule. "
                    f"Requested="
                    f"{product.canonical_smiles!r}; "
                    f"returned="
                    f"{reaction.product.canonical_smiles!r}."
                )

            if reaction.key in seen:
                continue

            seen.add(
                reaction.key
            )

            unique.append(
                reaction
            )

            if len(unique) >= k:
                break

        return tuple(unique)

    # --------------------------------------------------------
    # Environment transition
    # --------------------------------------------------------

    def step(
        self,
        state: RetroState,
        reaction: Reaction,
        *,
        selected_index: int,
    ) -> StepResult:
        """
        Execute one deterministic retrosynthetic action.
        """

        next_state = apply_reaction(
            state,
            reaction,
            selected_index=(
                selected_index
            ),
            purchasable_db=(
                self.purchasable_db
            ),
        )

        terminated = (
            next_state.is_terminal
        )

        reward = (
            self.reward_fn.step_reward(
                state,
                reaction,
                next_state,
                terminated=terminated,
            )
        )

        return StepResult(
            state=state,
            next_state=next_state,
            reaction=reaction,
            selected_index=(
                selected_index
            ),
            reward=float(reward),
            terminated=terminated,
        )

    # --------------------------------------------------------
    # External planning failure
    # --------------------------------------------------------

    def failure_reward(
        self,
        state: RetroState,
    ) -> float:
        """
        Called by the planner when search ends without a route,
        e.g. after exhausting the single-step call budget.
        """

        return float(
            self.reward_fn.failure_reward(
                state
            )
        )
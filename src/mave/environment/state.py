from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from mave.chemistry.molecule import Molecule
from mave.chemistry.reaction import Reaction
from mave.chemistry.route import RouteNode, SynthesisRoute
from mave.core.types import PlanningState
from mave.core.utils import make_context_id


@dataclass(frozen=True, slots=True)
class RetroState:
    """
    Retrosynthetic planning state.

    The paper defines s_t as the current set of unsolved
    molecules. We additionally retain the partial synthesis route
    so that the final route can be reconstructed.

    The MDP-relevant state representation is exposed through
    `unsolved_molecules`.
    """

    target: Molecule
    route: SynthesisRoute
    depth: int = 0

    metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError(
                "RetroState.depth must be non-negative."
            )

        if self.route.target != self.target:
            raise ValueError(
                "Route target must match state target."
            )

    # --------------------------------------------------------
    # MDP state
    # --------------------------------------------------------

    @property
    def unsolved_molecules(
        self,
    ) -> tuple[Molecule, ...]:
        """
        Molecules that still require retrosynthetic expansion.

        Purchasable leaves are not included.
        """
        return self.route.unsolved_leaves

    @property
    def num_unsolved(self) -> int:
        return len(
            self.unsolved_molecules
        )

    @property
    def is_terminal(self) -> bool:
        """
        Successful terminal state:
        all route leaves are purchasable.
        """
        return self.route.is_complete

    @property
    def is_solved(self) -> bool:
        return self.is_terminal

    # --------------------------------------------------------
    # Stable state identity
    # --------------------------------------------------------

    @property
    def state_id(self) -> str:
        """
        Stable identifier based on the MDP-visible state.

        Route history is deliberately excluded because the paper
        defines the planning state through the currently unsolved
        molecules.
        """
        return make_context_id(
            target_smiles=self.target.canonical_smiles,
            unsolved_molecules=[
                molecule.canonical_smiles
                for molecule
                in self.unsolved_molecules
            ],
            depth=self.depth,
        )

    # --------------------------------------------------------
    # Access
    # --------------------------------------------------------

    def molecule_at(
        self,
        index: int,
    ) -> Molecule:
        molecules = (
            self.unsolved_molecules
        )

        if not 0 <= index < len(molecules):
            raise IndexError(
                f"Unsolved molecule index {index} "
                f"is outside [0, {len(molecules)})."
            )

        return molecules[index]

    # --------------------------------------------------------
    # Compatibility with core.types.PlanningState
    # --------------------------------------------------------

    def to_core_state(
        self,
    ) -> PlanningState:
        return PlanningState(
            target_smiles=(
                self.target.canonical_smiles
            ),
            unsolved_molecules=tuple(
                molecule.canonical_smiles
                for molecule
                in self.unsolved_molecules
            ),
            depth=self.depth,
            state_id=self.state_id,
            metadata=dict(self.metadata),
        )

    # --------------------------------------------------------
    # Constructors
    # --------------------------------------------------------

    @classmethod
    def initial(
        cls,
        target: Molecule,
    ) -> "RetroState":
        """
        Construct an initial planning state.

        Purchasability of the target should already have been
        annotated by the environment.
        """
        route = SynthesisRoute(
            root=RouteNode(
                molecule=target,
            )
        )

        return cls(
            target=target,
            route=route,
            depth=0,
        )


@dataclass(frozen=True, slots=True)
class StepResult:
    """
    Result of one retrosynthetic reaction transition.
    """

    state: RetroState
    next_state: RetroState

    reaction: Reaction
    selected_index: int

    reward: float

    terminated: bool

    metadata: Mapping[str, Any] = field(
        default_factory=dict,
    )
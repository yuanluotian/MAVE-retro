from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from mave.chemistry.molecule import Molecule
from mave.chemistry.reaction import Reaction
from mave.chemistry.route import (
    RouteNode,
    SynthesisRoute,
)
from mave.core.types import Feedback
from mave.environment.state import RetroState


# ============================================================
# Node status
# ============================================================


class NodeStatus(str, Enum):
    OPEN = "open"
    SOLVED = "solved"
    DEAD = "dead"


# ============================================================
# Molecule OR node
# ============================================================


@dataclass(slots=True)
class MoleculeNode:
    """
    OR node in the retrosynthetic AND-OR tree.

    A molecule is solved when:
        - it is directly purchasable, or
        - at least one child reaction is solved.

    A non-purchasable expanded molecule is dead when all
    candidate reaction children are dead.
    """

    node_id: str
    molecule: Molecule

    parent_reaction: "ReactionNode | None" = None

    reactions: list[
        "ReactionNode"
    ] = field(
        default_factory=list
    )

    expanded: bool = False

    visit_count: int = 0

    status: NodeStatus = (
        NodeStatus.OPEN
    )

    metadata: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.molecule.is_purchasable:
            self.status = (
                NodeStatus.SOLVED
            )

    @property
    def is_open(self) -> bool:
        return (
            self.status
            == NodeStatus.OPEN
        )

    @property
    def is_solved(self) -> bool:
        return (
            self.status
            == NodeStatus.SOLVED
        )

    @property
    def is_dead(self) -> bool:
        return (
            self.status
            == NodeStatus.DEAD
        )


# ============================================================
# Reaction AND node
# ============================================================


@dataclass(slots=True)
class ReactionNode:
    """
    AND node / reaction edge.

    All precursor molecules must be solved for this reaction to
    form a complete synthesis route.

    Search statistics correspond to the MCTS edge (s, a):

        prior       = P(s,a)
        q_value     = Q(s,a)
        visit_count = N(s,a)
    """

    node_id: str

    reaction: Reaction

    parent_molecule: MoleculeNode

    precursors: list[
        MoleculeNode
    ] = field(
        default_factory=list
    )

    prior: float = 0.0
    q_value: float = 0.0
    visit_count: int = 0

    feedback: tuple[
        Feedback,
        ...
    ] = ()

    status: NodeStatus = (
        NodeStatus.OPEN
    )

    # Used to recover the first route actually solved by search.
    solved_at: int | None = None

    metadata: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    @property
    def is_open(self) -> bool:
        return (
            self.status
            == NodeStatus.OPEN
        )

    @property
    def is_solved(self) -> bool:
        return (
            self.status
            == NodeStatus.SOLVED
        )

    @property
    def is_dead(self) -> bool:
        return (
            self.status
            == NodeStatus.DEAD
        )


# ============================================================
# Selected tree path
# ============================================================


@dataclass(frozen=True, slots=True)
class TreePath:
    """
    Path from root molecule to one frontier molecule.

    For a path:

        M0 -> R0 -> M1 -> R1 -> M2

    molecule_nodes:
        (M0, M1, M2)

    reaction_nodes:
        (R0, R1)
    """

    molecule_nodes: tuple[
        MoleculeNode,
        ...
    ]

    reaction_nodes: tuple[
        ReactionNode,
        ...
    ]

    def __post_init__(self) -> None:
        if not self.molecule_nodes:
            raise ValueError(
                "TreePath must contain at least "
                "the root molecule."
            )

        if (
            len(self.molecule_nodes)
            != len(self.reaction_nodes)
            + 1
        ):
            raise ValueError(
                "TreePath must alternate "
                "molecule/reaction nodes."
            )

    @property
    def frontier(
        self,
    ) -> MoleculeNode:
        return self.molecule_nodes[-1]

    @property
    def depth(self) -> int:
        return len(
            self.reaction_nodes
        )

    def reaction_for_molecule(
        self,
        molecule: MoleculeNode,
    ) -> ReactionNode | None:

        for index, node in enumerate(
            self.molecule_nodes[:-1]
        ):
            if (
                node.node_id
                == molecule.node_id
            ):
                return (
                    self.reaction_nodes[
                        index
                    ]
                )

        return None


# ============================================================
# AND-OR tree
# ============================================================


class AndOrTree:
    """
    Retrosynthetic AND-OR search tree.

    Molecules are OR nodes:
        choose one reaction.

    Reactions are AND nodes:
        every precursor must be solved.
    """

    def __init__(
        self,
        target: Molecule,
    ) -> None:

        self._molecule_counter = 0
        self._reaction_counter = 0
        self._solution_counter = 0

        self.root = MoleculeNode(
            node_id=self._new_molecule_id(),
            molecule=target,
        )

    # ========================================================
    # IDs
    # ========================================================

    def _new_molecule_id(
        self,
    ) -> str:

        identifier = (
            f"M{self._molecule_counter}"
        )

        self._molecule_counter += 1

        return identifier

    def _new_reaction_id(
        self,
    ) -> str:

        identifier = (
            f"R{self._reaction_counter}"
        )

        self._reaction_counter += 1

        return identifier

    # ========================================================
    # Expansion
    # ========================================================

    def expand(
        self,
        molecule_node: MoleculeNode,
        reactions: Sequence[Reaction],
        priors: Sequence[float],
        *,
        feedback: Sequence[
            Feedback
        ] = (),
    ) -> tuple[
        ReactionNode,
        ...
    ]:
        """
        Expand one molecule OR node.

        Paper Eq. (28):

            P(s,a) = pi(a | s,f)
            Q_0(s,a) = P(s,a)
        """

        if molecule_node.expanded:
            raise ValueError(
                "Molecule node has already "
                "been expanded."
            )

        if molecule_node.is_solved:
            raise ValueError(
                "Cannot expand a solved molecule."
            )

        if len(reactions) != len(
            priors
        ):
            raise ValueError(
                "Number of reactions and priors "
                "must match."
            )

        if not reactions:
            molecule_node.expanded = True
            molecule_node.status = (
                NodeStatus.DEAD
            )

            self.refresh_upwards(
                molecule_node
            )

            return ()

        created: list[
            ReactionNode
        ] = []

        for reaction, prior in zip(
            reactions,
            priors,
            strict=True,
        ):

            if (
                reaction.product
                != molecule_node.molecule
            ):
                raise ValueError(
                    "Reaction product does not match "
                    "the molecule being expanded."
                )

            probability = float(
                prior
            )

            if not (
                0.0
                <= probability
                <= 1.0
            ):
                raise ValueError(
                    "Reaction prior must lie "
                    "in [0, 1]."
                )

            reaction_node = (
                ReactionNode(
                    node_id=(
                        self._new_reaction_id()
                    ),
                    reaction=reaction,
                    parent_molecule=(
                        molecule_node
                    ),
                    prior=probability,

                    # Paper Eq. (28).
                    q_value=probability,

                    visit_count=0,

                    feedback=tuple(
                        feedback
                    ),
                )
            )

            precursors: list[
                MoleculeNode
            ] = []

            for reactant in (
                reaction.reactants
            ):

                precursor = MoleculeNode(
                    node_id=(
                        self._new_molecule_id()
                    ),
                    molecule=reactant,
                    parent_reaction=(
                        reaction_node
                    ),
                )

                precursors.append(
                    precursor
                )

            reaction_node.precursors = (
                precursors
            )

            self._refresh_reaction(
                reaction_node
            )

            molecule_node.reactions.append(
                reaction_node
            )

            created.append(
                reaction_node
            )

        molecule_node.expanded = True

        self.refresh_upwards(
            molecule_node
        )

        return tuple(created)

    # ========================================================
    # Status propagation
    # ========================================================

    def _refresh_reaction(
        self,
        node: ReactionNode,
    ) -> None:

        previous = node.status

        # AND:
        # any dead precursor -> reaction dead
        if any(
            child.is_dead
            for child
            in node.precursors
        ):
            node.status = (
                NodeStatus.DEAD
            )

        # all precursors solved -> reaction solved
        elif (
            node.precursors
            and all(
                child.is_solved
                for child
                in node.precursors
            )
        ):
            node.status = (
                NodeStatus.SOLVED
            )

        else:
            node.status = (
                NodeStatus.OPEN
            )

        if (
            previous
            != NodeStatus.SOLVED
            and node.status
            == NodeStatus.SOLVED
        ):
            node.solved_at = (
                self._solution_counter
            )

            self._solution_counter += 1

    def _refresh_molecule(
        self,
        node: MoleculeNode,
    ) -> None:

        if node.molecule.is_purchasable:
            node.status = (
                NodeStatus.SOLVED
            )
            return

        if any(
            reaction.is_solved
            for reaction
            in node.reactions
        ):
            node.status = (
                NodeStatus.SOLVED
            )
            return

        if (
            node.expanded
            and (
                not node.reactions
                or all(
                    reaction.is_dead
                    for reaction
                    in node.reactions
                )
            )
        ):
            node.status = (
                NodeStatus.DEAD
            )
            return

        node.status = (
            NodeStatus.OPEN
        )

    def refresh_upwards(
        self,
        start: MoleculeNode,
    ) -> None:
        """
        Recompute AND/OR statuses from a molecule toward root.
        """

        current_molecule: (
            MoleculeNode | None
        ) = start

        while current_molecule is not None:

            self._refresh_molecule(
                current_molecule
            )

            parent_reaction = (
                current_molecule
                .parent_reaction
            )

            if parent_reaction is None:
                break

            self._refresh_reaction(
                parent_reaction
            )

            current_molecule = (
                parent_reaction
                .parent_molecule
            )

    # ========================================================
    # Route extraction
    # ========================================================

    @staticmethod
    def _solution_reaction(
        node: MoleculeNode,
    ) -> ReactionNode | None:

        solved = [
            reaction
            for reaction
            in node.reactions
            if reaction.is_solved
        ]

        if not solved:
            return None

        # Prefer the reaction that first became solved.
        return min(
            solved,
            key=lambda reaction: (
                float("inf")
                if reaction.solved_at
                is None
                else reaction.solved_at
            ),
        )

    def _build_solved_route_node(
        self,
        node: MoleculeNode,
    ) -> RouteNode:

        if node.molecule.is_purchasable:
            return RouteNode(
                molecule=node.molecule,
                metadata={
                    "tree_node_id":
                        node.node_id,
                },
            )

        reaction_node = (
            self._solution_reaction(
                node
            )
        )

        if reaction_node is None:
            raise RuntimeError(
                "Cannot extract complete route "
                "from an unsolved molecule node."
            )

        children = tuple(
            self._build_solved_route_node(
                precursor
            )
            for precursor
            in reaction_node.precursors
        )

        return RouteNode(
            molecule=node.molecule,
            reaction=(
                reaction_node.reaction
            ),
            children=children,
            metadata={
                "tree_node_id":
                    node.node_id,
            },
        )

    def solved_route(
        self,
    ) -> SynthesisRoute | None:

        if not self.root.is_solved:
            return None

        return SynthesisRoute(
            root=(
                self._build_solved_route_node(
                    self.root
                )
            )
        )

    # ========================================================
    # Partial route for policy/oracle context
    # ========================================================

    def _build_partial_node(
        self,
        node: MoleculeNode,
        preferred: Mapping[
            str,
            str,
        ],
    ) -> RouteNode:
        """
        Build one partial synthesis tree.

        Search-path reactions are preserved. Solved sibling
        branches are also reconstructed; unresolved branches stay
        as leaves.
        """

        metadata = {
            "tree_node_id":
                node.node_id,
        }

        if node.molecule.is_purchasable:
            return RouteNode(
                molecule=node.molecule,
                metadata=metadata,
            )

        selected: (
            ReactionNode | None
        ) = None

        preferred_reaction_id = (
            preferred.get(
                node.node_id
            )
        )

        if (
            preferred_reaction_id
            is not None
        ):
            selected = next(
                (
                    reaction
                    for reaction
                    in node.reactions
                    if reaction.node_id
                    == preferred_reaction_id
                ),
                None,
            )

        elif node.is_solved:
            selected = (
                self._solution_reaction(
                    node
                )
            )

        if selected is None:
            return RouteNode(
                molecule=node.molecule,
                metadata=metadata,
            )

        children = tuple(
            self._build_partial_node(
                precursor,
                preferred,
            )
            for precursor
            in selected.precursors
        )

        return RouteNode(
            molecule=node.molecule,
            reaction=selected.reaction,
            children=children,
            metadata=metadata,
        )

    def state_for_path(
        self,
        path: TreePath,
    ) -> tuple[
        RetroState,
        int,
    ]:
        """
        Convert current selected search path into the
        RetroState/selected_index representation expected by the
        oracle layer.

        The `tree_node_id` marker disambiguates repeated identical
        molecules in different branches.
        """

        preferred = {
            molecule.node_id:
                reaction.node_id
            for molecule, reaction
            in zip(
                path.molecule_nodes[:-1],
                path.reaction_nodes,
                strict=True,
            )
        }

        route = SynthesisRoute(
            root=(
                self._build_partial_node(
                    self.root,
                    preferred,
                )
            )
        )

        state = RetroState(
            target=self.root.molecule,
            route=route,
            depth=path.depth,
        )

        frontier_id = (
            path.frontier.node_id
        )

        selected_index: int | None = (
            None
        )

        unsolved_index = 0

        for leaf in (
            state.route.root.iter_leaves()
        ):

            if leaf.is_purchasable_leaf:
                continue

            if (
                leaf.metadata.get(
                    "tree_node_id"
                )
                == frontier_id
            ):
                selected_index = (
                    unsolved_index
                )
                break

            unsolved_index += 1

        if selected_index is None:
            raise RuntimeError(
                "Selected frontier could not be "
                "recovered from partial route."
            )

        return (
            state,
            selected_index,
        )
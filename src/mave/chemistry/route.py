from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from .molecule import Molecule
from .reaction import Reaction


# ============================================================
# Route node
# ============================================================


@dataclass(frozen=True, slots=True)
class RouteNode:
    """
    One molecule node in a completed or partially completed
    retrosynthetic route.

    If reaction is None, the molecule is currently a leaf.

    If reaction is present, its reactants must match the child
    molecules.
    """

    molecule: Molecule

    reaction: Reaction | None = None

    children: tuple["RouteNode", ...] = ()

    metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.molecule,
            Molecule,
        ):
            raise TypeError(
                "RouteNode.molecule must be a Molecule."
            )

        # Leaf node.
        if self.reaction is None:
            if self.children:
                raise ValueError(
                    "A RouteNode without a reaction "
                    "cannot contain children."
                )

            return

        # Expanded node.
        if (
            self.reaction.product
            != self.molecule
        ):
            raise ValueError(
                "Reaction product must match "
                "RouteNode molecule."
            )

        if len(self.children) != len(
            self.reaction.reactants
        ):
            raise ValueError(
                "Number of RouteNode children must match "
                "number of reaction reactants."
            )

        child_molecules = sorted(
            (
                child.molecule.canonical_smiles
                for child in self.children
            )
        )

        reaction_reactants = sorted(
            (
                reactant.canonical_smiles
                for reactant
                in self.reaction.reactants
            )
        )

        if (
            child_molecules
            != reaction_reactants
        ):
            raise ValueError(
                "RouteNode children do not match "
                "Reaction reactants."
            )

    # --------------------------------------------------------
    # Node type
    # --------------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        return self.reaction is None

    @property
    def is_purchasable_leaf(self) -> bool:
        return (
            self.is_leaf
            and self.molecule.is_purchasable
        )

    @property
    def is_unsolved_leaf(self) -> bool:
        return (
            self.is_leaf
            and not self.molecule.is_purchasable
        )

    # --------------------------------------------------------
    # Tree traversal
    # --------------------------------------------------------

    def iter_nodes(
        self,
    ) -> Iterator["RouteNode"]:
        """
        Depth-first traversal over molecule nodes.
        """
        yield self

        for child in self.children:
            yield from child.iter_nodes()

    def iter_reactions(
        self,
    ) -> Iterator[Reaction]:
        """
        Depth-first traversal over reactions.
        """
        if self.reaction is not None:
            yield self.reaction

        for child in self.children:
            yield from child.iter_reactions()

    def iter_leaves(
        self,
    ) -> Iterator["RouteNode"]:
        if self.is_leaf:
            yield self
            return

        for child in self.children:
            yield from child.iter_leaves()

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    @property
    def depth(self) -> int:
        """
        Maximum number of reaction steps from this molecule to
        any leaf.
        """
        if self.is_leaf:
            return 0

        return (
            1
            + max(
                child.depth
                for child in self.children
            )
        )

    @property
    def num_reactions(self) -> int:
        return sum(
            1
            for _ in self.iter_reactions()
        )

    @property
    def num_molecules(self) -> int:
        return sum(
            1
            for _ in self.iter_nodes()
        )

    # --------------------------------------------------------
    # Root-to-leaf reaction paths
    # --------------------------------------------------------

    def reaction_paths(
        self,
    ) -> tuple[
        tuple[Reaction, ...],
        ...,
    ]:
        """
        Return all root-to-leaf reaction paths.

        This is useful for the route-quality definition:

            S(T) = min_p prod_{r in p} q(r)

        but no scoring logic is implemented here.
        """

        if self.is_leaf:
            return ((),)

        paths: list[
            tuple[Reaction, ...]
        ] = []

        assert self.reaction is not None

        for child in self.children:

            child_paths = (
                child.reaction_paths()
            )

            for child_path in child_paths:
                paths.append(
                    (
                        self.reaction,
                        *child_path,
                    )
                )

        return tuple(paths)

    # --------------------------------------------------------
    # Serialization
    # --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "molecule": self.molecule.to_dict(),
            "reaction": (
                None
                if self.reaction is None
                else self.reaction.to_dict()
            ),
            "children": [
                child.to_dict()
                for child in self.children
            ],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "RouteNode":

        reaction_data = data.get(
            "reaction"
        )

        return cls(
            molecule=Molecule.from_dict(
                data["molecule"]
            ),
            reaction=(
                None
                if reaction_data is None
                else Reaction.from_dict(
                    reaction_data
                )
            ),
            children=tuple(
                cls.from_dict(child)
                for child
                in data.get(
                    "children",
                    [],
                )
            ),
            metadata=data.get(
                "metadata",
                {},
            ),
        )


# ============================================================
# Route
# ============================================================


@dataclass(frozen=True, slots=True)
class SynthesisRoute:
    """
    Final or partial retrosynthetic synthesis tree.

    The root molecule is the target.

    Every internal molecule node is associated with one chosen
    retrosynthetic reaction, and the reaction's reactants become
    child molecule nodes.
    """

    root: RouteNode

    metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    @property
    def target(self) -> Molecule:
        return self.root.molecule

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """
        A route is complete iff every leaf is a purchasable
        building block.
        """
        leaves = tuple(
            self.root.iter_leaves()
        )

        return (
            bool(leaves)
            and all(
                leaf.is_purchasable_leaf
                for leaf in leaves
            )
        )

    @property
    def is_partial(self) -> bool:
        return not self.is_complete

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    @property
    def depth(self) -> int:
        return self.root.depth

    @property
    def num_reactions(self) -> int:
        return (
            self.root.num_reactions
        )

    @property
    def num_molecules(self) -> int:
        return (
            self.root.num_molecules
        )

    @property
    def leaves(
        self,
    ) -> tuple[Molecule, ...]:
        return tuple(
            node.molecule
            for node
            in self.root.iter_leaves()
        )

    @property
    def unsolved_leaves(
        self,
    ) -> tuple[Molecule, ...]:
        return tuple(
            node.molecule
            for node
            in self.root.iter_leaves()
            if node.is_unsolved_leaf
        )

    @property
    def purchasable_leaves(
        self,
    ) -> tuple[Molecule, ...]:
        return tuple(
            node.molecule
            for node
            in self.root.iter_leaves()
            if node.is_purchasable_leaf
        )

    # --------------------------------------------------------
    # Reactions / paths
    # --------------------------------------------------------

    @property
    def reactions(
        self,
    ) -> tuple[Reaction, ...]:
        return tuple(
            self.root.iter_reactions()
        )

    @property
    def reaction_paths(
        self,
    ) -> tuple[
        tuple[Reaction, ...],
        ...,
    ]:
        return (
            self.root.reaction_paths()
        )

    # --------------------------------------------------------
    # Serialization
    # --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "SynthesisRoute":
        return cls(
            root=RouteNode.from_dict(
                data["root"]
            ),
            metadata=data.get(
                "metadata",
                {},
            ),
        )
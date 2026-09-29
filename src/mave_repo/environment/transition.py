from __future__ import annotations

from dataclasses import dataclass

from mave_repro.chemistry.molecule import Molecule
from mave_repro.chemistry.reaction import Reaction
from mave_repro.chemistry.route import (
    RouteNode,
    SynthesisRoute,
)
from mave_repro.environment.purchasable import (
    PurchasableDatabase,
)
from mave_repro.environment.state import (
    RetroState,
)


class TransitionError(RuntimeError):
    """
    Base exception for invalid retrosynthetic transitions.
    """


class InvalidReactionProductError(
    TransitionError
):
    pass


class InvalidSelectedMoleculeError(
    TransitionError
):
    pass


def _annotate_reactant(
    molecule: Molecule,
    purchasable_db: (
        PurchasableDatabase | None
    ),
) -> Molecule:
    """
    Annotate purchasability of a newly introduced reactant.
    """

    if purchasable_db is None:
        return molecule

    return purchasable_db.annotate(
        molecule
    )


def _expand_unsolved_leaf(
    node: RouteNode,
    *,
    target_unsolved_index: int,
    reaction: Reaction,
    purchasable_db: (
        PurchasableDatabase | None
    ),
) -> tuple[
    RouteNode,
    int,
    bool,
]:
    """
    Expand the N-th unsolved leaf in depth-first order.

    Returns
    -------
    node:
        Updated subtree.

    remaining_index:
        Updated index counter.

    expanded:
        Whether the target leaf has already been expanded.
    """

    # --------------------------------------------------------
    # Leaf
    # --------------------------------------------------------

    if node.is_leaf:

        # Purchasable leaves do not belong to the MDP's unsolved
        # molecule list.
        if node.is_purchasable_leaf:
            return (
                node,
                target_unsolved_index,
                False,
            )

        # Not the selected unsolved molecule yet.
        if target_unsolved_index > 0:
            return (
                node,
                target_unsolved_index - 1,
                False,
            )

        # This is the selected frontier molecule.
        if (
            node.molecule
            != reaction.product
        ):
            raise InvalidReactionProductError(
                "Reaction product does not match "
                "the selected unsolved molecule. "
                f"Selected={node.molecule.canonical_smiles!r}, "
                f"reaction product="
                f"{reaction.product.canonical_smiles!r}."
            )

        children = tuple(
            RouteNode(
                molecule=_annotate_reactant(
                    reactant,
                    purchasable_db,
                )
            )
            for reactant
            in reaction.reactants
        )

        expanded_node = RouteNode(
            molecule=node.molecule,
            reaction=reaction,
            children=children,
            metadata=node.metadata,
        )

        return (
            expanded_node,
            0,
            True,
        )

    # --------------------------------------------------------
    # Internal node
    # --------------------------------------------------------

    new_children: list[RouteNode] = []

    remaining_index = (
        target_unsolved_index
    )

    expanded = False

    for child in node.children:

        if expanded:
            new_children.append(
                child
            )
            continue

        (
            updated_child,
            remaining_index,
            child_expanded,
        ) = _expand_unsolved_leaf(
            child,
            target_unsolved_index=(
                remaining_index
            ),
            reaction=reaction,
            purchasable_db=(
                purchasable_db
            ),
        )

        new_children.append(
            updated_child
        )

        if child_expanded:
            expanded = True

    if not expanded:
        return (
            node,
            remaining_index,
            False,
        )

    return (
        RouteNode(
            molecule=node.molecule,
            reaction=node.reaction,
            children=tuple(
                new_children
            ),
            metadata=node.metadata,
        ),
        remaining_index,
        True,
    )


def apply_reaction(
    state: RetroState,
    reaction: Reaction,
    *,
    selected_index: int,
    purchasable_db: (
        PurchasableDatabase | None
    ) = None,
) -> RetroState:
    """
    Apply a retrosynthetic reaction action.

    Following the paper's deterministic transition:

        s_{t+1} = P(s_t, a_t)

    the selected product is replaced by its predicted reactants.

    Parameters
    ----------
    state:
        Current retrosynthetic planning state.

    reaction:
        Chosen retrosynthetic reaction.

    selected_index:
        Index into `state.unsolved_molecules`.

        This is explicit because the same molecule may appear
        multiple times in a synthesis tree.

    purchasable_db:
        Optional database used to annotate newly introduced
        reactants.
    """

    if state.is_terminal:
        raise TransitionError(
            "Cannot apply a reaction to a terminal state."
        )

    unsolved = (
        state.unsolved_molecules
    )

    if not 0 <= selected_index < len(
        unsolved
    ):
        raise InvalidSelectedMoleculeError(
            f"selected_index={selected_index} "
            f"outside [0, {len(unsolved)})."
        )

    selected_molecule = (
        unsolved[selected_index]
    )

    if (
        selected_molecule
        != reaction.product
    ):
        raise InvalidReactionProductError(
            "Reaction product does not match "
            "selected unsolved molecule. "
            f"Selected="
            f"{selected_molecule.canonical_smiles!r}; "
            f"reaction product="
            f"{reaction.product.canonical_smiles!r}."
        )

    new_root, _, expanded = (
        _expand_unsolved_leaf(
            state.route.root,
            target_unsolved_index=(
                selected_index
            ),
            reaction=reaction,
            purchasable_db=(
                purchasable_db
            ),
        )
    )

    if not expanded:
        raise TransitionError(
            "Unable to locate selected unsolved "
            "molecule in synthesis route."
        )

    new_route = SynthesisRoute(
        root=new_root,
        metadata=state.route.metadata,
    )

    return RetroState(
        target=state.target,
        route=new_route,
        depth=state.depth + 1,
        metadata=state.metadata,
    )
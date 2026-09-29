from __future__ import annotations

import math
from dataclasses import dataclass

from mave_repro.planning.and_or_tree import (
    AndOrTree,
    MoleculeNode,
    NodeStatus,
    ReactionNode,
    TreePath,
)


# ============================================================
# PUCT
# ============================================================


def puct_score(
    *,
    q_value: float,
    prior: float,
    parent_visits: int,
    edge_visits: int,
    cpuct: float,
) -> float:
    """
    Paper Eq. (27):

        Q(s,a)
        +
        c_puct P(s,a)
        sqrt(N(s))
        / (1 + N(s,a))
    """

    if cpuct < 0:
        raise ValueError(
            "cpuct must be non-negative."
        )

    if parent_visits < 0:
        raise ValueError(
            "parent_visits must be "
            "non-negative."
        )

    if edge_visits < 0:
        raise ValueError(
            "edge_visits must be "
            "non-negative."
        )

    exploration = (
        float(cpuct)
        * float(prior)
        * math.sqrt(
            float(parent_visits)
        )
        / (
            1.0
            + float(edge_visits)
        )
    )

    return (
        float(q_value)
        + exploration
    )


# ============================================================
# OR-node reaction selection
# ============================================================


def select_reaction(
    node: MoleculeNode,
    *,
    cpuct: float,
) -> ReactionNode:
    """
    Select one non-dead reaction child according to PUCT.
    """

    candidates = [
        reaction
        for reaction
        in node.reactions
        if not reaction.is_dead
    ]

    if not candidates:
        raise RuntimeError(
            "No viable reaction children."
        )

    # N(s). Maintaining a dedicated molecule visit count keeps
    # the implementation aligned with Eq. (27).
    parent_visits = max(
        node.visit_count,
        sum(
            reaction.visit_count
            for reaction
            in candidates
        ),
    )

    scored = [
        (
            puct_score(
                q_value=(
                    reaction.q_value
                ),
                prior=(
                    reaction.prior
                ),
                parent_visits=(
                    parent_visits
                ),
                edge_visits=(
                    reaction
                    .visit_count
                ),
                cpuct=cpuct,
            ),
            reaction.prior,
            -reaction.visit_count,
            reaction,
        )
        for reaction
        in candidates
    ]

    # Deterministic tie-breaking:
    # PUCT -> prior -> fewer visits.
    return max(
        scored,
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        ),
    )[3]


# ============================================================
# AND-node precursor selection
# ============================================================


def select_precursor(
    reaction: ReactionNode,
) -> MoleculeNode:
    """
    Choose one unresolved precursor beneath an AND node.

    IMPORTANT
    ---------
    The paper requires all precursor molecules to be solved but
    does not state how MCTS chooses among multiple unresolved AND
    children.

    Reproduction choice:
        expand the least-visited unresolved precursor first.
    """

    unresolved = [
        precursor
        for precursor
        in reaction.precursors
        if precursor.status
        == NodeStatus.OPEN
    ]

    if not unresolved:
        raise RuntimeError(
            "Reaction has no unresolved "
            "precursor child."
        )

    return min(
        unresolved,
        key=lambda node: (
            node.visit_count,
            node.node_id,
        ),
    )


# ============================================================
# Selection
# ============================================================


def select_frontier(
    tree: AndOrTree,
    *,
    cpuct: float,
) -> TreePath:
    """
    Recursively traverse:

        molecule OR node
          -> PUCT reaction
          -> unresolved AND precursor
          -> ...

    until reaching an unexpanded frontier or terminal node.
    """

    molecules: list[
        MoleculeNode
    ] = [
        tree.root
    ]

    reactions: list[
        ReactionNode
    ] = []

    current = tree.root

    while True:

        # Terminal or dead molecule.
        if (
            current.is_solved
            or current.is_dead
        ):
            break

        # Frontier molecule.
        if not current.expanded:
            break

        viable = [
            reaction
            for reaction
            in current.reactions
            if not reaction.is_dead
        ]

        if not viable:
            current.status = (
                NodeStatus.DEAD
            )

            tree.refresh_upwards(
                current
            )

            break

        reaction = select_reaction(
            current,
            cpuct=cpuct,
        )

        # This normally cannot occur for an OPEN molecule,
        # because a solved reaction would make its molecule
        # solved. Keep the guard for consistency.
        if reaction.is_solved:
            tree.refresh_upwards(
                current
            )
            break

        precursor = (
            select_precursor(
                reaction
            )
        )

        reactions.append(
            reaction
        )

        molecules.append(
            precursor
        )

        current = precursor

    return TreePath(
        molecule_nodes=tuple(
            molecules
        ),
        reaction_nodes=tuple(
            reactions
        ),
    )


# ============================================================
# Backup
# ============================================================


@dataclass(frozen=True, slots=True)
class BackupResult:
    leaf_value: float
    root_value: float


def backup(
    path: TreePath,
    *,
    leaf_value: float,
    gamma: float,
) -> BackupResult:
    """
    Paper Eq. (30):

        N(s,a) <- N(s,a) + 1

        Q(s,a) <- Q(s,a)
                  + [gamma v - Q(s,a)] / N(s,a)

    The paper states that the updated value is propagated to the
    next edge toward the root. We therefore set:

        v <- updated Q(s,a)

    after each edge update.
    """

    if not (
        0.0
        <= gamma
        <= 1.0
    ):
        raise ValueError(
            "gamma must lie in [0, 1]."
        )

    value = float(
        leaf_value
    )

    # Every visited molecule contributes to N(s).
    for molecule in (
        path.molecule_nodes
    ):
        molecule.visit_count += 1

    # Leaf -> root.
    for reaction in reversed(
        path.reaction_nodes
    ):

        reaction.visit_count += 1

        target = (
            float(gamma)
            * value
        )

        reaction.q_value += (
            target
            - reaction.q_value
        ) / reaction.visit_count

        # "The updated value is then propagated to the next
        # edge toward the root."
        value = (
            reaction.q_value
        )

    return BackupResult(
        leaf_value=float(
            leaf_value
        ),
        root_value=value,
    )
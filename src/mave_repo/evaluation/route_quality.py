from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Mapping,
    Protocol,
)

from mave_repro.chemistry.reaction import (
    Reaction,
)
from mave_repro.chemistry.route import (
    RouteNode,
    SynthesisRoute,
)


# ============================================================
# Reaction quality interface
# ============================================================


class ReactionQualityScorer(
    Protocol
):
    """
    Per-reaction surrogate quality score:

        q(r) in [0, 1]

    The paper uses the same surrogate reaction-yield predictor
    that provides yield feedback.
    """

    def __call__(
        self,
        reaction: Reaction,
    ) -> float:
        ...


# ============================================================
# Optimal-route reference interface
# ============================================================


class OptimalRouteScoreProvider(
    Protocol
):
    """
    Provides:

        S(T_i*)

    for a target.

    According to the paper, this value must come from exhaustive
    traversal over the same defined reaction search space.
    """

    def best_score(
        self,
        target_smiles: str,
    ) -> float:
        ...


# ============================================================
# Simple dictionary provider
# ============================================================


class DictOptimalRouteScoreProvider:
    """
    Useful when exhaustive optimal scores are precomputed once
    and stored for evaluation.
    """

    def __init__(
        self,
        scores: Mapping[
            str,
            float,
        ],
    ) -> None:

        self.scores = {
            str(target): float(score)
            for target, score
            in scores.items()
        }

    def best_score(
        self,
        target_smiles: str,
    ) -> float:

        try:
            score = self.scores[
                target_smiles
            ]

        except KeyError as exc:
            raise KeyError(
                "No optimal route score available "
                f"for target {target_smiles!r}."
            ) from exc

        if score <= 0:
            raise ValueError(
                "Optimal route score must be > 0."
            )

        return score


# ============================================================
# Result
# ============================================================


@dataclass(frozen=True, slots=True)
class RouteQualityResult:
    target_smiles: str

    returned_score: float

    optimal_score: float

    normalized_quality: float

    num_reactions: int

    num_paths: int


# ============================================================
# Recursive tree score
# ============================================================


def molecule_quality(
    node: RouteNode,
    *,
    reaction_quality: (
        ReactionQualityScorer
    ),
) -> float:
    """
    Paper Eq. (36).

    For purchasable molecule m:

        Q(m) = 1

    For a molecule synthesized through reaction r:

        Q(m)
        =
        q(r)
        min_{m' in C(r)}
        Q(m')
    """

    # --------------------------------------------------------
    # Purchasable leaf
    # --------------------------------------------------------

    if node.is_leaf:

        if not (
            node.molecule
            .is_purchasable
        ):
            raise ValueError(
                "Route-quality evaluation requires "
                "a complete synthesis route. "
                "Encountered non-purchasable leaf "
                f"{node.molecule.canonical_smiles!r}."
            )

        return 1.0

    if node.reaction is None:
        raise RuntimeError(
            "Internal route node is missing reaction."
        )

    q = float(
        reaction_quality(
            node.reaction
        )
    )

    if not 0.0 <= q <= 1.0:
        raise ValueError(
            "Per-reaction quality score q(r) "
            "must lie in [0, 1]."
        )

    child_scores = [
        molecule_quality(
            child,
            reaction_quality=(
                reaction_quality
            ),
        )
        for child
        in node.children
    ]

    if not child_scores:
        raise RuntimeError(
            "Expanded route node contains no children."
        )

    return float(
        q
        * min(
            child_scores
        )
    )


def synthesis_tree_score(
    route: SynthesisRoute,
    *,
    reaction_quality: (
        ReactionQualityScorer
    ),
) -> float:
    """
    Paper Eq. (37):

        S(T)
        =
        min_{p in P(T)}
        prod_{r in p} q(r)

    The recursive Eq. (36) implementation above is equivalent.
    """

    if not route.is_complete:
        raise ValueError(
            "Route quality can only be computed "
            "for a complete synthesis route."
        )

    score = molecule_quality(
        route.root,
        reaction_quality=(
            reaction_quality
        ),
    )

    if not 0.0 <= score <= 1.0:
        raise RuntimeError(
            "Invalid synthesis-tree score."
        )

    return float(score)


# ============================================================
# Path-form implementation
# ============================================================


def synthesis_tree_score_by_paths(
    route: SynthesisRoute,
    *,
    reaction_quality: (
        ReactionQualityScorer
    ),
) -> float:
    """
    Explicit implementation of Eq. (37).

    Primarily useful as a unit-test cross-check for the recursive
    Eq. (36) implementation.
    """

    if not route.is_complete:
        raise ValueError(
            "Route must be complete."
        )

    paths = (
        route.reaction_paths
    )

    path_scores: list[
        float
    ] = []

    for path in paths:

        score = 1.0

        for reaction in path:

            q = float(
                reaction_quality(
                    reaction
                )
            )

            if not 0.0 <= q <= 1.0:
                raise ValueError(
                    "Reaction score outside [0, 1]."
                )

            score *= q

        path_scores.append(
            score
        )

    if not path_scores:
        # Target itself is already purchasable.
        return 1.0

    return float(
        min(
            path_scores
        )
    )


# ============================================================
# Target-normalized route quality
# ============================================================


def normalized_route_quality(
    *,
    returned_score: float,
    optimal_score: float,
    tolerance: float = 1e-8,
) -> float:
    """
    Paper Eq. (39):

        Q_route,i
        =
        S(T_hat_i)
        / S(T_i*)

    provided S(T_i*) > 0 and both routes belong to the same
    reaction search space.
    """

    returned_score = float(
        returned_score
    )

    optimal_score = float(
        optimal_score
    )

    if returned_score < 0:
        raise ValueError(
            "returned_score must be non-negative."
        )

    if optimal_score <= 0:
        raise ValueError(
            "optimal_score must be > 0."
        )

    quality = (
        returned_score
        / optimal_score
    )

    # Same search-space assumption implies Q_route <= 1.
    if quality > 1.0 + tolerance:
        raise ValueError(
            "Returned route scores higher than the "
            "provided target-specific optimum. "
            "The returned route and optimal reference may "
            "not belong to the same search space."
        )

    return float(
        min(
            quality,
            1.0,
        )
    )


# ============================================================
# Full evaluator
# ============================================================


class RouteQualityEvaluator:
    """
    Evaluate one returned synthesis route against the
    target-specific exhaustive optimum.
    """

    def __init__(
        self,
        *,
        reaction_quality: (
            ReactionQualityScorer
        ),
        optimal_scores: (
            OptimalRouteScoreProvider
        ),
    ) -> None:

        self.reaction_quality = (
            reaction_quality
        )

        self.optimal_scores = (
            optimal_scores
        )

    def evaluate(
        self,
        route: SynthesisRoute,
    ) -> RouteQualityResult:

        target = (
            route.target
            .canonical_smiles
        )

        returned = (
            synthesis_tree_score(
                route,
                reaction_quality=(
                    self.reaction_quality
                ),
            )
        )

        optimum = (
            self.optimal_scores
            .best_score(
                target
            )
        )

        quality = (
            normalized_route_quality(
                returned_score=(
                    returned
                ),
                optimal_score=(
                    optimum
                ),
            )
        )

        return RouteQualityResult(
            target_smiles=target,
            returned_score=(
                returned
            ),
            optimal_score=(
                optimum
            ),
            normalized_quality=(
                quality
            ),
            num_reactions=(
                route.num_reactions
            ),
            num_paths=len(
                route.reaction_paths
            ),
        )
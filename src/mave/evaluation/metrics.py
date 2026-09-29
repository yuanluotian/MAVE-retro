from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import math


# ============================================================
# Per-target metrics
# ============================================================


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """
    Metrics for one retrosynthetic planning target.

    Paper definitions
    -----------------
    success:
        Whether at least one complete retrosynthetic route was
        found.

    route_quality:
        Target-normalized route quality in [0, 1].
        None when no route was returned or when the reference
        optimum is unavailable.

    query_rate:
        Fraction of planning iterations that invoked external
        feedback.

    query_cost:
        Cumulative feedback acquisition cost over the episode.
    """

    target_smiles: str

    success: bool

    route_quality: float | None

    query_rate: float | None

    query_cost: float

    planning_iterations: int | None = None

    queried_iterations: int | None = None

    feedback_queries: int | None = None

    single_step_calls: int | None = None

    def __post_init__(self) -> None:

        if self.route_quality is not None:
            if not (
                0.0
                <= self.route_quality
                <= 1.0 + 1e-8
            ):
                raise ValueError(
                    "route_quality must lie in [0, 1]."
                )

        if self.query_rate is not None:
            if not (
                0.0
                <= self.query_rate
                <= 1.0 + 1e-8
            ):
                raise ValueError(
                    "query_rate must lie in [0, 1]."
                )

        if self.query_cost < 0:
            raise ValueError(
                "query_cost must be non-negative."
            )

        if (
            self.planning_iterations
            is not None
            and self.planning_iterations < 0
        ):
            raise ValueError(
                "planning_iterations must be non-negative."
            )

        if (
            self.queried_iterations
            is not None
            and self.queried_iterations < 0
        ):
            raise ValueError(
                "queried_iterations must be non-negative."
            )

        if (
            self.planning_iterations is not None
            and self.queried_iterations is not None
            and self.queried_iterations
            > self.planning_iterations
        ):
            raise ValueError(
                "queried_iterations cannot exceed "
                "planning_iterations."
            )


# ============================================================
# Aggregate metrics
# ============================================================


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """
    Dataset-level metrics reported in the paper.
    """

    num_targets: int

    num_solved: int

    success_rate: float

    route_quality: float | None

    query_rate: float | None

    query_cost: float

    num_route_quality_samples: int

    num_query_rate_samples: int


# ============================================================
# Basic utilities
# ============================================================


def mean(
    values: Sequence[float],
) -> float:
    if not values:
        raise ValueError(
            "Cannot compute mean of empty sequence."
        )

    return float(
        sum(values)
        / len(values)
    )


def optional_mean(
    values: Iterable[
        float | None
    ],
) -> float | None:

    valid = [
        float(value)
        for value in values
        if value is not None
        and math.isfinite(
            float(value)
        )
    ]

    if not valid:
        return None

    return mean(valid)


# ============================================================
# Individual metrics
# ============================================================


def success_rate(
    episodes: Sequence[
        EpisodeMetrics
    ],
) -> float:
    """
    Fraction of targets for which a route was found.
    """

    if not episodes:
        raise ValueError(
            "No evaluation episodes."
        )

    return mean(
        [
            float(
                episode.success
            )
            for episode
            in episodes
        ]
    )


def episode_query_rate(
    *,
    planning_iterations: int,
    queried_iterations: int,
) -> float:
    """
    Paper definition:

        queried planning iterations
        ---------------------------
        total planning iterations

    IMPORTANT:
    This is NOT:

        number of feedback queries
        --------------------------
        number of planning iterations

    because one planning iteration may acquire multiple feedback
    levels.
    """

    if planning_iterations < 0:
        raise ValueError(
            "planning_iterations must be non-negative."
        )

    if queried_iterations < 0:
        raise ValueError(
            "queried_iterations must be non-negative."
        )

    if (
        queried_iterations
        > planning_iterations
    ):
        raise ValueError(
            "queried_iterations cannot exceed "
            "planning_iterations."
        )

    if planning_iterations == 0:
        return 0.0

    return float(
        queried_iterations
        / planning_iterations
    )


def average_query_rate(
    episodes: Sequence[
        EpisodeMetrics
    ],
) -> float | None:
    """
    Mean of per-target query rates.
    """

    return optional_mean(
        episode.query_rate
        for episode
        in episodes
    )


def average_query_cost(
    episodes: Sequence[
        EpisodeMetrics
    ],
) -> float:
    """
    Mean cumulative acquisition cost per target.
    """

    if not episodes:
        raise ValueError(
            "No evaluation episodes."
        )

    return mean(
        [
            episode.query_cost
            for episode
            in episodes
        ]
    )


def average_route_quality(
    episodes: Sequence[
        EpisodeMetrics
    ],
) -> float | None:
    """
    Mean over episodes for which route-quality values exist.

    The paper defines per-route quality but does not explicitly
    state how failed targets are included in the route-quality
    average. We therefore keep failures as None rather than
    silently assigning zero.
    """

    return optional_mean(
        episode.route_quality
        for episode
        in episodes
    )


# ============================================================
# Full aggregation
# ============================================================


def aggregate_metrics(
    episodes: Sequence[
        EpisodeMetrics
    ],
) -> AggregateMetrics:

    if not episodes:
        raise ValueError(
            "No evaluation episodes."
        )

    route_quality_values = [
        episode.route_quality
        for episode
        in episodes
        if episode.route_quality
        is not None
    ]

    query_rate_values = [
        episode.query_rate
        for episode
        in episodes
        if episode.query_rate
        is not None
    ]

    solved = sum(
        int(
            episode.success
        )
        for episode
        in episodes
    )

    return AggregateMetrics(
        num_targets=len(
            episodes
        ),

        num_solved=solved,

        success_rate=(
            success_rate(
                episodes
            )
        ),

        route_quality=(
            average_route_quality(
                episodes
            )
        ),

        query_rate=(
            average_query_rate(
                episodes
            )
        ),

        query_cost=(
            average_query_cost(
                episodes
            )
        ),

        num_route_quality_samples=(
            len(
                route_quality_values
            )
        ),

        num_query_rate_samples=(
            len(
                query_rate_values
            )
        ),
    )
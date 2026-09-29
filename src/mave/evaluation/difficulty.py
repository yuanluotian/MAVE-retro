from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import (
    Mapping,
    Sequence,
)


# ============================================================
# Labels
# ============================================================


class Difficulty(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


# ============================================================
# Reference-planner result
# ============================================================


@dataclass(frozen=True, slots=True)
class DifficultyRecord:
    """
    Difficulty label determined independently using Retro*+.

    first_solution_calls:
        Number of single-step model calls required by Retro*+ to
        identify the first complete retrosynthetic route.

        None means that the target was not solved before the
        reference search reached the Hard threshold / limit.
    """

    target_smiles: str

    first_solution_calls: (
        int | None
    )

    difficulty: Difficulty


# ============================================================
# Paper thresholds
# ============================================================


EASY_THRESHOLD = 100
HARD_THRESHOLD = 500


def classify_difficulty(
    first_solution_calls: (
        int | None
    ),
) -> Difficulty:
    """
    Paper Eq. (35):

        Easy:
            n_i < 100

        Medium:
            100 <= n_i < 500

        Hard:
            n_i >= 500

    Targets not solved before the Hard-category threshold are
    also assigned Hard.
    """

    if first_solution_calls is None:
        return Difficulty.HARD

    n = int(
        first_solution_calls
    )

    if n < 0:
        raise ValueError(
            "first_solution_calls must "
            "be non-negative."
        )

    if n < EASY_THRESHOLD:
        return Difficulty.EASY

    if n < HARD_THRESHOLD:
        return Difficulty.MEDIUM

    return Difficulty.HARD


# ============================================================
# Build records
# ============================================================


def make_difficulty_record(
    *,
    target_smiles: str,
    first_solution_calls: (
        int | None
    ),
) -> DifficultyRecord:

    return DifficultyRecord(
        target_smiles=(
            target_smiles
        ),
        first_solution_calls=(
            first_solution_calls
        ),
        difficulty=(
            classify_difficulty(
                first_solution_calls
            )
        ),
    )


def build_difficulty_records(
    reference_results: Mapping[
        str,
        int | None,
    ],
) -> tuple[
    DifficultyRecord,
    ...
]:
    """
    Convert target -> Retro*+ first-solution-call count into
    fixed difficulty labels.
    """

    return tuple(
        make_difficulty_record(
            target_smiles=target,
            first_solution_calls=calls,
        )
        for target, calls
        in reference_results.items()
    )


def difficulty_lookup(
    records: Sequence[
        DifficultyRecord
    ],
) -> dict[
    str,
    Difficulty,
]:
    return {
        record.target_smiles:
            record.difficulty
        for record in records
    }


# ============================================================
# Stopping-depth statistics
# ============================================================


def average_stopping_depth(
    levels: Sequence[int],
) -> float | None:
    """
    Average stopping level used for analyses such as Fig. 4.
    """

    if not levels:
        return None

    for level in levels:
        if not 0 <= int(level) <= 4:
            raise ValueError(
                "Feedback stopping level must "
                "lie in [0, 4]."
            )

    return float(
        sum(
            int(level)
            for level
            in levels
        )
        / len(levels)
    )


def stopping_level_distribution(
    levels: Sequence[int],
    *,
    num_levels: int = 5,
) -> tuple[
    float,
    ...,
]:
    """
    Percentage/proportion of planning instances stopping at each
    feedback level.

    Returns probabilities summing to 1.
    """

    if num_levels <= 0:
        raise ValueError(
            "num_levels must be positive."
        )

    counts = [
        0
        for _ in range(
            num_levels
        )
    ]

    for value in levels:

        level = int(value)

        if not (
            0
            <= level
            < num_levels
        ):
            raise ValueError(
                f"Invalid stopping level {level}."
            )

        counts[level] += 1

    if not levels:
        return tuple(
            0.0
            for _ in range(
                num_levels
            )
        )

    total = float(
        len(levels)
    )

    return tuple(
        count / total
        for count
        in counts
    )
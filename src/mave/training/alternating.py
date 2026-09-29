from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import (
    Callable,
    Sequence,
)

from mave.training.collector import (
    CollectedEscalationGroup,
    ReactionTrainingGroup,
)

from mave.training.train_escalation import (
    EscalationTrainer,
    EscalationTrainStats,
)

from mave.training.train_reaction import (
    ReactionTrainer,
    ReactionTrainStats,
)


# ============================================================
# Phase
# ============================================================


class TrainingPhase(str, Enum):
    ESCALATION = "escalation"
    REACTION = "reaction"


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True, slots=True)
class AlternatingTrainConfig:
    """
    Paper Table 7:
        alternating_interval = 1
        epochs = 10000
        batch_size = 6
    """

    epochs: int = 10000

    alternating_interval: int = 1

    batch_size: int = 6

    start_phase: TrainingPhase = (
        TrainingPhase.ESCALATION
    )


# ============================================================
# Epoch record
# ============================================================


@dataclass(frozen=True, slots=True)
class AlternatingEpochRecord:
    epoch: int

    phase: TrainingPhase

    escalation_stats: (
        EscalationTrainStats
        | None
    ) = None

    reaction_stats: (
        ReactionTrainStats
        | None
    ) = None


# ============================================================
# Batch-provider signatures
# ============================================================


EscalationBatchProvider = Callable[
    [int, int],
    Sequence[
        CollectedEscalationGroup
    ],
]

ReactionBatchProvider = Callable[
    [int, int],
    Sequence[
        ReactionTrainingGroup
    ],
]


# ============================================================
# Alternating trainer
# ============================================================


class AlternatingTrainer:
    """
    Alternate optimization between mu and pi.

    Batch collection itself remains outside this coordinator:
    this class controls which policy is updated and when.
    """

    def __init__(
        self,
        *,
        escalation_trainer: (
            EscalationTrainer
        ),
        reaction_trainer: (
            ReactionTrainer
        ),
        escalation_batch_provider: (
            EscalationBatchProvider
        ),
        reaction_batch_provider: (
            ReactionBatchProvider
        ),
        config: (
            AlternatingTrainConfig
        ) = AlternatingTrainConfig(),
    ) -> None:

        if (
            config.alternating_interval
            <= 0
        ):
            raise ValueError(
                "alternating_interval must "
                "be positive."
            )

        if config.epochs <= 0:
            raise ValueError(
                "epochs must be positive."
            )

        if config.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        self.escalation_trainer = (
            escalation_trainer
        )

        self.reaction_trainer = (
            reaction_trainer
        )

        self.escalation_batch_provider = (
            escalation_batch_provider
        )

        self.reaction_batch_provider = (
            reaction_batch_provider
        )

        self.config = config

    # ========================================================
    # Phase selection
    # ========================================================

    def phase_for_epoch(
        self,
        epoch: int,
    ) -> TrainingPhase:

        block = (
            epoch
            // self.config
            .alternating_interval
        )

        even_block = (
            block % 2 == 0
        )

        if (
            self.config.start_phase
            == TrainingPhase.ESCALATION
        ):
            return (
                TrainingPhase.ESCALATION
                if even_block
                else TrainingPhase.REACTION
            )

        return (
            TrainingPhase.REACTION
            if even_block
            else TrainingPhase.ESCALATION
        )

    # ========================================================
    # Run one epoch
    # ========================================================

    def train_epoch(
        self,
        epoch: int,
    ) -> AlternatingEpochRecord:

        phase = self.phase_for_epoch(
            epoch
        )

        if (
            phase
            == TrainingPhase.ESCALATION
        ):

            batch = (
                self
                .escalation_batch_provider(
                    epoch,
                    self.config.batch_size,
                )
            )

            stats = (
                self.escalation_trainer
                .train_batch(
                    batch
                )
            )

            return (
                AlternatingEpochRecord(
                    epoch=epoch,
                    phase=phase,
                    escalation_stats=(
                        stats
                    ),
                )
            )

        batch = (
            self
            .reaction_batch_provider(
                epoch,
                self.config.batch_size,
            )
        )

        stats = (
            self.reaction_trainer
            .train_batch(
                batch
            )
        )

        return AlternatingEpochRecord(
            epoch=epoch,
            phase=phase,
            reaction_stats=stats,
        )

    # ========================================================
    # Full training
    # ========================================================

    def train(
        self,
        *,
        callback: (
            Callable[
                [
                    AlternatingEpochRecord
                ],
                None,
            ]
            | None
        ) = None,
    ) -> tuple[
        AlternatingEpochRecord,
        ...
    ]:

        history: list[
            AlternatingEpochRecord
        ] = []

        for epoch in range(
            self.config.epochs
        ):

            record = (
                self.train_epoch(
                    epoch
                )
            )

            history.append(
                record
            )

            if callback is not None:
                callback(
                    record
                )

        return tuple(history)
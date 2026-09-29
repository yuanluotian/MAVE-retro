from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
    field,
)
from typing import (
    Any,
    Mapping,
    Protocol,
    Sequence,
)

from mave_repro.core.types import (
    PlannerResult,
)

from mave_repro.evaluation.difficulty import (
    Difficulty,
)

from mave_repro.evaluation.metrics import (
    AggregateMetrics,
    EpisodeMetrics,
    aggregate_metrics,
    episode_query_rate,
)

from mave_repro.evaluation.route_quality import (
    RouteQualityEvaluator,
    RouteQualityResult,
)


# ============================================================
# Planner protocol
# ============================================================


class EvaluationPlanner(
    Protocol
):
    def plan(
        self,
        target_smiles: str,
    ) -> PlannerResult:
        ...


# ============================================================
# Target-level record
# ============================================================


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    target_smiles: str

    planner_result: PlannerResult

    metrics: EpisodeMetrics

    route_quality_details: (
        RouteQualityResult | None
    )

    difficulty: (
        Difficulty | None
    )

    stopping_levels: tuple[
        int,
        ...
    ] = ()

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )


# ============================================================
# Difficulty-group result
# ============================================================


@dataclass(frozen=True, slots=True)
class DifficultyMetrics:
    difficulty: Difficulty

    metrics: AggregateMetrics

    num_targets: int

    mean_stopping_depth: (
        float | None
    )


# ============================================================
# Full report
# ============================================================


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    benchmark: str

    overall: AggregateMetrics

    records: tuple[
        EvaluationRecord,
        ...
    ]

    by_difficulty: Mapping[
        Difficulty,
        DifficultyMetrics,
    ]

    warnings: tuple[
        str,
        ...
    ] = ()


# ============================================================
# Evaluator
# ============================================================


class Evaluator:
    """
    Dataset-level retrosynthetic-planning evaluator.

    Expected PlannerResult.metadata entries
    ---------------------------------------
    For exact paper query-rate computation:

        planning_iterations: int
        queried_iterations: int

    Optional escalation analysis:

        stopping_levels: list[int]

    IMPORTANT
    ---------
    feedback_queries is NOT sufficient to reconstruct query rate
    because one queried planning iteration may acquire multiple
    hierarchy levels.
    """

    def __init__(
        self,
        *,
        planner: EvaluationPlanner,

        route_quality: (
            RouteQualityEvaluator
            | None
        ) = None,

        difficulty_labels: (
            Mapping[
                str,
                Difficulty,
            ]
            | None
        ) = None,

        benchmark: str = "unknown",
    ) -> None:

        self.planner = planner

        self.route_quality = (
            route_quality
        )

        self.difficulty_labels = dict(
            difficulty_labels
            or {}
        )

        self.benchmark = benchmark

    # ========================================================
    # Evaluate one target
    # ========================================================

    def evaluate_target(
        self,
        target_smiles: str,
    ) -> tuple[
        EvaluationRecord,
        tuple[str, ...],
    ]:

        result = self.planner.plan(
            target_smiles
        )

        warnings: list[str] = []

        # ----------------------------------------------------
        # Route quality
        # ----------------------------------------------------

        route_quality_details: (
            RouteQualityResult | None
        ) = None

        route_quality_value: (
            float | None
        ) = None

        if result.success:

            if result.route is None:
                raise RuntimeError(
                    "Planner reports success but "
                    "returned no synthesis route."
                )

            if (
                self.route_quality
                is not None
            ):
                route_quality_details = (
                    self.route_quality
                    .evaluate(
                        result.route
                    )
                )

                route_quality_value = (
                    route_quality_details
                    .normalized_quality
                )

        # ----------------------------------------------------
        # Query rate
        # ----------------------------------------------------

        planning_iterations = (
            result.metadata.get(
                "planning_iterations"
            )
        )

        queried_iterations = (
            result.metadata.get(
                "queried_iterations"
            )
        )

        query_rate: (
            float | None
        )

        if (
            planning_iterations
            is None
            or queried_iterations
            is None
        ):
            query_rate = None

            warnings.append(
                f"{target_smiles}: exact query rate "
                "cannot be computed because planner "
                "metadata does not contain both "
                "'planning_iterations' and "
                "'queried_iterations'."
            )

        else:
            planning_iterations = int(
                planning_iterations
            )

            queried_iterations = int(
                queried_iterations
            )

            query_rate = (
                episode_query_rate(
                    planning_iterations=(
                        planning_iterations
                    ),
                    queried_iterations=(
                        queried_iterations
                    ),
                )
            )

        # ----------------------------------------------------
        # Stopping levels
        # ----------------------------------------------------

        raw_stopping_levels = (
            result.metadata.get(
                "stopping_levels",
                (),
            )
        )

        stopping_levels = tuple(
            int(value)
            for value
            in raw_stopping_levels
        )

        # ----------------------------------------------------
        # Difficulty
        # ----------------------------------------------------

        difficulty = (
            self.difficulty_labels
            .get(
                target_smiles
            )
        )

        # ----------------------------------------------------
        # Episode metrics
        # ----------------------------------------------------

        metrics = EpisodeMetrics(
            target_smiles=(
                target_smiles
            ),

            success=(
                result.success
            ),

            route_quality=(
                route_quality_value
            ),

            query_rate=(
                query_rate
            ),

            query_cost=float(
                result.feedback_cost
            ),

            planning_iterations=(
                planning_iterations
            ),

            queried_iterations=(
                queried_iterations
            ),

            feedback_queries=(
                result.feedback_queries
            ),

            single_step_calls=(
                result.single_step_calls
            ),
        )

        record = EvaluationRecord(
            target_smiles=(
                target_smiles
            ),

            planner_result=result,

            metrics=metrics,

            route_quality_details=(
                route_quality_details
            ),

            difficulty=difficulty,

            stopping_levels=(
                stopping_levels
            ),
        )

        return (
            record,
            tuple(warnings),
        )

    # ========================================================
    # Dataset evaluation
    # ========================================================

    def evaluate(
        self,
        targets: Sequence[str],
    ) -> EvaluationReport:

        if not targets:
            raise ValueError(
                "Target set is empty."
            )

        records: list[
            EvaluationRecord
        ] = []

        warnings: list[
            str
        ] = []

        for target in targets:

            record, target_warnings = (
                self.evaluate_target(
                    target
                )
            )

            records.append(
                record
            )

            warnings.extend(
                target_warnings
            )

        overall = aggregate_metrics(
            [
                record.metrics
                for record
                in records
            ]
        )

        by_difficulty = (
            self._aggregate_by_difficulty(
                records
            )
        )

        return EvaluationReport(
            benchmark=(
                self.benchmark
            ),
            overall=overall,
            records=tuple(
                records
            ),
            by_difficulty=(
                by_difficulty
            ),
            warnings=tuple(
                warnings
            ),
        )

    # ========================================================
    # Difficulty aggregation
    # ========================================================

    @staticmethod
    def _aggregate_by_difficulty(
        records: Sequence[
            EvaluationRecord
        ],
    ) -> dict[
        Difficulty,
        DifficultyMetrics,
    ]:

        output: dict[
            Difficulty,
            DifficultyMetrics,
        ] = {}

        for difficulty in Difficulty:

            subset = [
                record
                for record
                in records
                if record.difficulty
                == difficulty
            ]

            if not subset:
                continue

            metrics = (
                aggregate_metrics(
                    [
                        record.metrics
                        for record
                        in subset
                    ]
                )
            )

            stopping_levels = [
                level
                for record
                in subset
                for level
                in record
                .stopping_levels
            ]

            if stopping_levels:
                mean_depth = float(
                    sum(
                        stopping_levels
                    )
                    / len(
                        stopping_levels
                    )
                )
            else:
                mean_depth = None

            output[
                difficulty
            ] = DifficultyMetrics(
                difficulty=(
                    difficulty
                ),
                metrics=metrics,
                num_targets=len(
                    subset
                ),
                mean_stopping_depth=(
                    mean_depth
                ),
            )

        return output


# ============================================================
# Serialization helpers
# ============================================================


def report_to_dict(
    report: EvaluationReport,
) -> dict[str, Any]:
    """
    Convert evaluation report into a JSON-friendly dictionary.
    """

    return {
        "benchmark":
            report.benchmark,

        "overall":
            asdict(
                report.overall
            ),

        "by_difficulty": {
            difficulty.value: {
                "metrics":
                    asdict(
                        value.metrics
                    ),

                "num_targets":
                    value.num_targets,

                "mean_stopping_depth":
                    value.mean_stopping_depth,
            }
            for difficulty, value
            in report.by_difficulty.items()
        },

        "records": [
            {
                "target_smiles":
                    record.target_smiles,

                "difficulty": (
                    None
                    if record.difficulty
                    is None
                    else record
                    .difficulty
                    .value
                ),

                "metrics":
                    asdict(
                        record.metrics
                    ),

                "stopping_levels":
                    list(
                        record
                        .stopping_levels
                    ),

                "route_quality": (
                    None
                    if record
                    .route_quality_details
                    is None
                    else asdict(
                        record
                        .route_quality_details
                    )
                ),
            }
            for record
            in report.records
        ],

        "warnings":
            list(
                report.warnings
            ),
    }
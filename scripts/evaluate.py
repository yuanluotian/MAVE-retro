from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
from typing import Any, Callable

from omegaconf import OmegaConf

from mave_repro.core.utils import (
    set_seed,
)

from mave_repro.evaluation.evaluator import (
    Evaluator,
    report_to_dict,
)


# ============================================================
# Configuration
# ============================================================


def load_yaml(
    path: Path,
) -> dict[str, Any]:

    config = OmegaConf.load(
        path
    )

    return OmegaConf.to_container(
        config,
        resolve=True,
    )


def build_config_bundle(
    *,
    config_dir: Path,
    benchmark: str,
    oracle_name: str,
    budget: int | None = None,
) -> dict[str, Any]:

    eval_config = load_yaml(
        config_dir
        / "eval"
        / f"{benchmark}.yaml"
    )

    planner_config = load_yaml(
        config_dir
        / "planner"
        / "mcts.yaml"
    )

    if budget is not None:
        planner_config[
            "search"
        ][
            "max_single_step_calls"
        ] = int(budget)

        eval_config[
            "search"
        ][
            "max_single_step_calls"
        ] = int(budget)

    return {
        "base":
            load_yaml(
                config_dir
                / "base.yaml"
            ),

        "model":
            load_yaml(
                config_dir
                / "model"
                / "llama3_8b.yaml"
            ),

        "planner":
            planner_config,

        "oracle":
            load_yaml(
                config_dir
                / "oracle"
                / f"{oracle_name}.yaml"
            ),

        "eval":
            eval_config,
    }


# ============================================================
# Runtime hook
# ============================================================


def load_symbol(
    spec: str,
) -> Callable[..., Any]:

    if ":" not in spec:
        raise ValueError(
            "Expected module:function."
        )

    module_name, symbol_name = (
        spec.split(
            ":",
            1,
        )
    )

    module = (
        importlib.import_module(
            module_name
        )
    )

    symbol = getattr(
        module,
        symbol_name,
    )

    if not callable(symbol):
        raise TypeError(
            f"{spec!r} is not callable."
        )

    return symbol


# ============================================================
# Targets
# ============================================================


def load_targets(
    path: Path,
) -> list[str]:

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        if (
            reader.fieldnames is None
            or "target_smiles"
            not in reader.fieldnames
        ):
            raise ValueError(
                f"{path} must contain "
                "'target_smiles'."
            )

        return [
            row[
                "target_smiles"
            ].strip()
            for row in reader
            if row.get(
                "target_smiles"
            )
        ]


# ============================================================
# Evaluation
# ============================================================


def evaluate_once(
    *,
    config: dict[str, Any],
    runtime_factory: Callable[
        [dict[str, Any]],
        dict[str, Any],
    ],
    targets: list[str],
    method_name: str,
) -> dict[str, Any]:

    runtime = (
        runtime_factory(
            config
        )
    )

    if "planner" not in runtime:
        raise KeyError(
            "Evaluation runtime factory "
            "must return 'planner'."
        )

    eval_cfg = (
        config["eval"]
    )

    benchmark_name = (
        eval_cfg[
            "benchmark"
        ]["name"]
    )

    evaluator = Evaluator(
        planner=runtime[
            "planner"
        ],

        route_quality=(
            runtime.get(
                "route_quality_evaluator"
            )
        ),

        difficulty_labels=(
            runtime.get(
                "difficulty_labels"
            )
        ),

        benchmark=(
            benchmark_name
        ),
    )

    report = evaluator.evaluate(
        targets
    )

    payload = report_to_dict(
        report
    )

    payload[
        "method"
    ] = method_name

    payload[
        "feedback_setting"
    ] = (
        config[
            "oracle"
        ][
            "oracle"
        ][
            "setting"
        ]
    )

    payload[
        "single_step_budget"
    ] = (
        config[
            "planner"
        ][
            "search"
        ][
            "max_single_step_calls"
        ]
    )

    return payload


# ============================================================
# Save
# ============================================================


def save_report(
    payload: dict[str, Any],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Main
# ============================================================


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(
            "configs"
        ),
    )

    parser.add_argument(
        "--benchmark",
        choices=[
            "uspto190",
            "pdbbind160",
        ],
        required=True,
    )

    parser.add_argument(
        "--oracle",
        choices=[
            "hierarchy",
            "yield_only",
        ],
        default="hierarchy",
    )

    parser.add_argument(
        "--runtime-factory",
        required=True,
    )

    parser.add_argument(
        "--method",
        default="MAVE",
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    config = build_config_bundle(
        config_dir=args.config_dir,
        benchmark=args.benchmark,
        oracle_name=args.oracle,
        budget=args.budget,
    )

    set_seed(
        int(
            config["base"].get(
                "seed",
                42,
            )
        )
    )

    if args.targets is None:

        target_path = Path(
            config[
                "eval"
            ][
                "benchmark"
            ][
                "target_file"
            ]
        )

    else:
        target_path = (
            args.targets
        )

    targets = load_targets(
        target_path
    )

    factory = load_symbol(
        args.runtime_factory
    )

    payload = evaluate_once(
        config=config,
        runtime_factory=factory,
        targets=targets,
        method_name=(
            args.method
        ),
    )

    save_report(
        payload,
        args.output,
    )

    overall = payload[
        "overall"
    ]

    print(
        f"Benchmark: "
        f"{payload['benchmark']}"
    )

    print(
        f"Success rate: "
        f"{overall['success_rate']:.4f}"
    )

    print(
        f"Route quality: "
        f"{overall['route_quality']}"
    )

    print(
        f"Query rate: "
        f"{overall['query_rate']}"
    )

    print(
        f"Query cost: "
        f"{overall['query_cost']:.4f}"
    )

    if payload.get(
        "warnings"
    ):
        print(
            "\nWarnings:"
        )

        for warning in (
            payload["warnings"]
        ):
            print(
                f"  - {warning}"
            )


if __name__ == "__main__":
    main()
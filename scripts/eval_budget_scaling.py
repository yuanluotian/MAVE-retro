from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import (
    build_config_bundle,
    evaluate_once,
    load_targets,
    save_report,
)

from mave.core.utils import (
    set_seed,
)


DEFAULT_BUDGETS = (
    100,
    200,
    500,
    1000,
)


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
        "--provider-factory",
        default=None,
        help=(
            "Optional provider builder in "
            "'module:function' format. "
            "Each budget is assembled through "
            "prepare_system()."
        ),
    )

    parser.add_argument(
        "--method",
        default="MAVE",
    )

    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(
            DEFAULT_BUDGETS
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    all_results = []

    for budget in args.budgets:

        config = build_config_bundle(
            config_dir=(
                args.config_dir
            ),
            benchmark=(
                args.benchmark
            ),
            oracle_name=(
                args.oracle
            ),
            budget=budget,
        )

        set_seed(
            int(
                config[
                    "base"
                ].get(
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

        print(
            f"\n=== Budget "
            f"{budget} ==="
        )

        payload = evaluate_once(
            config=config,
            targets=targets,
            method_name=(
                args.method
            ),
            provider_factory=(
                args.provider_factory
            ),
        )

        output_file = (
            args.output_dir
            / (
                f"{args.benchmark}"
                f"_{args.oracle}"
                f"_budget{budget}.json"
            )
        )

        save_report(
            payload,
            output_file,
        )

        overall = (
            payload["overall"]
        )

        all_results.append(
            {
                "budget":
                    budget,

                "success_rate":
                    overall[
                        "success_rate"
                    ],

                "route_quality":
                    overall[
                        "route_quality"
                    ],

                "query_rate":
                    overall[
                        "query_rate"
                    ],

                "query_cost":
                    overall[
                        "query_cost"
                    ],
            }
        )

    # --------------------------------------------------------
    # Combined summary
    # --------------------------------------------------------

    summary = {
        "benchmark":
            args.benchmark,

        "method":
            args.method,

        "feedback_setting":
            args.oracle,

        "results":
            all_results,
    }

    summary_path = (
        args.output_dir
        / (
            f"{args.benchmark}"
            f"_{args.oracle}"
            "_budget_scaling.json"
        )
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\nBudget scaling summary:"
    )

    for result in all_results:

        print(
            f"B={result['budget']:4d} | "
            f"SR={result['success_rate']:.4f} | "
            f"RQ={result['route_quality']} | "
            f"QR={result['query_rate']} | "
            f"QC={result['query_cost']:.2f}"
        )


if __name__ == "__main__":
    main()

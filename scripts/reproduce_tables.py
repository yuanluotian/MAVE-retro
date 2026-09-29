from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


# ============================================================
# Load reports
# ============================================================


def find_reports(
    root: Path,
) -> list[Path]:

    return sorted(
        path
        for path
        in root.rglob(
            "*.json"
        )
        if path.is_file()
    )


def load_report(
    path: Path,
) -> dict[str, Any] | None:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    # Evaluation reports have these keys.
    required = {
        "benchmark",
        "overall",
        "method",
    }

    if not required.issubset(
        data
    ):
        return None

    return data


# ============================================================
# Flatten
# ============================================================


def flatten_report(
    report: dict[str, Any],
) -> dict[str, Any]:

    overall = (
        report["overall"]
    )

    return {
        "method":
            report.get(
                "method",
                "unknown",
            ),

        "feedback_setting":
            report.get(
                "feedback_setting",
                "unknown",
            ),

        "benchmark":
            report[
                "benchmark"
            ],

        "budget":
            report.get(
                "single_step_budget"
            ),

        "num_targets":
            overall.get(
                "num_targets"
            ),

        "success_rate":
            overall.get(
                "success_rate"
            ),

        "route_quality":
            overall.get(
                "route_quality"
            ),

        "query_rate":
            overall.get(
                "query_rate"
            ),

        "query_cost":
            overall.get(
                "query_cost"
            ),
    }


# ============================================================
# Formatting
# ============================================================


def format_percent(
    value: float | None,
) -> str:

    if value is None:
        return "--"

    return (
        f"{100.0 * value:.1f}%"
    )


def format_float(
    value: float | None,
    digits: int = 3,
) -> str:

    if value is None:
        return "--"

    return f"{value:.{digits}f}"


def format_cost(
    value: float | None,
) -> str:

    if value is None:
        return "--"

    return f"{value:.1f}"


# ============================================================
# Long CSV
# ============================================================


def write_long_csv(
    rows: list[
        dict[str, Any]
    ],
    output: Path,
) -> None:

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "method",
        "feedback_setting",
        "benchmark",
        "budget",
        "num_targets",
        "success_rate",
        "route_quality",
        "query_rate",
        "query_cost",
    ]

    with output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Paper-style wide table
# ============================================================


def build_wide_table(
    rows: list[
        dict[str, Any]
    ],
    *,
    budget: int | None,
) -> tuple[
    list[str],
    list[list[str]],
]:

    selected = [
        row
        for row in rows
        if (
            budget is None
            or row[
                "budget"
            ] == budget
        )
    ]

    benchmarks = sorted(
        {
            row[
                "benchmark"
            ]
            for row
            in selected
        }
    )

    keys = sorted(
        {
            (
                row["method"],
                row[
                    "feedback_setting"
                ],
            )
            for row
            in selected
        }
    )

    lookup = {
        (
            row["method"],
            row[
                "feedback_setting"
            ],
            row[
                "benchmark"
            ],
        ): row
        for row in selected
    }

    header = [
        "Method",
        "Feedback",
    ]

    for benchmark in benchmarks:

        header.extend(
            [
                f"{benchmark} SR",
                f"{benchmark} RQ",
                f"{benchmark} QR",
                f"{benchmark} QC",
            ]
        )

    body: list[
        list[str]
    ] = []

    for method, feedback in keys:

        line = [
            method,
            feedback,
        ]

        for benchmark in benchmarks:

            row = lookup.get(
                (
                    method,
                    feedback,
                    benchmark,
                )
            )

            if row is None:

                line.extend(
                    [
                        "--",
                        "--",
                        "--",
                        "--",
                    ]
                )

                continue

            line.extend(
                [
                    format_percent(
                        row[
                            "success_rate"
                        ]
                    ),

                    format_float(
                        row[
                            "route_quality"
                        ]
                    ),

                    format_percent(
                        row[
                            "query_rate"
                        ]
                    ),

                    format_cost(
                        row[
                            "query_cost"
                        ]
                    ),
                ]
            )

        body.append(
            line
        )

    return (
        header,
        body,
    )


# ============================================================
# Markdown
# ============================================================


def write_markdown(
    *,
    header: list[str],
    body: list[
        list[str]
    ],
    output: Path,
) -> None:

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "| "
            + " | ".join(
                header
            )
            + " |\n"
        )

        file.write(
            "| "
            + " | ".join(
                [
                    "---"
                    for _ in header
                ]
            )
            + " |\n"
        )

        for row in body:

            file.write(
                "| "
                + " | ".join(
                    row
                )
                + " |\n"
            )


# ============================================================
# LaTeX
# ============================================================


def latex_escape(
    value: str,
) -> str:

    return (
        value
        .replace(
            "&",
            r"\&",
        )
        .replace(
            "%",
            r"\%",
        )
        .replace(
            "_",
            r"\_",
        )
    )


def write_latex(
    *,
    header: list[str],
    body: list[
        list[str]
    ],
    output: Path,
) -> None:

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    num_columns = len(
        header
    )

    alignment = (
        "ll"
        + "c"
        * (
            num_columns - 2
        )
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\\begin{tabular}{"
            + alignment
            + "}\n"
        )

        file.write(
            "\\toprule\n"
        )

        file.write(
            " & ".join(
                latex_escape(
                    value
                )
                for value in header
            )
            + " \\\\\n"
        )

        file.write(
            "\\midrule\n"
        )

        for row in body:

            file.write(
                " & ".join(
                    latex_escape(
                        value
                    )
                    for value in row
                )
                + " \\\\\n"
            )

        file.write(
            "\\bottomrule\n"
        )

        file.write(
            "\\end{tabular}\n"
        )


# ============================================================
# Budget-scaling table
# ============================================================


def write_budget_scaling_csv(
    rows: list[
        dict[str, Any]
    ],
    output: Path,
) -> None:

    rows = sorted(
        rows,
        key=lambda row: (
            row[
                "benchmark"
            ],
            row[
                "method"
            ],
            row[
                "feedback_setting"
            ],
            (
                -1
                if row[
                    "budget"
                ] is None
                else row[
                    "budget"
                ]
            ),
        ),
    )

    write_long_csv(
        rows,
        output,
    )


# ============================================================
# Main
# ============================================================


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/tables"
        ),
    )

    parser.add_argument(
        "--main-budget",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    reports: list[
        dict[str, Any]
    ] = []

    for path in find_reports(
        args.input_dir
    ):

        report = load_report(
            path
        )

        if report is not None:
            reports.append(
                report
            )

    if not reports:
        raise RuntimeError(
            "No evaluation reports found."
        )

    rows = [
        flatten_report(
            report
        )
        for report
        in reports
    ]

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Machine-readable all-results table
    # --------------------------------------------------------

    write_long_csv(
        rows,
        args.output_dir
        / "all_results.csv",
    )

    # --------------------------------------------------------
    # Main 100-call paper-style table
    # --------------------------------------------------------

    header, body = (
        build_wide_table(
            rows,
            budget=(
                args.main_budget
            ),
        )
    )

    write_markdown(
        header=header,
        body=body,
        output=(
            args.output_dir
            / "main_table.md"
        ),
    )

    write_latex(
        header=header,
        body=body,
        output=(
            args.output_dir
            / "main_table.tex"
        ),
    )

    # --------------------------------------------------------
    # Budget scaling
    # --------------------------------------------------------

    write_budget_scaling_csv(
        rows,
        args.output_dir
        / "budget_scaling.csv",
    )

    print(
        f"Loaded {len(reports)} "
        f"evaluation reports."
    )

    print(
        f"Tables written to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
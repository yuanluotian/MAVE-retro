from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from mave.chemistry.canonicalize import (
    canonicalize_smiles,
)


# ============================================================
# I/O
# ============================================================


def read_smiles(
    path: Path,
    *,
    smiles_column: str = "smiles",
) -> list[str]:
    """
    Read SMILES from TXT / CSV / TSV.

    TXT:
        one SMILES per line

    CSV/TSV:
        uses --smiles-column
    """

    suffix = path.suffix.lower()

    if suffix in {
        ".txt",
        ".smi",
        ".smiles",
    }:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return [
                line.strip()
                for line in file
                if line.strip()
            ]

    if suffix in {
        ".csv",
        ".tsv",
    }:

        delimiter = (
            "\t"
            if suffix == ".tsv"
            else ","
        )

        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file,
                delimiter=delimiter,
            )

            if (
                reader.fieldnames is None
                or smiles_column
                not in reader.fieldnames
            ):
                raise ValueError(
                    f"Column {smiles_column!r} "
                    f"not found in {path}."
                )

            values: list[str] = []

            for row in reader:
                value = row.get(
                    smiles_column
                )

                if value:
                    values.append(
                        value.strip()
                    )

            return values

    raise ValueError(
        f"Unsupported input format: {path}"
    )


# ============================================================
# Canonicalization
# ============================================================


def canonicalize_dataset(
    smiles: Iterable[str],
    *,
    remove_duplicates: bool = True,
    ignore_invalid: bool = False,
) -> tuple[
    list[str],
    list[str],
]:
    """
    Returns:
        valid canonical SMILES
        rejected raw SMILES
    """

    valid: list[str] = []
    rejected: list[str] = []

    seen: set[str] = set()

    for raw in smiles:

        raw = raw.strip()

        if not raw:
            continue

        try:
            canonical = (
                canonicalize_smiles(
                    raw
                )
            )

        except Exception:

            if ignore_invalid:
                rejected.append(
                    raw
                )
                continue

            raise

        if (
            remove_duplicates
            and canonical in seen
        ):
            continue

        seen.add(
            canonical
        )

        valid.append(
            canonical
        )

    return (
        valid,
        rejected,
    )


# ============================================================
# Targets
# ============================================================


def prepare_targets(
    *,
    input_path: Path,
    output_path: Path,
    benchmark: str,
    smiles_column: str,
    expected_count: int | None,
    ignore_invalid: bool,
) -> None:

    raw = read_smiles(
        input_path,
        smiles_column=smiles_column,
    )

    canonical, rejected = (
        canonicalize_dataset(
            raw,
            remove_duplicates=True,
            ignore_invalid=(
                ignore_invalid
            ),
        )
    )

    if (
        expected_count is not None
        and len(canonical)
        != expected_count
    ):
        raise ValueError(
            f"{benchmark}: expected "
            f"{expected_count} unique targets, "
            f"but obtained {len(canonical)}."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "target_id",
                "target_smiles",
            ],
        )

        writer.writeheader()

        for index, smiles in enumerate(
            canonical
        ):
            writer.writerow(
                {
                    "target_id":
                        f"{benchmark}_{index:04d}",

                    "target_smiles":
                        smiles,
                }
            )

    manifest = {
        "benchmark":
            benchmark,

        "source_file":
            str(input_path),

        "output_file":
            str(output_path),

        "num_raw":
            len(raw),

        "num_targets":
            len(canonical),

        "num_rejected":
            len(rejected),

        "rejected":
            rejected,
    }

    manifest_path = (
        output_path
        .with_suffix(
            ".manifest.json"
        )
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[prepare_data] "
        f"{benchmark}: "
        f"{len(canonical)} targets "
        f"-> {output_path}"
    )


# ============================================================
# Purchasable building blocks
# ============================================================


def prepare_purchasable(
    *,
    input_path: Path,
    output_path: Path,
    smiles_column: str,
    ignore_invalid: bool,
) -> None:

    raw = read_smiles(
        input_path,
        smiles_column=smiles_column,
    )

    canonical, rejected = (
        canonicalize_dataset(
            raw,
            remove_duplicates=True,
            ignore_invalid=(
                ignore_invalid
            ),
        )
    )

    canonical.sort()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for smiles in canonical:
            file.write(
                smiles + "\n"
            )

    manifest = {
        "source_file":
            str(input_path),

        "output_file":
            str(output_path),

        "num_raw":
            len(raw),

        "num_unique":
            len(canonical),

        "num_rejected":
            len(rejected),
    }

    manifest_path = (
        output_path
        .with_suffix(
            output_path.suffix
            + ".manifest.json"
        )
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "[prepare_data] "
        f"purchasable building blocks: "
        f"{len(canonical)} "
        f"-> {output_path}"
    )


# ============================================================
# CLI
# ============================================================


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Prepare canonicalized datasets "
            "for MAVE reproduction."
        )
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    targets = subparsers.add_parser(
        "targets"
    )

    targets.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    targets.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    targets.add_argument(
        "--benchmark",
        type=str,
        required=True,
    )

    targets.add_argument(
        "--smiles-column",
        type=str,
        default="smiles",
    )

    targets.add_argument(
        "--expected-count",
        type=int,
        default=None,
    )

    targets.add_argument(
        "--ignore-invalid",
        action="store_true",
    )

    # --------------------------------------------------------
    # Purchasable
    # --------------------------------------------------------

    purchasable = (
        subparsers.add_parser(
            "purchasable"
        )
    )

    purchasable.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    purchasable.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    purchasable.add_argument(
        "--smiles-column",
        type=str,
        default="smiles",
    )

    purchasable.add_argument(
        "--ignore-invalid",
        action="store_true",
    )

    return parser


def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

    if args.command == "targets":

        prepare_targets(
            input_path=args.input,
            output_path=args.output,
            benchmark=args.benchmark,
            smiles_column=(
                args.smiles_column
            ),
            expected_count=(
                args.expected_count
            ),
            ignore_invalid=(
                args.ignore_invalid
            ),
        )

        return

    if args.command == "purchasable":

        prepare_purchasable(
            input_path=args.input,
            output_path=args.output,
            smiles_column=(
                args.smiles_column
            ),
            ignore_invalid=(
                args.ignore_invalid
            ),
        )

        return

    raise RuntimeError(
        "Unknown command."
    )


if __name__ == "__main__":
    main()
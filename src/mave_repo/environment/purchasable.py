from __future__ import annotations

import csv

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from mave_repro.chemistry.canonicalize import (
    canonicalize_smiles,
)
from mave_repro.chemistry.molecule import Molecule


class PurchasableDatabase(ABC):
    """
    Interface for purchasable building-block databases.

    The paper uses eMolecules, but the environment depends only
    on this interface so alternative databases can be substituted
    without changing planning logic.
    """

    @abstractmethod
    def contains_canonical(
        self,
        canonical_smiles: str,
    ) -> bool:
        """
        Query using already-canonicalized SMILES.
        """

    def is_purchasable(
        self,
        molecule: Molecule | str,
    ) -> bool:
        """
        Query purchasability.
        """
        if isinstance(
            molecule,
            Molecule,
        ):
            smiles = (
                molecule.canonical_smiles
            )

        elif isinstance(
            molecule,
            str,
        ):
            smiles = (
                canonicalize_smiles(
                    molecule
                )
            )

        else:
            raise TypeError(
                "Expected Molecule or SMILES string, "
                f"got {type(molecule)!r}."
            )

        return self.contains_canonical(
            smiles
        )

    def annotate(
        self,
        molecule: Molecule,
    ) -> Molecule:
        """
        Return a Molecule with purchasability annotated.
        """
        return molecule.with_purchasable(
            self.is_purchasable(
                molecule
            )
        )

    def __contains__(
        self,
        molecule: object,
    ) -> bool:
        if isinstance(
            molecule,
            (Molecule, str),
        ):
            return self.is_purchasable(
                molecule
            )

        return False


class InMemoryPurchasableDatabase(
    PurchasableDatabase
):
    """
    Hash-set implementation.

    Suitable for reproduction experiments when the building-block
    list fits comfortably in memory.
    """

    def __init__(
        self,
        smiles: Iterable[str],
        *,
        ignore_invalid: bool = False,
    ) -> None:

        canonical: set[str] = set()

        for value in smiles:
            value = value.strip()

            if not value:
                continue

            try:
                normalized = (
                    canonicalize_smiles(
                        value
                    )
                )

            except ValueError:
                if ignore_invalid:
                    continue

                raise

            canonical.add(
                normalized
            )

        self._smiles = frozenset(
            canonical
        )

    def contains_canonical(
        self,
        canonical_smiles: str,
    ) -> bool:
        return (
            canonical_smiles
            in self._smiles
        )

    def __len__(self) -> int:
        return len(self._smiles)

    # --------------------------------------------------------
    # File loaders
    # --------------------------------------------------------

    @classmethod
    def from_txt(
        cls,
        path: str | Path,
        *,
        ignore_invalid: bool = False,
    ) -> "InMemoryPurchasableDatabase":
        """
        Load one SMILES per line.
        """
        path = Path(path)

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            smiles = [
                line.strip()
                for line in file
                if line.strip()
            ]

        return cls(
            smiles,
            ignore_invalid=ignore_invalid,
        )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        smiles_column: str = "smiles",
        delimiter: str = ",",
        ignore_invalid: bool = False,
    ) -> "InMemoryPurchasableDatabase":
        """
        Load SMILES from CSV/TSV.
        """
        path = Path(path)

        smiles: list[str] = []

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

            for row in reader:
                value = row.get(
                    smiles_column
                )

                if value:
                    smiles.append(
                        value
                    )

        return cls(
            smiles,
            ignore_invalid=ignore_invalid,
        )


class EmptyPurchasableDatabase(
    PurchasableDatabase
):
    """
    Utility backend for tests: nothing is purchasable.
    """

    def contains_canonical(
        self,
        canonical_smiles: str,
    ) -> bool:
        return False


class AllPurchasableDatabase(
    PurchasableDatabase
):
    """
    Utility backend for tests: everything is purchasable.
    """

    def contains_canonical(
        self,
        canonical_smiles: str,
    ) -> bool:
        return True
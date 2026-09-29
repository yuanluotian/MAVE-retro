from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .canonicalize import (
    canonicalize_smiles,
    is_valid_smiles,
    num_atoms,
    num_heavy_atoms,
)


@dataclass(frozen=True, slots=True)
class Molecule:
    """
    Canonical molecular representation used throughout the
    retrosynthetic planning environment.

    Equality and hashing are based on canonical SMILES.

    Parameters
    ----------
    smiles:
        Input SMILES string.

    purchasable:
        Whether the molecule belongs to the configured
        purchasable building-block database.

        None means the status has not yet been checked.

    metadata:
        Optional auxiliary information. It should not affect
        chemical identity.
    """

    smiles: str

    purchasable: bool | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )

    canonical_smiles: str = field(
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.smiles, str):
            raise TypeError(
                "Molecule.smiles must be a string."
            )

        canonical = canonicalize_smiles(
            self.smiles
        )

        object.__setattr__(
            self,
            "canonical_smiles",
            canonical,
        )

    # --------------------------------------------------------
    # Chemical identity
    # --------------------------------------------------------

    def __hash__(self) -> int:
        return hash(
            self.canonical_smiles
        )

    def __eq__(
        self,
        other: object,
    ) -> bool:
        if not isinstance(
            other,
            Molecule,
        ):
            return NotImplemented

        return (
            self.canonical_smiles
            == other.canonical_smiles
        )

    # --------------------------------------------------------
    # Convenience properties
    # --------------------------------------------------------

    @property
    def is_purchasable(self) -> bool:
        """
        Return True only when purchasability has been explicitly
        confirmed.
        """
        return self.purchasable is True

    @property
    def atom_count(self) -> int:
        return num_atoms(
            self.canonical_smiles
        )

    @property
    def heavy_atom_count(self) -> int:
        return num_heavy_atoms(
            self.canonical_smiles
        )

    # --------------------------------------------------------
    # Immutable updates
    # --------------------------------------------------------

    def with_purchasable(
        self,
        purchasable: bool,
    ) -> "Molecule":
        """
        Return a new Molecule with updated purchasability.
        """
        return Molecule(
            smiles=self.canonical_smiles,
            purchasable=bool(purchasable),
            metadata=self.metadata,
        )

    def with_metadata(
        self,
        **metadata: Any,
    ) -> "Molecule":
        merged = {
            **dict(self.metadata),
            **metadata,
        }

        return Molecule(
            smiles=self.canonical_smiles,
            purchasable=self.purchasable,
            metadata=merged,
        )

    # --------------------------------------------------------
    # Serialization
    # --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.canonical_smiles,
            "purchasable": self.purchasable,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "Molecule":
        return cls(
            smiles=str(
                data["smiles"]
            ),
            purchasable=data.get(
                "purchasable"
            ),
            metadata=data.get(
                "metadata",
                {},
            ),
        )

    # --------------------------------------------------------
    # Representation
    # --------------------------------------------------------

    def __str__(self) -> str:
        return self.canonical_smiles

    def __repr__(self) -> str:
        purchasable = (
            ""
            if self.purchasable is None
            else f", purchasable={self.purchasable}"
        )

        return (
            "Molecule("
            f"smiles={self.canonical_smiles!r}"
            f"{purchasable}"
            ")"
        )


def ensure_molecule(
    value: str | Molecule,
) -> Molecule:
    """
    Normalize either a SMILES string or Molecule to Molecule.
    """
    if isinstance(
        value,
        Molecule,
    ):
        return value

    if isinstance(
        value,
        str,
    ):
        return Molecule(value)

    raise TypeError(
        "Expected SMILES string or Molecule, "
        f"got {type(value)!r}."
    )
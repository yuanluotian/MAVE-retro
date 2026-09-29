from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .canonicalize import (
    canonicalize_components,
)
from .molecule import (
    Molecule,
    ensure_molecule,
)


@dataclass(frozen=True, slots=True)
class Reaction:
    """
    One retrosynthetic reaction action.

    In forward-synthesis notation:

        reactants -> product

    In retrosynthetic planning, applying this action performs:

        product -> reactants

    Parameters
    ----------
    product:
        Molecule being disconnected.

    reactants:
        Predicted precursor molecules.

    score:
        Optional score from the single-step retrosynthesis model.

    template_id:
        Optional reaction-template identifier.

    reaction_id:
        Optional externally supplied reaction identifier.
    """

    product: Molecule

    reactants: tuple[Molecule, ...]

    score: float | None = None

    template_id: str | None = None

    reaction_id: str | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.product,
            Molecule,
        ):
            raise TypeError(
                "Reaction.product must be a Molecule."
            )

        if not isinstance(
            self.reactants,
            tuple,
        ):
            raise TypeError(
                "Reaction.reactants must be a tuple."
            )

        if not self.reactants:
            raise ValueError(
                "Reaction requires at least one reactant."
            )

        if not all(
            isinstance(reactant, Molecule)
            for reactant in self.reactants
        ):
            raise TypeError(
                "Every reactant must be a Molecule."
            )

        # Canonicalize ordering so that reactant permutation
        # does not define a different chemical action.
        sorted_reactants = tuple(
            sorted(
                self.reactants,
                key=lambda molecule:
                    molecule.canonical_smiles,
            )
        )

        object.__setattr__(
            self,
            "reactants",
            sorted_reactants,
        )

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------

    @property
    def key(self) -> tuple[
        str,
        tuple[str, ...],
    ]:
        return (
            self.product.canonical_smiles,
            tuple(
                reactant.canonical_smiles
                for reactant in self.reactants
            ),
        )

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(
        self,
        other: object,
    ) -> bool:
        if not isinstance(
            other,
            Reaction,
        ):
            return NotImplemented

        return self.key == other.key

    # --------------------------------------------------------
    # Convenience
    # --------------------------------------------------------

    @property
    def num_reactants(self) -> int:
        return len(self.reactants)

    @property
    def reactant_smiles(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            reactant.canonical_smiles
            for reactant in self.reactants
        )

    @property
    def product_smiles(self) -> str:
        return (
            self.product.canonical_smiles
        )

    @property
    def reaction_smiles(self) -> str:
        """
        Forward-form reaction SMILES:

            reactants>>product
        """
        reactants = ".".join(
            self.reactant_smiles
        )

        return (
            f"{reactants}"
            f">>"
            f"{self.product_smiles}"
        )

    @property
    def retrosynthetic_smiles(self) -> str:
        """
        Human-readable retrosynthetic representation:

            product>>reactants
        """
        reactants = ".".join(
            self.reactant_smiles
        )

        return (
            f"{self.product_smiles}"
            f">>"
            f"{reactants}"
        )

    # --------------------------------------------------------
    # Serialization
    # --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "product": self.product.to_dict(),
            "reactants": [
                reactant.to_dict()
                for reactant in self.reactants
            ],
            "score": self.score,
            "template_id": self.template_id,
            "reaction_id": self.reaction_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "Reaction":
        return cls(
            product=Molecule.from_dict(
                data["product"]
            ),
            reactants=tuple(
                Molecule.from_dict(item)
                for item in data["reactants"]
            ),
            score=data.get("score"),
            template_id=data.get(
                "template_id"
            ),
            reaction_id=data.get(
                "reaction_id"
            ),
            metadata=data.get(
                "metadata",
                {},
            ),
        )

    # --------------------------------------------------------
    # Constructors
    # --------------------------------------------------------

    @classmethod
    def from_smiles(
        cls,
        *,
        product: str,
        reactants: Iterable[str],
        score: float | None = None,
        template_id: str | None = None,
        reaction_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Reaction":
        """
        Construct a Reaction directly from SMILES.
        """
        return cls(
            product=Molecule(
                product
            ),
            reactants=tuple(
                Molecule(smiles)
                for smiles in reactants
            ),
            score=score,
            template_id=template_id,
            reaction_id=reaction_id,
            metadata=metadata or {},
        )


def ensure_reaction(
    *,
    product: str | Molecule,
    reactants: Iterable[
        str | Molecule
    ],
    score: float | None = None,
    template_id: str | None = None,
    reaction_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Reaction:
    """
    Convenience constructor accepting Molecule objects or SMILES.
    """
    product_molecule = ensure_molecule(
        product
    )

    reactant_molecules = tuple(
        ensure_molecule(value)
        for value in reactants
    )

    return Reaction(
        product=product_molecule,
        reactants=reactant_molecules,
        score=score,
        template_id=template_id,
        reaction_id=reaction_id,
        metadata=metadata or {},
    )
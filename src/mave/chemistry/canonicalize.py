from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

try:
    from rdkit import Chem
except ImportError as exc:  # pragma: no cover
    Chem = None
    _RDKIT_IMPORT_ERROR = exc
else:
    _RDKIT_IMPORT_ERROR = None


# ============================================================
# Internal helpers
# ============================================================


def require_rdkit() -> None:
    """
    Raise a clear error when chemistry utilities are used without
    RDKit installed.
    """
    if Chem is None:
        raise ImportError(
            "RDKit is required for chemistry canonicalization. "
            "Install it before using mave.chemistry."
        ) from _RDKIT_IMPORT_ERROR


# ============================================================
# Molecule-level canonicalization
# ============================================================


def mol_from_smiles(
    smiles: str,
    *,
    sanitize: bool = True,
):
    """
    Parse a SMILES string into an RDKit molecule.

    Parameters
    ----------
    smiles:
        Input SMILES.

    sanitize:
        Whether RDKit sanitization should be performed.

    Returns
    -------
    rdkit.Chem.Mol

    Raises
    ------
    ValueError
        If the SMILES string is empty or cannot be parsed.
    """
    require_rdkit()

    if not isinstance(smiles, str):
        raise TypeError(
            f"SMILES must be a string, got {type(smiles)!r}."
        )

    smiles = smiles.strip()

    if not smiles:
        raise ValueError("SMILES must not be empty.")

    mol = Chem.MolFromSmiles(
        smiles,
        sanitize=sanitize,
    )

    if mol is None:
        raise ValueError(
            f"Invalid SMILES: {smiles!r}"
        )

    return mol


def is_valid_smiles(
    smiles: str,
) -> bool:
    """
    Check whether a SMILES string can be parsed by RDKit.
    """
    try:
        mol_from_smiles(smiles)
    except (TypeError, ValueError, ImportError):
        return False

    return True


def canonicalize_smiles(
    smiles: str,
    *,
    isomeric: bool = True,
    remove_atom_mapping: bool = False,
) -> str:
    """
    Convert a SMILES string to canonical RDKit SMILES.

    Parameters
    ----------
    smiles:
        Input SMILES.

    isomeric:
        Preserve stereochemical information.

    remove_atom_mapping:
        Remove atom-map numbers before serialization.

    Notes
    -----
    The paper does not specify the exact SMILES canonicalization
    procedure. Using RDKit canonical SMILES is therefore a
    reproduction implementation choice and should be kept fixed
    across all experiments.
    """
    mol = mol_from_smiles(smiles)

    if remove_atom_mapping:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=isomeric,
    )


# ============================================================
# Multi-component molecules
# ============================================================


def split_smiles_components(
    smiles: str,
    *,
    canonicalize: bool = True,
) -> tuple[str, ...]:
    """
    Split dot-separated molecular components.

    Example
    -------
    "CCO.O" -> ("CCO", "O")
    """
    if not isinstance(smiles, str):
        raise TypeError(
            f"SMILES must be a string, got {type(smiles)!r}."
        )

    parts = tuple(
        part.strip()
        for part in smiles.split(".")
        if part.strip()
    )

    if not parts:
        raise ValueError(
            "No valid SMILES components found."
        )

    if canonicalize:
        parts = tuple(
            canonicalize_smiles(part)
            for part in parts
        )

    return parts


def canonicalize_components(
    components: Iterable[str],
    *,
    sort_components: bool = True,
) -> tuple[str, ...]:
    """
    Canonicalize a collection of molecule SMILES.

    Sorting is useful when the order of reactants should not
    affect equality, hashing, or cache keys.
    """
    canonical = [
        canonicalize_smiles(smiles)
        for smiles in components
    ]

    if sort_components:
        canonical.sort()

    return tuple(canonical)


def join_smiles_components(
    components: Iterable[str],
    *,
    canonicalize: bool = True,
    sort_components: bool = True,
) -> str:
    """
    Serialize multiple molecular components as dot-separated
    SMILES.
    """
    components = tuple(components)

    if canonicalize:
        components = canonicalize_components(
            components,
            sort_components=sort_components,
        )
    elif sort_components:
        components = tuple(
            sorted(components)
        )

    return ".".join(components)


# ============================================================
# Reaction SMILES helpers
# ============================================================


def canonicalize_reaction_smiles(
    reaction_smiles: str,
    *,
    remove_atom_mapping: bool = False,
    sort_reactants: bool = True,
) -> str:
    """
    Canonicalize a retrosynthetic reaction SMILES.

    Supported forms
    ---------------
    reactants>>product
    reactants>reagents>product

    The component convention here is:

        reactants >> product

    even though the retrosynthetic action is conceptually
    product -> reactants.
    """
    if not isinstance(reaction_smiles, str):
        raise TypeError(
            "reaction_smiles must be a string."
        )

    reaction_smiles = reaction_smiles.strip()

    if ">>" in reaction_smiles:
        reactants, product = reaction_smiles.split(
            ">>",
            maxsplit=1,
        )

        reagents = None

    else:
        parts = reaction_smiles.split(">")

        if len(parts) != 3:
            raise ValueError(
                "Reaction SMILES must have either "
                "'reactants>>product' or "
                "'reactants>reagents>product' format."
            )

        reactants, reagents, product = parts

    reactant_parts = split_smiles_components(
        reactants,
        canonicalize=False,
    )

    reactant_parts = tuple(
        canonicalize_smiles(
            smiles,
            remove_atom_mapping=remove_atom_mapping,
        )
        for smiles in reactant_parts
    )

    if sort_reactants:
        reactant_parts = tuple(
            sorted(reactant_parts)
        )

    product = canonicalize_smiles(
        product,
        remove_atom_mapping=remove_atom_mapping,
    )

    canonical_reactants = ".".join(
        reactant_parts
    )

    if reagents is None:
        return (
            f"{canonical_reactants}"
            f">>"
            f"{product}"
        )

    reagent_parts = (
        split_smiles_components(
            reagents,
            canonicalize=False,
        )
        if reagents.strip()
        else ()
    )

    canonical_reagents = ".".join(
        canonicalize_smiles(
            smiles,
            remove_atom_mapping=remove_atom_mapping,
        )
        for smiles in reagent_parts
    )

    return (
        f"{canonical_reactants}"
        f">"
        f"{canonical_reagents}"
        f">"
        f"{product}"
    )


# ============================================================
# Simple molecular descriptors
# ============================================================


def num_atoms(
    smiles: str,
) -> int:
    """
    Number of atoms including hydrogens only when explicitly
    represented in the SMILES.
    """
    mol = mol_from_smiles(smiles)
    return int(mol.GetNumAtoms())


def num_heavy_atoms(
    smiles: str,
) -> int:
    mol = mol_from_smiles(smiles)
    return int(mol.GetNumHeavyAtoms())
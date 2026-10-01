from __future__ import annotations

from typing import Any, Mapping, Sequence

from mave.chemistry.molecule import Molecule
from mave.chemistry.reaction import Reaction
from mave.core.types import (
    Feedback,
    FeedbackLevel,
    OracleContext,
    ReactionCandidate,
)
from mave.environment.state import RetroState


_NATIVE_STATE = "_native_state"
_NATIVE_SELECTED_MOLECULE = "_native_selected_molecule"
_NATIVE_CANDIDATES = "_native_candidates"


def reaction_to_candidate(reaction: Reaction) -> ReactionCandidate:
    """Convert an executable chemistry reaction to its policy-facing form."""
    return ReactionCandidate(
        product_smiles=reaction.product_smiles,
        reactant_smiles=reaction.reactant_smiles,
        single_step_score=(
            0.0 if reaction.score is None else float(reaction.score)
        ),
        template_id=reaction.template_id,
        reaction_id=reaction.reaction_id,
        metadata=dict(reaction.metadata),
    )


def build_oracle_context(
    *,
    state: RetroState,
    selected_index: int,
    candidates: Sequence[Reaction],
    current_level: FeedbackLevel = FeedbackLevel.L0,
    feedback_history: Sequence[Feedback] = (),
    context_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> OracleContext:
    """Build the canonical oracle context from planner-native objects.

    Oracle and policy-facing dataclasses intentionally use serializable core
    representations.  Backend execution still needs the chemistry objects, so
    the adapter retains them in private metadata instead of leaking the
    planner's ``selected_index`` into the public OracleContext contract.
    """
    selected_molecule = state.molecule_at(selected_index)
    native_candidates = tuple(candidates)

    if not native_candidates:
        raise ValueError("OracleContext requires at least one reaction candidate.")

    for reaction in native_candidates:
        if reaction.product != selected_molecule:
            raise ValueError(
                "Every oracle candidate must disconnect the selected molecule."
            )

    merged_metadata = dict(metadata or {})
    merged_metadata.update(
        {
            _NATIVE_STATE: state,
            _NATIVE_SELECTED_MOLECULE: selected_molecule,
            _NATIVE_CANDIDATES: native_candidates,
        }
    )

    return OracleContext(
        state=state.to_core_state(),
        selected_molecule=selected_molecule.canonical_smiles,
        candidates=tuple(
            reaction_to_candidate(reaction) for reaction in native_candidates
        ),
        current_level=current_level,
        feedback_history=tuple(feedback_history),
        context_id=context_id,
        metadata=merged_metadata,
    )


def native_state(context: OracleContext) -> RetroState:
    value = context.metadata.get(_NATIVE_STATE)
    if not isinstance(value, RetroState):
        raise TypeError("OracleContext does not contain a native RetroState.")
    return value


def native_selected_molecule(context: OracleContext) -> Molecule:
    value = context.metadata.get(_NATIVE_SELECTED_MOLECULE)
    if not isinstance(value, Molecule):
        raise TypeError("OracleContext does not contain a native selected molecule.")
    return value


def native_candidates(context: OracleContext) -> tuple[Reaction, ...]:
    value = context.metadata.get(_NATIVE_CANDIDATES)
    if not isinstance(value, tuple) or not all(
        isinstance(candidate, Reaction) for candidate in value
    ):
        raise TypeError("OracleContext does not contain native reaction candidates.")
    return value

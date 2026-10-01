from __future__ import annotations

from typing import Any, Iterable


def materialize_parameters(parameters: Iterable[Any]) -> tuple[Any, ...]:
    """Materialize one policy's parameter iterable without duplicate objects."""
    materialized = tuple(parameters)
    identities = [id(parameter) for parameter in materialized]

    if len(identities) != len(set(identities)):
        raise ValueError("A policy parameter collection contains duplicates.")

    return materialized


def optimizer_parameters(optimizer: Any) -> tuple[Any, ...]:
    """Return the exact parameters managed by an optimizer."""
    return materialize_parameters(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def ensure_disjoint_parameters(
    escalation_parameters: Iterable[Any],
    reaction_parameters: Iterable[Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Validate that mu and pi have no shared trainable parameter objects."""
    escalation = materialize_parameters(escalation_parameters)
    reaction = materialize_parameters(reaction_parameters)
    overlap = {id(parameter) for parameter in escalation}.intersection(
        id(parameter) for parameter in reaction
    )

    if overlap:
        raise ValueError(
            "Escalation and reaction policies must have disjoint trainable "
            "parameters; shared parameters would receive independent optimizer "
            "states and violate alternating-policy freezing."
        )

    return escalation, reaction


def activate_parameter_set(
    *,
    active: Iterable[Any],
    inactive: Iterable[Any],
) -> None:
    """Enable gradients only for the policy updated in the current phase."""
    active_parameters, inactive_parameters = ensure_disjoint_parameters(
        active,
        inactive,
    )

    for parameter in active_parameters:
        parameter.requires_grad_(True)

    for parameter in inactive_parameters:
        parameter.requires_grad_(False)

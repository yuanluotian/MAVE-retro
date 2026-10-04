from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class CheckpointLoadError(RuntimeError):
    pass


def load_policy_checkpoint(
    system: Any,
    path: Path,
    *,
    strict: bool = True,
) -> tuple[str, ...]:
    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint file does not exist: {checkpoint_path}"
        )

    import torch

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(state, Mapping):
        raise CheckpointLoadError(
            "Checkpoint must contain a mapping of named state dictionaries."
        )

    targets = {
        "escalation_backbone": system.escalation_backbone,
        "reaction_backbone": system.reaction_backbone,
    }

    missing = [
        name
        for name in targets
        if name not in state
    ]

    if missing:
        raise CheckpointLoadError(
            "Checkpoint is missing required policy state: "
            + ", ".join(missing)
        )

    for name, backbone in targets.items():
        model = getattr(backbone, "model", None)
        load_state_dict = getattr(model, "load_state_dict", None)

        if load_state_dict is None:
            raise CheckpointLoadError(
                f"Policy backbone {name!r} has no model.load_state_dict()."
            )

        policy_state = state[name]

        if not isinstance(policy_state, Mapping):
            raise CheckpointLoadError(
                f"Checkpoint entry {name!r} is not a state dictionary."
            )

        try:
            load_state_dict(
                policy_state,
                strict=strict,
            )
        except RuntimeError as error:
            raise CheckpointLoadError(
                f"Unable to load checkpoint entry {name!r}: {error}"
            ) from error

        backbone.eval()

    return tuple(targets)

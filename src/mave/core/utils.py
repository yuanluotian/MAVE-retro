from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


# ============================================================
# Randomness / reproducibility
# ============================================================


def set_seed(
    seed: int,
    *,
    deterministic: bool = True,
) -> None:
    """
    Seed Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Global random seed.

    deterministic:
        If True, request deterministic PyTorch behavior where
        possible.
    """

    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is None:
        return

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(
                True,
                warn_only=True,
            )
        except AttributeError:
            pass


# ============================================================
# Numerical helpers
# ============================================================


def standardize_numpy(
    values: Sequence[float] | np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Group normalization:

        (x - mean(x)) / (std(x) + eps)

    Population standard deviation (ddof=0) is used here.

    NOTE:
    The paper specifies group normalization but does not state
    whether population or sample standard deviation is used.
    Therefore ddof=0 is a reproduction choice and should remain
    consistent across all experiments.
    """

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(
            "standardize_numpy expects a 1D array."
        )

    if len(array) == 0:
        raise ValueError(
            "Cannot standardize an empty array."
        )

    mean = array.mean()
    std = array.std(ddof=0)

    return (
        array - mean
    ) / (
        std + eps
    )


def standardize_tensor(
    values: "torch.Tensor",
    *,
    eps: float = 1e-8,
) -> "torch.Tensor":
    """
    PyTorch equivalent of within-group normalization.

    Uses population standard deviation (unbiased=False).
    """

    if torch is None:
        raise ImportError(
            "PyTorch is required for standardize_tensor."
        )

    if values.ndim != 1:
        raise ValueError(
            "standardize_tensor expects a 1D tensor."
        )

    if values.numel() == 0:
        raise ValueError(
            "Cannot standardize an empty tensor."
        )

    mean = values.mean()

    std = values.std(
        unbiased=False
    )

    return (
        values - mean
    ) / (
        std + eps
    )


def safe_mean(
    values: Sequence[float],
    *,
    default: float = 0.0,
) -> float:
    if len(values) == 0:
        return default

    return float(
        sum(values) / len(values)
    )


# ============================================================
# Stable identifiers
# ============================================================


def stable_hash(
    value: Any,
    *,
    length: int = 16,
) -> str:
    """
    Generate deterministic identifier from a JSON-serializable
    representation of an object.
    """

    serialized = json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    digest = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()

    return digest[:length]


def make_context_id(
    target_smiles: str,
    unsolved_molecules: Sequence[str],
    *,
    depth: int,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """
    Create stable ID for one planning context.

    This is useful because all K MAVE rollouts belonging to the
    same group should share the same context_id.
    """

    payload = {
        "target_smiles": target_smiles,
        "unsolved_molecules": sorted(
            unsolved_molecules
        ),
        "depth": depth,
        "extra": extra or {},
    }

    return stable_hash(payload)


def make_query_id(
    *,
    state_id: str | None,
    level: int,
    feedback_type: str,
    payload: Mapping[str, Any],
) -> str:
    """
    Generate stable query ID for feedback caching.
    """

    return stable_hash(
        {
            "state_id": state_id,
            "level": level,
            "feedback_type": feedback_type,
            "payload": payload,
        }
    )


# ============================================================
# Serialization
# ============================================================


def to_jsonable(value: Any) -> Any:
    """
    Convert common project objects to structures compatible
    with json.dumps().
    """

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value):
        return {
            key: to_jsonable(item)
            for key, item in asdict(value).items()
        }

    if isinstance(value, Mapping):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            to_jsonable(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if (
        torch is not None
        and isinstance(value, torch.Tensor)
    ):
        if value.numel() == 1:
            return value.detach().cpu().item()

        return (
            value
            .detach()
            .cpu()
            .tolist()
        )

    return str(value)


def save_json(
    data: Any,
    path: str | Path,
    *,
    indent: int = 2,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            to_jsonable(data),
            file,
            indent=indent,
            ensure_ascii=False,
            sort_keys=True,
        )


def load_json(
    path: str | Path,
) -> Any:
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# Dictionary / config helpers
# ============================================================


def deep_update(
    base: dict[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge override into base.

    Nested dictionaries are merged rather than replaced.
    """

    for key, value in override.items():

        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, Mapping)
        ):
            deep_update(
                base[key],
                value,
            )

        else:
            base[key] = value

    return base


def get_nested(
    mapping: Mapping[str, Any],
    path: str,
    *,
    default: Any = None,
) -> Any:
    """
    Read nested config values using dot notation.

    Example:

        get_nested(config, "train.grpo.learning_rate")
    """

    current: Any = mapping

    for key in path.split("."):

        if not isinstance(
            current,
            Mapping,
        ):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


# ============================================================
# File-system helpers
# ============================================================


def ensure_dir(
    path: str | Path,
) -> Path:
    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


# ============================================================
# Collection helpers
# ============================================================


def ensure_tuple(
    value: Any,
) -> tuple[Any, ...]:
    if value is None:
        return ()

    if isinstance(value, tuple):
        return value

    if isinstance(value, list):
        return tuple(value)

    return (value,)


def flatten_once(
    nested: Sequence[Sequence[Any]],
) -> list[Any]:
    return [
        item
        for sequence in nested
        for item in sequence
    ]
from __future__ import annotations

import hashlib
import json
from dataclasses import (
    fields,
    is_dataclass,
)
from enum import Enum
from typing import Any, Mapping

from mave.oracle.base import (
    Feedback,
    FeedbackQuery,
    OracleContext,
)


# ============================================================
# Stable serialization
# ============================================================


def _normalize(
    value: Any,
) -> Any:
    """
    Convert common project objects into deterministic,
    JSON-serializable representations for cache keys.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        Enum,
    ):
        return value.value

    # --------------------------------------------------------
    # Molecule-like objects
    # --------------------------------------------------------

    if hasattr(
        value,
        "canonical_smiles",
    ):
        return {
            "type":
                value.__class__.__name__,

            "canonical_smiles":
                str(
                    value.canonical_smiles
                ),
        }

    # --------------------------------------------------------
    # Reaction-like objects
    # --------------------------------------------------------

    if hasattr(
        value,
        "key",
    ):
        return {
            "type":
                value.__class__.__name__,

            "key":
                str(
                    value.key
                ),
        }

    # --------------------------------------------------------
    # State-like objects
    # --------------------------------------------------------

    if hasattr(
        value,
        "state_id",
    ):
        return {
            "type":
                value.__class__.__name__,

            "state_id":
                str(
                    value.state_id
                ),
        }

    # --------------------------------------------------------
    # Route-like objects
    # --------------------------------------------------------

    if hasattr(
        value,
        "route_id",
    ):
        return {
            "type":
                value.__class__.__name__,

            "route_id":
                str(
                    value.route_id
                ),
        }

    # --------------------------------------------------------
    # Mappings
    # --------------------------------------------------------

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): _normalize(
                item
            )
            for key, item
            in sorted(
                value.items(),
                key=lambda pair: str(
                    pair[0]
                ),
            )
        }

    # --------------------------------------------------------
    # Sequences
    # --------------------------------------------------------

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        return [
            _normalize(
                item
            )
            for item
            in value
        ]

    # --------------------------------------------------------
    # Dataclasses
    # --------------------------------------------------------

    if is_dataclass(
        value
    ):
        return {
            item.name:
                _normalize(
                    getattr(
                        value,
                        item.name,
                    )
                )
            for item
            in fields(
                value
            )
        }

    # Last-resort representation.
    return repr(
        value
    )


# ============================================================
# Cache key
# ============================================================


def make_cache_key(
    query: FeedbackQuery,
    context: OracleContext,
) -> str:
    """
    Build a deterministic key for one feedback acquisition.
    """

    if query.cache_key is not None:

        return str(
            query.cache_key
        )

    state_id = getattr(
        context.state,
        "state_id",
        None,
    )

    payload = {

        "feedback_type":
            query.feedback_type.value,

        "state":
            (
                str(state_id)
                if state_id is not None
                else None
            ),

        "selected_index":
            context.selected_index,

        "selected_molecule":
            _normalize(
                context.selected_molecule
            ),

        "candidates":
            _normalize(
                context.candidates
            ),

        "target":
            _normalize(
                query.target
            ),

        "alternatives":
            _normalize(
                query.alternatives
            ),

        "params":
            _normalize(
                query.params
            ),
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=True,
    )

    return hashlib.sha256(
        serialized.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# In-memory cache
# ============================================================


class InMemoryFeedbackCache:
    """
    Cache final acquired natural-language feedback.

    Important:
        The cached object stores the original acquisition cost.
        MultiFidelityOracleSystem returns it through as_cached(),
        which sets the additional cost to zero.
    """

    def __init__(
        self,
    ) -> None:

        self._store: dict[
            str,
            Feedback,
        ] = {}

    def get(
        self,
        query: FeedbackQuery,
        context: OracleContext,
    ) -> Feedback | None:

        key = make_cache_key(
            query,
            context,
        )

        return self._store.get(
            key
        )

    def set(
        self,
        query: FeedbackQuery,
        context: OracleContext,
        feedback: Feedback,
    ) -> None:

        key = make_cache_key(
            query,
            context,
        )

        self._store[
            key
        ] = feedback

    def contains(
        self,
        query: FeedbackQuery,
        context: OracleContext,
    ) -> bool:

        key = make_cache_key(
            query,
            context,
        )

        return (
            key
            in self._store
        )

    def clear(
        self,
    ) -> None:

        self._store.clear()

    def __len__(
        self,
    ) -> int:

        return len(
            self._store
        )


# ============================================================
# No-op cache
# ============================================================


class NullFeedbackCache:
    """
    Cache implementation for experiments where feedback reuse
    should be disabled explicitly.
    """

    def get(
        self,
        query: FeedbackQuery,
        context: OracleContext,
    ) -> None:
        return None

    def set(
        self,
        query: FeedbackQuery,
        context: OracleContext,
        feedback: Feedback,
    ) -> None:
        return None

    def clear(
        self,
    ) -> None:
        return None

    def __len__(
        self,
    ) -> int:
        return 0
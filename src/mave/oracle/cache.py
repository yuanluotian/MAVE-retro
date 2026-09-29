from __future__ import annotations

from mave.core.types import (
    Feedback,
)


class FeedbackCache:
    """
    In-memory feedback cache.

    Algorithm 2 states that previously acquired feedback is
    cached and reused without additional acquisition cost.
    """

    def __init__(self) -> None:
        self._cache: dict[
            str,
            Feedback,
        ] = {}

    def get(
        self,
        query_id: str,
    ) -> Feedback | None:
        return self._cache.get(
            query_id
        )

    def put(
        self,
        query_id: str,
        feedback: Feedback,
    ) -> None:

        if not query_id:
            raise ValueError(
                "query_id must not be empty."
            )

        self._cache[
            query_id
        ] = feedback

    def contains(
        self,
        query_id: str,
    ) -> bool:
        return (
            query_id
            in self._cache
        )

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(
            self._cache
        )

    def __contains__(
        self,
        query_id: object,
    ) -> bool:

        if not isinstance(
            query_id,
            str,
        ):
            return False

        return self.contains(
            query_id
        )
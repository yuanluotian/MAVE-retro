from __future__ import annotations

from typing import (
    Any,
    Callable,
    TypeVar,
)


T = TypeVar("T")


class Registry:
    """
    Lightweight name -> class registry.
    """

    def __init__(
        self,
        name: str,
    ) -> None:

        self.name = name

        self._items: dict[
            str,
            Any,
        ] = {}

    def register(
        self,
        name: str,
    ) -> Callable[[T], T]:

        def decorator(
            obj: T,
        ) -> T:

            if name in self._items:

                existing = (
                    self._items[
                        name
                    ]
                )

                if existing is not obj:
                    raise KeyError(
                        f"{name!r} is already registered "
                        f"in {self.name} registry."
                    )

            self._items[
                name
            ] = obj

            return obj

        return decorator

    def get(
        self,
        name: str,
    ) -> Any:

        if name not in self._items:
            raise KeyError(
                f"Unknown {self.name}: {name!r}. "
                f"Available: {sorted(self._items)}"
            )

        return self._items[
            name
        ]

    def create(
        self,
        name: str,
        /,
        **kwargs: Any,
    ) -> Any:

        cls = self.get(
            name
        )

        return cls(
            **kwargs
        )

    def names(
        self,
    ) -> tuple[str, ...]:

        return tuple(
            sorted(
                self._items
            )
        )

    def contains(
        self,
        name: str,
    ) -> bool:

        return (
            name
            in self._items
        )


# ============================================================
# Global registries
# ============================================================


BACKEND_REGISTRY = Registry(
    "backend"
)

ORACLE_REGISTRY = Registry(
    "oracle"
)


# ============================================================
# Backend registration
# ============================================================


def register_backend(
    name: str,
):
    return BACKEND_REGISTRY.register(
        name
    )


def get_backend(
    name: str,
):
    return BACKEND_REGISTRY.get(
        name
    )


def create_backend(
    name: str,
    /,
    **kwargs: Any,
):
    return BACKEND_REGISTRY.create(
        name,
        **kwargs,
    )


# ============================================================
# Oracle registration
# ============================================================


def register_oracle(
    name: str,
):
    return ORACLE_REGISTRY.register(
        name
    )


def get_oracle(
    name: str,
):
    return ORACLE_REGISTRY.get(
        name
    )


def create_oracle(
    name: str,
    /,
    **kwargs: Any,
):
    return ORACLE_REGISTRY.create(
        name,
        **kwargs,
    )
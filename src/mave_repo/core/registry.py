from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    """
    Lightweight name -> object registry.

    Typical usage:

        MAPPING_REGISTRY = Registry("mapping")

        @MAPPING_REGISTRY.register("tanh")
        class TanhMapping:
            ...

        cls = MAPPING_REGISTRY.get("tanh")
    """

    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, T] = {}

    # --------------------------------------------------------
    # Registration
    # --------------------------------------------------------

    def register(
        self,
        name: str,
        *,
        overwrite: bool = False,
    ) -> Callable[[T], T]:
        """
        Register an object using decorator syntax.
        """

        normalized_name = self._normalize_name(name)

        def decorator(obj: T) -> T:
            self.add(
                normalized_name,
                obj,
                overwrite=overwrite,
            )
            return obj

        return decorator

    def add(
        self,
        name: str,
        obj: T,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register an object directly.
        """

        key = self._normalize_name(name)

        if key in self._items and not overwrite:
            raise KeyError(
                f"{self.name!r} registry already contains "
                f"an entry named {key!r}."
            )

        self._items[key] = obj

    # --------------------------------------------------------
    # Lookup
    # --------------------------------------------------------

    def get(self, name: str) -> T:
        key = self._normalize_name(name)

        try:
            return self._items[key]

        except KeyError as exc:
            available = ", ".join(
                sorted(self._items.keys())
            )

            raise KeyError(
                f"Unknown {self.name} {name!r}. "
                f"Available: [{available}]"
            ) from exc

    def maybe_get(
        self,
        name: str,
        default: T | None = None,
    ) -> T | None:
        key = self._normalize_name(name)

        return self._items.get(
            key,
            default,
        )

    def contains(self, name: str) -> bool:
        return self._normalize_name(name) in self._items

    # --------------------------------------------------------
    # Object construction
    # --------------------------------------------------------

    def create(
        self,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Retrieve a registered callable/class and instantiate it.

        Example:

            mapping = MAPPING_REGISTRY.create(
                "tanh",
                regularization=1e-3,
            )
        """

        obj = self.get(name)

        if not callable(obj):
            raise TypeError(
                f"Registered object {name!r} in "
                f"{self.name!r} is not callable."
            )

        return obj(
            *args,
            **kwargs,
        )

    # --------------------------------------------------------
    # Introspection
    # --------------------------------------------------------

    def keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(self._items.keys())
        )

    def values(self) -> tuple[T, ...]:
        return tuple(
            self._items[key]
            for key in sorted(self._items.keys())
        )

    def items(self) -> tuple[tuple[str, T], ...]:
        return tuple(
            (key, self._items[key])
            for key in sorted(self._items.keys())
        )

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False

        return self.contains(name)

    def __iter__(self) -> Iterator[str]:
        return iter(
            sorted(self._items.keys())
        )

    def __repr__(self) -> str:
        entries = ", ".join(
            sorted(self._items.keys())
        )

        return (
            f"Registry("
            f"name={self.name!r}, "
            f"items=[{entries}]"
            f")"
        )

    # --------------------------------------------------------
    # Internal
    # --------------------------------------------------------

    @staticmethod
    def _normalize_name(name: str) -> str:
        if not isinstance(name, str):
            raise TypeError(
                "Registry names must be strings."
            )

        normalized = name.strip().lower()

        if not normalized:
            raise ValueError(
                "Registry name must not be empty."
            )

        return normalized


# ============================================================
# Project-level registries
# ============================================================

MODEL_REGISTRY: Registry[Any] = Registry("model")

POLICY_REGISTRY: Registry[Any] = Registry("policy")

ORACLE_REGISTRY: Registry[Any] = Registry("oracle")

ORACLE_BACKEND_REGISTRY: Registry[Any] = Registry(
    "oracle_backend"
)

PLANNER_REGISTRY: Registry[Any] = Registry("planner")

MAPPING_REGISTRY: Registry[Any] = Registry(
    "reward_cost_mapping"
)

METRIC_REGISTRY: Registry[Any] = Registry("metric")

BASELINE_REGISTRY: Registry[Any] = Registry("baseline")


# ============================================================
# Convenience lookup
# ============================================================

_REGISTRIES: dict[str, Registry[Any]] = {
    "model": MODEL_REGISTRY,
    "policy": POLICY_REGISTRY,
    "oracle": ORACLE_REGISTRY,
    "oracle_backend": ORACLE_BACKEND_REGISTRY,
    "planner": PLANNER_REGISTRY,
    "mapping": MAPPING_REGISTRY,
    "metric": METRIC_REGISTRY,
    "baseline": BASELINE_REGISTRY,
}


def get_registry(name: str) -> Registry[Any]:
    """
    Get one of the global project registries.
    """

    key = name.strip().lower()

    try:
        return _REGISTRIES[key]

    except KeyError as exc:
        available = ", ".join(
            sorted(_REGISTRIES.keys())
        )

        raise KeyError(
            f"Unknown registry {name!r}. "
            f"Available: [{available}]"
        ) from exc
from __future__ import annotations

from typing import Any, Callable, TypeVar

from mave_repro.core.registry import (
    ORACLE_BACKEND_REGISTRY,
    ORACLE_REGISTRY,
)


T = TypeVar("T")


# ============================================================
# Oracle registration
# ============================================================


def register_oracle(
    name: str,
    *,
    overwrite: bool = False,
) -> Callable[[T], T]:
    return ORACLE_REGISTRY.register(
        name,
        overwrite=overwrite,
    )


def register_backend(
    name: str,
    *,
    overwrite: bool = False,
) -> Callable[[T], T]:
    return ORACLE_BACKEND_REGISTRY.register(
        name,
        overwrite=overwrite,
    )


# ============================================================
# Lookup
# ============================================================


def get_oracle_class(
    name: str,
) -> Any:
    return ORACLE_REGISTRY.get(
        name
    )


def get_backend_class(
    name: str,
) -> Any:
    return ORACLE_BACKEND_REGISTRY.get(
        name
    )


# ============================================================
# Construction
# ============================================================


def build_oracle(
    name: str,
    **kwargs: Any,
) -> Any:
    return ORACLE_REGISTRY.create(
        name,
        **kwargs,
    )


def build_backend(
    name: str,
    **kwargs: Any,
) -> Any:
    return ORACLE_BACKEND_REGISTRY.create(
        name,
        **kwargs,
    )
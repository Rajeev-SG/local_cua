"""Adapter registry."""
from __future__ import annotations

_REGISTRY = {}


def register(key):
    def deco(cls):
        _REGISTRY[key] = cls
        return cls
    return deco


def get_adapter(key: str, viewport: dict):
    # import side-effects populate the registry
    from . import uground, showui, uitars, tongui, fara  # noqa: F401
    if key not in _REGISTRY:
        raise KeyError(f"no adapter {key!r}; have {sorted(_REGISTRY)}")
    cls = _REGISTRY[key]
    return cls(viewport)


def list_adapters():
    from . import uground, showui, uitars, tongui, fara  # noqa: F401
    return sorted(_REGISTRY)

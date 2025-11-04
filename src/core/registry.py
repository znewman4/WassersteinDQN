# src/core/registry.py
from typing import Callable, Dict

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._entries: Dict[str, Callable] = {}

    def register(self, key: str):
        """Decorator for registering components"""
        def _inner(fn_or_cls):
            if key in self._entries:
                raise KeyError(f"{self.name} already has '{key}' registered.")
            self._entries[key] = fn_or_cls
            return fn_or_cls
        return _inner

    def get(self, key: str) -> Callable:
        if key not in self._entries:
            raise KeyError(
                f"{self.name} registry missing '{key}'. "
                f"Available: {list(self._entries.keys())}"
            )
        return self._entries[key]

    def all(self) -> Dict[str, Callable]:
        return self._entries.copy()


# ✅ Global registries
AGENTS = Registry("AGENTS")
ENVS = Registry("ENVS")
DATASETS = Registry("DATASETS")

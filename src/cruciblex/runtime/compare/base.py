from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True)
class ComparisonRequest:
    expected: object
    actual: object
    tolerance: dict[str, float]
    metadata: dict[str, object]


@dataclass(slots=True)
class ComparisonReport:
    passed: bool
    max_abs_diff: float
    mean_abs_diff: float
    detail: str


class Comparator(ABC):
    @abstractmethod
    def compare(self, request: ComparisonRequest) -> ComparisonReport:
        raise NotImplementedError


class ComparatorRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Comparator]] = {}

    def register(self, name: str):
        def decorator(factory: Callable[[], Comparator] | type[Comparator]):
            self._factories[name] = factory
            return factory

        return decorator

    def resolve(self, name: str) -> Comparator:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(f"unknown comparator: {name}") from exc
        return factory()

    def known(self) -> list[str]:
        return sorted(self._factories)


COMPARATOR_REGISTRY = ComparatorRegistry()
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from cruciblex.domain.case import CaseSpec
from cruciblex.domain.plan import ExecutionPlan
from cruciblex.runtime.backends.base import DeviceContext


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    case: CaseSpec
    plan: ExecutionPlan
    context: DeviceContext | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class InputGenerator(ABC):
    @abstractmethod
    def generate(self, request: GenerationRequest) -> list[object]:
        raise NotImplementedError


class GeneratorRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], InputGenerator]] = {}

    def register(self, name: str):
        def decorator(factory: Callable[[], InputGenerator] | type[InputGenerator]):
            self._factories[name] = factory
            return factory

        return decorator

    def resolve(self, name: str) -> InputGenerator:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(f"unknown generator: {name}") from exc
        return factory()

    def known(self) -> list[str]:
        return sorted(self._factories)


GENERATOR_REGISTRY = GeneratorRegistry()

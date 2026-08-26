from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from cruciblex.domain.case import CaseSpec
from cruciblex.domain.enums import ExecutionRole
from cruciblex.domain.plan import ExecutionPlan
from cruciblex.runtime.backends.base import DeviceContext


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    case: CaseSpec
    inputs: list[object]
    plan: ExecutionPlan
    context: DeviceContext | None = None
    role: ExecutionRole = ExecutionRole.CANDIDATE


class ExecutionNotSupportedError(RuntimeError):
    pass


class BackendExecutor(ABC):
    @abstractmethod
    def execute(self, request: ExecutionRequest) -> object:
        raise NotImplementedError


class ExecutorRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], BackendExecutor]] = {}

    def register(self, name: str):
        def decorator(factory: Callable[[], BackendExecutor] | type[BackendExecutor]):
            self._factories[name] = factory
            return factory

        return decorator

    def resolve(self, name: str) -> BackendExecutor:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(f"unknown executor: {name}") from exc
        return factory()

    def known(self) -> list[str]:
        return sorted(self._factories)


EXECUTOR_REGISTRY = ExecutorRegistry()
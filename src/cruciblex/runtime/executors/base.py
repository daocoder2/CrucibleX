from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from cruciblex.domain.case import CaseSpec, InvocationBindingSpec
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

    def call_arguments(self, values: list[object]) -> tuple[list[object], dict[str, object]]:
        binding = self._binding()
        if binding.mode == "positional":
            return [value for index, value in enumerate(values) if not self._omitted(index, binding)], {}
        names = binding.names or [parameter.name for parameter in self.case.parameters]
        if len(names) < len(values):
            raise ValueError("binding names must cover every input")
        positional_indexes = set(binding.positional) if binding.mode == "mixed" else set()
        if any(index < 0 or index >= len(values) for index in positional_indexes):
            raise ValueError("mixed positional index is out of range")
        if positional_indexes and positional_indexes != set(range(max(positional_indexes) + 1)):
            raise ValueError("mixed positional indexes must form a contiguous prefix")
        if any(self.case.parameters[index].metadata.get("keyword_only") for index in positional_indexes):
            raise ValueError("keyword-only parameters cannot be positional")
        positional = [value for index, value in enumerate(values) if index in positional_indexes and not self._omitted(index, binding)]
        kwargs = {str(name): values[index] for index, name in enumerate(names) if index < len(values) and index not in positional_indexes and not self._omitted(index, binding)}
        return positional, kwargs

    def _binding(self) -> InvocationBindingSpec:
        if self.case.invocation.binding is not None:
            return self.case.invocation.binding
        legacy_binding = self.case.invocation.metadata.get("binding")
        if isinstance(legacy_binding, dict):
            return InvocationBindingSpec.model_validate(legacy_binding)
        return InvocationBindingSpec()

    def _omitted(self, index: int, binding: InvocationBindingSpec) -> bool:
        if index >= len(self.case.parameters):
            return False
        return self.case.parameters[index].name in binding.omit or index in binding.omit


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
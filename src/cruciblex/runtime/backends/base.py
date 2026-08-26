from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from cruciblex.domain.enums import BackendKind
from cruciblex.domain.node import DeviceSpec, NodeSpec


class DeviceContext(BaseModel):
    host: str
    node_name: str
    device: DeviceSpec
    output_root: Path
    env: dict[str, str] = Field(default_factory=dict)
    labels: set[str] = Field(default_factory=set)

    @property
    def backend(self) -> BackendKind:
        return self.device.backend

    @classmethod
    def from_node(cls, node: NodeSpec, device: DeviceSpec, output_root: Path) -> DeviceContext:
        return cls(
            host=node.host,
            node_name=node.display_name,
            device=device,
            output_root=output_root,
            labels=set(node.labels) | set(device.labels),
        )


class BackendRuntime(ABC):
    def prepare(self, context: DeviceContext) -> DeviceContext:
        return context

    def cleanup(self, context: DeviceContext) -> None:
        return None


class BackendRegistry:
    def __init__(self) -> None:
        self._factories: dict[BackendKind, Callable[[], BackendRuntime]] = {}

    def register(self, backend: BackendKind):
        def decorator(factory: Callable[[], BackendRuntime] | type[BackendRuntime]):
            self._factories[backend] = factory
            return factory

        return decorator

    def resolve(self, backend: BackendKind) -> BackendRuntime:
        try:
            factory = self._factories[backend]
        except KeyError as exc:
            raise KeyError(f"unknown backend runtime: {backend}") from exc
        return factory()

    def known(self) -> list[BackendKind]:
        return sorted(self._factories)


BACKEND_REGISTRY = BackendRegistry()

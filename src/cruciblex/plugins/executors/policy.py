from __future__ import annotations

from abc import ABC, abstractmethod

from cruciblex.domain.enums import BackendKind
from cruciblex.runtime.backends.base import DeviceContext


class DevicePolicy(ABC):
    @abstractmethod
    def torch_device(self, context: DeviceContext) -> str | None:
        raise NotImplementedError


class DefaultDevicePolicy(DevicePolicy):
    def torch_device(self, context: DeviceContext) -> str | None:
        if context.backend == BackendKind.CPU:
            return "cpu"
        if context.backend == BackendKind.GPU:
            return f"cuda:{context.device.id}"
        if context.backend == BackendKind.NPU:
            return f"npu:{context.device.id}"
        return None


DEFAULT_DEVICE_POLICY = DefaultDevicePolicy()

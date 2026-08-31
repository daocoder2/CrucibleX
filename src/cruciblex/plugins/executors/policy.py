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
        device_index = 0 if context.env.get("CX_DEVICE_INDEX_MODE") == "actor_local" else context.device.id
        if context.backend == BackendKind.GPU:
            return f"cuda:{device_index}"
        if context.backend == BackendKind.NPU:
            return f"npu:{device_index}"
        if context.backend == BackendKind.ACLNN:
            return f"npu:{device_index}"
        if context.backend == BackendKind.DCU:
            return f"cuda:{device_index}"
        return None


DEFAULT_DEVICE_POLICY = DefaultDevicePolicy()

from __future__ import annotations

from cruciblex.domain.enums import BackendKind
from cruciblex.runtime.backends.base import BACKEND_REGISTRY, BackendRuntime, DeviceContext


@BACKEND_REGISTRY.register(BackendKind.GPU)
class GpuBackendRuntime(BackendRuntime):
    def prepare(self, context: DeviceContext) -> DeviceContext:
        context.env["CX_BACKEND"] = BackendKind.GPU.value
        context.env["CX_DEVICE_ID"] = str(context.device.id)
        context.env.setdefault("CUDA_VISIBLE_DEVICES", str(context.device.id))
        context.env.setdefault("NVIDIA_VISIBLE_DEVICES", str(context.device.id))
        return context

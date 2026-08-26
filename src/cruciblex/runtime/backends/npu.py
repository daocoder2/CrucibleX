from __future__ import annotations

from cruciblex.domain.enums import BackendKind
from cruciblex.runtime.backends.base import BACKEND_REGISTRY, BackendRuntime, DeviceContext


@BACKEND_REGISTRY.register(BackendKind.NPU)
class NpuBackendRuntime(BackendRuntime):
    def prepare(self, context: DeviceContext) -> DeviceContext:
        context.env["CX_BACKEND"] = BackendKind.NPU.value
        context.env["CX_DEVICE_ID"] = str(context.device.id)
        context.env.setdefault("ASCEND_DEVICE_ID", str(context.device.id))
        return context


@BACKEND_REGISTRY.register(BackendKind.ACLNN)
class AclnnBackendRuntime(BackendRuntime):
    def prepare(self, context: DeviceContext) -> DeviceContext:
        context.env["CX_BACKEND"] = BackendKind.ACLNN.value
        context.env["CX_DEVICE_ID"] = str(context.device.id)
        context.env.setdefault("ASCEND_DEVICE_ID", str(context.device.id))
        return context

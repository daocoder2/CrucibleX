from __future__ import annotations

from cruciblex.domain.enums import BackendKind
from cruciblex.runtime.backends.base import BACKEND_REGISTRY, BackendRuntime, DeviceContext


@BACKEND_REGISTRY.register(BackendKind.DCU)
class DcuBackendRuntime(BackendRuntime):
    def prepare(self, context: DeviceContext) -> DeviceContext:
        context.env["CX_BACKEND"] = BackendKind.DCU.value
        context.env["CX_DEVICE_ID"] = str(context.device.id)
        context.env.setdefault("HIP_VISIBLE_DEVICES", str(context.device.id))
        context.env.setdefault("ROCR_VISIBLE_DEVICES", str(context.device.id))
        return context

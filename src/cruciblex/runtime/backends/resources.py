from __future__ import annotations

from pydantic import BaseModel, Field

from cruciblex.domain.enums import BackendKind
from cruciblex.domain.node import DeviceSpec


class RayResourceSpec(BaseModel):
    num_cpus: float | None = None
    num_gpus: float | None = None
    resources: dict[str, float] = Field(default_factory=dict)


def ray_resources_for(device: DeviceSpec) -> RayResourceSpec:
    if device.backend == BackendKind.GPU:
        return RayResourceSpec(num_gpus=1, resources=device.resources)
    if device.backend == BackendKind.NPU:
        resources = {"npu": 1.0, **device.resources}
        return RayResourceSpec(resources=resources)
    if device.backend == BackendKind.CPU:
        return RayResourceSpec(num_cpus=device.resources.get("cpu", 1.0), resources={})
    return RayResourceSpec(resources={device.backend.value: 1.0, **device.resources})

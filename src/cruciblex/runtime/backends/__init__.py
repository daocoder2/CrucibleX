from cruciblex.runtime.backends.base import BACKEND_REGISTRY, BackendRuntime, DeviceContext
from cruciblex.runtime.backends.cpu import CpuBackendRuntime
from cruciblex.runtime.backends.dcu import DcuBackendRuntime
from cruciblex.runtime.backends.gpu import GpuBackendRuntime
from cruciblex.runtime.backends.npu import AclnnBackendRuntime, NpuBackendRuntime
from cruciblex.runtime.backends.resources import RayResourceSpec, ray_resources_for

__all__ = [
    "BACKEND_REGISTRY",
    "AclnnBackendRuntime",
    "BackendRuntime",
    "CpuBackendRuntime",
    "DcuBackendRuntime",
    "DeviceContext",
    "GpuBackendRuntime",
    "NpuBackendRuntime",
    "RayResourceSpec",
    "ray_resources_for",
]

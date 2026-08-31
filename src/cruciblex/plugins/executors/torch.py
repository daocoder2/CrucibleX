from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from cruciblex.plugins.executors.policy import DEFAULT_DEVICE_POLICY, DevicePolicy
from cruciblex.runtime.executors.base import (
    EXECUTOR_REGISTRY,
    BackendExecutor,
    ExecutionNotSupportedError,
    ExecutionRequest,
)


class TorchFunctionExecutor(BackendExecutor):
    def __init__(self, device_policy: DevicePolicy | None = None) -> None:
        self._device_policy = device_policy or DEFAULT_DEVICE_POLICY

    def execute(self, request: ExecutionRequest) -> object:
        torch = self._torch()
        target = self._resolve_api(torch, request.case.invocation.api)
        torch_inputs = [self._to_torch(torch, value, request) for value in request.inputs]
        positional, keywords = request.call_arguments(torch_inputs)
        with torch.no_grad():
            output = target(*positional, **keywords)
        self.last_execution_evidence = self._device_output_evidence(output)
        return self._to_numpy(output)

    def _torch(self):
        try:
            return importlib.import_module("torch")
        except ImportError as exc:
            raise ExecutionNotSupportedError("torch executor requires the torch package") from exc

    def _resolve_api(self, torch, api: str):
        if api.startswith("torch."):
            attr_path = api.split(".")[1:]
            target: Any = torch
            for attr in attr_path:
                target = getattr(target, attr)
            return target
        raise ExecutionNotSupportedError(f"unsupported torch api path: {api}")

    def _to_torch(self, torch, value: object, request: ExecutionRequest):
        if isinstance(value, np.ndarray):
            if not value.flags.writeable:
                value = np.array(value, copy=True)
            tensor = torch.from_numpy(value)
            device = self._device_policy.torch_device(request.context) if request.context is not None else None
            if device is None:
                return tensor
            self._ensure_device_available(torch, device)
            return tensor.to(device)
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _ensure_device_available(self, torch, device: str) -> None:
        if device.startswith("cuda"):
            cuda = getattr(torch, "cuda", None)
            is_available = getattr(cuda, "is_available", None)
            if not callable(is_available) or not is_available():
                raise ExecutionNotSupportedError(f"torch device is unavailable: {device}")
        if device.startswith("npu"):
            self._ensure_torch_npu(torch, device)

    def _ensure_torch_npu(self, torch, device: str) -> None:
        try:
            importlib.import_module("torch_npu")
        except ImportError as exc:
            raise ExecutionNotSupportedError("torch NPU execution requires torch_npu") from exc
        npu = getattr(torch, "npu", None)
        is_available = getattr(npu, "is_available", None)
        if callable(is_available) and not is_available():
            raise ExecutionNotSupportedError(f"torch device is unavailable: {device}")

    def _device_output_evidence(self, value: object) -> dict[str, object]:
        if isinstance(value, (list, tuple)):
            values = [self._device_output_evidence(item) for item in value]
            return {
                "backend_output_dtype": [item.get("backend_output_dtype") for item in values],
                "backend_output_device": [item.get("backend_output_device") for item in values],
                "backend_dtype_source": "device_tensor",
            }
        return {
            "backend_output_dtype": str(getattr(value, "dtype", "unknown")),
            "backend_output_device": str(getattr(value, "device", "unknown")),
            "backend_dtype_source": "device_tensor",
        }

    def _to_numpy(self, value: object) -> object:
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, (list, tuple)):
            return [self._to_numpy(item) for item in value]
        return value


EXECUTOR_REGISTRY.register("torch")(TorchFunctionExecutor)

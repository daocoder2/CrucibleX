from __future__ import annotations

import importlib

import numpy as np

from cruciblex.plugins.executors.aclnn_bridge import ACLNN_ADAPTER_REGISTRY
from cruciblex.runtime.executors.base import (
    EXECUTOR_REGISTRY,
    BackendExecutor,
    ExecutionNotSupportedError,
    ExecutionRequest,
)


class AclnnFunctionExecutor(BackendExecutor):
    def execute(self, request: ExecutionRequest) -> object:
        if ACLNN_ADAPTER_REGISTRY.supports(request.case.invocation.api_type):
            return ACLNN_ADAPTER_REGISTRY.resolve(request.case.invocation.api_type).execute(request)
        target = self._resolve_api(request.case.invocation.api)
        positional, keywords = request.call_arguments(request.inputs)
        return self._to_numpy(target(*positional, **keywords))

    def _resolve_api(self, api: str):
        if "." not in api:
            raise ExecutionNotSupportedError(f"unsupported ACLNN api path: {api}")
        module_name, attr_name = api.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ExecutionNotSupportedError(
                f"ACLNN executor requires importable runtime module: {module_name}"
            ) from exc
        try:
            return getattr(module, attr_name)
        except AttributeError as exc:
            raise ExecutionNotSupportedError(f"unknown ACLNN api: {api}") from exc

    def _to_numpy(self, value: object) -> object:
        if isinstance(value, np.ndarray):
            return value
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy = getattr(value, "numpy", None)
        if callable(numpy):
            return numpy()
        if isinstance(value, (list, tuple)):
            return [self._to_numpy(item) for item in value]
        return value


EXECUTOR_REGISTRY.register("aclnn")(AclnnFunctionExecutor)

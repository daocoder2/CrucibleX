from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cruciblex.runtime.executors.base import (
    EXECUTOR_REGISTRY,
    BackendExecutor,
    ExecutionNotSupportedError,
    ExecutionRequest,
)


class NumpyFunctionExecutor(BackendExecutor):
    def execute(self, request: ExecutionRequest) -> object:
        api = request.case.invocation.api
        executor = _EXECUTORS.get(api) or _EXECUTORS.get(request.case.operator.name)
        if executor is None:
            raise ExecutionNotSupportedError(f"unsupported operator: {request.case.operator.name}")
        args, kwargs = request.call_arguments(request.inputs)
        return executor(args, kwargs)


_EXECUTORS: dict[str, Callable[[list[object], dict[str, object]], object]] = {
    "torch.abs": lambda args, kwargs: np.abs(_require_one(args)),
    "abs": lambda args, kwargs: np.abs(_require_one(args)),
    "torch.add": lambda args, kwargs: np.add(_require_one(args, 0), _require_one(args, 1)),
    "numpy.add": lambda args, kwargs: np.add(_require_one(args, 0), _require_one(args, 1)),
    "numpy.sum": lambda args, kwargs: np.sum(*args, **kwargs),
    "numpy.mean": lambda args, kwargs: np.mean(*args, **kwargs),
    "sum": lambda args, kwargs: np.sum(*args, **kwargs),
}


for alias in ("function", "method", "tensor", "numpy"):
    EXECUTOR_REGISTRY.register(alias)(NumpyFunctionExecutor)


def _require_one(inputs: list[object], index: int = 0) -> object:
    try:
        return inputs[index]
    except IndexError as exc:
        raise ValueError("operator requires more inputs") from exc
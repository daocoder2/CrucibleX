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
        return executor(request.inputs)


_EXECUTORS: dict[str, Callable[[list[object]], object]] = {
    "torch.abs": lambda inputs: np.abs(_require_one(inputs)),
    "abs": lambda inputs: np.abs(_require_one(inputs)),
    "torch.add": lambda inputs: np.add(_require_one(inputs, 0), _require_one(inputs, 1)),
}


for alias in ("function", "method", "tensor", "numpy"):
    EXECUTOR_REGISTRY.register(alias)(NumpyFunctionExecutor)


def _require_one(inputs: list[object], index: int = 0) -> object:
    try:
        return inputs[index]
    except IndexError as exc:
        raise ValueError("operator requires more inputs") from exc
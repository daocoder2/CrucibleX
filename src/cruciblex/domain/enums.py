from __future__ import annotations

from enum import StrEnum


class BackendKind(StrEnum):
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    ACLNN = "aclnn"
    DIST = "dist"


class TaskKind(StrEnum):
    ACCURACY = "accuracy"
    ACCURACY_LOAD = "accuracy_load"
    ACCURACY_DC = "accuracy_dc"
    PERFORMANCE_DEVICE = "performance_device"
    PERFORMANCE_DEVICE_PTA = "performance_device_pta"
    PERFORMANCE_E2E = "performance_e2e"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    MEMORY_DEVICE = "memory_device"
    RUN = "run"
    FUZZ = "fuzz"


class SchedulerKind(StrEnum):
    LOCAL = "local"
    RAY = "ray"


class ParameterKind(StrEnum):
    TENSOR = "tensor"
    TENSOR_LIST = "tensor_list"
    TENSOR_TUPLE = "tensor_tuple"
    SCALAR = "scalar"
    SCALAR_LIST = "scalar_list"
    SCALAR_TUPLE = "scalar_tuple"
    ATTRIBUTE = "attribute"
    ATTRIBUTE_LIST = "attribute_list"
    ATTRIBUTE_TUPLE = "attribute_tuple"


class ExecutionRole(StrEnum):
    CANDIDATE = "candidate"
    REFERENCE = "reference"


class ResultStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
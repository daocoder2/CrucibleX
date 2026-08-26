from __future__ import annotations

import numpy as np

from cruciblex.domain.case import ParameterSpec
from cruciblex.domain.enums import ParameterKind
from cruciblex.runtime.generation import GENERATOR_REGISTRY, GenerationRequest, InputGenerator


class DefaultInputGenerator(InputGenerator):
    def generate(self, request: GenerationRequest) -> list[object]:
        values: list[object] = []
        for parameter in request.case.parameters:
            values.append(self._generate_parameter(parameter))
        return values

    def _generate_parameter(self, parameter: ParameterSpec) -> object:
        if parameter.kind in {ParameterKind.TENSOR, ParameterKind.TENSOR_LIST, ParameterKind.TENSOR_TUPLE}:
            return self._generate_tensor(parameter)
        if parameter.kind in {ParameterKind.SCALAR, ParameterKind.SCALAR_LIST, ParameterKind.SCALAR_TUPLE}:
            return self._generate_scalar(parameter)
        if parameter.kind in {ParameterKind.ATTRIBUTE, ParameterKind.ATTRIBUTE_LIST, ParameterKind.ATTRIBUTE_TUPLE}:
            return self._generate_attribute(parameter)
        raise ValueError(f"unsupported parameter kind: {parameter.kind}")

    def _generate_tensor(self, parameter: ParameterSpec):
        dtype = self._dtype(parameter)
        shape = self._shape(parameter)
        value_range = self._tensor_range(parameter)
        total = int(np.prod(shape)) if shape else 1
        data = np.linspace(value_range[0], value_range[1], num=total, dtype=dtype)
        return data.reshape(shape) if shape else data[0]

    def _generate_scalar(self, parameter: ParameterSpec):
        dtype = self._dtype(parameter)
        value_range = self._tensor_range(parameter)
        midpoint = (value_range[0] + value_range[1]) / 2
        return dtype.type(midpoint) if hasattr(dtype, "type") else midpoint

    def _generate_attribute(self, parameter: ParameterSpec):
        if parameter.dtypes and parameter.dtypes[0] in {"bool", "true", "false"}:
            return True
        return self._generate_scalar(parameter)

    def _shape(self, parameter: ParameterSpec) -> tuple[int, ...]:
        shape = parameter.shape
        if shape is None:
            return ()
        if shape.dims:
            return tuple(shape.dims)
        if shape.dim_count and shape.dim_values:
            if len(shape.dim_count) == 1 and len(shape.dim_values) == 1:
                return tuple([shape.dim_values[0]] * shape.dim_count[0])
            if len(shape.dim_count) == len(shape.dim_values):
                return tuple(shape.dim_values)
        if shape.dim_values:
            return tuple(shape.dim_values)
        return ()

    def _tensor_range(self, parameter: ParameterSpec) -> tuple[float, float]:
        if parameter.value_range.valid:
            first = parameter.value_range.valid[0]
            if isinstance(first, list) and len(first) >= 2:
                return float(first[0]), float(first[1])
            if isinstance(first, tuple) and len(first) >= 2:
                return float(first[0]), float(first[1])
        return (-1.0, 1.0)

    def _dtype(self, parameter: ParameterSpec):
        name = parameter.dtypes[0] if parameter.dtypes else "fp32"
        mapping = {
            "fp64": np.float64,
            "fp32": np.float32,
            "fp16": np.float16,
            "bf16": np.float32,
            "int64": np.int64,
            "int32": np.int32,
            "int16": np.int16,
            "int8": np.int8,
            "uint8": np.uint8,
            "bool": np.bool_,
        }
        return mapping.get(name, np.float32)


GENERATOR_REGISTRY.register("default")(DefaultInputGenerator)

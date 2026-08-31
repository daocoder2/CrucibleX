from __future__ import annotations

import numpy as np

from cruciblex.domain.case import ParameterSpec, ShapeSpec, ValueRange
from cruciblex.domain.enums import ParameterKind
from cruciblex.runtime.generation import GENERATOR_REGISTRY, GenerationRequest, InputGenerator


class DefaultInputGenerator(InputGenerator):
    def generate(self, request: GenerationRequest) -> list[object]:
        values: list[object] = []
        for parameter in request.case.parameters:
            values.append(self._generate_parameter(parameter))
        return values

    def _generate_parameter(self, parameter: ParameterSpec) -> object:
        if parameter.kind == ParameterKind.TENSOR:
            return self._generate_tensor(parameter)
        if parameter.kind == ParameterKind.TENSOR_LIST:
            return self._generate_collection(parameter, self._generate_tensor, tuple_result=False)
        if parameter.kind == ParameterKind.TENSOR_TUPLE:
            return self._generate_collection(parameter, self._generate_tensor, tuple_result=True)
        if parameter.kind == ParameterKind.SCALAR:
            return self._generate_scalar(parameter)
        if parameter.kind == ParameterKind.SCALAR_LIST:
            return self._generate_collection(parameter, self._generate_scalar, tuple_result=False)
        if parameter.kind == ParameterKind.SCALAR_TUPLE:
            return self._generate_collection(parameter, self._generate_scalar, tuple_result=True)
        if parameter.kind == ParameterKind.ATTRIBUTE:
            return self._generate_attribute(parameter)
        if parameter.kind == ParameterKind.ATTRIBUTE_LIST:
            return self._generate_collection(parameter, self._generate_attribute, tuple_result=False)
        if parameter.kind == ParameterKind.ATTRIBUTE_TUPLE:
            return self._generate_collection(parameter, self._generate_attribute, tuple_result=True)
        raise ValueError(f"unsupported parameter kind: {parameter.kind}")

    def _generate_collection(self, parameter: ParameterSpec, generator, tuple_result: bool) -> object:
        items = parameter.metadata.get("items")
        if isinstance(items, list) and items:
            values = [self._generate_parameter(self._parameter_from_item(parameter, item)) for item in items]
            return tuple(values) if tuple_result else values

        length = self._collection_length(parameter, tuple_result=tuple_result)
        values = []
        for index in range(length):
            values.append(generator(self._parameter_for_index(parameter, index)))
        return tuple(values) if tuple_result else values

    def _parameter_from_item(self, parameter: ParameterSpec, item: object) -> ParameterSpec:
        if not isinstance(item, dict):
            return self._parameter_for_index(parameter, 0)
        metadata = {**parameter.metadata, **dict(item.get("metadata") or {})}
        metadata.pop("items", None)
        kind = item.get("kind") or self._element_kind(parameter.kind)
        return ParameterSpec(
            name=item.get("name", parameter.name),
            kind=ParameterKind(kind),
            required=bool(item.get("required", parameter.required)),
            dtypes=list(item.get("dtypes", parameter.dtypes)),
            shape=ShapeSpec.model_validate(item["shape"]) if item.get("shape") is not None else parameter.shape,
            value_range=ValueRange.model_validate(item.get("value_range") or parameter.value_range.model_dump(mode="json")),
            requires_grad=bool(item.get("requires_grad", parameter.requires_grad)),
            metadata=metadata,
        )

    def _parameter_for_index(self, parameter: ParameterSpec, index: int) -> ParameterSpec:
        metadata = dict(parameter.metadata)
        shape = parameter.shape
        item_shapes = metadata.get("item_shapes")
        if isinstance(item_shapes, list) and item_shapes:
            selected_shape = item_shapes[index % len(item_shapes)]
            if isinstance(selected_shape, list) and all(isinstance(item, int) for item in selected_shape):
                shape = ShapeSpec(dims=[int(item) for item in selected_shape])

        item_values = metadata.get("item_values")
        if isinstance(item_values, list) and item_values:
            metadata["selected_random_value"] = item_values[index % len(item_values)]

        item_dtypes = metadata.get("item_dtypes")
        dtypes = parameter.dtypes
        if isinstance(item_dtypes, list) and item_dtypes:
            dtypes = [str(item_dtypes[index % len(item_dtypes)])]

        return parameter.model_copy(update={"shape": shape, "dtypes": dtypes, "metadata": metadata})

    def _collection_length(self, parameter: ParameterSpec, tuple_result: bool) -> int:
        metadata = parameter.metadata
        candidates = ["tuple_length" if tuple_result else "list_length", "length", "item_count"]
        for key in candidates:
            value = metadata.get(key)
            if value is not None:
                return max(0, int(value))
        return 2

    def _element_kind(self, kind: ParameterKind) -> ParameterKind:
        if kind in {ParameterKind.TENSOR_LIST, ParameterKind.TENSOR_TUPLE}:
            return ParameterKind.TENSOR
        if kind in {ParameterKind.SCALAR_LIST, ParameterKind.SCALAR_TUPLE}:
            return ParameterKind.SCALAR
        if kind in {ParameterKind.ATTRIBUTE_LIST, ParameterKind.ATTRIBUTE_TUPLE}:
            return ParameterKind.ATTRIBUTE
        return kind

    def _generate_tensor(self, parameter: ParameterSpec):
        dtype = self._dtype(parameter)
        shape = self._shape(parameter)
        policy_value = self._policy_value(parameter, dtype)
        if policy_value is not None:
            data = np.full(shape or (), policy_value, dtype=dtype)
            return data if shape else data[()]
        total = int(np.prod(shape)) if shape else 1
        policy_values = self._boundary_values(parameter, dtype, total)
        if policy_values is not None:
            return policy_values.reshape(shape) if shape else policy_values[0]
        value_range = self._tensor_range(parameter)
        data = self._distribution_values(parameter, value_range, total, dtype)
        return data.reshape(shape) if shape else data[0]

    def _generate_scalar(self, parameter: ParameterSpec):
        dtype = self._dtype(parameter)
        policy_value = self._policy_value(parameter, dtype)
        if policy_value is not None:
            return dtype.type(policy_value) if hasattr(dtype, "type") else policy_value
        value_range = self._tensor_range(parameter)
        policy = parameter.metadata.get("value_policy")
        if isinstance(policy, dict) and policy.get("kind") in {"uniform", "normal", "sparsity"}:
            return self._distribution_values(parameter, value_range, 1, dtype)[0]
        midpoint = (value_range[0] + value_range[1]) / 2
        return dtype.type(midpoint) if hasattr(dtype, "type") else midpoint

    def _generate_attribute(self, parameter: ParameterSpec):
        metadata = parameter.metadata
        if "default_value" in metadata:
            return metadata["default_value"]
        enum_values = metadata.get("enum_values") or metadata.get("optional_values")
        if isinstance(enum_values, list) and enum_values:
            return enum_values[0]
        selected_value = metadata.get("selected_random_value")
        if selected_value is not None and not isinstance(selected_value, (list, tuple)):
            return selected_value
        if parameter.value_range.valid:
            first = parameter.value_range.valid[0]
            if not isinstance(first, (list, tuple)):
                return first
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

    def _boundary_values(self, parameter: ParameterSpec, dtype, total: int):
        policy = parameter.metadata.get("value_policy")
        if not isinstance(policy, dict):
            return None
        kind = policy.get("kind")
        if kind == "integer_bounds" and np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return np.resize(np.asarray([info.min, info.max], dtype=dtype), total)
        if kind == "float_bounds" and np.issubdtype(dtype, np.floating):
            info = np.finfo(dtype)
            scale = min(1.0, max(1e-6, float(policy.get("scale", 0.5))))
            return np.resize(np.asarray([-info.max * scale, 0, info.max * scale], dtype=dtype), total)
        return None

    def _distribution_values(self, parameter: ParameterSpec, value_range: tuple[float, float], total: int, dtype):
        policy = parameter.metadata.get("value_policy") or {}
        rng = np.random.default_rng(int(parameter.metadata.get("value_policy_seed", 0)))
        kind = policy.get("kind")
        if kind == "uniform":
            data = rng.uniform(float(policy.get("low", value_range[0])), float(policy.get("high", value_range[1])), total)
        elif kind == "normal":
            data = rng.normal(float(policy.get("mean", 0)), float(policy.get("std", 1)), total)
        elif kind == "sparsity":
            ratio = min(1.0, max(0.0, float(policy.get("ratio", 0.5))))
            data = rng.uniform(value_range[0], value_range[1], total)
            data[rng.random(total) < ratio] = 0
        else:
            data = np.linspace(value_range[0], value_range[1], num=total)
        return np.asarray(data, dtype=dtype)

    def _policy_value(self, parameter: ParameterSpec, dtype):
        policy = parameter.metadata.get("value_policy")
        if not isinstance(policy, dict):
            return None
        kind = policy.get("kind")
        if kind == "constant":
            return policy.get("value", 0)
        if kind == "zero":
            return 0
        if kind == "one":
            return 1
        if kind == "nan" and np.issubdtype(dtype, np.floating):
            return np.nan
        if kind == "inf" and np.issubdtype(dtype, np.floating):
            return np.inf
        return None

    def _tensor_range(self, parameter: ParameterSpec) -> tuple[float, float]:
        selected_random = parameter.metadata.get("selected_random_value")
        if isinstance(selected_random, list) and len(selected_random) >= 2:
            return float(selected_random[0]), float(selected_random[1])
        if isinstance(selected_random, tuple) and len(selected_random) >= 2:
            return float(selected_random[0]), float(selected_random[1])
        if selected_random is not None and not isinstance(selected_random, (list, tuple)):
            value = float(selected_random)
            return value, value
        selected_invalid = parameter.metadata.get("selected_invalid_value")
        if isinstance(selected_invalid, list) and len(selected_invalid) >= 2:
            return float(selected_invalid[0]), float(selected_invalid[1])
        if isinstance(selected_invalid, tuple) and len(selected_invalid) >= 2:
            return float(selected_invalid[0]), float(selected_invalid[1])
        if selected_invalid is not None and not isinstance(selected_invalid, (list, tuple)):
            value = float(selected_invalid)
            return value, value
        if parameter.value_range.valid:
            first = parameter.value_range.valid[0]
            if isinstance(first, list) and len(first) >= 2:
                return float(first[0]), float(first[1])
            if isinstance(first, tuple) and len(first) >= 2:
                return float(first[0]), float(first[1])
            if not isinstance(first, (list, tuple)):
                value = float(first)
                return value, value
        return (-1.0, 1.0)

    def _dtype(self, parameter: ParameterSpec):
        name = parameter.dtypes[0] if parameter.dtypes else "fp32"
        mapping = {
            "fp64": np.float64,
            "float64": np.float64,
            "fp32": np.float32,
            "float32": np.float32,
            "fp16": np.float16,
            "float16": np.float16,
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

from __future__ import annotations

import numpy as np

from cruciblex.domain.case import ParameterSpec, ShapeSpec, ValueRange
from cruciblex.domain.enums import ParameterKind
from cruciblex.generation.dtypes import quantize_bf16_reference, validate_value_policy
from cruciblex.runtime.generation import GENERATOR_REGISTRY, GenerationRequest, InputGenerator


class DefaultInputGenerator(InputGenerator):
    def generate(self, request: GenerationRequest) -> list[object]:
        values: list[object] = []
        for parameter in request.case.parameters:
            values.append(self._generate_parameter(parameter))
        return values

    def _generate_parameter(self, parameter: ParameterSpec) -> object:
        if parameter.values is not None:
            return self._generate_exact_values(parameter)
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

    def _generate_exact_values(self, parameter: ParameterSpec) -> object:
        if parameter.kind == ParameterKind.TENSOR:
            value = np.asarray(parameter.values, dtype=self._dtype(parameter))
            expected_shape = self._shape(parameter)
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"exact values shape {list(value.shape)} does not match parameter {parameter.name} shape {list(expected_shape)}")
            return self._apply_dtype_contract(value, parameter)
        if parameter.kind == ParameterKind.SCALAR:
            dtype = self._dtype(parameter)
            return dtype.type(parameter.values) if hasattr(dtype, "type") else parameter.values
        if parameter.kind == ParameterKind.ATTRIBUTE:
            return parameter.values
        if parameter.kind in {ParameterKind.TENSOR_LIST, ParameterKind.TENSOR_TUPLE}:
            if not isinstance(parameter.values, (list, tuple)):
                raise ValueError(f"exact values for {parameter.kind.value} must be a list or tuple")
            item_kind = ParameterKind.TENSOR
            values = [self._generate_exact_values(parameter.model_copy(update={"kind": item_kind, "values": value})) for value in parameter.values]
            return tuple(values) if parameter.kind == ParameterKind.TENSOR_TUPLE else values
        if parameter.kind in {ParameterKind.SCALAR_LIST, ParameterKind.SCALAR_TUPLE, ParameterKind.ATTRIBUTE_LIST, ParameterKind.ATTRIBUTE_TUPLE}:
            if not isinstance(parameter.values, (list, tuple)):
                raise ValueError(f"exact values for {parameter.kind.value} must be a list or tuple")
            return tuple(parameter.values) if parameter.kind in {ParameterKind.SCALAR_TUPLE, ParameterKind.ATTRIBUTE_TUPLE} else list(parameter.values)
        raise ValueError(f"exact values are unsupported for parameter kind: {parameter.kind}")

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
            values=item.get("values", parameter.values),
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
        policy = parameter.metadata.get("value_policy")
        if isinstance(policy, dict):
            validation = validate_value_policy(policy, self._declared_dtype(parameter))
            if validation["rejected"]:
                raise ValueError(f"{validation['rejected']}: {validation['requested']} for {self._declared_dtype(parameter)}")
        shape = self._shape(parameter)
        policy_value = self._policy_value(parameter, dtype)
        if policy_value is not None:
            data = np.full(shape or (), policy_value, dtype=dtype)
            return self._apply_layout_policy(data if shape else data[()], parameter)
        matrix = self._matrix_profile(parameter, shape, dtype)
        if matrix is not None:
            return self._apply_layout_policy(matrix, parameter)
        total = int(np.prod(shape)) if shape else 1
        policy_values = self._boundary_values(parameter, dtype, total)
        if policy_values is not None:
            data = policy_values.reshape(shape) if shape else policy_values[0]
            return self._apply_layout_policy(data, parameter)
        value_range = self._tensor_range(parameter)
        data = self._distribution_values(parameter, value_range, total, dtype)
        return self._apply_layout_policy(data.reshape(shape) if shape else data[0], parameter)

    def _apply_layout_policy(self, value, parameter: ParameterSpec):
        policy = parameter.metadata.get("shape_policy")
        if not isinstance(policy, dict) or not isinstance(value, np.ndarray):
            return self._apply_dtype_contract(value, parameter)
        storage_shape = policy.get("storage_shape")
        if isinstance(storage_shape, list) and policy.get("strides") is None and all(isinstance(dim, int) and dim >= 0 for dim in storage_shape):
            storage = np.zeros(tuple(storage_shape), dtype=value.dtype)
            slices = policy.get("slice")
            if not isinstance(slices, list):
                raise ValueError("storage_shape requires a slice list")
            index = tuple(slice(*(item if isinstance(item, list) else [item])) for item in slices)
            view = storage[index]
            if view.shape != value.shape:
                raise ValueError(f"storage slice shape {list(view.shape)} does not match generated shape {list(value.shape)}")
            view[...] = value
            value = view
        strides = policy.get("strides")
        if strides is not None:
            if not isinstance(storage_shape, list) or not isinstance(strides, list):
                raise ValueError("strides requires storage_shape")
            if len(strides) != value.ndim:
                raise ValueError("strides rank must match generated tensor rank")
            storage = np.zeros(tuple(storage_shape), dtype=value.dtype)
            offset = int(policy.get("storage_offset", 0)) * value.dtype.itemsize
            byte_strides = tuple(int(stride) * value.dtype.itemsize for stride in strides)
            try:
                view = np.ndarray(value.shape, dtype=value.dtype, buffer=storage, offset=offset, strides=byte_strides)
            except (TypeError, ValueError) as exc:
                raise ValueError("storage_offset/strides exceed storage_shape") from exc
            view[...] = value
            value = view
        if policy.get("non_contiguous"):
            if value.ndim < 2:
                raise ValueError("non_contiguous shape_policy requires rank at least 2")
            value = np.swapaxes(value, -1, -2)
        return self._apply_dtype_contract(value, parameter)

    def _matrix_profile(self, parameter: ParameterSpec, shape: list[int], dtype):
        policy = parameter.metadata.get("value_policy")
        if not isinstance(policy, dict) or policy.get("kind") != "matrix_profile":
            return None
        if len(shape) != 2:
            raise ValueError("matrix_profile requires a rank-2 tensor shape")
        if not np.issubdtype(dtype, np.floating):
            raise ValueError("matrix_profile requires a floating dtype")
        profile = str(policy.get("profile", "well_conditioned"))
        rows, columns = shape
        rank_limit = min(rows, columns)
        if rank_limit == 0:
            return np.empty(shape, dtype=dtype)
        rng = np.random.default_rng(int(parameter.metadata.get("value_policy_seed", 0)))
        if profile == "well_conditioned":
            left, _ = np.linalg.qr(rng.normal(size=(rows, rank_limit)))
            right, _ = np.linalg.qr(rng.normal(size=(columns, rank_limit)))
            condition = float(policy.get("condition_number", 4.0))
            if condition < 1.0:
                raise ValueError("well_conditioned matrix_profile condition_number must be at least 1")
            singular = np.linspace(1.0, 1.0 / condition, rank_limit)
            return ((left * singular) @ right.T).astype(dtype)
        if profile == "identity":
            return np.eye(rows, columns, dtype=dtype)
        if profile == "diagonal":
            diagonal = rng.normal(size=min(rows, columns))
            return np.diag(diagonal).astype(dtype) if rows == columns else np.eye(rows, columns, dtype=dtype) * diagonal[:1]
        if profile == "symmetric":
            if rows != columns:
                raise ValueError("symmetric matrix_profile requires a square shape")
            data = rng.normal(size=shape)
            return ((data + data.T) / 2).astype(dtype)
        if profile == "rank_deficient":
            rank = int(policy.get("rank", max(1, rank_limit - 1)))
            if rank < 0 or rank >= rank_limit:
                raise ValueError("rank_deficient matrix_profile rank must satisfy 0 <= rank < min(shape)")
            if rank == 0:
                return np.zeros(shape, dtype=dtype)
            return (rng.normal(size=(rows, rank)) @ rng.normal(size=(rank, columns))).astype(dtype)
        raise ValueError(f"unknown matrix_profile: {profile}")

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

    def _boundary_set(self, values, dtype, total: int):
        if np.issubdtype(dtype, np.bool_):
            return np.resize(np.asarray([bool(value) for value in values], dtype=dtype), total)
        component_dtype = np.empty((), dtype=dtype).real.dtype if np.issubdtype(dtype, np.complexfloating) else dtype
        info = np.iinfo(component_dtype) if np.issubdtype(component_dtype, np.integer) else np.finfo(component_dtype)
        resolved = []
        for value in values:
            if value == "min":
                resolved.append(info.min)
            elif value == "max":
                resolved.append(info.max)
            elif value in {"-max", "extreme_negative"}:
                resolved.append(-info.max)
            elif value in {"-one", "one"}:
                resolved.append(-1 if value == "-one" else 1)
            elif value in {"-zero", "zero"}:
                resolved.append(0)
            elif value == "subnormal":
                resolved.append(np.nextafter(dtype(0), dtype(1), dtype=dtype))
            elif value == "-subnormal":
                resolved.append(-np.nextafter(dtype(0), dtype(1), dtype=dtype))
            elif value in {"inf", "-inf", "nan"} and (np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating)):
                resolved.append({"inf": np.inf, "-inf": -np.inf, "nan": np.nan}[value])
            else:
                resolved.append(value)
        return np.resize(np.asarray(resolved, dtype=dtype), total)

    def _boundary_values(self, parameter: ParameterSpec, dtype, total: int):
        policy = parameter.metadata.get("value_policy")
        if not isinstance(policy, dict):
            return None
        kind = policy.get("kind")
        if kind == "boundary_set":
            return self._boundary_set(policy.get("values", []), dtype, total)
        if kind == "integer_bounds" and np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return np.resize(np.asarray([info.min, info.max], dtype=dtype), total)
        if kind in {"float_bounds", "extreme"} and np.issubdtype(dtype, np.floating):
            info = np.finfo(dtype)
            scale = min(1.0, max(1e-6, float(policy.get("scale", 0.5))))
            return np.resize(np.asarray([-info.max * scale, 0, info.max * scale], dtype=dtype), total)
        if kind == "subnormal" and np.issubdtype(dtype, np.floating):
            smallest = np.nextafter(dtype(0), dtype(1), dtype=dtype)
            return np.resize(np.asarray([-smallest, 0, smallest], dtype=dtype), total)
        return None

    def _distribution_values(self, parameter: ParameterSpec, value_range: tuple[float, float], total: int, dtype):
        policy = parameter.metadata.get("value_policy") or {}
        rng = np.random.default_rng(int(parameter.metadata.get("value_policy_seed", 0)))
        kind = policy.get("kind")
        if kind == "uniform":
            data = rng.uniform(float(policy.get("low", value_range[0])), float(policy.get("high", value_range[1])), total)
        elif kind == "normal":
            data = rng.normal(float(policy.get("mean", 0)), float(policy.get("std", 1)), total)
        elif kind == "exponential":
            data = rng.exponential(float(policy.get("scale", 1.0)), total)
            if policy.get("signed"):
                data *= rng.choice([-1.0, 1.0], total)
        elif kind == "complex_normal":
            data = rng.normal(float(policy.get("mean", 0)), float(policy.get("std", 1)), total) + 1j * rng.normal(0, float(policy.get("imag_std", policy.get("std", 1))), total)
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
        if kind == "nan" and (np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating)):
            return np.nan
        if kind == "inf" and (np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating)):
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

    def _declared_dtype(self, parameter: ParameterSpec) -> str:
        return str(parameter.dtypes[0] if parameter.dtypes else "fp32").lower()

    def _apply_dtype_contract(self, value, parameter: ParameterSpec):
        if self._declared_dtype(parameter) == "bf16":
            return quantize_bf16_reference(value)
        return value

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
            "complex64": np.complex64,
            "complex128": np.complex128,
            "c64": np.complex64,
            "c128": np.complex128,
            "int64": np.int64,
            "int32": np.int32,
            "int16": np.int16,
            "int8": np.int8,
            "uint8": np.uint8,
            "bool": np.bool_,
        }
        return mapping.get(name, np.float32)


GENERATOR_REGISTRY.register("default")(DefaultInputGenerator)

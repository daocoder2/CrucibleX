from __future__ import annotations

import random
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass

from cruciblex.domain.case import CaseSpec, ParameterSpec, ShapeSpec
from cruciblex.domain.enums import ParameterKind


@dataclass(frozen=True, slots=True)
class GenerationContext:
    seed: int
    case_index: int
    source_case_id: int
    task: str | None = None


class ConstraintPlugin(ABC):
    def after_parameter(self, parameter: ParameterSpec, context: GenerationContext) -> ParameterSpec:
        return parameter

    def after_case(self, case: CaseSpec, context: GenerationContext) -> CaseSpec:
        return case


class ConstraintRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], ConstraintPlugin]] = {}

    def register(self, name: str):
        def decorator(factory: Callable[[], ConstraintPlugin] | type[ConstraintPlugin]):
            self._factories[name] = factory
            return factory

        return decorator

    def resolve(self, name: str) -> ConstraintPlugin:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(f"unknown constraint plugin: {name}") from exc
        return factory()

    def known(self) -> list[str]:
        return sorted(self._factories)


CONSTRAINT_REGISTRY = ConstraintRegistry()


class LinkedParametersConstraint(ConstraintPlugin):
    def after_case(self, case: CaseSpec, context: GenerationContext) -> CaseSpec:
        parameter_map = {parameter.name: parameter for parameter in case.parameters if parameter.name}
        updated: list[ParameterSpec] = []
        for parameter in case.parameters:
            metadata = dict(parameter.metadata)
            dtype_source = metadata.get("same_dtype_as")
            shape_source = metadata.get("same_shape_as")
            if dtype_source is not None:
                source = parameter_map.get(str(dtype_source))
                if source is not None and source.dtypes:
                    metadata.setdefault("resolved_dtype_from", str(dtype_source))
                    parameter = parameter.model_copy(update={"dtypes": list(source.dtypes), "metadata": metadata})
            if shape_source is not None:
                source = parameter_map.get(str(shape_source))
                if source is not None and source.shape is not None:
                    metadata.setdefault("resolved_shape_from", str(shape_source))
                    parameter = parameter.model_copy(update={"shape": source.shape, "metadata": metadata})
            updated.append(parameter)
        return case.model_copy(update={"parameters": updated})


class ShapeRelationshipsConstraint(ConstraintPlugin):
    """Resolve declarative shape relationships after individual parameters are generated."""

    def after_case(self, case: CaseSpec, context: GenerationContext) -> CaseSpec:
        parameters = {parameter.name: parameter for parameter in case.parameters if parameter.name}
        updated: list[ParameterSpec] = []
        for parameter in case.parameters:
            relation = parameter.metadata.get("shape_relationship")
            if not isinstance(relation, dict) or parameter.shape is None:
                updated.append(parameter)
                continue
            source_name = relation.get("source")
            source = parameters.get(str(source_name))
            source_dims = _shape_dims(source.shape if source else None)
            target_dims = _shape_dims(parameter.shape)
            kind = relation.get("kind")
            resolved = _resolve_shape_relationship(kind, source_dims, target_dims, relation)
            metadata = dict(parameter.metadata)
            if resolved is not None:
                metadata["resolved_shape_relationship"] = str(kind)
                parameter = parameter.model_copy(update={"shape": ShapeSpec(dims=resolved), "metadata": metadata})
            updated.append(parameter)
            if parameter.name:
                parameters[parameter.name] = parameter
        return case.model_copy(update={"parameters": updated})


def _shape_dims(shape: ShapeSpec | None) -> list[int] | None:
    if shape is None:
        return None
    if shape.dims is not None:
        return list(shape.dims)
    if len(shape.dim_count) == 1 and len(shape.dim_values) == shape.dim_count[0]:
        return list(shape.dim_values)
    return None


def _resolve_shape_relationship(kind: object, source: list[int] | None, target: list[int] | None, relation: dict[object, object]) -> list[int] | None:
    if kind == "rank_range":
        if target is None:
            return None
        minimum = max(0, int(relation.get("min_rank", 0)))
        maximum = max(minimum, int(relation.get("max_rank", minimum)))
        rank = min(max(len(target), minimum), maximum)
        if len(target) > rank:
            return target[:rank]
        return [*target, *([1] * (rank - len(target)))]
    if kind == "transpose_of":
        if not source:
            return None
        axes = relation.get("axes")
        if axes is None:
            return list(reversed(source))
        if not isinstance(axes, list) or sorted(axes) != list(range(len(source))):
            return target
        return [source[index] for index in axes]
    if kind == "dimension_alias":
        if not source or not target:
            return None
        source_dimension = int(relation.get("source_dimension", -1)) % len(source)
        dimension = int(relation.get("dimension", -1)) % len(target)
        target[dimension] = source[source_dimension]
        return target
    if kind == "divisible_by":
        if not target:
            return None
        divisor = int(relation.get("divisor", 1))
        if divisor < 1:
            return target
        dimension = int(relation.get("dimension", -1))
        dimension %= len(target)
        target[dimension] = max(divisor, ((target[dimension] + divisor - 1) // divisor) * divisor)
        return target
    if not source:
        return None
    if kind == "same_rank":
        return target if target is not None and len(target) == len(source) else list(source)
    if kind == "same_numel":
        if target:
            prefix = 1
            for value in target[:-1]:
                prefix *= value
            source_numel = 1
            for value in source:
                source_numel *= value
            if prefix and source_numel % prefix == 0:
                return [*target[:-1], source_numel // prefix]
        return list(source)
    if kind == "broadcastable_with":
        values = list(target or source)
        if len(values) > len(source):
            values = values[-len(source):]
        values = [1] * (len(source) - len(values)) + values
        return [value if value in {1, source[index]} else 1 for index, value in enumerate(values)]
    if kind == "dim_equal":
        if not target:
            return None
        source_dim = int(relation.get("source_dimension", -1)) % len(source)
        target_dim = int(relation.get("dimension", -1)) % len(target)
        target[target_dim] = source[source_dim]
        return target
    return None


class BoundaryCoverageConstraint(ConstraintPlugin):
    def after_parameter(self, parameter: ParameterSpec, context: GenerationContext) -> ParameterSpec:
        metadata = dict(parameter.metadata)
        if not metadata.get("cycle_on_index"):
            return parameter
        if parameter.kind not in {ParameterKind.TENSOR, ParameterKind.TENSOR_LIST, ParameterKind.TENSOR_TUPLE, ParameterKind.SCALAR, ParameterKind.SCALAR_LIST, ParameterKind.SCALAR_TUPLE, ParameterKind.ATTRIBUTE, ParameterKind.ATTRIBUTE_LIST, ParameterKind.ATTRIBUTE_TUPLE}:
            return parameter
        dtypes = list(metadata.get("boundary_dtypes") or parameter.dtypes)
        if dtypes:
            parameter = parameter.model_copy(update={"dtypes": [dtypes[context.case_index % len(dtypes)]], "metadata": metadata})
        boundary_values = metadata.get("boundary_values")
        if isinstance(boundary_values, list) and boundary_values:
            selected = boundary_values[context.case_index % len(boundary_values)]
            if parameter.kind in {ParameterKind.TENSOR, ParameterKind.TENSOR_LIST, ParameterKind.TENSOR_TUPLE}:
                shape = _shape_from_boundary(selected, parameter.shape)
                if shape is not None:
                    parameter = parameter.model_copy(update={"shape": shape, "metadata": metadata})
        return parameter


def _shape_from_boundary(selected: object, current: ShapeSpec | None) -> ShapeSpec | None:
    if isinstance(selected, list) and all(isinstance(item, int) for item in selected):
        return ShapeSpec(dims=[int(item) for item in selected])
    if isinstance(selected, tuple) and all(isinstance(item, int) for item in selected):
        return ShapeSpec(dims=[int(item) for item in selected])
    return current


class RandomCoverageConstraint(ConstraintPlugin):
    def after_parameter(self, parameter: ParameterSpec, context: GenerationContext) -> ParameterSpec:
        metadata = dict(parameter.metadata)
        if not metadata.get("random_coverage"):
            return parameter
        rng = random.Random(context.seed + context.case_index + sum(ord(ch) for ch in parameter.name or ""))
        dtypes = list(metadata.get("random_dtypes") or parameter.dtypes)
        if dtypes:
            parameter = parameter.model_copy(update={"dtypes": [dtypes[rng.randrange(len(dtypes))]], "metadata": metadata})
        if parameter.kind in {ParameterKind.TENSOR, ParameterKind.TENSOR_LIST, ParameterKind.TENSOR_TUPLE}:
            shape_choices = metadata.get("random_shapes")
            if isinstance(shape_choices, list) and shape_choices:
                selected_shape = shape_choices[rng.randrange(len(shape_choices))]
                shape = _shape_from_boundary(selected_shape, parameter.shape)
                if shape is not None:
                    parameter = parameter.model_copy(update={"shape": shape, "metadata": metadata})
        value_choices = metadata.get("random_values")
        if isinstance(value_choices, list) and value_choices:
            selected_value = value_choices[rng.randrange(len(value_choices))]
            metadata["selected_random_value"] = selected_value
            parameter = parameter.model_copy(update={"metadata": metadata})
        return parameter


class ValuePolicyConstraint(ConstraintPlugin):
    def after_parameter(self, parameter: ParameterSpec, context: GenerationContext) -> ParameterSpec:
        policy = parameter.metadata.get("value_policy")
        if not isinstance(policy, dict):
            return parameter
        metadata = dict(parameter.metadata)
        kind = str(policy.get("kind", ""))
        dtype = str(parameter.dtypes[0]) if parameter.dtypes else ""
        if kind in {"nan", "inf"} and not dtype.startswith(("fp", "float", "bf16")):
            metadata["resolved_value_policy"] = "filtered_non_floating_dtype"
        elif kind in {"uniform", "normal", "sparsity"}:
            metadata["value_policy_seed"] = context.seed + context.case_index + sum(ord(ch) for ch in parameter.name or "")
            metadata["resolved_value_policy"] = kind
        return parameter.model_copy(update={"metadata": metadata})


class OperatorFactsConstraint(ConstraintPlugin):
    """Project declarative operator facts into parameter-level generation policies."""

    def after_case(self, case: CaseSpec, context: GenerationContext) -> CaseSpec:
        facts = case.generation.metadata.get("operator_facts")
        if not isinstance(facts, dict):
            return case
        facts_by_name = facts.get("parameters")
        if not isinstance(facts_by_name, dict):
            return case
        updated: list[ParameterSpec] = []
        groups: dict[str, str] = {}
        for parameter in case.parameters:
            fact = facts_by_name.get(parameter.name)
            if not isinstance(fact, dict):
                updated.append(parameter)
                continue
            metadata = dict(parameter.metadata)
            for key in ("dtype_policy", "value_policy", "shape_policy", "dtype_promotion"):
                if key in fact:
                    metadata.setdefault(key, fact[key])
            shape_policy = fact.get("shape_policy")
            if isinstance(shape_policy, dict):
                rank_range = shape_policy.get("rank_range")
                if isinstance(rank_range, list) and len(rank_range) == 2:
                    metadata.setdefault("shape_relationship", {"kind": "rank_range", "min_rank": rank_range[0], "max_rank": rank_range[1]})
                broadcast_group = shape_policy.get("broadcast_group")
                if isinstance(broadcast_group, str):
                    source = groups.setdefault(broadcast_group, parameter.name or "")
                    if source and source != parameter.name:
                        metadata.setdefault("shape_relationship", {"kind": "broadcastable_with", "source": source})
            dtypes = fact.get("dtypes") or fact.get("dtype_families")
            if isinstance(dtypes, list) and not parameter.dtypes:
                parameter = parameter.model_copy(update={"dtypes": [str(dtype) for dtype in dtypes]})
            metadata["resolved_operator_facts"] = True
            updated.append(parameter.model_copy(update={"metadata": metadata}))
        return case.model_copy(update={"parameters": updated})


class DtypePolicyConstraint(ConstraintPlugin):
    def after_parameter(self, parameter: ParameterSpec, context: GenerationContext) -> ParameterSpec:
        policy = parameter.metadata.get("dtype_policy")
        if not isinstance(policy, dict):
            return parameter
        groups = policy.get("groups")
        group = policy.get("group")
        candidates = groups.get(group) if isinstance(groups, dict) and group is not None else parameter.dtypes
        if not isinstance(candidates, list) or not candidates:
            return parameter
        allowed = policy.get("allowed")
        if isinstance(allowed, list):
            candidates = [dtype for dtype in candidates if dtype in allowed]
        denied = policy.get("denied")
        if isinstance(denied, list):
            candidates = [dtype for dtype in candidates if dtype not in denied]
        backend = policy.get("backend")
        backend_allowed = policy.get("backend_allowed")
        if isinstance(backend_allowed, dict) and backend is not None:
            selected = backend_allowed.get(str(backend))
            if isinstance(selected, list):
                candidates = [dtype for dtype in candidates if dtype in selected]
        backend_denied = policy.get("backend_denied")
        if isinstance(backend_denied, dict) and backend is not None:
            selected = backend_denied.get(str(backend))
            if isinstance(selected, list):
                candidates = [dtype for dtype in candidates if dtype not in selected]
        if not candidates:
            return parameter
        metadata = dict(parameter.metadata)
        metadata["resolved_dtype_policy"] = str(group or "explicit")
        return parameter.model_copy(update={"dtypes": [str(candidates[context.case_index % len(candidates)])], "metadata": metadata})


_DTYPE_BYTES = {
    "fp64": 8,
    "float64": 8,
    "int64": 8,
    "fp32": 4,
    "float32": 4,
    "bf16": 4,
    "int32": 4,
    "fp16": 2,
    "float16": 2,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}


class DtypePromotionConstraint(ConstraintPlugin):
    def after_case(self, case: CaseSpec, context: GenerationContext) -> CaseSpec:
        parameters = list(case.parameters)
        by_name = {parameter.name: index for index, parameter in enumerate(parameters) if parameter.name}
        for index, parameter in enumerate(parameters):
            relation = parameter.metadata.get("dtype_promotion")
            if not isinstance(relation, dict):
                continue
            sources = relation.get("sources", [])
            if not isinstance(sources, list):
                continue
            dtypes = [parameters[by_name[str(name)]].dtypes[0] for name in sources if str(name) in by_name and parameters[by_name[str(name)]].dtypes]
            if not dtypes:
                continue
            promoted = max(dtypes, key=lambda dtype: _dtype_rank(str(dtype)))
            metadata = dict(parameter.metadata)
            metadata["resolved_dtype_promotion"] = list(dtypes)
            parameters[index] = parameter.model_copy(update={"dtypes": [promoted], "metadata": metadata})
        return case.model_copy(update={"parameters": parameters})


class ProductLimitsConstraint(ConstraintPlugin):
    def after_case(self, case: CaseSpec, context: GenerationContext) -> CaseSpec:
        limits = case.generation.metadata.get("product_limits", [])
        if not isinstance(limits, list):
            return case
        parameters = list(case.parameters)
        by_name = {parameter.name: index for index, parameter in enumerate(parameters) if parameter.name}
        for item in limits:
            if not isinstance(item, dict):
                continue
            names = item.get("parameters")
            maximum = item.get("max_elements")
            if not isinstance(names, list) or not names or maximum is None:
                continue
            indexes = [by_name[str(name)] for name in names if str(name) in by_name]
            while indexes and _parameter_product(parameters, indexes) > int(maximum):
                index = max(indexes, key=lambda position: _num_elements(_shape_list(parameters[position])))
                parameter = parameters[index]
                dims = _shape_list(parameter)
                if not dims:
                    break
                largest = max(range(len(dims)), key=dims.__getitem__)
                dims[largest] = max(1, dims[largest] // 2)
                metadata = dict(parameter.metadata)
                metadata["product_limit"] = int(maximum)
                parameters[index] = parameter.model_copy(update={"shape": ShapeSpec(dims=dims), "metadata": metadata})
        return case.model_copy(update={"parameters": parameters})


class MaxElementsConstraint(ConstraintPlugin):
    def after_case(self, case: CaseSpec, context: GenerationContext) -> CaseSpec:
        max_elements = case.generation.max_elements
        if max_elements is None:
            return case
        parameters = [self._cap_parameter(parameter, max_elements) for parameter in case.parameters]
        return case.model_copy(update={"parameters": parameters})

    def _cap_parameter(self, parameter: ParameterSpec, max_elements: int) -> ParameterSpec:
        if parameter.kind not in {ParameterKind.TENSOR, ParameterKind.TENSOR_LIST, ParameterKind.TENSOR_TUPLE}:
            return parameter
        if parameter.shape is None:
            return parameter
        dims = parameter.shape.dims
        if not dims:
            return parameter
        capped = list(dims)
        while _num_elements(capped) > max_elements and capped:
            largest_index = max(range(len(capped)), key=capped.__getitem__)
            capped[largest_index] = max(1, capped[largest_index] // 2)
        return parameter.model_copy(update={"shape": parameter.shape.model_copy(update={"dims": capped})})


class MaxBytesConstraint(ConstraintPlugin):
    def after_case(self, case: CaseSpec, context: GenerationContext) -> CaseSpec:
        max_bytes = case.generation.max_bytes
        if max_bytes is None:
            return case
        parameters = [self._cap_parameter(parameter, max_bytes) for parameter in case.parameters]
        return case.model_copy(update={"parameters": parameters})

    def _cap_parameter(self, parameter: ParameterSpec, max_bytes: int) -> ParameterSpec:
        if parameter.kind not in {ParameterKind.TENSOR, ParameterKind.TENSOR_LIST, ParameterKind.TENSOR_TUPLE}:
            return parameter
        if parameter.shape is None:
            return parameter
        dims = parameter.shape.dims
        if not dims:
            return parameter
        dtype_bytes = _max_dtype_bytes(parameter.dtypes)
        capped = list(dims)
        while _num_elements(capped) * dtype_bytes > max_bytes and capped:
            largest_index = max(range(len(capped)), key=capped.__getitem__)
            capped[largest_index] = max(1, capped[largest_index] // 2)
        metadata = dict(parameter.metadata)
        metadata["max_bytes"] = max_bytes
        metadata["estimated_bytes"] = _num_elements(capped) * dtype_bytes
        return parameter.model_copy(update={"shape": parameter.shape.model_copy(update={"dims": capped}), "metadata": metadata})


def _dtype_rank(dtype: str) -> int:
    return {"bool": 0, "int8": 1, "int16": 2, "int32": 3, "int64": 4, "fp16": 5, "bf16": 6, "fp32": 7, "fp64": 8}.get(dtype, 0)


def _shape_list(parameter: ParameterSpec) -> list[int]:
    return list(parameter.shape.dims) if parameter.shape and parameter.shape.dims else []


def _parameter_product(parameters: list[ParameterSpec], indexes: list[int]) -> int:
    total = 1
    for index in indexes:
        total *= _num_elements(_shape_list(parameters[index]))
    return total


def _max_dtype_bytes(dtypes: list[str]) -> int:
    if not dtypes:
        return _DTYPE_BYTES["fp32"]
    return max(_DTYPE_BYTES.get(dtype, _DTYPE_BYTES["fp32"]) for dtype in dtypes)


def _num_elements(dims: list[int]) -> int:
    total = 1
    for dim in dims:
        total *= max(1, int(dim))
    return total


CONSTRAINT_REGISTRY.register("boundary_coverage")(BoundaryCoverageConstraint)
CONSTRAINT_REGISTRY.register("operator_facts")(OperatorFactsConstraint)
CONSTRAINT_REGISTRY.register("value_policy")(ValuePolicyConstraint)
CONSTRAINT_REGISTRY.register("dtype_policy")(DtypePolicyConstraint)
CONSTRAINT_REGISTRY.register("linked_parameters")(LinkedParametersConstraint)
CONSTRAINT_REGISTRY.register("shape_relationships")(ShapeRelationshipsConstraint)
CONSTRAINT_REGISTRY.register("max_bytes")(MaxBytesConstraint)
CONSTRAINT_REGISTRY.register("dtype_promotion")(DtypePromotionConstraint)
CONSTRAINT_REGISTRY.register("product_limits")(ProductLimitsConstraint)
CONSTRAINT_REGISTRY.register("max_elements")(MaxElementsConstraint)
CONSTRAINT_REGISTRY.register("random_coverage")(RandomCoverageConstraint)

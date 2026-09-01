from __future__ import annotations

import random
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass

from cruciblex.domain.case import CaseSpec, ParameterSpec, ShapeSpec
from cruciblex.domain.enums import ParameterKind
from cruciblex.generation.dtypes import validate_value_policy
from cruciblex.generation.policies import merge_facts, operator_facts, resolve_policy


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
            collection_relation = metadata.get("collection_relationship")
            if isinstance(collection_relation, dict):
                source = parameter_map.get(str(collection_relation.get("source")))
                resolved = _resolve_collection_relationship(parameter, source, collection_relation)
                if resolved:
                    metadata.update(resolved)
                    metadata["resolved_collection_relationship"] = str(collection_relation.get("kind"))
                    parameter = parameter.model_copy(update={"metadata": metadata})
            updated.append(parameter)
            if parameter.name:
                parameter_map[parameter.name] = parameter
        return case.model_copy(update={"parameters": updated})


def _resolve_collection_relationship(parameter: ParameterSpec, source: ParameterSpec | None, relation: dict[str, object]) -> dict[str, object]:
    if source is None:
        return {}
    kind = relation.get("kind")
    source_metadata = source.metadata
    source_items = source_metadata.get("items")
    if kind == "same_length_as":
        if isinstance(source_items, list):
            return {"length": len(source_items)}
        for key in ("length", "list_length", "tuple_length", "item_count"):
            if isinstance(source_metadata.get(key), int):
                return {"length": int(source_metadata[key])}
        return {}
    if kind == "same_item_dtype_as":
        dtypes = source_metadata.get("item_dtypes")
        if isinstance(dtypes, list) and dtypes:
            return {"item_dtypes": list(dtypes)}
        if source.dtypes:
            return {"item_dtypes": list(source.dtypes)}
    if kind == "same_item_shape_as":
        shapes = source_metadata.get("item_shapes")
        if isinstance(shapes, list) and shapes:
            return {"item_shapes": list(shapes)}
        if source.shape is not None and source.shape.dims is not None:
            return {"item_shapes": [list(source.shape.dims)]}
    if kind == "broadcast_items_with":
        source_shapes = source_metadata.get("item_shapes")
        target_shapes = parameter.metadata.get("item_shapes")
        if not isinstance(source_shapes, list):
            source_shapes = [source.shape.dims] if source.shape and source.shape.dims else []
        if not isinstance(target_shapes, list):
            target_shapes = [parameter.shape.dims] if parameter.shape and parameter.shape.dims else []
        shapes = [_broadcast_item_shape(left, right) for left, right in zip(target_shapes, source_shapes, strict=False)]
        if shapes:
            return {"item_shapes": shapes}
    if kind == "zip_with":
        length = _collection_length(source)
        if length is not None:
            return {"length": length, "collection_pairing": "zip"}
    if kind == "cartesian_with":
        source_length = _collection_length(source)
        target_length = _collection_length(parameter)
        if source_length is not None and target_length is not None:
            return {"length": source_length * target_length, "collection_pairing": "cartesian"}
    return {}


def _collection_length(parameter: ParameterSpec) -> int | None:
    items = parameter.metadata.get("items")
    if isinstance(items, list):
        return len(items)
    for key in ("length", "list_length", "tuple_length", "item_count"):
        value = parameter.metadata.get(key)
        if isinstance(value, int):
            return value
    return None


def _broadcast_item_shape(left: object, right: object) -> list[int]:
    left_dims = list(left) if isinstance(left, list) else []
    right_dims = list(right) if isinstance(right, list) else []
    width = max(len(left_dims), len(right_dims))
    left_dims = [1] * (width - len(left_dims)) + left_dims
    right_dims = [1] * (width - len(right_dims)) + right_dims
    result = []
    for value, other in zip(left_dims, right_dims, strict=True):
        if value not in {1, other} and other != 1:
            return []
        result.append(max(value, other))
    return result


class ShapeRelationshipsConstraint(ConstraintPlugin):
    """Resolve declarative shape relationships after individual parameters are generated."""

    def after_case(self, case: CaseSpec, context: GenerationContext) -> CaseSpec:
        parameters = {parameter.name: parameter for parameter in case.parameters if parameter.name}
        updated: list[ParameterSpec] = []
        for parameter in case.parameters:
            relation = parameter.metadata.get("shape_relationship")
            if not isinstance(relation, dict):
                updated.append(parameter)
                continue
            source_name = relation.get("source")
            source = parameters.get(str(source_name))
            source_dims = _shape_dims(source.shape if source else None)
            target_dims = _shape_dims(parameter.shape)
            kind = relation.get("kind")
            if kind == "attention_mask" and source_dims:
                key = parameters.get(str(relation.get("key", "key")))
                key_dims = _shape_dims(key.shape if key else None)
                resolved = [source_dims[0], source_dims[1], source_dims[2], key_dims[2]] if key_dims and len(source_dims) == 4 and len(key_dims) == 4 else target_dims
            elif kind in {"last_dimension_as", "last_k_dimensions_as"} and source_dims and parameter.kind in {ParameterKind.TENSOR, ParameterKind.ATTRIBUTE, ParameterKind.ATTRIBUTE_LIST, ParameterKind.ATTRIBUTE_TUPLE}:
                width = int(relation.get("k", 1)) if kind == "last_k_dimensions_as" else 1
                resolved = source_dims[-max(1, width):]
            elif kind == "conv_weight_channels" and source_dims and target_dims:
                groups = parameters.get(str(relation.get("groups", "groups")))
                group_value = groups.values if groups else 1
                group_value = int(group_value) if isinstance(group_value, int) and group_value > 0 else 1
                resolved = list(target_dims)
                resolved[int(relation.get("dimension", 1)) % len(resolved)] = source_dims[int(relation.get("source_dimension", 1)) % len(source_dims)] // group_value
            else:
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


def _positive_pair(value: object, default: int) -> tuple[int, int]:
    if isinstance(value, int) and value > 0:
        return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(item, int) and item > 0 for item in value):
        return int(value[0]), int(value[1])
    return default, default


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
    if kind == "dimension_aliases":
        if target is None:
            return None
        aliases = relation.get("aliases", [])
        if not isinstance(aliases, list):
            return target
        for item in aliases:
            if not isinstance(item, dict):
                continue
            source_dimension = int(item.get("source_dimension", -1)) % len(source)
            dimension = int(item.get("dimension", -1)) % len(target)
            target[dimension] = source[source_dimension]
        return target
    if kind == "same_shape_as":
        return list(source)
    if kind == "last_dimension_as":
        return [source[-1]]
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
        validation = validate_value_policy(policy, dtype)
        metadata["value_policy_validation"] = validation
        if validation["rejected"]:
            metadata["resolved_value_policy"] = validation["rejected"]
            return parameter.model_copy(update={"metadata": metadata})
        if kind in {"nan", "inf"} and not dtype.startswith(("fp", "float", "bf16")):
            metadata["resolved_value_policy"] = "filtered_non_floating_dtype"
        elif kind in {"uniform", "normal", "sparsity"}:
            metadata["value_policy_seed"] = context.seed + context.case_index + sum(ord(ch) for ch in parameter.name or "")
            metadata["resolved_value_policy"] = kind
        return parameter.model_copy(update={"metadata": metadata})


class OperatorFactsConstraint(ConstraintPlugin):
    """Project declarative operator facts into parameter-level generation policies."""

    def after_case(self, case: CaseSpec, context: GenerationContext) -> CaseSpec:
        inline_facts = case.generation.metadata.get("operator_facts")
        library_names = case.generation.metadata.get("operator_fact_library", [])
        if isinstance(library_names, str):
            library_names = [library_names]
        facts: dict[str, object] = {}
        if case.operator.name in ("torch.add", "torch.matmul", "torch.softmax", "torch.sum", "torch.mean", "torch.norm", "torch.sort", "torch.topk", "torch.index_select", "torch.select", "torch.gather", "torch.scatter", "torch.bmm", "torch.where", "torch.masked_fill", "torch.reshape", "torch.view", "torch.transpose", "torch.conv2d", "torch.layer_norm", "torch.scaled_dot_product_attention"):
            library_names = [*library_names, case.operator.name]
        if isinstance(library_names, list):
            for name in library_names:
                if isinstance(name, str):
                    facts = merge_facts(facts, operator_facts(name))
        if isinstance(inline_facts, dict):
            facts = merge_facts(facts, inline_facts)
        case_metadata = dict(case.metadata)
        contract = facts.get("contract")
        explicit_contract = case.generation.metadata.get("operator_contract")
        if isinstance(contract, dict) or isinstance(explicit_contract, dict):
            case_metadata["resolved_operator_contract"] = merge_facts(
                dict(contract) if isinstance(contract, dict) else {},
                dict(explicit_contract) if isinstance(explicit_contract, dict) else {},
            )
            case = case.model_copy(update={"metadata": case_metadata})
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
            policy_kinds = {"dtype_policy": "dtype", "value_policy": "value", "shape_policy": "shape"}
            for key in ("dtype_policy", "value_policy", "shape_policy", "shape_relationship", "collection_relationship", "dtype_promotion"):
                value = metadata.get(key, fact.get(key))
                if key in policy_kinds and isinstance(value, dict):
                    value = resolve_policy(policy_kinds[key], value)
                if value is not None:
                    metadata[key] = value
            shape_policy = metadata.get("shape_policy")
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


class OperatorContractConstraint(ConstraintPlugin):
    """Resolve case-level operator facts into auditable shape and dtype evidence."""

    def after_case(self, case: CaseSpec, context: GenerationContext) -> CaseSpec:
        contract = case.metadata.get("resolved_operator_contract")
        if not isinstance(contract, dict):
            return case
        parameters = {parameter.name: parameter for parameter in case.parameters if parameter.name}
        resolved = dict(contract)
        family = contract.get("family")
        input_name = str(contract.get("input", "query" if family == "attention" else "input"))
        input_parameter = parameters.get(input_name)
        input_shape = _shape_dims(input_parameter.shape if input_parameter else None)
        dim = _contract_attribute(contract, parameters, "dim")
        if family == "reduce" and input_shape is not None:
            resolved["output_shape"] = _reduce_shape(input_shape, dim, bool(_contract_attribute(contract, parameters, "keepdim")))
            resolved["output_dtype"] = _contract_dtype(contract, input_parameter)
        elif family == "sort" and input_shape is not None:
            resolved["output_shape"] = list(input_shape)
            resolved["values_dtype"] = _contract_dtype(contract, input_parameter)
            resolved["indices_dtype"] = "int64"
        elif family == "topk" and input_shape is not None:
            k = _contract_attribute(contract, parameters, "k")
            if isinstance(k, int) and 0 < k <= input_shape[dim % len(input_shape) if isinstance(dim, int) else -1]:
                output_shape = list(input_shape)
                output_shape[dim % len(output_shape) if isinstance(dim, int) else -1] = k
                resolved["output_shape"] = output_shape
                resolved["values_dtype"] = _contract_dtype(contract, input_parameter)
                resolved["indices_dtype"] = "int64"
        elif family == "index" and input_shape is not None:
            index_dim = dim if isinstance(dim, int) else 0
            resolved["index_range"] = [0, input_shape[index_dim % len(input_shape)] - 1]
            resolved["index_dtype"] = "int64"
            index_parameter = parameters.get(str(contract.get("index", "index")))
            index_shape = _shape_dims(index_parameter.shape if index_parameter else None)
            mode = contract.get("mode", "index_select")
            if mode == "gather" and index_shape is not None:
                resolved["output_shape"] = index_shape
            elif mode == "scatter":
                resolved["output_shape"] = list(input_shape)
            elif mode == "select":
                resolved["output_shape"] = [value for position, value in enumerate(input_shape) if position != index_dim % len(input_shape)]
        elif family in {"where", "masked_fill"} and input_shape is not None:
            shapes = [input_shape]
            if family == "where":
                names = (contract.get("condition", "condition"), contract.get("other", "other"))
            else:
                names = (contract.get("mask", "mask"),)
            for name in names:
                parameter = parameters.get(str(name))
                shape = _shape_dims(parameter.shape if parameter else None)
                if shape is not None:
                    shapes.append(shape)
            output_shape = shapes[0]
            for shape in shapes[1:]:
                output_shape = _broadcast_item_shape(output_shape, shape)
            if output_shape:
                resolved["output_shape"] = output_shape
                resolved["broadcast_shape"] = output_shape
            resolved["output_dtype"] = _contract_dtype(contract, input_parameter)
        elif family == "reshape" and input_shape is not None:
            target = _contract_attribute(contract, parameters, "shape")
            if (
                isinstance(target, (list, tuple))
                and all(isinstance(value, int) and value >= 0 for value in target)
                and _num_elements(input_shape) == _num_elements(list(target))
            ):
                resolved["output_shape"] = list(target)
                resolved["output_dtype"] = _contract_dtype(contract, input_parameter)
        elif family == "transpose" and input_shape is not None:
            dim0 = _contract_attribute(contract, parameters, "dim0")
            dim1 = _contract_attribute(contract, parameters, "dim1")
            if isinstance(dim0, int) and isinstance(dim1, int):
                output_shape = list(input_shape)
                first, second = dim0 % len(output_shape), dim1 % len(output_shape)
                output_shape[first], output_shape[second] = output_shape[second], output_shape[first]
                resolved["output_shape"] = output_shape
                resolved["output_dtype"] = _contract_dtype(contract, input_parameter)
        elif family == "conv" and input_shape is not None:
            weight_parameter = parameters.get(str(contract.get("weight", "weight")))
            weight_shape = _shape_dims(weight_parameter.shape if weight_parameter else None)
            if len(input_shape) == 4 and weight_shape and len(weight_shape) == 4:
                stride = _positive_pair(_contract_attribute(contract, parameters, "stride"), 1)
                padding = _positive_pair(_contract_attribute(contract, parameters, "padding"), 0)
                dilation = _positive_pair(_contract_attribute(contract, parameters, "dilation"), 1)
                groups = _contract_attribute(contract, parameters, "groups")
                groups = int(groups) if isinstance(groups, int) else 1
                valid_groups = groups > 0 and input_shape[1] % groups == 0 and weight_shape[0] % groups == 0 and weight_shape[1] * groups == input_shape[1]
                effective_kernel = [dilation[0] * (weight_shape[2] - 1) + 1, dilation[1] * (weight_shape[3] - 1) + 1]
                valid_geometry = all(value > 0 for value in stride + dilation) and all(value >= 0 for value in padding) and effective_kernel[0] <= input_shape[2] + 2 * padding[0] and effective_kernel[1] <= input_shape[3] + 2 * padding[1]
                resolved["groups"] = groups
                resolved["valid_groups"] = valid_groups
                resolved["effective_kernel"] = effective_kernel
                resolved["valid_geometry"] = valid_geometry
                height = (input_shape[2] + 2 * padding[0] - effective_kernel[0]) // stride[0] + 1
                width = (input_shape[3] + 2 * padding[1] - effective_kernel[1]) // stride[1] + 1
                if height > 0 and width > 0:
                    resolved["output_shape"] = [input_shape[0], weight_shape[0], height, width]
                    resolved["output_dtype"] = _contract_dtype(contract, input_parameter)
        elif family == "norm" and input_shape is not None:
            resolved["output_shape"] = list(input_shape)
            resolved["output_dtype"] = _contract_dtype(contract, input_parameter)
            normalized = parameters.get(str(contract.get("normalized_shape", "normalized_shape")))
            normalized_values = normalized.values if normalized else None
            resolved["normalized_shape"] = list(normalized_values) if isinstance(normalized_values, (list, tuple)) else input_shape[-1:]
        elif family == "attention" and input_shape is not None:
            key_parameter = parameters.get(str(contract.get("key", "key")))
            value_parameter = parameters.get(str(contract.get("value", "value")))
            key_shape = _shape_dims(key_parameter.shape if key_parameter else None)
            value_shape = _shape_dims(value_parameter.shape if value_parameter else None)
            if len(input_shape) == 4 and key_shape and value_shape and len(key_shape) == 4 and len(value_shape) == 4:
                resolved["qk_embedding_compatible"] = input_shape[3] == key_shape[3]
                resolved["kv_sequence_compatible"] = key_shape[2] == value_shape[2]
                resolved["batch_compatible"] = input_shape[0] == key_shape[0] == value_shape[0]
                resolved["head_compatible"] = input_shape[1] == key_shape[1] == value_shape[1]
                mask_parameter = parameters.get(str(contract.get("mask", "attn_mask")))
                mask_shape = _shape_dims(mask_parameter.shape if mask_parameter else None)
                resolved["mask_shape"] = mask_shape
                resolved["mask_broadcast_compatible"] = mask_shape is None or bool(_broadcast_item_shape([input_shape[0], input_shape[1], input_shape[2], key_shape[2]], mask_shape))
                resolved["valid_attention"] = all(resolved[key] for key in ("qk_embedding_compatible", "kv_sequence_compatible", "batch_compatible", "head_compatible", "mask_broadcast_compatible"))
                if resolved["valid_attention"]:
                    resolved["output_shape"] = [input_shape[0], input_shape[1], input_shape[2], value_shape[3]]
                    resolved["output_dtype"] = _contract_dtype(contract, input_parameter)
        elif family == "matmul":
            left = _shape_dims(parameters.get(str(contract.get("left", "input"))).shape if parameters.get(str(contract.get("left", "input"))) else None)
            right = _shape_dims(parameters.get(str(contract.get("right", "other"))).shape if parameters.get(str(contract.get("right", "other"))) else None)
            if left and right and len(left) >= 2 and len(right) >= 2:
                resolved["batch_shape"] = _broadcast_dims(left[:-2], right[:-2])
                resolved["inner_dimension"] = left[-1]
        metadata = dict(case.metadata)
        metadata["resolved_operator_contract"] = resolved
        return case.model_copy(update={"metadata": metadata})


def _contract_attribute(contract: dict[str, object], parameters: dict[str, ParameterSpec], name: str) -> object:
    parameter_name = contract.get(f"{name}_parameter", name)
    parameter = parameters.get(str(parameter_name))
    if parameter is not None and parameter.values is not None:
        return parameter.values
    return contract.get(name)


def _contract_dtype(contract: dict[str, object], parameter: ParameterSpec | None) -> str | None:
    selected = contract.get("output_dtype")
    if selected == "input":
        return parameter.dtypes[0] if parameter and parameter.dtypes else None
    return str(selected) if isinstance(selected, str) else None


def _reduce_shape(shape: list[int], dim: object, keepdim: bool) -> list[int]:
    dimensions = range(len(shape)) if dim is None else ([dim] if isinstance(dim, int) else dim)
    selected = {int(item) % len(shape) for item in dimensions if isinstance(item, int)}
    return [1 if index in selected and keepdim else value for index, value in enumerate(shape) if keepdim or index not in selected]


def _broadcast_dims(left: list[int], right: list[int]) -> list[int]:
    width = max(len(left), len(right))
    left = [1] * (width - len(left)) + left
    right = [1] * (width - len(right)) + right
    return [max(first, second) if first in {1, second} or second == 1 else 1 for first, second in zip(left, right, strict=True)]


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
CONSTRAINT_REGISTRY.register("operator_contract")(OperatorContractConstraint)
CONSTRAINT_REGISTRY.register("value_policy")(ValuePolicyConstraint)
CONSTRAINT_REGISTRY.register("dtype_policy")(DtypePolicyConstraint)
CONSTRAINT_REGISTRY.register("linked_parameters")(LinkedParametersConstraint)
CONSTRAINT_REGISTRY.register("shape_relationships")(ShapeRelationshipsConstraint)
CONSTRAINT_REGISTRY.register("max_bytes")(MaxBytesConstraint)
CONSTRAINT_REGISTRY.register("dtype_promotion")(DtypePromotionConstraint)
CONSTRAINT_REGISTRY.register("product_limits")(ProductLimitsConstraint)
CONSTRAINT_REGISTRY.register("max_elements")(MaxElementsConstraint)
CONSTRAINT_REGISTRY.register("random_coverage")(RandomCoverageConstraint)

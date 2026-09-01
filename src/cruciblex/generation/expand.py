from __future__ import annotations

from pathlib import Path
from typing import Any

from cruciblex.domain.case import CaseSpec
from cruciblex.runtime.generation.constraints import (
    CONSTRAINT_REGISTRY,
    GenerationContext,
)
from cruciblex.storage.artifacts import ArtifactStore


def expand_cases(cases: list[CaseSpec]) -> list[CaseSpec]:
    expanded: list[CaseSpec] = []
    for case in cases:
        count = max(1, case.generation.count)
        invalid_count = max(0, case.generation.invalid_count)
        for index in range(count):
            expanded.append(_apply_constraints(case, index, count, invalid=False, invalid_index=None))
        for invalid_index in range(invalid_count):
            expanded.append(_apply_constraints(case, invalid_index, invalid_count, invalid=True, invalid_index=invalid_index))
    return expanded


def _apply_constraints(case: CaseSpec, index: int, count: int, invalid: bool, invalid_index: int | None) -> CaseSpec:
    context = GenerationContext(
        seed=case.generation.seed,
        case_index=index,
        source_case_id=case.id,
    )
    generated = _generated_case(case, index, count, invalid=invalid, invalid_index=invalid_index)
    for constraint_name in _constraint_names(generated):
        constraint = CONSTRAINT_REGISTRY.resolve(constraint_name)
        parameters = [
            constraint.after_parameter(parameter, context)
            for parameter in generated.parameters
        ]
        generated = generated.model_copy(update={"parameters": parameters})
        generated = constraint.after_case(generated, context)
    if invalid:
        generated = _mark_invalid_case(generated, invalid_index or 0)
        generated = _apply_invalid_values(generated, invalid_index or 0)
        generated = _apply_contract_invalid_value(generated, invalid_index or 0)
    return generated


def persist_generated_cases(cases: list[CaseSpec], output_root: str | Path) -> Path:
    payload: dict[str, Any] = {
        "cases": [case.model_dump(mode="json") for case in cases],
    }
    store = ArtifactStore(output_root)
    return store.write_json(store.ensure() / "generated_cases.json", payload)


def _generated_case(case: CaseSpec, index: int, count: int, invalid: bool, invalid_index: int | None) -> CaseSpec:
    if count == 1 and not invalid:
        return case
    metadata = dict(case.metadata)
    metadata.update(
        {
            "source_case_id": case.id,
            "generation_index": index,
            "generation_seed": case.generation.seed,
        }
    )
    if invalid:
        metadata["expected_invalid"] = True
        metadata["invalid_index"] = invalid_index
    suffix = index if not invalid else 500000 + (invalid_index or 0)
    return case.model_copy(update={"id": case.id * 100000 + suffix, "metadata": metadata})


def _mark_invalid_case(case: CaseSpec, invalid_index: int) -> CaseSpec:
    metadata = dict(case.metadata)
    metadata["expected_invalid"] = True
    metadata["invalid_index"] = invalid_index
    return case.model_copy(update={"metadata": metadata})


def _apply_invalid_values(case: CaseSpec, invalid_index: int) -> CaseSpec:
    parameters = []
    for parameter in case.parameters:
        values = list(parameter.value_range.invalid)
        if not values:
            parameters.append(parameter)
            continue
        selected = values[invalid_index % len(values)]
        metadata = dict(parameter.metadata)
        metadata["selected_invalid_value"] = selected
        parameters.append(parameter.model_copy(update={"metadata": metadata}))
    return case.model_copy(update={"parameters": parameters})


def _apply_contract_invalid_value(case: CaseSpec, invalid_index: int) -> CaseSpec:
    contract = case.metadata.get("resolved_operator_contract")
    if not isinstance(contract, dict):
        return case
    parameters = {parameter.name: parameter for parameter in case.parameters if parameter.name}
    input_name = str(contract.get("input", "query" if contract.get("family") == "attention" else "input"))
    input_parameter = parameters.get(input_name)
    input_shape = list(input_parameter.shape.dims) if input_parameter and input_parameter.shape and input_parameter.shape.dims else None
    if not input_shape:
        return case
    family = contract.get("family")
    dim_name = str(contract.get("dim_parameter", "dim"))
    mutations: list[tuple[str, object, str]] = []
    if family in {"reduce", "sort", "topk", "index", "transpose"}:
        mutations.append((dim_name, len(input_shape), "dim_out_of_range"))
    if family == "topk":
        dim_parameter = parameters.get(dim_name)
        dim = dim_parameter.values if dim_parameter else None
        axis = int(dim) % len(input_shape) if isinstance(dim, int) else len(input_shape) - 1
        mutations.append((str(contract.get("k_parameter", "k")), input_shape[axis] + 1, "k_exceeds_axis"))
    if family == "index":
        dim_parameter = parameters.get(dim_name)
        dim = dim_parameter.values if dim_parameter else None
        axis = int(dim) % len(input_shape) if isinstance(dim, int) else 0
        mutations.append((str(contract.get("index", "index")), input_shape[axis], "index_out_of_range"))
    if family == "conv":
        weight = parameters.get(str(contract.get("weight", "weight")))
        if weight and weight.shape and weight.shape.dims and len(weight.shape.dims) == 4:
            invalid_shape = list(weight.shape.dims)
            invalid_shape[1] = max(1, invalid_shape[1] + 1)
            mutations.append((str(contract.get("weight", "weight")), invalid_shape, "conv_channel_mismatch"))
    if family == "norm":
        normalized_name = contract.get("normalized_shape", "normalized_shape")
        normalized = parameters.get(str(normalized_name)) or parameters.get("normalized_shape")
        if normalized:
            if normalized.shape and normalized.shape.dims:
                invalid_shape = list(normalized.shape.dims)
            elif isinstance(normalized.values, (list, tuple)) and normalized.values:
                invalid_shape = list(normalized.values)
            else:
                invalid_shape = []
            if invalid_shape:
                invalid_shape[-1] += 1
                mutations.append((str(normalized_name) if isinstance(normalized_name, str) else "normalized_shape", invalid_shape, "normalized_shape_mismatch"))
    if family == "attention":
        key = parameters.get(str(contract.get("key", "key")))
        if key and key.shape and key.shape.dims and len(key.shape.dims) == 4:
            invalid_shape = list(key.shape.dims)
            invalid_shape[1] += 1
            mutations.append((str(contract.get("key", "key")), invalid_shape, "attention_head_mismatch"))
    if family == "reshape":
        target_name = str(contract.get("shape", "shape"))
        target = parameters.get(target_name)
        if target and isinstance(target.values, (list, tuple)) and target.values:
            invalid_shape = list(target.values)
            invalid_shape[-1] = int(invalid_shape[-1]) + 1
            mutations.append((target_name, invalid_shape, "reshape_numel_mismatch"))
    if not mutations:
        return case
    name, value, reason = mutations[invalid_index % len(mutations)]
    updated = []
    for parameter in case.parameters:
        if parameter.name != name:
            updated.append(parameter)
            continue
        metadata = dict(parameter.metadata)
        metadata["contract_invalid_reason"] = reason
        if parameter.kind.value in {"attribute", "scalar", "attribute_list", "attribute_tuple", "scalar_list", "scalar_tuple"}:
            parameter = parameter.model_copy(update={"values": value, "metadata": metadata})
        else:
            metadata["selected_invalid_value"] = value
            parameter = parameter.model_copy(update={"metadata": metadata})
        updated.append(parameter)
    metadata = dict(case.metadata)
    metadata["contract_invalid_reason"] = reason
    return case.model_copy(update={"parameters": updated, "metadata": metadata})


def _constraint_names(case: CaseSpec) -> list[str]:
    names = list(case.generation.constraints)
    if isinstance(case.generation.metadata.get("operator_facts"), dict) or case.generation.metadata.get("operator_fact_library") or case.operator.name in {"torch.add", "torch.matmul", "torch.softmax", "torch.sum", "torch.mean", "torch.norm", "torch.sort", "torch.topk", "torch.index_select", "torch.select", "torch.gather", "torch.scatter", "torch.bmm", "torch.where", "torch.masked_fill", "torch.reshape", "torch.view", "torch.transpose", "torch.conv2d", "torch.layer_norm", "torch.scaled_dot_product_attention"}:
        for name in ("operator_facts", "dtype_policy", "value_policy", "shape_relationships", "operator_contract", "dtype_promotion"):
            if name not in names:
                names.append(name)
    if case.generation.max_elements is not None and "max_elements" not in names:
        names.append("max_elements")
    if case.generation.max_bytes is not None and "max_bytes" not in names:
        names.append("max_bytes")
    return names

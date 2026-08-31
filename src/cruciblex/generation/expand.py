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


def _constraint_names(case: CaseSpec) -> list[str]:
    names = list(case.generation.constraints)
    if case.generation.max_elements is not None and "max_elements" not in names:
        names.append("max_elements")
    if case.generation.max_bytes is not None and "max_bytes" not in names:
        names.append("max_bytes")
    return names

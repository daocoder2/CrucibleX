from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any


def semantic_reduction_candidates(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate deterministic shape, dtype, and value reduction candidates."""
    candidates: list[dict[str, Any]] = []
    shape_candidate = deepcopy(case)
    shape_changed = False
    for parameter in shape_candidate.get("parameters", []):
        if not isinstance(parameter, dict):
            continue
        shape = parameter.get("shape_rules")
        if isinstance(shape, dict) and isinstance(shape.get("dims"), list):
            dims = shape["dims"]
            reduced = [1 for _ in dims]
            if reduced != dims:
                shape["dims"] = reduced
                shape_changed = True
    if shape_changed:
        candidates.append(_annotate(shape_candidate, "shape_to_one", ["rank", "dtype", "value_policy", "fuzz_seed", "invalid_marker"]))

    dtype_candidate = deepcopy(case)
    dtype_changed = False
    for parameter in dtype_candidate.get("parameters", []):
        if not isinstance(parameter, dict):
            continue
        dtypes = parameter.get("dtypes")
        if isinstance(dtypes, list) and dtypes and dtypes != ["fp32"]:
            parameter["dtypes"] = ["fp32"]
            dtype_changed = True
    if dtype_changed:
        candidates.append(_annotate(dtype_candidate, "dtype_to_fp32", ["shape", "value_policy", "fuzz_seed", "invalid_marker"]))

    value_candidate = deepcopy(case)
    value_changed = False
    for parameter in value_candidate.get("parameters", []):
        if not isinstance(parameter, dict):
            continue
        value_range = parameter.get("value_range")
        if isinstance(value_range, dict) and value_range.get("valid"):
            value_range["valid"] = [[0, 0]]
            value_changed = True
        metadata = parameter.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("enum_values"), list) and metadata["enum_values"]:
            metadata["enum_values"] = [metadata["enum_values"][0]]
            value_changed = True
    if value_changed:
        candidates.append(_annotate(value_candidate, "value_to_zero_enum_first", ["shape", "dtype", "fuzz_seed", "invalid_marker"]))
    return candidates


def reduce_with_predicate(
    case: dict[str, Any],
    predicate: Callable[[dict[str, Any]], bool],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Accept reduction candidates only when replay predicate still holds."""
    current = deepcopy(case)
    attempts: list[dict[str, Any]] = []
    for candidate in semantic_reduction_candidates(current):
        preserved = bool(predicate(candidate))
        attempts.append({"strategy": candidate["reduction"]["strategy"], "accepted": preserved})
        if preserved:
            current = candidate
    return current, attempts


def _annotate(candidate: dict[str, Any], strategy: str, preserves: list[str]) -> dict[str, Any]:
    candidate.setdefault("reduction", {})
    candidate["reduction"].update({"strategy": strategy, "preserves": preserves, "requires_replay": True})
    return candidate

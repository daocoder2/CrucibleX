from __future__ import annotations

from collections.abc import Iterable

from cruciblex.domain.case import CaseSpec

_DIMENSIONS = ("operator", "backend", "task", "dtype", "tag")


def filter_cases(
    cases: Iterable[CaseSpec],
    *,
    include: dict[str, set[str]] | None = None,
    exclude: dict[str, set[str]] | None = None,
) -> list[CaseSpec]:
    include = include or {}
    exclude = exclude or {}
    return [case for case in cases if _matches(case, include, positive=True) and _matches(case, exclude, positive=False)]


def case_dimensions(case: CaseSpec) -> dict[str, set[str]]:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    invocation_metadata = case.invocation.metadata if isinstance(case.invocation.metadata, dict) else {}
    dtypes = {dtype for parameter in case.parameters for dtype in parameter.dtypes}
    tags = metadata.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    declared_backends = metadata.get("manifest_backends", [])
    if not isinstance(declared_backends, list):
        declared_backends = []
    backends = {str(case.invocation.executor or invocation_metadata.get("backend", ""))}
    backends.update(str(backend) for backend in declared_backends)
    return {
        "operator": {case.operator.name, case.invocation.api},
        "backend": backends,
        "task": {str(invocation_metadata.get("task", metadata.get("task", "")))},
        "dtype": dtypes,
        "tag": {str(tag) for tag in tags},
    }


def _matches(case: CaseSpec, selectors: dict[str, set[str]], *, positive: bool) -> bool:
    dimensions = case_dimensions(case)
    for dimension in _DIMENSIONS:
        requested = selectors.get(dimension, set())
        if not requested:
            continue
        matched = bool(dimensions[dimension] & requested)
        if positive and not matched:
            return False
        if not positive and matched:
            return False
    return True

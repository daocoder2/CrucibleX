from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cruciblex.domain.case import (
    CaseSpec,
    InvocationSpec,
    OperatorSpec,
    OracleSpec,
    ParameterSpec,
    ShapeSpec,
    ValueRange,
)
from cruciblex.domain.enums import ParameterKind, SchedulerKind, TaskKind
from cruciblex.domain.node import NodeSpec
from cruciblex.domain.plan import ArtifactPolicy, JobSpec
from cruciblex.domain.run import RunContext
from cruciblex.generation.expand import expand_cases, persist_generated_cases


def load_case(path: str | Path) -> CaseSpec:
    data = _require_mapping(_load_structured_file(path), "case file")
    return _parse_case(data)


def load_cases(path: str | Path) -> list[CaseSpec]:
    root = _require_mapping(_load_structured_file(path), "case file")
    cases = root.get("cases")
    if not isinstance(cases, list):
        raise TypeError("case file must contain a 'cases' list")
    return [_parse_case(_require_mapping(case, "case entry")) for case in cases]


def load_nodes(path: str | Path) -> list[NodeSpec]:
    root = _require_mapping(_load_structured_file(path), "node file")
    nodes = root.get("nodes")
    if not isinstance(nodes, list):
        raise TypeError("node file must contain a 'nodes' list")
    return [_parse_node(_require_mapping(node, "node entry")) for node in nodes]


def load_job(
    case_path: str | Path,
    node_path: str | Path,
    tasks: list[TaskKind] | None = None,
    scheduler: SchedulerKind = SchedulerKind.RAY,
    output_path: str | Path = "cx_output",
) -> JobSpec:
    output_root = Path(output_path).resolve()
    cases = expand_cases(load_cases(case_path))
    persist_generated_cases(cases, output_root)
    return JobSpec(
        cases=cases,
        nodes=load_nodes(node_path),
        tasks=tasks or [TaskKind.ACCURACY],
        scheduler=scheduler,
        artifacts=ArtifactPolicy(output_root=output_root),
    )


def load_job_from_context(context: RunContext) -> JobSpec:
    return load_job(
        context.case_path,
        context.node_path,
        tasks=context.tasks,
        scheduler=context.scheduler,
        output_path=context.output_root,
    )


def _load_structured_file(path: str | Path) -> Any:
    raw_path = Path(path)
    text = raw_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"empty structured file: {raw_path}")
    loaded = yaml.safe_load(text)
    if loaded is None:
        raise ValueError(f"empty structured file: {raw_path}")
    return loaded


def _require_mapping(data: Any, label: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise TypeError(f"{label} must contain a mapping")
    return data


def _parse_case(data: dict[str, Any]) -> CaseSpec:
    return CaseSpec(
        id=data["id"],
        operator=OperatorSpec.model_validate(_require_mapping(data["operator"], "operator")),
        invocation=InvocationSpec.model_validate(_require_mapping(data["invocation"], "invocation")),
        parameters=[_parse_parameter(item) for item in data.get("parameters", [])],
        oracle=OracleSpec.model_validate(_require_mapping(data.get("oracle", {}), "oracle")),
        generator=data.get("generator", "default"),
        generation=data.get("generation", {}),
        metadata=data.get("metadata", {}),
    )


def _parse_node(data: dict[str, Any]) -> NodeSpec:
    return NodeSpec.model_validate(data)


def _parse_parameter(data: dict[str, Any]) -> ParameterSpec:
    kind = data.get("kind")
    if kind is None:
        raise ValueError("parameter kind is required")
    shape_data = data.get("shape")
    shape = ShapeSpec.model_validate(shape_data) if shape_data is not None else None
    value_range_data = data.get("value_range") or {}
    return ParameterSpec(
        name=data.get("name"),
        kind=ParameterKind(kind),
        required=data.get("required", True),
        dtypes=list(data.get("dtypes", [])),
        shape=shape,
        value_range=ValueRange.model_validate(value_range_data),
        requires_grad=data.get("requires_grad", False),
        metadata=data.get("metadata", {}),
    )

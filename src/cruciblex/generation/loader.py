from __future__ import annotations

import json
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
from cruciblex.domain.enums import BackendKind, ParameterKind, SchedulerKind, TaskKind
from cruciblex.domain.manifest import Manifest
from cruciblex.domain.node import NodeSpec
from cruciblex.domain.plan import ArtifactPolicy, JobSpec
from cruciblex.domain.run import RunContext
from cruciblex.generation.expand import expand_cases, persist_generated_cases
from cruciblex.generation.filtering import filter_cases


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


def load_manifest(path: str | Path) -> Manifest:
    return Manifest.model_validate(_load_structured_file(path))


def validate_manifest_references(path: str | Path) -> tuple[Manifest, int]:
    """Validate manifest schema, included cases, runtime policy, and filters without artifacts."""
    manifest_file = Path(path).resolve()
    manifest_model = load_manifest(manifest_file)
    manifest = manifest_model.model_dump(mode="python")
    runtime_policy = _manifest_runtime_policy(manifest["runtime"])
    cases: list[CaseSpec] = []
    for lane_number, lane_data in enumerate(manifest["lanes"]):
        lane = _require_mapping(lane_data, "manifest lane")
        lane_name = str(lane.get("name", f"lane-{lane_number}"))
        cases.extend(
            _load_manifest_lane_cases(
                manifest_file,
                lane,
                lane_name,
                str(lane.get("kind", "contract")),
                _parse_manifest_backends(lane.get("backends")),
                runtime_policy,
            )
        )
    selected_cases = filter_cases(
        cases,
        include=_manifest_selectors(manifest["filters"], "include"),
        exclude=_manifest_selectors(manifest["filters"], "exclude"),
    )
    if not selected_cases:
        raise ValueError("manifest filters selected no cases")
    return manifest_model, len(selected_cases)


def _manifest_task_name(manifest: dict[str, Any]) -> str:
    task = manifest.get("task")
    if isinstance(task, dict) and isinstance(task.get("name"), str):
        return task["name"]
    return "manifest"


def _manifest_bool(mapping: dict[str, Any], name: str, default: bool) -> bool:
    value = mapping.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"manifest runtime.{name} must be a boolean")
    return value


def _manifest_string_list(mapping: dict[str, Any], name: str) -> list[str]:
    value = mapping.get(name, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"manifest {name} must be a list")
    return [str(item) for item in value]


def _resolve_manifest_include(manifest_file: Path, include: object) -> Path:
    include_path = Path(str(include))
    if include_path.is_absolute():
        return include_path
    return (manifest_file.parent / include_path).resolve()


def manifest_include_paths(path: str | Path) -> list[Path]:
    """Return every declared case include in stable order, before filters apply."""
    manifest_file = Path(path).resolve()
    manifest = load_manifest(manifest_file)
    return sorted({
        _resolve_manifest_include(manifest_file, entry.include)
        for lane in manifest.lanes
        for entry in lane.cases
    })


def _parse_manifest_backends(value: object) -> list[BackendKind] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError("manifest backends must be a list when declared")
    backends = [BackendKind(str(backend)) for backend in value]
    return backends or None


def load_job_from_manifest(
    manifest_path: str | Path,
    node_path: str | Path,
    tasks: list[TaskKind] | None = None,
    scheduler: SchedulerKind = SchedulerKind.RAY,
    output_path: str | Path = "cx_output",
) -> JobSpec:
    manifest_file = Path(manifest_path)
    manifest_model = load_manifest(manifest_file)
    manifest = manifest_model.model_dump(mode="python")
    lanes = manifest["lanes"]

    runtime = manifest["runtime"]
    filters = manifest["filters"]
    reporting = manifest["reporting"]
    runtime_policy = _manifest_runtime_policy(runtime)
    output_root = Path(output_path).resolve()

    cases: list[CaseSpec] = []
    lane_index: list[dict[str, Any]] = []
    for lane_number, lane_data in enumerate(lanes):
        lane = _require_mapping(lane_data, "manifest lane")
        lane_name = str(lane.get("name", f"lane-{lane_number}"))
        lane_kind = str(lane.get("kind", "contract"))
        lane_backends = _parse_manifest_backends(lane.get("backends"))
        lane_cases = _load_manifest_lane_cases(
            manifest_file,
            lane,
            lane_name,
            lane_kind,
            lane_backends,
            runtime_policy,
        )
        lane_index.append({
            "name": lane_name,
            "kind": lane_kind,
            "backends": [backend.value for backend in lane_backends] if lane_backends else None,
            "case_count": len(lane_cases),
        })
        cases.extend(lane_cases)

    selected_cases = filter_cases(
        cases,
        include=_manifest_selectors(filters, "include"),
        exclude=_manifest_selectors(filters, "exclude"),
    )
    if not selected_cases:
        raise ValueError(
            "manifest filters selected no cases: "
            f"included_cases={len(cases)} include={_manifest_selectors(filters, 'include')} "
            f"exclude={_manifest_selectors(filters, 'exclude')}"
        )
    expanded = expand_cases(selected_cases)
    if not expanded:
        raise ValueError(f"manifest expanded no cases: selected_cases={len(selected_cases)}")
    persist_generated_cases(expanded, output_root)
    _write_manifest_reporting(output_root, reporting, manifest, lane_index, selected_cases, expanded)
    return JobSpec(
        cases=expanded,
        nodes=load_nodes(node_path),
        tasks=tasks or [TaskKind.ACCURACY],
        scheduler=scheduler,
        artifacts=ArtifactPolicy(output_root=output_root),
    )


def _manifest_runtime_policy(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "allow_generated_cases": _manifest_bool(runtime, "allow_generated_cases", True),
        "allow_invalid_cases": _manifest_bool(runtime, "allow_invalid_cases", True),
        "require_real_evidence": _manifest_bool(runtime, "require_real_evidence", False),
        "require_backend_dtype_source": runtime.get("require_backend_dtype_source"),
    }


def _load_manifest_lane_cases(
    manifest_file: Path,
    lane: dict[str, Any],
    lane_name: str,
    lane_kind: str,
    lane_backends: list[BackendKind] | None,
    runtime_policy: dict[str, Any],
) -> list[CaseSpec]:
    entries = lane.get("cases")
    if not isinstance(entries, list) or not entries:
        raise TypeError("manifest lane must contain a non-empty 'cases' list")

    cases: list[CaseSpec] = []
    for include_index, entry_data in enumerate(entries):
        entry = _require_mapping(entry_data, "manifest case include")
        include = entry.get("include")
        if not isinstance(include, str) or not include.strip():
            raise ValueError("manifest case entry must declare a non-empty include path")
        include_path = _resolve_manifest_include(manifest_file, include)
        if not include_path.is_file():
            raise FileNotFoundError(f"manifest include not found: {include_path}")
        for case in load_cases(include_path):
            metadata = dict(case.metadata)
            metadata.update({
                "manifest_path": str(manifest_file),
                "manifest_lane": lane_name,
                "manifest_lane_kind": lane_kind,
                "manifest_case_include": str(include_path),
                "manifest_case_index": include_index,
                "manifest_runtime": runtime_policy,
            })
            if lane_backends is not None:
                metadata["manifest_backends"] = [backend.value for backend in lane_backends]
            cases.append(_apply_manifest_runtime(case.model_copy(update={"metadata": metadata}), runtime_policy))
    return cases


def _apply_manifest_runtime(case: CaseSpec, runtime_policy: dict[str, Any]) -> CaseSpec:
    updates: dict[str, Any] = {}
    if not runtime_policy["allow_generated_cases"]:
        updates["count"] = 1
    if not runtime_policy["allow_invalid_cases"]:
        updates["invalid_count"] = 0
    if not updates:
        return case
    return case.model_copy(update={"generation": case.generation.model_copy(update=updates)})


def _manifest_selectors(filters: dict[str, Any], prefix: str) -> dict[str, set[str]]:
    selectors: dict[str, set[str]] = {}
    mapping = {
        f"{prefix}_operators": "operator",
        f"{prefix}_backends": "backend",
        f"{prefix}_tasks": "task",
        f"{prefix}_dtypes": "dtype",
        f"{prefix}_tags": "tag",
    }
    for key, dimension in mapping.items():
        values = set(_manifest_string_list(filters, key))
        if values:
            selectors[dimension] = values
    return selectors


def _manifest_reporting_root(output_root: Path, reporting: dict[str, Any]) -> Path:
    output_dir = reporting.get("output_dir")
    if output_dir is None:
        return output_root
    path = Path(str(output_dir))
    return path if path.is_absolute() else (output_root / path).resolve()


def _write_manifest_reporting(
    output_root: Path,
    reporting: dict[str, Any],
    manifest: dict[str, Any],
    lane_index: list[dict[str, Any]],
    selected_cases: list[CaseSpec],
    expanded_cases: list[CaseSpec],
) -> None:
    reporting_root = _manifest_reporting_root(output_root, reporting)
    reporting_root.mkdir(parents=True, exist_ok=True)
    base = {
        "task_name": _manifest_task_name(manifest),
        "selected_case_count": len(selected_cases),
        "expanded_case_count": len(expanded_cases),
    }
    if _manifest_bool(reporting, "emit_lane_index", False):
        (reporting_root / "manifest_lane_index.json").write_text(
            json.dumps({**base, "lanes": lane_index}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if _manifest_bool(reporting, "emit_case_index", False):
        case_rows = [
            {
                "id": case.id,
                "operator": case.operator.name,
                "lane": case.metadata.get("manifest_lane"),
                "lane_kind": case.metadata.get("manifest_lane_kind"),
                "backends": case.metadata.get("manifest_backends"),
                "include": case.metadata.get("manifest_case_include"),
            }
            for case in expanded_cases
        ]
        (reporting_root / "manifest_case_index.json").write_text(
            json.dumps({**base, "cases": case_rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
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
    unsupported = set(data) - {"name", "kind", "required", "dtypes", "shape", "values", "value_range", "requires_grad", "metadata"}
    if unsupported:
        raise ValueError(f"unsupported parameter fields: {', '.join(sorted(unsupported))}")
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
        values=data.get("values"),
        value_range=ValueRange.model_validate(value_range_data),
        requires_grad=data.get("requires_grad", False),
        metadata=data.get("metadata", {}),
    )

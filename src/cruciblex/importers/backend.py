from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from cruciblex.domain.case import CaseSpec

CONVERTER_VERSION = "cruciblex.importers.backend:v1"
_DTYPE_ALIASES = {
    "float16": "fp16",
    "float32": "fp32",
    "float64": "fp64",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "bool": "bool",
}


def import_backend_case(
    source: str | Path,
    *,
    source_format: str,
    case_id: int | None = None,
    executor: str | None = None,
    reference_executor: str | None = None,
) -> CaseSpec:
    normalized_format = source_format.lower()
    if normalized_format not in {"atb", "temu"}:
        raise ValueError(f"unsupported backend source format: {source_format}")
    source_path = Path(source)
    payload = _load_structured(source_path)
    operator = _operator_payload(payload)
    operator_name = str(operator.get("name") or payload.get("name") or payload.get("api") or f"imported.{normalized_format}")
    api = str(operator.get("api") or payload.get("api") or operator_name)
    parameters = _parameters(payload)
    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {}
    backend_config = payload.get("backend_config") or payload.get("config") or payload.get(normalized_format) or {}
    metadata["backend_import"] = {
        "source_format": normalized_format,
        "executor": executor or payload.get("executor") or normalized_format,
        "config": backend_config if isinstance(backend_config, dict) else {"value": backend_config},
        "plugin_skeleton": {
            "module": f"cruciblex.plugins.executors.{normalized_format}",
            "executor_name": executor or payload.get("executor") or normalized_format,
        },
    }
    metadata["provenance"] = {
        "source_path": str(source_path),
        "source_format": normalized_format,
        "converter_version": CONVERTER_VERSION,
        "warnings": [
            "backend-specific execution semantics are preserved in metadata.backend_import and require a matching executor plugin"
        ],
        "lossy_fields": _lossy_fields(payload, normalized_format),
    }
    case_data = {
        "id": int(case_id if case_id is not None else payload.get("id", payload.get("case_id", 1))),
        "operator": {"name": operator_name, "version": operator.get("version"), "backward": bool(operator.get("backward", False))},
        "invocation": {
            "api": api,
            "api_type": str(operator.get("api_type") or payload.get("api_type") or normalized_format),
            "executor": executor or payload.get("executor") or normalized_format,
            "metadata": {"backend_source_format": normalized_format},
        },
        "oracle": _oracle_payload(payload, reference_executor=reference_executor),
        "generator": "default",
        "generation": {"count": int(payload.get("count", 1)), "metadata": {"backend_source_format": normalized_format}},
        "parameters": parameters,
        "metadata": metadata,
    }
    return CaseSpec.model_validate(case_data)


def write_imported_backend(path: str | Path, case: CaseSpec) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump({"cases": [case.model_dump(mode="json")]}, sort_keys=False), encoding="utf-8")
    return output_path


def _load_structured(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"empty backend import source: {path}")
    loaded = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise TypeError("backend import source must contain a mapping")
    return loaded


def _operator_payload(payload: dict[str, Any]) -> dict[str, Any]:
    operator = payload.get("operator", {})
    if isinstance(operator, str):
        return {"name": operator}
    if isinstance(operator, dict):
        return dict(operator)
    return {}


def _parameters(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_parameters = payload.get("parameters") or payload.get("inputs") or payload.get("args") or []
    if not isinstance(raw_parameters, list):
        raise TypeError("backend import parameters must be a list when provided")
    return [_parameter_payload(item, index) for index, item in enumerate(raw_parameters)]


def _parameter_payload(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("backend import parameter entries must be mappings")
    shape = [int(dim) for dim in raw.get("shape", [])] if isinstance(raw.get("shape", []), list) else []
    dtype = _dtype_alias(raw.get("dtype"))
    return {
        "name": str(raw.get("name", f"input_{index}")),
        "kind": str(raw.get("kind", "tensor")),
        "dtypes": [dtype] if dtype else [],
        "shape": {"dims": shape, "dim_count": [len(shape)], "dim_values": sorted(set(shape))},
        "value_range": {},
        "metadata": {"source": "backend_import", "backend_role": raw.get("role", "input")},
    }


def _oracle_payload(payload: dict[str, Any], *, reference_executor: str | None) -> dict[str, Any]:
    oracle = dict(payload.get("oracle", {})) if isinstance(payload.get("oracle"), dict) else {}
    return {
        "comparison": oracle.get("comparison", "allclose"),
        "reference_executor": reference_executor or oracle.get("reference_executor"),
        "expected_error": oracle.get("expected_error"),
        "tolerance": oracle.get("tolerance", {}),
        "accuracy_policy": oracle.get("accuracy_policy", {}),
        "metadata": oracle.get("metadata", {}),
    }


def _dtype_alias(dtype: Any) -> str | None:
    if dtype is None:
        return None
    text = str(dtype)
    return _DTYPE_ALIASES.get(text, text)


def _lossy_fields(payload: dict[str, Any], source_format: str) -> list[str]:
    supported = {
        "id",
        "case_id",
        "name",
        "operator",
        "api",
        "api_type",
        "executor",
        "parameters",
        "inputs",
        "args",
        "oracle",
        "metadata",
        "count",
        "backend_config",
        "config",
        source_format,
    }
    return sorted(set(payload) - supported)

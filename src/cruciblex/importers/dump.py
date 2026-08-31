from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from cruciblex.domain.case import CaseSpec

CONVERTER_VERSION = "cruciblex.importers.dump:v1"
_DTYPE_ALIASES = {
    "float16": "fp16",
    "float32": "fp32",
    "float64": "fp64",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "bool": "bool",
}


def import_dump_case(
    source: str | Path,
    *,
    output: str | Path,
    case_id: int | None = None,
    executor: str | None = None,
    reference_executor: str | None = None,
) -> tuple[CaseSpec, Path]:
    source_path = Path(source)
    output_path = Path(output)
    payload = _load_structured(source_path)
    inputs = _inputs(payload)
    snapshot_path = output_path.with_name("inputs.json").resolve()
    case = CaseSpec.model_validate(
        _case_payload(
            payload,
            source_path=source_path,
            snapshot_path=snapshot_path,
            case_id=case_id,
            executor=executor,
            reference_executor=reference_executor,
            inputs=inputs,
        )
    )
    return case, snapshot_path


def write_imported_dump(path: str | Path, case: CaseSpec, snapshot_path: Path, inputs: list[dict[str, Any]]) -> dict[str, Path]:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump({"cases": [case.model_dump(mode="json")]}, sort_keys=False),
        encoding="utf-8",
    )
    snapshot_path.write_text(json.dumps({"inputs": inputs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"case": output_path, "inputs": snapshot_path}


def load_dump_inputs(source: str | Path) -> list[dict[str, Any]]:
    return _inputs(_load_structured(Path(source)))


def _load_structured(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"empty dump source: {path}")
    loaded = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise TypeError("dump source must contain a mapping")
    return loaded


def _inputs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_inputs = payload.get("inputs") or payload.get("parameters") or payload.get("args")
    if not isinstance(raw_inputs, list):
        raise TypeError("dump source must contain an inputs list")
    return [_input_snapshot(item, index) for index, item in enumerate(raw_inputs)]


def _input_snapshot(item: Any, index: int) -> dict[str, Any]:
    if isinstance(item, dict):
        value = item.get("data", item.get("value"))
        dtype = _dtype(item.get("dtype"), value)
        shape = _shape(item.get("shape"), value)
        return {
            "name": item.get("name", f"input_{index}"),
            "kind": item.get("kind", "tensor" if shape else "scalar"),
            "dtype": dtype,
            "shape": shape,
            "data": value,
        }
    return {
        "name": f"input_{index}",
        "kind": "scalar",
        "dtype": _dtype(None, item),
        "shape": [],
        "data": item,
    }


def _case_payload(
    payload: dict[str, Any],
    *,
    source_path: Path,
    snapshot_path: Path,
    case_id: int | None,
    executor: str | None,
    reference_executor: str | None,
    inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    operator = _operator_payload(payload)
    operator_name = str(operator.get("name") or payload.get("name") or payload.get("api") or "imported.dump")
    api = str(operator.get("api") or payload.get("api") or operator_name)
    api_type = str(operator.get("api_type") or payload.get("api_type") or "function")
    selected_executor = executor or operator.get("executor") or payload.get("executor")
    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {}
    metadata["provenance"] = {
        "source_path": str(source_path),
        "source_format": "dump",
        "converter_version": CONVERTER_VERSION,
        "warnings": [],
        "lossy_fields": _lossy_fields(payload),
        "input_snapshot_path": str(snapshot_path),
    }
    return {
        "id": int(case_id if case_id is not None else payload.get("id", payload.get("case_id", 1))),
        "operator": {"name": operator_name, "version": operator.get("version"), "backward": bool(operator.get("backward", False))},
        "invocation": {"api": api, "api_type": api_type, "executor": selected_executor},
        "oracle": _oracle_payload(payload, reference_executor=reference_executor),
        "generator": "dump_replay",
        "generation": {"count": 1, "metadata": {"input_snapshot_path": str(snapshot_path)}},
        "parameters": [_parameter_payload(item) for item in inputs],
        "metadata": metadata,
    }


def _operator_payload(payload: dict[str, Any]) -> dict[str, Any]:
    operator = payload.get("operator", {})
    if isinstance(operator, str):
        return {"name": operator}
    if isinstance(operator, dict):
        return dict(operator)
    return {}


def _oracle_payload(payload: dict[str, Any], *, reference_executor: str | None) -> dict[str, Any]:
    oracle = dict(payload.get("oracle", {})) if isinstance(payload.get("oracle"), dict) else {}
    return {
        "comparison": oracle.get("comparison", "allclose"),
        "reference_executor": reference_executor or oracle.get("reference_executor"),
        "expected_error": oracle.get("expected_error"),
        "tolerance": oracle.get("tolerance", {}),
        "metadata": oracle.get("metadata", {}),
    }


def _parameter_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": item["name"],
        "kind": item.get("kind", "tensor"),
        "dtypes": [_dtype_alias(item.get("dtype"))] if item.get("dtype") else [],
        "shape": {"dims": item.get("shape", []), "dim_count": [len(item.get("shape", []))], "dim_values": sorted(set(item.get("shape", [])))},
        "value_range": {},
        "metadata": {"source": "dump", "snapshot_name": item["name"]},
    }


def _shape(shape: Any, value: Any) -> list[int]:
    if isinstance(shape, list):
        return [int(dim) for dim in shape]
    if isinstance(value, list):
        dims: list[int] = []
        current = value
        while isinstance(current, list):
            dims.append(len(current))
            current = current[0] if current else []
        return dims
    return []


def _dtype(dtype: Any, value: Any) -> str | None:
    if dtype is not None:
        return str(dtype)
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int64"
    if isinstance(value, float):
        return "float32"
    return None


def _dtype_alias(dtype: Any) -> str:
    text = str(dtype)
    return _DTYPE_ALIASES.get(text, text)


def _lossy_fields(payload: dict[str, Any]) -> list[str]:
    supported = {"id", "case_id", "name", "operator", "api", "api_type", "executor", "inputs", "parameters", "args", "oracle", "metadata"}
    return sorted(set(payload) - supported)

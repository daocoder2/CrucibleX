from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from cruciblex.domain.case import CaseSpec

CONVERTER_VERSION = "cruciblex.importers.profile:v1"
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


def import_profile_case(
    source: str | Path,
    *,
    case_id: int | None = None,
    executor: str | None = None,
    reference_executor: str | None = None,
) -> CaseSpec:
    source_path = Path(source)
    payload = _load_structured(source_path)
    samples = _samples(payload)
    parameters = _parameters(samples)
    operator = _operator_payload(payload)
    operator_name = str(operator.get("name") or payload.get("name") or payload.get("api") or "imported.profile")
    api = str(operator.get("api") or payload.get("api") or operator_name)
    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {}
    metadata["provenance"] = {
        "source_path": str(source_path),
        "source_format": "profile",
        "converter_version": CONVERTER_VERSION,
        "warnings": [],
        "lossy_fields": _lossy_fields(payload),
        "sample_count": len(samples),
    }
    case_data = {
        "id": int(case_id if case_id is not None else payload.get("id", payload.get("case_id", 1))),
        "operator": {"name": operator_name, "version": operator.get("version"), "backward": bool(operator.get("backward", False))},
        "invocation": {"api": api, "api_type": str(operator.get("api_type") or payload.get("api_type") or "function"), "executor": executor or operator.get("executor") or payload.get("executor")},
        "oracle": _oracle_payload(payload, reference_executor=reference_executor),
        "generator": "default",
        "generation": {
            "count": int(payload.get("count", len(samples) or 1)),
            "metadata": {
                "profile_shapes": {name: data["shapes"] for name, data in parameters.items()},
                "profile_dtypes": {name: data["dtypes"] for name, data in parameters.items()},
            },
        },
        "parameters": [_parameter_payload(name, data) for name, data in parameters.items()],
        "metadata": metadata,
    }
    return CaseSpec.model_validate(case_data)


def write_imported_profile(path: str | Path, case: CaseSpec) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump({"cases": [case.model_dump(mode="json")]}, sort_keys=False), encoding="utf-8")
    return output_path


def _load_structured(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"empty profile source: {path}")
    loaded = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise TypeError("profile source must contain a mapping")
    return loaded


def _samples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    samples = payload.get("samples") or payload.get("records") or payload.get("observations")
    if isinstance(samples, list):
        return [_sample(item) for item in samples]
    if isinstance(payload.get("parameters"), list):
        return [{"parameters": payload["parameters"]}]
    raise TypeError("profile source must contain samples, records, observations, or parameters")


def _sample(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise TypeError("profile samples must be mappings")
    parameters = item.get("parameters") or item.get("inputs") or item.get("args")
    if not isinstance(parameters, list):
        raise TypeError("profile sample must contain parameters, inputs, or args list")
    return {"parameters": parameters}


def _parameters(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    seen_shapes: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    seen_dtypes: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        for index, raw in enumerate(sample["parameters"]):
            item = _parameter_sample(raw, index)
            name = item["name"]
            if name not in grouped:
                grouped[name] = {"kind": item["kind"], "dtypes": [], "shapes": []}
            dtype = item.get("dtype")
            if dtype is not None and dtype not in seen_dtypes[name]:
                seen_dtypes[name].add(dtype)
                grouped[name]["dtypes"].append(dtype)
            shape = tuple(item.get("shape", []))
            if shape not in seen_shapes[name]:
                seen_shapes[name].add(shape)
                grouped[name]["shapes"].append(list(shape))
    return grouped


def _parameter_sample(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("profile parameter samples must be mappings")
    return {
        "name": str(raw.get("name", f"input_{index}")),
        "kind": str(raw.get("kind", "tensor")),
        "dtype": _dtype_alias(raw.get("dtype")),
        "shape": [int(dim) for dim in raw.get("shape", [])] if isinstance(raw.get("shape", []), list) else [],
    }


def _parameter_payload(name: str, data: dict[str, Any]) -> dict[str, Any]:
    shapes = data["shapes"] or [[]]
    dim_values = sorted({int(dim) for shape in shapes for dim in shape})
    ranks = sorted({len(shape) for shape in shapes})
    return {
        "name": name,
        "kind": data.get("kind", "tensor"),
        "dtypes": data.get("dtypes", []),
        "shape": {"dim_count": ranks, "dim_values": dim_values},
        "value_range": {},
        "metadata": {"source": "profile", "observed_shapes": shapes},
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
        "accuracy_policy": oracle.get("accuracy_policy", {}),
        "metadata": oracle.get("metadata", {}),
    }


def _dtype_alias(dtype: Any) -> str | None:
    if dtype is None:
        return None
    text = str(dtype)
    return _DTYPE_ALIASES.get(text, text)


def _lossy_fields(payload: dict[str, Any]) -> list[str]:
    supported = {"id", "case_id", "name", "operator", "api", "api_type", "executor", "samples", "records", "observations", "parameters", "oracle", "metadata", "count"}
    return sorted(set(payload) - supported)

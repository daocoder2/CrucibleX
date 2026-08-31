from __future__ import annotations

from copy import deepcopy
from typing import Any

DTYPE_POLICY_LIBRARY: dict[str, dict[str, Any]] = {
    "floating": {"group": "floating", "groups": {"floating": ["fp32", "fp16", "bf16", "fp64"]}},
    "low_precision_floating": {"group": "low_precision", "groups": {"low_precision": ["fp16", "bf16"]}},
    "complex": {"group": "complex", "groups": {"complex": ["complex64", "complex128"]}},
    "integer": {"group": "integer", "groups": {"integer": ["int8", "int16", "int32", "int64", "uint8"]}},
    "boolean": {"allowed": ["bool"]},
}

VALUE_POLICY_LIBRARY: dict[str, dict[str, Any]] = {
    "float_edges": {"kind": "boundary_set", "values": ["-inf", "-max", "-subnormal", "-zero", "zero", "subnormal", "max", "inf", "nan"]},
    "integer_edges": {"kind": "boundary_set", "values": ["min", "-one", "zero", "one", "max"]},
    "boolean_edges": {"kind": "boundary_set", "values": [False, True]},
    "complex_edges": {"kind": "boundary_set", "values": ["zero", "one", "inf", "nan"]},
    "signed_exponential": {"kind": "exponential", "scale": 1.0, "signed": True},
    "extreme": {"kind": "extreme", "scale": 0.5},
    "sparse": {"kind": "sparsity", "ratio": 0.8},
}

SHAPE_POLICY_LIBRARY: dict[str, dict[str, Any]] = {
    "non_contiguous": {"non_contiguous": True},
    "offset_stride": {"storage_offset": 0},
    "sliced_view": {},
}

OPERATOR_FACT_LIBRARY: dict[str, dict[str, Any]] = {
    "torch.add": {
        "parameters": {
            "input": {"dtype_policy": {"library": "floating"}, "shape_policy": {"broadcast_group": "binary"}, "value_policy": {"library": "float_edges"}},
            "other": {"dtype_policy": {"library": "floating"}, "shape_policy": {"broadcast_group": "binary"}, "value_policy": {"library": "float_edges"}},
        },
    },
    "torch.matmul": {
        "parameters": {
            "input": {"dtype_policy": {"library": "floating"}, "value_policy": {"kind": "matrix_profile", "profile": "well_conditioned"}},
            "other": {
                "dtype_policy": {"library": "floating"},
                "shape_relationship": {"kind": "dimension_alias", "source": "input", "source_dimension": 1, "dimension": 0},
                "value_policy": {"kind": "matrix_profile", "profile": "well_conditioned"},
            },
        },
    },
    "torch.softmax": {
        "parameters": {
            "input": {"dtype_policy": {"library": "floating"}, "value_policy": {"library": "signed_exponential"}},
        },
    },
}


def resolve_policy(kind: str, policy: dict[str, Any]) -> dict[str, Any]:
    library = {"dtype": DTYPE_POLICY_LIBRARY, "value": VALUE_POLICY_LIBRARY, "shape": SHAPE_POLICY_LIBRARY}[kind]
    name = policy.get("library")
    if name is not None and str(name) not in library:
        raise ValueError(f"unknown {kind} policy library: {name}")
    base = deepcopy(library.get(str(name), {})) if name is not None else {}
    return _merge(base, {key: value for key, value in policy.items() if key != "library"})


def operator_facts(name: str) -> dict[str, Any]:
    return deepcopy(OPERATOR_FACT_LIBRARY.get(name, {}))


def merge_facts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    return _merge(deepcopy(base), override)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge(dict(base[key]), value)
        else:
            base[key] = value
    return base

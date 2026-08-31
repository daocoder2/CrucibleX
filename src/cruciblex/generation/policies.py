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
        "contract": {"family": "matmul", "left": "input", "right": "other", "output_dtype": "input"},
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
    "torch.sum": {
        "contract": {"family": "reduce", "input": "input", "dim_parameter": "dim", "keepdim_parameter": "keepdim", "output_dtype": "input"},
        "parameters": {
            "input": {"dtype_policy": {"library": "floating"}, "shape_policy": {"rank_range": [1, 4]}, "value_policy": {"kind": "normal", "scale": 1.0}},
        },
    },
    "torch.mean": {
        "contract": {"family": "reduce", "input": "input", "dim_parameter": "dim", "keepdim_parameter": "keepdim", "output_dtype": "input"},
        "parameters": {
            "input": {"dtype_policy": {"library": "floating"}, "shape_policy": {"rank_range": [1, 4]}, "value_policy": {"kind": "normal", "scale": 1.0}},
        },
    },
    "torch.norm": {
        "parameters": {
            "input": {"dtype_policy": {"library": "floating"}, "shape_policy": {"rank_range": [1, 4]}, "value_policy": {"kind": "normal", "scale": 1.0}},
        },
    },
    "torch.sort": {
        "contract": {"family": "topk", "input": "input", "dim_parameter": "dim", "output_dtype": "input", "k": None},
        "parameters": {
            "input": {"dtype_policy": {"library": "floating"}, "shape_policy": {"rank_range": [1, 4]}, "value_policy": {"kind": "normal", "scale": 1.0}},
        },
    },
    "torch.topk": {
        "contract": {"family": "topk", "input": "input", "k_parameter": "k", "dim_parameter": "dim", "largest_parameter": "largest", "sorted_parameter": "sorted", "output_dtype": "input"},
        "parameters": {
            "input": {"dtype_policy": {"library": "floating"}, "shape_policy": {"rank_range": [1, 4]}, "value_policy": {"kind": "normal", "scale": 1.0}},
        },
    },
    "torch.index_select": {
        "contract": {"family": "index", "input": "input", "index": "index", "dim_parameter": "dim"},
        "parameters": {
            "input": {"dtype_policy": {"library": "floating"}, "value_policy": {"kind": "normal", "scale": 1.0}},
            "index": {"dtype_policy": {"library": "integer", "allowed": ["int64"]}},
        },
    },
    "torch.select": {
        "contract": {"family": "index", "mode": "select", "input": "input", "index": "index", "dim_parameter": "dim"},
        "parameters": {
            "input": {"dtype_policy": {"library": "floating"}},
            "index": {"dtype_policy": {"library": "integer", "allowed": ["int64"]}},
        },
    },
    "torch.gather": {
        "contract": {"family": "index", "mode": "gather", "input": "input", "index": "index", "dim_parameter": "dim"},
        "parameters": {
            "input": {"dtype_policy": {"library": "floating"}},
            "index": {"dtype_policy": {"library": "integer", "allowed": ["int64"]}},
        },
    },
    "torch.scatter": {
        "contract": {"family": "index", "mode": "scatter", "input": "input", "index": "index", "dim_parameter": "dim"},
        "parameters": {
            "input": {"dtype_policy": {"library": "floating"}},
            "index": {"dtype_policy": {"library": "integer", "allowed": ["int64"]}},
            "src": {"dtype_policy": {"library": "floating"}, "shape_relationship": {"kind": "same_shape_as", "source": "input"}},
        },
    },
    "torch.bmm": {
        "contract": {"family": "matmul", "left": "input", "right": "mat2", "output_dtype": "input", "batch_mode": "equal"},
        "parameters": {
            "input": {"dtype_policy": {"library": "floating"}, "shape_policy": {"rank_range": [3, 3]}},
            "mat2": {"dtype_policy": {"library": "floating"}, "shape_policy": {"rank_range": [3, 3]}},
        },
    },
    "torch.conv2d": {"contract": {"family": "conv", "input": "input", "weight": "weight", "bias": "bias", "attributes": ["stride", "padding", "dilation", "groups"], "shape_formula": "NCHW_OIHW", "runtime_supported": False}},
    "torch.layer_norm": {"contract": {"family": "norm", "input": "input", "normalized_shape": "normalized_shape", "weight": "weight", "bias": "bias", "eps": "eps", "runtime_supported": False}},
    "torch.scaled_dot_product_attention": {"contract": {"family": "attention", "query": "query", "key": "key", "value": "value", "mask": "attn_mask", "dropout": "dropout_p", "causal": "is_causal", "runtime_supported": False}},
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

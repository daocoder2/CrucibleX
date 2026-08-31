from __future__ import annotations

from typing import Any

import numpy as np


def quantize_bf16_reference(value: object) -> object:
    """Round fp32 values to the nearest representable bfloat16 value."""
    array = np.asarray(value, dtype=np.float32)
    bits = array.view(np.uint32)
    rounding_bias = ((bits >> 16) & 1) + 0x7FFF
    quantized = ((bits + rounding_bias) & np.uint32(0xFFFF0000)).view(np.float32)
    return quantized.item() if array.ndim == 0 else quantized


def dtype_contract(declared_dtype: str) -> dict[str, str]:
    if declared_dtype == "bf16":
        return {"declared_dtype": "bf16", "reference_dtype": "fp32", "quantization": "bfloat16_rne"}
    return {"declared_dtype": declared_dtype, "reference_dtype": declared_dtype, "quantization": "none"}


def validate_value_policy(policy: dict[str, Any], declared_dtype: str) -> dict[str, Any]:
    kind = str(policy.get("kind", ""))
    boundary_values = set(policy.get("values", [])) if kind == "boundary_set" else set()
    floating = declared_dtype.startswith(("fp", "float", "bf16"))
    complex_dtype = declared_dtype.startswith(("complex", "c64", "c128"))
    special_boundary = {"-inf", "inf", "nan", "subnormal", "-subnormal"} & boundary_values
    integer_boundary = {"min"} & boundary_values
    if kind in {"nan", "inf", "subnormal", "float_bounds", "extreme", "exponential"} and not (floating or complex_dtype):
        return {"requested": kind, "effective": None, "rejected": "unsupported_policy_for_dtype"}
    if special_boundary and not (floating or complex_dtype):
        return {"requested": sorted(special_boundary), "effective": None, "rejected": "unsupported_policy_for_dtype"}
    if complex_dtype and integer_boundary:
        return {"requested": sorted(integer_boundary), "effective": None, "rejected": "integer_boundary_requires_integer_dtype"}
    if kind == "complex_normal" and not complex_dtype:
        return {"requested": kind, "effective": None, "rejected": "complex_policy_requires_complex_dtype"}
    if kind == "subnormal" and complex_dtype:
        return {"requested": kind, "effective": None, "rejected": "subnormal_policy_requires_real_floating_dtype"}
    return {"requested": kind, "effective": kind, "rejected": None}

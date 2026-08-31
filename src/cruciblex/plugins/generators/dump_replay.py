from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cruciblex.runtime.generation import GENERATOR_REGISTRY, GenerationRequest, InputGenerator

_DTYPE_MAP = {
    "bool": np.bool_,
    "fp16": np.float16,
    "float16": np.float16,
    "fp32": np.float32,
    "float32": np.float32,
    "fp64": np.float64,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
}


@GENERATOR_REGISTRY.register("dump_replay")
class DumpReplayGenerator(InputGenerator):
    def generate(self, request: GenerationRequest) -> list[object]:
        path = request.case.generation.metadata.get("input_snapshot_path") or request.case.metadata.get("provenance", {}).get("input_snapshot_path")
        if not path:
            raise ValueError("dump_replay generator requires input_snapshot_path metadata")
        snapshot = json.loads(Path(str(path)).read_text(encoding="utf-8"))
        inputs = snapshot.get("inputs")
        if not isinstance(inputs, list):
            raise TypeError("dump replay snapshot must contain an inputs list")
        return [self._value(item) for item in inputs]

    def _value(self, item: Any) -> object:
        if not isinstance(item, dict):
            return item
        data = item.get("data")
        shape = item.get("shape") or []
        dtype = item.get("dtype")
        if shape:
            array = np.asarray(data, dtype=_DTYPE_MAP.get(str(dtype), None))
            return array.reshape([int(dim) for dim in shape])
        if dtype in _DTYPE_MAP:
            return np.asarray(data, dtype=_DTYPE_MAP[str(dtype)]).item()
        return data

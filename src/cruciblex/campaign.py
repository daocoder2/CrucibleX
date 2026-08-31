from __future__ import annotations

import hashlib
import itertools
import random
from typing import Any


def expand_campaign_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        raise TypeError("campaign file runs must be a list")
    expanded = [dict(item) for item in runs if isinstance(item, dict)]
    matrix = payload.get("matrix")
    if matrix is None:
        return expanded
    if not isinstance(matrix, dict):
        raise TypeError("campaign matrix must be a mapping")
    controls = {"base", "dimensions", "max_runs", "seed"}
    dimensions = matrix.get("dimensions")
    if dimensions is None:
        dimensions = {key: value for key, value in matrix.items() if key not in controls}
    if not isinstance(dimensions, dict):
        raise TypeError("campaign matrix dimensions must be a mapping")
    base = matrix.get("base", {})
    if not isinstance(base, dict):
        raise TypeError("campaign matrix base must be a mapping")
    keys = sorted(dimensions)
    values = []
    for key in keys:
        options = dimensions[key]
        if not isinstance(options, list) or not options:
            raise TypeError(f"campaign matrix dimension {key!r} must be a non-empty list")
        values.append(options)
    candidates = []
    for index, combination in enumerate(itertools.product(*values)):
        item = {**base, **dict(zip(keys, combination))}
        item.setdefault("name", "matrix-" + "-".join(_slug(str(item[key])) for key in keys))
        item["matrix_id"] = _matrix_id(item, index)
        candidates.append(item)
    total = len(candidates)
    max_runs = matrix.get("max_runs")
    if max_runs is not None:
        limit = int(max_runs)
        if limit < 1:
            raise ValueError("campaign matrix max_runs must be positive")
        if len(candidates) > limit:
            rng = random.Random(int(matrix.get("seed", 0)))
            selected_ids = {item["matrix_id"] for item in rng.sample(candidates, limit)}
            candidates = [item for item in candidates if item["matrix_id"] in selected_ids]
    for item in candidates:
        item["matrix_total"] = total
        item["matrix_selected"] = len(candidates)
    expanded.extend(candidates)
    return expanded


def select_campaign_shard(runs: list[dict[str, Any]], shard_index: int = 0, shard_count: int = 1) -> list[dict[str, Any]]:
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return [item for index, item in enumerate(runs) if index % shard_count == shard_index]


def _matrix_id(item: dict[str, Any], index: int) -> str:
    identity = "|".join(f"{key}={item[key]}" for key in sorted(item) if key not in {"output", "matrix_id"})
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16] or f"matrix-{index}"


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)

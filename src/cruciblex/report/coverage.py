from __future__ import annotations

from collections import Counter
from typing import Any

_DIMENSIONS = ("operator", "backend", "task", "dtype", "shape")


def summarize_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = {dimension: Counter() for dimension in _DIMENSIONS}
    combinations: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        metrics = row.get("metrics") or {}
        values = {
            "operator": str(row.get("case_name", "")),
            "backend": str(row.get("backend", "")),
            "task": str(row.get("task", "")),
            "dtype": str(metrics.get("dtype") or metrics.get("resolved_dtype") or ""),
            "shape": _shape_key(metrics.get("output_shape")),
        }
        for dimension, value in values.items():
            if value:
                counts[dimension][value] += 1
        combinations[tuple(values[dimension] for dimension in ("operator", "backend", "task"))] += 1
    return {
        "total": len(rows),
        "dimensions": {dimension: dict(sorted(counter.items())) for dimension, counter in counts.items()},
        "combinations": [
            {"operator": key[0], "backend": key[1], "task": key[2], "count": count}
            for key, count in sorted(combinations.items())
        ],
    }


def evaluate_coverage_policy(summary: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    dimensions = summary.get("dimensions", {})
    missing: list[dict[str, Any]] = []
    for dimension, required in (policy.get("required") or {}).items():
        observed = set((dimensions.get(dimension) or {}).keys())
        for value in required if isinstance(required, list) else []:
            if str(value) not in observed:
                missing.append({"dimension": dimension, "value": str(value)})
    observed_combinations = {
        (row["operator"], row["backend"], row["task"]): row["count"]
        for row in summary.get("combinations", [])
    }
    for combination in policy.get("required_combinations") or []:
        if not isinstance(combination, dict):
            continue
        key = tuple(str(combination.get(dimension, "")) for dimension in ("operator", "backend", "task"))
        if key not in observed_combinations:
            missing.append({"combination": dict(combination)})
    return {"status": "passed" if not missing else "failed", "missing": missing, "summary": summary}


def _shape_key(value: object) -> str:
    if isinstance(value, list):
        return "x".join(str(item) for item in value)
    return str(value or "")

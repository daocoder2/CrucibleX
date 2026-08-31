from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from cruciblex.domain.result import ExecutionResult
from cruciblex.storage.results import ResultStore

DEFAULT_KEY = ("case_id", "case_name", "backend", "task", "device_id")


def load_gate_policy(path: str | Path) -> dict[str, Any]:
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise TypeError("performance gate policy must be a mapping")
    return loaded


def evaluate_performance_gate(baseline: str | Path, candidate: str | Path, policy: dict[str, Any]) -> dict[str, Any]:
    baseline_root = Path(baseline)
    candidate_root = Path(candidate)
    baseline_results = ResultStore(baseline_root).read_results_jsonl()
    candidate_results = ResultStore(candidate_root).read_results_jsonl()
    metric_policies = policy.get("metrics", {})
    if not isinstance(metric_policies, dict):
        raise TypeError("performance gate metrics must be a mapping")
    comparisons: list[dict[str, Any]] = []
    baseline_by_key = {_result_key(result): result for result in baseline_results}
    candidate_by_key = {_result_key(result): result for result in candidate_results}
    for key in sorted(set(baseline_by_key) | set(candidate_by_key)):
        base = baseline_by_key.get(key)
        current = candidate_by_key.get(key)
        if base is None or current is None:
            comparisons.append({"key": list(key), "status": "insufficient_data", "reason": "missing baseline or candidate result"})
            continue
        for metric, metric_policy in metric_policies.items():
            if _metric_applies(metric_policy, current.task.value):
                comparisons.append(_compare_metric(key, metric, base, current, metric_policy))
    profiler = _compare_profiler(baseline_root, candidate_root, policy.get("profiler", {}))
    failed = [item for item in comparisons if item["status"] == "regressed"]
    insufficient = [item for item in comparisons if item["status"] == "insufficient_data"]
    profiler_failed = profiler.get("status") == "regressed"
    profiler_insufficient = profiler.get("status") == "insufficient_data"
    return {
        "status": "regressed" if failed or profiler_failed else ("insufficient_data" if insufficient or profiler_insufficient else "passed"),
        "baseline": str(baseline_root),
        "candidate": str(candidate_root),
        "metrics": comparisons,
        "profiler": profiler,
        "regressions": len(failed) + int(profiler_failed),
        "insufficient_data": len(insufficient) + int(profiler_insufficient),
    }


def write_performance_gate(result: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _result_key(result: ExecutionResult) -> tuple[str, str, str, str, str]:
    return (str(result.case_id), result.case_name, result.backend.value, result.task.value, str(result.device_id))


def _metric_applies(policy: Any, task: str) -> bool:
    if not isinstance(policy, dict):
        return True
    tasks = policy.get("tasks")
    return tasks is None or task in tasks


def _compare_metric(key: tuple[str, str, str, str, str], metric: str, base: ExecutionResult, current: ExecutionResult, policy: Any) -> dict[str, Any]:
    base_value = base.metrics.get(metric)
    current_value = current.metrics.get(metric)
    item: dict[str, Any] = {"key": list(key), "metric": metric, "baseline": base_value, "candidate": current_value}
    if not isinstance(base_value, (int, float)) or not isinstance(current_value, (int, float)):
        item.update({"status": "insufficient_data", "reason": "metric missing or non-numeric"})
        return item
    delta = float(current_value) - float(base_value)
    delta_percent = (delta / abs(float(base_value)) * 100.0) if base_value else (0.0 if delta == 0 else None)
    item.update({"delta": delta, "delta_percent": delta_percent, "status": "passed"})
    if not isinstance(policy, dict):
        return item
    max_regression = policy.get("max_regression_percent")
    min_regression = policy.get("min_regression_percent")
    if delta_percent is None:
        item.update({"status": "insufficient_data", "reason": "baseline metric is zero"})
    elif max_regression is not None and delta_percent > float(max_regression):
        item.update({"status": "regressed", "threshold": {"max_regression_percent": float(max_regression)}})
    elif min_regression is not None and delta_percent < -float(min_regression):
        item.update({"status": "regressed", "threshold": {"min_regression_percent": float(min_regression)}})
    return item


def _compare_profiler(baseline_root: Path, candidate_root: Path, policy: Any) -> dict[str, Any]:
    if isinstance(policy, dict) and policy.get("required") is False:
        return {"status": "not_required"}
    base = _read_json(baseline_root / "profiler_summary.json")
    current = _read_json(candidate_root / "profiler_summary.json")
    if base is None or current is None:
        return {"status": "insufficient_data", "reason": "profiler_summary.json missing"}
    base_ops = {str(row.get("op_type")): row for row in base.get("top_operators", [])}
    current_ops = {str(row.get("op_type")): row for row in current.get("top_operators", [])}
    rows = []
    for name in sorted(set(base_ops) | set(current_ops)):
        old = base_ops.get(name)
        new = current_ops.get(name)
        if old is None or new is None:
            rows.append({"op_type": name, "status": "insufficient_data"})
            continue
        old_time = float(old.get("total_time_us", 0.0))
        new_time = float(new.get("total_time_us", 0.0))
        rows.append({"op_type": name, "baseline_total_time_us": old_time, "candidate_total_time_us": new_time, "delta_us": new_time - old_time, "delta_percent": ((new_time - old_time) / abs(old_time) * 100.0) if old_time else None, "status": "passed"})
    threshold = policy.get("max_operator_regression_percent") if isinstance(policy, dict) else None
    if threshold is not None:
        for row in rows:
            if row.get("delta_percent") is not None and row["delta_percent"] > float(threshold):
                row.update({"status": "regressed", "threshold": {"max_operator_regression_percent": float(threshold)}})
    status = "regressed" if any(row.get("status") == "regressed" for row in rows) else "compared"
    return {"status": status, "operators": rows}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cruciblex.domain.enums import TaskKind
from cruciblex.domain.result import ExecutionResult
from cruciblex.report.cross_compare import CrossDeviceComparator

_PERFORMANCE_TASKS = {
    TaskKind.PERFORMANCE_DEVICE,
    TaskKind.PERFORMANCE_DEVICE_PTA,
    TaskKind.PERFORMANCE_E2E,
    TaskKind.PERFORMANCE_BENCHMARK,
}
_MEMORY_TASKS = {TaskKind.MEMORY_DEVICE}


class ResultPostProcessor:
    """Apply driver-side post-processing after scheduler collection."""

    def __init__(self, output_root: str | Path | None = None) -> None:
        self.output_root = Path(output_root) if output_root is not None else None

    def process(self, results: list[ExecutionResult]) -> list[ExecutionResult]:
        processed: list[ExecutionResult] = list(results)
        processed.extend(CrossDeviceComparator(self.output_root).compare(results))
        if self.output_root is not None:
            self.write_summary(processed)
        return processed

    def write_summary(self, results: list[ExecutionResult]) -> Path:
        if self.output_root is None:
            raise ValueError("output_root is required to write postprocess summary")
        payload = self.summarize(results)
        path = self.output_root / "postprocess.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def summarize(self, results: list[ExecutionResult]) -> dict[str, Any]:
        status_by_task_backend: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        performance: list[dict[str, Any]] = []
        memory: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []
        invalid_cases: list[dict[str, Any]] = []
        fuzz_cases: list[dict[str, Any]] = []
        failure_rows: list[dict[str, Any]] = []

        for result in results:
            row = self._row(result)
            if result.status.value != "passed":
                failure_rows.append(row)
            if result.metrics.get("stage") == "cross_device_compare":
                comparisons.append(row)
                continue
            if result.metrics.get("expected_invalid"):
                invalid_cases.append(row)
            if result.metrics.get("fuzz_case"):
                fuzz_cases.append(row)
            task = result.task.value
            backend = result.backend.value
            status_by_task_backend[task][backend][result.status.value] += 1
            if result.task in _PERFORMANCE_TASKS:
                performance.append(row)
            elif result.task in _MEMORY_TASKS:
                memory.append(row)

        return {
            "status_by_task_backend": {
                task: {backend: dict(counts) for backend, counts in by_backend.items()}
                for task, by_backend in status_by_task_backend.items()
            },
            "performance": performance,
            "memory": memory,
            "comparisons": comparisons,
            "invalid_cases": invalid_cases,
            "fuzz_cases": fuzz_cases,
            "failure_clusters": self._failure_clusters(failure_rows),
        }

    def _failure_clusters(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        clusters: dict[str, dict[str, Any]] = {}
        for row in rows:
            signature = self._failure_signature(row)
            cluster = clusters.setdefault(
                signature,
                {
                    "signature": signature,
                    "count": 0,
                    "statuses": Counter(),
                    "task": row["task"],
                    "backend": row["backend"],
                    "expected_invalid": bool(row["metrics"].get("expected_invalid")),
                    "failure_kind": row["failure_kind"],
                    "failure_stage": row["failure_stage"],
                    "skip_reason": row["skip_reason"],
                    "error": self._failure_message(row),
                    "compare_detail": row["metrics"].get("compare_detail"),
                    "cases": [],
                },
            )
            cluster["count"] += 1
            cluster["statuses"][row["status"]] += 1
            cluster["cases"].append(
                {
                    "plan_id": row["plan_id"],
                    "case_id": row["case_id"],
                    "case_name": row["case_name"],
                    "node_name": row["node_name"],
                    "device_id": row["device_id"],
                    "source_case_id": row["metrics"].get("source_case_id"),
                    "generation_index": row["metrics"].get("generation_index"),
                    "invalid_index": row["metrics"].get("invalid_index"),
                }
            )
        return [
            {**cluster, "statuses": dict(cluster["statuses"])}
            for cluster in sorted(clusters.values(), key=lambda item: (-item["count"], item["signature"]))
        ]

    def _failure_signature(self, row: dict[str, Any]) -> str:
        metrics = row["metrics"]
        return "|".join(
            [
                row["case_name"],
                row["task"],
                row["backend"],
                row["status"],
                row["failure_kind"],
                str(bool(metrics.get("expected_invalid"))),
                str(metrics.get("dtype") or metrics.get("resolved_dtype") or ""),
                self._shape_key(metrics.get("output_shape")),
                str(metrics.get("error_type") or ""),
                str(metrics.get("compare_detail") or ""),
            ]
        )

    def _shape_key(self, value: object) -> str:
        if isinstance(value, list):
            return "x".join(str(item) for item in value)
        return str(value or "")

    def _failure_message(self, row: dict[str, Any]) -> str:
        metrics = row["metrics"]
        message = row.get("error") or metrics.get("actual_error") or metrics.get("compare_detail") or row["status"]
        return str(message).splitlines()[0]

    def _row(self, result: ExecutionResult) -> dict[str, Any]:
        return {
            "plan_id": result.plan_id,
            "case_id": result.case_id,
            "case_name": result.case_name,
            "node_name": result.node_name,
            "backend": result.backend.value,
            "device_id": result.device_id,
            "task": result.task.value,
            "status": result.status.value,
            "failure_kind": result.metrics.get("failure_kind", result.status.value),
            "failure_stage": result.metrics.get("failure_stage"),
            "skip_reason": result.metrics.get("skip_reason"),
            "metrics": result.metrics,
            "error": result.error,
        }

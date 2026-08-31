from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from cruciblex.domain.result import ExecutionResult
from cruciblex.domain.run import RunManifest


class ResultStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def ensure(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def manifest_path(self, name: str = "manifest.json") -> Path:
        return self.ensure() / name

    def write_manifest(self, manifest: RunManifest, name: str = "manifest.json") -> Path:
        path = self.manifest_path(name)
        path.write_text(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def read_manifest(self, name: str = "manifest.json") -> RunManifest:
        path = self.manifest_path(name)
        return RunManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def write_results_jsonl(self, results: list[ExecutionResult], name: str = "results.jsonl") -> Path:
        path = self.ensure() / name
        lines = [json.dumps(result.with_derived_evidence().model_dump(mode="json"), ensure_ascii=False) for result in results]
        content = "\n".join(lines)
        if content:
            content += "\n"
        path.write_text(content, encoding="utf-8")
        return path

    def write_results_csv(self, results: list[ExecutionResult], name: str = "results.csv") -> Path:
        path = self.ensure() / name
        fieldnames = [
            "result_schema_version",
            "plan_id",
            "case_id",
            "case_name",
            "node_name",
            "backend",
            "device_id",
            "task",
            "status",
            "error",
            "metrics_json",
            "evidence_json",
            "artifact_count",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(self._result_row(result.with_derived_evidence()))
        return path

    def read_results_jsonl(self, name: str = "results.jsonl") -> list[ExecutionResult]:
        path = self.ensure() / name
        if not path.exists():
            return []
        return [ExecutionResult.model_validate(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def write_summary_json(self, summary: dict[str, Any], name: str = "summary.json") -> Path:
        path = self.ensure() / name
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _result_row(self, result: ExecutionResult) -> dict[str, str]:
        return {
            "result_schema_version": str(result.result_schema_version),
            "plan_id": result.plan_id,
            "case_id": str(result.case_id),
            "case_name": result.case_name,
            "node_name": result.node_name,
            "backend": result.backend.value,
            "device_id": str(result.device_id),
            "task": result.task.value,
            "status": result.status.value,
            "error": result.error or "",
            "metrics_json": json.dumps(result.metrics, ensure_ascii=False, sort_keys=True),
            "evidence_json": json.dumps(result.evidence.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) if result.evidence else "",
            "artifact_count": str(len(result.artifacts)),
        }

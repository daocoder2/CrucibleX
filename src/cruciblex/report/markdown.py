from __future__ import annotations

import json
from pathlib import Path

from cruciblex.domain.result import ExecutionResult
from cruciblex.storage.results import ResultStore


class MarkdownReportWriter:
    def __init__(self, output_root: str | Path) -> None:
        self._store = ResultStore(output_root)

    def write(self, name: str = "report.md") -> Path:
        manifest = self._store.read_manifest()
        results = self._store.read_results_jsonl()
        summary = self._read_summary()
        path = self._store.ensure() / name
        path.write_text(self._render(manifest, summary, results), encoding="utf-8")
        return path

    def _read_summary(self) -> dict[str, object]:
        path = self._store.ensure() / "summary.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _render(self, manifest, summary: dict[str, object], results: list[ExecutionResult]) -> str:
        tasks = ", ".join(task.value for task in manifest.tasks)
        results_path = manifest.results_path or ""
        summary_path = manifest.summary_path or ""
        total = summary.get("total", 0)
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        lines: list[str] = [
            "# CrucibleX Report",
            "",
            "## Run",
            f"- run_id: {manifest.run_id}",
            f"- created_at: {manifest.created_at.isoformat()}",
            f"- case_path: {manifest.case_path}",
            f"- node_path: {manifest.node_path}",
            f"- tasks: {tasks}",
            f"- scheduler: {manifest.scheduler.value}",
            f"- plan_count: {manifest.plan_count}",
            f"- submitted_count: {manifest.submitted_count}",
            f"- skipped_count: {manifest.skipped_count}",
            f"- results_path: {results_path}",
            f"- summary_path: {summary_path}",
            "",
            "## Summary",
            f"- total: {total}",
            f"- passed: {passed}",
            f"- failed: {failed}",
            "",
            "## Failures",
        ]
        failures = [result for result in results if result.status.value != "passed"]
        if not failures:
            lines.append("- none")
        else:
            for result in failures:
                error = result.error or ""
                detail = result.metrics.get("compare_detail", "")
                lines.extend([
                    f"- plan_id: {result.plan_id}",
                    f"  case_name: {result.case_name}",
                    f"  task: {result.task.value}",
                    f"  backend: {result.backend.value}",
                    f"  device_id: {result.device_id}",
                    f"  error: {error}",
                    f"  compare_detail: {detail}",
                ])
        lines.extend(["", "## Artifacts"])
        if not results:
            lines.append("- none")
        else:
            for result in results:
                lines.append(f"- {result.plan_id}")
                for artifact in result.artifacts:
                    lines.append(f"  - {artifact.name}: {artifact.path}")
        lines.append("")
        return "\n".join(lines)
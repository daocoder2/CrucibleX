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
        postprocess = self._read_postprocess()
        path = self._store.ensure() / name
        path.write_text(self._render(manifest, summary, results, postprocess), encoding="utf-8")
        return path

    def _read_summary(self) -> dict[str, object]:
        path = self._store.ensure() / "summary.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_postprocess(self) -> dict[str, object]:
        path = self._store.ensure() / "postprocess.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_campaign_summary(self) -> dict[str, object]:
        path = self._store.ensure() / "campaign_summary.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _render(self, manifest, summary: dict[str, object], results: list[ExecutionResult], postprocess: dict[str, object]) -> str:
        tasks = ", ".join(task.value for task in manifest.tasks)
        results_path = manifest.results_path or ""
        summary_path = manifest.summary_path or ""
        total = summary.get("total", 0)
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        fuzz_cases = postprocess.get("fuzz_cases", [])
        failure_clusters = postprocess.get("failure_clusters", [])
        compatibility = manifest.metadata.get("runtime_compatibility", {})
        compatibility_status = compatibility.get("status", "unavailable") if isinstance(compatibility, dict) else "unavailable"
        version_policy = manifest.metadata.get("version_policy", "warn")
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
            f"- input_schema_version: {manifest.input_schema_version}",
            f"- runtime_compatibility: {compatibility_status}",
            f"- version_policy: {version_policy}",
            "",
            "## Summary",
            f"- total: {total}",
            f"- passed: {passed}",
            f"- failed: {failed}",
            f"- fuzz_cases: {len(fuzz_cases)}",
            f"- failure_clusters: {len(failure_clusters)}",
            "",
            "## Failures",
        ]
        lines.extend(["", "## Hardware Evidence"])
        evidence_results = [result for result in results if result.evidence is not None]
        if not evidence_results:
            lines.append("- none")
        else:
            for result in evidence_results:
                evidence = result.evidence
                lines.extend([
                    f"- plan_id: {result.plan_id}",
                    f"  status: {result.status.value}",
                    f"  backend: {evidence.backend.value}",
                    f"  device_id: {evidence.device_id}",
                    f"  resolved_device: {evidence.resolved_device or ''}",
                    f"  probe_status: {evidence.probe_status}",
                    f"  runtime: {json.dumps(evidence.runtime, sort_keys=True)}",
                    f"  evidence_fingerprint: {evidence.fingerprint or ''}",
                    f"  evidence_artifacts: {', '.join(evidence.artifact_refs)}",
                ])
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
                    f"  failure_kind: {result.metrics.get('failure_kind', result.status.value)}",
                    f"  failure_stage: {result.metrics.get('failure_stage', '')}",
                    f"  skip_reason: {result.metrics.get('skip_reason', '')}",
                    f"  error: {error}",
                    f"  compare_detail: {detail}",
                ])
        lines.extend(["", "## Failure Clusters"])
        if not failure_clusters:
            lines.append("- none")
        else:
            for cluster in failure_clusters:
                lines.extend(
                    [
                        f"- signature: {cluster.get('signature', '')}",
                        f"  count: {cluster.get('count', 0)}",
                        f"  task: {cluster.get('task', '')}",
                        f"  backend: {cluster.get('backend', '')}",
                        f"  failure_kind: {cluster.get('failure_kind', '')}",
                        f"  failure_stage: {cluster.get('failure_stage', '')}",
                        f"  error: {cluster.get('error', '')}",
                    ]
                )

        lines.extend(["", "## Fuzz Cases"])
        if not fuzz_cases:
            lines.append("- none")
        else:
            for row in fuzz_cases:
                metrics = row.get("metrics", {})
                lines.extend(
                    [
                        f"- plan_id: {row.get('plan_id', '')}",
                        f"  case_name: {row.get('case_name', '')}",
                        f"  status: {row.get('status', '')}",
                        f"  backend: {row.get('backend', '')}",
                        f"  source_case_id: {metrics.get('source_case_id', '')}",
                        f"  generation_index: {metrics.get('generation_index', '')}",
                        f"  generation_seed: {metrics.get('generation_seed', '')}",
                        f"  failure_kind: {row.get('failure_kind', '')}",
                        f"  error: {row.get('error', '') or metrics.get('actual_error', '')}",
                    ]
                )

        lines.extend(["", "## Artifacts"])
        campaign_summary = self._read_campaign_summary()
        if campaign_summary:
            lines.extend(["", "## Batch Summary"])
            for row in campaign_summary.get("runs", []):
                lines.extend([
                    f"- {row.get('name', '')}",
                    f"  output: {row.get('output', '')}",
                    f"  total: {row.get('total', 0)}",
                    f"  passed: {row.get('passed', 0)}",
                    f"  failed: {row.get('failed', 0)}",
                    f"  fuzz_cases: {row.get('fuzz_cases', 0)}",
                    f"  failure_clusters: {row.get('failure_clusters', 0)}",
                ])
        if not results:
            lines.append("- none")
        else:
            for result in results:
                lines.append(f"- {result.plan_id}")
                for artifact in result.artifacts:
                    lines.append(f"  - {artifact.name}: {artifact.path}")
        lines.append("")
        return "\n".join(lines)
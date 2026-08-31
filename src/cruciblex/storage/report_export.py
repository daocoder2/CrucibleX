from __future__ import annotations

import json
from typing import Any

from cruciblex.domain.result import ExecutionResult
from cruciblex.domain.run import RunManifest

REPORT_EXPORT_SCHEMA_VERSION = 1

REPORT_EXPORT_FIELDS = [
    "report_schema_version",
    "run_id",
    "result_schema_version",
    "plan_id",
    "case_id",
    "case_name",
    "task",
    "status",
    "node_name",
    "backend",
    "device_id",
    "resolved_device",
    "candidate_executor",
    "case_fingerprint",
    "matrix_id",
    "source_case_id",
    "generation_index",
    "generation_seed",
    "comparison",
    "max_abs_diff",
    "mean_abs_diff",
    "max_relative_error",
    "mean_relative_error",
    "rmse",
    "matched_ratio",
    "latency_ms",
    "throughput_items_per_s",
    "memory_peak_bytes",
    "failure_kind",
    "failure_stage",
    "error",
    "hardware_probe_status",
    "hardware_fingerprint",
    "runtime_policy_json",
    "metrics_json",
    "evidence_json",
    "artifact_count",
]


def project_result(manifest: RunManifest, result: ExecutionResult) -> dict[str, Any]:
    result = result.with_derived_evidence()
    metrics = result.metrics
    evidence = result.evidence
    input_artifact = next((artifact for artifact in result.artifacts if artifact.kind == "inputs"), None)
    return {
        "report_schema_version": REPORT_EXPORT_SCHEMA_VERSION,
        "run_id": manifest.run_id,
        "result_schema_version": result.result_schema_version,
        "plan_id": result.plan_id,
        "case_id": result.case_id,
        "case_name": result.case_name,
        "task": result.task.value,
        "status": result.status.value,
        "node_name": result.node_name,
        "backend": result.backend.value,
        "device_id": result.device_id,
        "resolved_device": metrics.get("resolved_device", ""),
        "candidate_executor": metrics.get("candidate_executor", ""),
        "case_fingerprint": input_artifact.metadata.get("case_fingerprint", "") if input_artifact else "",
        "matrix_id": metrics.get("matrix_id", ""),
        "source_case_id": metrics.get("source_case_id", ""),
        "generation_index": metrics.get("generation_index", ""),
        "generation_seed": metrics.get("generation_seed", ""),
        "comparison": metrics.get("comparison", ""),
        "max_abs_diff": metrics.get("max_abs_diff", ""),
        "mean_abs_diff": metrics.get("mean_abs_diff", ""),
        "max_relative_error": metrics.get("max_relative_error", ""),
        "mean_relative_error": metrics.get("mean_relative_error", ""),
        "rmse": metrics.get("rmse", ""),
        "matched_ratio": metrics.get("matched_ratio", ""),
        "latency_ms": metrics.get("latency_ms", ""),
        "throughput_items_per_s": metrics.get("throughput_items_per_s", ""),
        "memory_peak_bytes": metrics.get("memory_peak_bytes", metrics.get("hardware_memory_peak_bytes", "")),
        "failure_kind": metrics.get("failure_kind", ""),
        "failure_stage": metrics.get("failure_stage", ""),
        "error": result.error or "",
        "hardware_probe_status": evidence.probe_status if evidence else "unknown",
        "hardware_fingerprint": evidence.fingerprint or "" if evidence else "",
        "runtime_policy_json": json.dumps(metrics.get("runtime_policy", {}), ensure_ascii=False, sort_keys=True),
        "metrics_json": json.dumps(metrics, ensure_ascii=False, sort_keys=True),
        "evidence_json": json.dumps(evidence.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) if evidence else "",
        "artifact_count": len(result.artifacts),
    }

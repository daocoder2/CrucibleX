from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from cruciblex.domain.enums import BackendKind, ExecutionRole, ResultStatus, TaskKind


class ArtifactPayload(BaseModel):
    name: str
    kind: str
    data: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactRef(BaseModel):
    name: str
    path: Path
    kind: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class HardwareEvidence(BaseModel):
    schema_version: Literal[1] = 1
    backend: BackendKind
    host: str | None = None
    node: str | None = None
    device_id: int | None = None
    resolved_device: str | None = None
    probe_status: str = "unknown"
    runtime: dict[str, Any] = Field(default_factory=dict)
    fingerprint: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    result_schema_version: Literal[1] = 1
    plan_id: str
    case_id: int
    case_name: str
    node_name: str
    backend: BackendKind
    device_id: int
    task: TaskKind
    status: ResultStatus
    candidate_role: ExecutionRole | None = None
    reference_role: ExecutionRole | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    evidence: HardwareEvidence | None = None
    artifact_payloads: list[ArtifactPayload] = Field(default_factory=list, exclude=True)
    error: str | None = None

    def with_derived_evidence(self) -> ExecutionResult:
        if self.evidence is not None or self.metrics.get("stage") == "cross_device_compare":
            return self
        metrics = self.metrics
        artifact_refs = [str(artifact.path) for artifact in self.artifacts if artifact.kind in {"gpu_evidence", "npu_evidence"}]
        for key in ("gpu_evidence_path", "npu_evidence_path"):
            if metrics.get(key):
                artifact_refs.append(str(metrics[key]))
        artifact_refs = list(dict.fromkeys(artifact_refs))
        probe_status = "unknown"
        runtime: dict[str, Any] = {}
        fingerprint = None
        if self.backend == BackendKind.GPU and "gpu_available" in metrics:
            probe_status = "available" if metrics.get("gpu_available") else "unavailable"
            runtime = {key: metrics[key] for key in ("cuda_version", "gpu_device_count") if metrics.get(key) is not None}
            fingerprint = metrics.get("gpu_evidence_fingerprint")
        if self.backend in {BackendKind.NPU, BackendKind.ACLNN} and "npu_available" in metrics:
            probe_status = "available" if metrics.get("npu_available") else "unavailable"
            runtime = {key: metrics[key] for key in ("npu_device_count", "torch_version", "torch_npu_version", "npu_device_name") if metrics.get(key) is not None}
            fingerprint = metrics.get("npu_evidence_fingerprint")
        return self.model_copy(
            update={
                "evidence": HardwareEvidence(
                    backend=self.backend,
                    host=metrics.get("host"),
                    node=metrics.get("node"),
                    device_id=metrics.get("device_id", self.device_id),
                    resolved_device=metrics.get("resolved_device"),
                    probe_status=probe_status,
                    runtime=runtime,
                    fingerprint=fingerprint,
                    artifact_refs=artifact_refs,
                )
            }
        )

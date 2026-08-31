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

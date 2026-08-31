from __future__ import annotations

from pathlib import Path
from typing import Any

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


class ExecutionResult(BaseModel):
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
    artifact_payloads: list[ArtifactPayload] = Field(default_factory=list, exclude=True)
    error: str | None = None

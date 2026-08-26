from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from cruciblex.domain.enums import SchedulerKind, TaskKind


def new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{timestamp}-{uuid4().hex[:8]}"


class RunContext(BaseModel):
    run_id: str = Field(default_factory=new_run_id)
    case_path: Path
    node_path: Path
    tasks: list[TaskKind]
    scheduler: SchedulerKind
    output_root: Path
    plugin_paths: list[Path] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_manifest(
        self,
        cruciblex_version: str,
        plan_count: int,
        submitted_count: int = 0,
        skipped_count: int = 0,
    ) -> RunManifest:
        return RunManifest(
            run_id=self.run_id,
            case_path=self.case_path,
            node_path=self.node_path,
            tasks=self.tasks,
            scheduler=self.scheduler,
            output_root=self.output_root,
            plugin_paths=self.plugin_paths,
            metadata=self.metadata,
            cruciblex_version=cruciblex_version,
            plan_count=plan_count,
            submitted_count=submitted_count,
            skipped_count=skipped_count,
        )


class RunManifest(BaseModel):
    run_id: str = Field(default_factory=new_run_id)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    case_path: Path
    node_path: Path
    tasks: list[TaskKind]
    scheduler: SchedulerKind
    output_root: Path
    plugin_paths: list[Path] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    cruciblex_version: str
    plan_count: int = 0
    submitted_count: int = 0
    skipped_count: int = 0
    results_path: Path | None = None
    summary_path: Path | None = None

    def with_outputs(self, results_path: Path, summary_path: Path) -> RunManifest:
        return self.model_copy(update={"results_path": results_path, "summary_path": summary_path})

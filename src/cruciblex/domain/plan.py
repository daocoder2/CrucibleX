from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from cruciblex.domain.case import CaseSpec
from cruciblex.domain.enums import SchedulerKind, TaskKind
from cruciblex.domain.node import DeviceSpec, NodeSpec


class ArtifactPolicy(BaseModel):
    output_root: Path = Path("cx_output")
    save_inputs: bool = False
    save_outputs: bool = False
    save_profiles: bool = False


class JobSpec(BaseModel):
    cases: list[CaseSpec]
    nodes: list[NodeSpec]
    tasks: list[TaskKind] = Field(default_factory=lambda: [TaskKind.ACCURACY])
    scheduler: SchedulerKind = SchedulerKind.RAY
    artifacts: ArtifactPolicy = Field(default_factory=ArtifactPolicy)

    @field_validator("cases", "nodes", "tasks", mode="after")
    @classmethod
    def ensure_non_empty(cls, value):
        if not value:
            raise ValueError("value must not be empty")
        return value


class ExecutionPlan(BaseModel):
    case: CaseSpec
    node: NodeSpec
    device: DeviceSpec
    task: TaskKind
    artifacts: ArtifactPolicy = Field(default_factory=ArtifactPolicy)

    @property
    def plan_id(self) -> str:
        return f"{self.case.id}:{self.node.display_name}:{self.device.slot}:{self.task.value}"
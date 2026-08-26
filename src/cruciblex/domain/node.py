from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from cruciblex.domain.enums import BackendKind, TaskKind


class DeviceSpec(BaseModel):
    id: int
    backend: BackendKind
    labels: set[str] = Field(default_factory=set)
    resources: dict[str, float] = Field(default_factory=dict)

    @property
    def slot(self) -> str:
        return f"{self.backend.value}:{self.id}"


class NodeSpec(BaseModel):
    name: str | None = None
    host: str = "127.0.0.1"
    devices: list[DeviceSpec] = Field(default_factory=list)
    output_path: Path | None = None
    role: str = "candidate"
    allowed_tasks: set[TaskKind] = Field(default_factory=set)
    labels: set[str] = Field(default_factory=set)

    @field_validator("devices", mode="after")
    @classmethod
    def ensure_devices(cls, devices: list[DeviceSpec]) -> list[DeviceSpec]:
        if not devices:
            raise ValueError("node must declare at least one device")
        return devices

    @property
    def display_name(self) -> str:
        return self.name or self.host

    def supports(self, task: TaskKind) -> bool:
        return not self.allowed_tasks or task in self.allowed_tasks

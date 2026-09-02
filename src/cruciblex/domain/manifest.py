from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cruciblex.domain.enums import BackendKind

MANIFEST_V1_TOP_LEVEL_FIELDS = ("version", "kind", "task", "lanes", "runtime", "filters", "reporting")
MANIFEST_V1_LANE_KINDS = ("contract", "hardware", "preflight_blocked")
MANIFEST_V1_VALIDATE_JSON_FIELDS = ("manifest", "task", "lanes", "cases", "lane_kinds")
MANIFEST_V1_PLAN_JSON_FIELDS = ("manifest", "task", "cases", "plans", "items")
MANIFEST_V1_PLAN_ITEM_FIELDS = ("plan_id", "lane", "lane_kind", "case_id", "case_name", "backend", "node", "device_id", "task")
HARDWARE_EVIDENCE_ARCHIVE_FILES = ("manifest.json", "results.jsonl", "report.jsonl", "summary.json", "postprocess.json")



class ManifestTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "manifest"
    description: str | None = None
    defaults: dict[str, object] = Field(default_factory=dict)


class ManifestCaseInclude(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include: str

    @field_validator("include")
    @classmethod
    def ensure_non_empty_include(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("manifest case include must not be empty")
        return value


class ManifestLane(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["contract", "hardware", "preflight_blocked"] = "contract"
    backends: list[BackendKind] | None = None
    cases: list[ManifestCaseInclude]

    @field_validator("cases", mode="after")
    @classmethod
    def ensure_cases(cls, value: list[ManifestCaseInclude]) -> list[ManifestCaseInclude]:
        if not value:
            raise ValueError("manifest lane must contain at least one case include")
        return value


class ManifestRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow_generated_cases: bool = True
    allow_invalid_cases: bool = True
    require_real_evidence: bool = False
    require_backend_dtype_source: str | None = None


class ManifestFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_operators: list[str] = Field(default_factory=list)
    exclude_operators: list[str] = Field(default_factory=list)
    include_backends: list[str] = Field(default_factory=list)
    exclude_backends: list[str] = Field(default_factory=list)
    include_tasks: list[str] = Field(default_factory=list)
    exclude_tasks: list[str] = Field(default_factory=list)
    include_dtypes: list[str] = Field(default_factory=list)
    exclude_dtypes: list[str] = Field(default_factory=list)
    include_tags: list[str] = Field(default_factory=list)
    exclude_tags: list[str] = Field(default_factory=list)


class ManifestReporting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str | None = None
    emit_case_index: bool = False
    emit_lane_index: bool = False


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    kind: Literal["manifest"] = "manifest"
    task: ManifestTask = Field(default_factory=ManifestTask)
    lanes: list[ManifestLane]
    runtime: ManifestRuntime = Field(default_factory=ManifestRuntime)
    filters: ManifestFilters = Field(default_factory=ManifestFilters)
    reporting: ManifestReporting = Field(default_factory=ManifestReporting)

    @field_validator("lanes", mode="after")
    @classmethod
    def ensure_lanes(cls, value: list[ManifestLane]) -> list[ManifestLane]:
        if not value:
            raise ValueError("manifest must contain at least one lane")
        names = [lane.name for lane in value]
        if len(names) != len(set(names)):
            raise ValueError("manifest lane names must be unique")
        return value

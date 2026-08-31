from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from cruciblex.domain.enums import ParameterKind


class ValueRange(BaseModel):
    valid: list[Any] = Field(default_factory=list)
    invalid: list[Any] = Field(default_factory=list)


class ShapeSpec(BaseModel):
    dims: list[int] | None = None
    dim_count: list[int] = Field(default_factory=list)
    dim_values: list[int] = Field(default_factory=list)
    max_elements: int | None = None


class ParameterSpec(BaseModel):
    name: str | None = None
    kind: ParameterKind
    required: bool = True
    dtypes: list[str] = Field(default_factory=list)
    shape: ShapeSpec | None = None
    value_range: ValueRange = Field(default_factory=ValueRange)
    requires_grad: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class OperatorSpec(BaseModel):
    name: str
    version: str | None = None
    backward: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class InvocationSpec(BaseModel):
    api: str
    api_type: str
    executor: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OracleSpec(BaseModel):
    comparison: str = "allclose"
    reference_executor: str | None = None
    expected_error: str | None = None
    tolerance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationSpec(BaseModel):
    count: int = 1
    invalid_count: int = 0
    seed: int = 0
    constraints: list[str] = Field(default_factory=list)
    max_elements: int | None = None
    max_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseSpec(BaseModel):
    id: int
    operator: OperatorSpec
    invocation: InvocationSpec
    parameters: list[ParameterSpec] = Field(default_factory=list)
    oracle: OracleSpec = Field(default_factory=OracleSpec)
    generator: str = "default"
    generation: GenerationSpec = Field(default_factory=GenerationSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.operator.name

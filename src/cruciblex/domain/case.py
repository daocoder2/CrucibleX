from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

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
    values: Any | None = None
    value_range: ValueRange = Field(default_factory=ValueRange)
    requires_grad: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class OperatorSpec(BaseModel):
    name: str
    version: str | None = None
    backward: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class InvocationBindingSpec(BaseModel):
    mode: Literal["positional", "keyword", "mixed"] = "positional"
    names: list[str] = Field(default_factory=list)
    positional: list[int] = Field(default_factory=list)
    omit: list[str | int] = Field(default_factory=list)


class InvocationSpec(BaseModel):
    api: str
    api_type: str
    executor: str | None = None
    binding: InvocationBindingSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AccuracyPolicySpec(BaseModel):
    category: Literal["non_computational", "integer", "quantized", "floating"] | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    small_value_threshold: float | None = None
    max_error_count: int | None = None
    relative_epsilon: float | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> AccuracyPolicySpec:
        allowed = {
            "non_computational": set(),
            "integer": set(),
            "quantized": {"ae", "mare", "mere", "rmse", "small_value_error_count"},
            "floating": {"mare", "mere", "rmse", "small_value_error_count"},
        }
        if self.category is not None and any(name not in allowed[self.category] for name in self.thresholds):
            raise ValueError(f"accuracy_policy metric is not applicable to category: {self.category}")
        if any(value < 0 for value in self.thresholds.values()):
            raise ValueError("accuracy_policy thresholds must be non-negative")
        if self.small_value_threshold is not None and self.small_value_threshold < 0:
            raise ValueError("small_value_threshold must be non-negative")
        if self.max_error_count is not None and self.max_error_count < 0:
            raise ValueError("max_error_count must be non-negative")
        if self.relative_epsilon is not None and self.relative_epsilon <= 0:
            raise ValueError("relative_epsilon must be positive")
        nested = self.thresholds.get("small_value_error_count")
        if self.max_error_count is not None and nested is not None and self.max_error_count != nested:
            raise ValueError("max_error_count conflicts with thresholds.small_value_error_count")
        return self


class OracleSpec(BaseModel):
    comparison: str = "allclose"
    reference_executor: str | None = None
    expected_error: str | None = None
    tolerance: dict[str, Any] = Field(default_factory=dict)
    accuracy_policy: AccuracyPolicySpec = Field(default_factory=AccuracyPolicySpec)
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

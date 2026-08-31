from __future__ import annotations

from dataclasses import dataclass

from cruciblex.domain.plan import ExecutionPlan
from cruciblex.domain.result import ArtifactRef
from cruciblex.runtime.pipeline import ExecutionPipeline
from cruciblex.storage.artifacts import ArtifactStore


def _input_sources(plan: ExecutionPlan) -> list[dict[str, object]]:
    sources = []
    for parameter in plan.case.parameters:
        if parameter.values is not None:
            source = "exact_values"
        elif parameter.metadata.get("value_policy") is not None:
            source = "value_policy"
        elif parameter.value_range.valid or parameter.value_range.invalid:
            source = "value_range"
        else:
            source = "default"
        sources.append({"parameter": parameter.name, "source": source})
    return sources


@dataclass(frozen=True, slots=True)
class InputBundle:
    inputs: list[object]
    artifacts: list[ArtifactRef]


class DriverInputMaterializer:
    """Generate case inputs once on the driver and reuse them across placements."""

    def __init__(self) -> None:
        self._pipeline = ExecutionPipeline()
        self._cache: dict[str, InputBundle] = {}

    def materialize(self, plan: ExecutionPlan) -> InputBundle:
        key = self._cache_key(plan)
        if key in self._cache:
            return self._cache[key]

        inputs = self._pipeline.generate_inputs(plan)
        store = ArtifactStore(plan.artifacts.output_root)
        path = store.ensure() / plan.case.name / "inputs.json"
        store.write_json(path, self._pipeline._serialize_inputs(inputs))
        artifact = ArtifactRef(
            name="inputs",
            path=path,
            kind="inputs",
            metadata={"role": "input", "scope": "case", "sources": _input_sources(plan)},
        )
        bundle = InputBundle(inputs=inputs, artifacts=[artifact])
        self._cache[key] = bundle
        return bundle

    def _cache_key(self, plan: ExecutionPlan) -> str:
        return plan.case.model_dump_json()

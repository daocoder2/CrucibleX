from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from cruciblex.domain.plan import ExecutionPlan
from cruciblex.domain.result import ArtifactRef
from cruciblex.generation.dtypes import dtype_contract
from cruciblex.runtime.pipeline import ExecutionPipeline
from cruciblex.storage.artifacts import ArtifactStore


def _fingerprint(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        sources.append({
            "parameter": parameter.name,
            "source": source,
            "spec_fingerprint": _fingerprint(parameter.model_dump(mode="json")),
        })
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
            metadata={
                "role": "input",
                "scope": "case",
                "input_schema_version": 1,
                "case_fingerprint": _fingerprint(plan.case.model_dump(mode="json")),
                "generator": plan.case.generator,
                "seed": plan.case.generation.seed,
                "sources": _input_sources(plan),
                "dtype_contracts": [dtype_contract(str(parameter.dtypes[0] if parameter.dtypes else "fp32")) for parameter in plan.case.parameters],
            },
        )
        bundle = InputBundle(inputs=inputs, artifacts=[artifact])
        self._cache[key] = bundle
        return bundle

    def _cache_key(self, plan: ExecutionPlan) -> str:
        return plan.case.model_dump_json()

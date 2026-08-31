from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cruciblex.domain.enums import BackendKind, ExecutionRole, ResultStatus, TaskKind
from cruciblex.domain.result import ArtifactRef, ExecutionResult
from cruciblex.runtime.compare import COMPARATOR_REGISTRY, ComparisonRequest
from cruciblex.storage.artifacts import ArtifactStore


class CrossDeviceComparator:
    """Compare device outputs against the CPU high-precision reference on the driver."""

    def __init__(self, output_root: str | Path | None = None) -> None:
        self.output_root = Path(output_root) if output_root is not None else None

    def compare(self, results: list[ExecutionResult]) -> list[ExecutionResult]:
        comparisons: list[ExecutionResult] = []
        grouped: dict[tuple[int, str], list[ExecutionResult]] = defaultdict(list)
        for result in results:
            if result.status == ResultStatus.PASSED and self._needs_compare(result.task):
                grouped[(result.case_id, result.task.value)].append(result)

        for group in grouped.values():
            reference = self._cpu_reference_result(group)
            if reference is None:
                continue
            reference_output = self._candidate_output(reference)
            if reference_output is None:
                continue
            for candidate in group:
                if candidate is reference or candidate.backend == BackendKind.CPU:
                    continue
                candidate_output = self._candidate_output(candidate)
                if candidate_output is None:
                    continue
                comparisons.append(self._compare_pair(reference, candidate, reference_output, candidate_output))
            comparisons.extend(self._compare_npu_against_gpu(reference, group, reference_output))
        return comparisons

    def _compare_npu_against_gpu(
        self,
        reference: ExecutionResult,
        group: list[ExecutionResult],
        reference_output: object,
    ) -> list[ExecutionResult]:
        gpu = next((result for result in group if result.backend == BackendKind.GPU), None)
        npu = next((result for result in group if result.backend in {BackendKind.NPU, BackendKind.ACLNN}), None)
        if gpu is None or npu is None:
            return []
        gpu_output = self._candidate_output(gpu)
        npu_output = self._candidate_output(npu)
        if gpu_output is None or npu_output is None:
            return []
        comparator = COMPARATOR_REGISTRY.resolve(npu.metrics.get("comparison", "allclose"))
        tolerance = self._tolerance(npu)
        metadata = {"reference_plan_id": reference.plan_id, "gpu_plan_id": gpu.plan_id, "npu_plan_id": npu.plan_id}
        gpu_report = comparator.compare(ComparisonRequest(expected=reference_output, actual=gpu_output, tolerance=tolerance, metadata=metadata))
        npu_report = comparator.compare(ComparisonRequest(expected=reference_output, actual=npu_output, tolerance=tolerance, metadata=metadata))
        baseline_passed = npu_report.passed and npu_report.max_abs_diff <= gpu_report.max_abs_diff
        path = self._comparison_path(reference, npu).with_name(
            f"{self._safe_path_part(reference.plan_id)}__npu-vs-gpu-baseline.json"
        )
        payload = {
            "reference_plan_id": reference.plan_id,
            "gpu_plan_id": gpu.plan_id,
            "npu_plan_id": npu.plan_id,
            "cpu_evidence": reference.evidence.model_dump(mode="json") if reference.evidence else None,
            "gpu_evidence": gpu.evidence.model_dump(mode="json") if gpu.evidence else None,
            "npu_evidence": npu.evidence.model_dump(mode="json") if npu.evidence else None,
            "tolerance": tolerance,
            "gpu_max_abs_diff": gpu_report.max_abs_diff,
            "npu_max_abs_diff": npu_report.max_abs_diff,
            "npu_within_gpu_baseline": npu_report.max_abs_diff <= gpu_report.max_abs_diff,
            "passed": baseline_passed,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return [ExecutionResult(
            plan_id=f"compare:{npu.case_id}:cpu->npu<=gpu:{npu.task.value}",
            case_id=npu.case_id,
            case_name=npu.case_name,
            node_name=f"{reference.node_name}->{npu.node_name}",
            backend=npu.backend,
            device_id=npu.device_id,
            task=npu.task,
            status=ResultStatus.PASSED if baseline_passed else ResultStatus.FAILED,
            candidate_role=ExecutionRole.CANDIDATE,
            reference_role=ExecutionRole.REFERENCE,
            metrics={
                "stage": "cpu_npu_gpu_accuracy_gate",
                "reference_plan_id": reference.plan_id,
                "gpu_plan_id": gpu.plan_id,
                "npu_plan_id": npu.plan_id,
                "gpu_max_abs_diff": gpu_report.max_abs_diff,
                "npu_max_abs_diff": npu_report.max_abs_diff,
                "npu_within_gpu_baseline": npu_report.max_abs_diff <= gpu_report.max_abs_diff,
                "failure_kind": "comparison_passed" if baseline_passed else "npu_exceeds_gpu_baseline",
                "compare_detail": "NPU error is within GPU baseline" if baseline_passed else "NPU error exceeds GPU baseline or absolute tolerance",
            },
            artifacts=[ArtifactRef(name="cpu_npu_gpu_gate", path=path, kind="comparison", metadata={"role": "driver"})],
            error=None if baseline_passed else "NPU error exceeds GPU baseline or absolute tolerance",
        )]

    def _needs_compare(self, task: TaskKind) -> bool:
        return task in {TaskKind.ACCURACY, TaskKind.ACCURACY_LOAD, TaskKind.ACCURACY_DC}

    def _cpu_reference_result(self, results: list[ExecutionResult]) -> ExecutionResult | None:
        cpu_results = [result for result in results if result.backend == BackendKind.CPU]
        return cpu_results[0] if cpu_results else None

    def _candidate_output(self, result: ExecutionResult) -> object | None:
        artifact = next((item for item in result.artifacts if item.name == "candidate_output"), None)
        if artifact is None or not artifact.path.exists():
            return None
        return self._deserialize(json.loads(artifact.path.read_text(encoding="utf-8")))

    def _compare_pair(
        self,
        reference: ExecutionResult,
        candidate: ExecutionResult,
        reference_output: object,
        candidate_output: object,
    ) -> ExecutionResult:
        comparator = COMPARATOR_REGISTRY.resolve(candidate.metrics.get("comparison", "allclose"))
        report = comparator.compare(
            ComparisonRequest(
                expected=reference_output,
                actual=candidate_output,
                tolerance=self._tolerance(candidate),
                metadata={
                    "case_id": candidate.case_id,
                    "reference_plan_id": reference.plan_id,
                    "candidate_plan_id": candidate.plan_id,
                },
            )
        )
        status = ResultStatus.PASSED if report.passed else ResultStatus.FAILED
        path = self._comparison_path(reference, candidate)
        payload = {
            "reference_plan_id": reference.plan_id,
            "candidate_plan_id": candidate.plan_id,
            "reference_backend": reference.backend.value,
            "candidate_backend": candidate.backend.value,
            "passed": report.passed,
            "max_abs_diff": report.max_abs_diff,
            "mean_abs_diff": report.mean_abs_diff,
            "detail": report.detail,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return ExecutionResult(
            plan_id=f"compare:{candidate.case_id}:{reference.backend.value}->{candidate.backend.value}:{candidate.task.value}",
            case_id=candidate.case_id,
            case_name=candidate.case_name,
            node_name=f"{reference.node_name}->{candidate.node_name}",
            backend=candidate.backend,
            device_id=candidate.device_id,
            task=candidate.task,
            status=status,
            candidate_role=ExecutionRole.CANDIDATE,
            reference_role=ExecutionRole.REFERENCE,
            metrics={
                "stage": "cross_device_compare",
                "comparison": candidate.metrics.get("comparison", "allclose"),
                "failure_kind": "comparison_passed" if report.passed else "comparison_mismatch",
                "reference_plan_id": reference.plan_id,
                "candidate_plan_id": candidate.plan_id,
                "reference_backend": reference.backend.value,
                "candidate_backend": candidate.backend.value,
                "max_abs_diff": report.max_abs_diff,
                "mean_abs_diff": report.mean_abs_diff,
                "compare_detail": report.detail,
            },
            artifacts=[ArtifactRef(name="cross_compare", path=path, kind="comparison", metadata={"role": "driver"})],
            error=None if report.passed else report.detail,
        )

    def _comparison_path(self, reference: ExecutionResult, candidate: ExecutionResult) -> Path:
        root = self._output_root(reference, candidate)
        filename = f"{self._safe_path_part(reference.plan_id)}__vs__{self._safe_path_part(candidate.plan_id)}.json"
        return root / candidate.case_name / "cross_compare" / filename

    def _output_root(self, reference: ExecutionResult, candidate: ExecutionResult) -> Path:
        if self.output_root is not None:
            return self.output_root
        artifact = next((item for item in [*reference.artifacts, *candidate.artifacts] if item.name == "candidate_output"), None)
        if artifact is None:
            raise ValueError("output_root is required when comparison artifacts are unavailable")
        return ArtifactStore(Path(artifact.path).parents[2]).root

    def _safe_path_part(self, value: str) -> str:
        return "".join(character if character.isalnum() or character in {"-", "_", "."} else "-" for character in value)

    def _tolerance(self, result: ExecutionResult) -> dict[str, float]:
        return {
            "atol": float(result.metrics.get("atol", 1e-6)),
            "rtol": float(result.metrics.get("rtol", 1e-6)),
        }

    def _deserialize(self, value: Any) -> object:
        if isinstance(value, dict) and {"dtype", "shape", "data"} <= set(value):
            return np.asarray(value["data"], dtype=value["dtype"]).reshape(value["shape"])
        if isinstance(value, list):
            return [self._deserialize(item) for item in value]
        return value

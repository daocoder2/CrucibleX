import csv
import json
from types import SimpleNamespace

import pytest

from cruciblex.domain import BackendKind, ExecutionResult, ResultStatus, TaskKind
from cruciblex.runtime.pipeline import ExecutionPipeline
from cruciblex.storage.results import ResultStore


@pytest.mark.parametrize("backend", [BackendKind.CPU, BackendKind.NPU, BackendKind.ACLNN])
def test_non_gpu_backends_emit_unknown_hardware_evidence(backend):
    plan = SimpleNamespace(device=SimpleNamespace(backend=backend))

    evidence = ExecutionPipeline()._hardware_evidence(
        plan,
        {"host": "worker", "node": "node-a", "device_id": 3, "resolved_device": "npu:3"},
    )

    assert evidence.schema_version == 1
    assert evidence.backend == backend
    assert evidence.probe_status == "unknown"
    assert evidence.host == "worker"
    assert evidence.resolved_device == "npu:3"
    assert evidence.runtime == {}
    assert evidence.fingerprint is None


def test_gpu_evidence_projects_runtime_probe_and_artifact_reference():
    plan = SimpleNamespace(device=SimpleNamespace(backend=BackendKind.GPU))

    evidence = ExecutionPipeline()._hardware_evidence(
        plan,
        {
            "host": "gpu-worker",
            "node": "gpu-node",
            "device_id": 2,
            "resolved_device": "cuda:0",
            "gpu_available": True,
            "cuda_version": "12.6",
            "gpu_device_count": 4,
            "gpu_evidence_fingerprint": "sha256:example",
            "gpu_evidence_path": "artifacts/gpu_evidence.json",
        },
    )

    assert evidence.probe_status == "available"
    assert evidence.runtime == {"cuda_version": "12.6", "gpu_device_count": 4}
    assert evidence.fingerprint == "sha256:example"
    assert evidence.artifact_refs == ["artifacts/gpu_evidence.json"]


def test_csv_projects_hardware_evidence(tmp_path):
    plan = SimpleNamespace(device=SimpleNamespace(backend=BackendKind.GPU))
    evidence = ExecutionPipeline()._hardware_evidence(plan, {"gpu_available": False})
    result = ExecutionResult(
        plan_id="1:gpu:gpu:0:run",
        case_id=1,
        case_name="torch.abs",
        node_name="gpu",
        backend=BackendKind.GPU,
        device_id=0,
        task=TaskKind.RUN,
        status=ResultStatus.SKIPPED,
        evidence=evidence,
    )

    path = ResultStore(tmp_path).write_results_csv([result])
    with path.open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert json.loads(row["evidence_json"]) == evidence.model_dump(mode="json")

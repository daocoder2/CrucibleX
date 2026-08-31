import json

from cruciblex.domain.enums import BackendKind, ResultStatus, TaskKind
from cruciblex.domain.result import ArtifactRef, ExecutionResult, HardwareEvidence
from cruciblex.plugins.comparators import allclose  # noqa: F401 - registers the built-in comparator
from cruciblex.report.cross_compare import CrossDeviceComparator


def _result(tmp_path, backend, values, plan_id):
    output = tmp_path / f"{plan_id}.json"
    output.write_text(json.dumps({"dtype": "float64", "shape": [2], "data": values}), encoding="utf-8")
    return ExecutionResult(
        plan_id=plan_id,
        case_id=1,
        case_name="tri_compare",
        node_name=backend.value,
        backend=backend,
        device_id=0,
        task=TaskKind.ACCURACY,
        status=ResultStatus.PASSED,
        metrics={"atol": 0.01, "rtol": 0.0},
        artifacts=[ArtifactRef(name="candidate_output", path=output, kind="candidate_output")],
        evidence=HardwareEvidence(backend=backend, probe_status="available", fingerprint=backend.value),
    )


def test_npu_passes_when_its_cpu_error_is_within_gpu_baseline(tmp_path):
    cpu = _result(tmp_path, BackendKind.CPU, [1.0, 2.0], "cpu")
    gpu = _result(tmp_path, BackendKind.GPU, [1.004, 2.004], "gpu")
    npu = _result(tmp_path, BackendKind.NPU, [1.002, 2.002], "npu")

    results = CrossDeviceComparator(tmp_path).compare([cpu, gpu, npu])
    gate = next(result for result in results if result.metrics["stage"] == "cpu_npu_gpu_accuracy_gate")

    assert gate.status == ResultStatus.PASSED
    assert gate.metrics["npu_within_gpu_baseline"] is True
    payload = json.loads(gate.artifacts[0].path.read_text(encoding="utf-8"))
    assert payload["npu_plan_id"] == "npu"
    assert payload["npu_evidence"]["fingerprint"] == "npu"


def test_npu_fails_when_its_cpu_error_exceeds_gpu_baseline(tmp_path):
    cpu = _result(tmp_path, BackendKind.CPU, [1.0, 2.0], "cpu")
    gpu = _result(tmp_path, BackendKind.GPU, [1.001, 2.001], "gpu")
    npu = _result(tmp_path, BackendKind.NPU, [1.005, 2.005], "npu")

    results = CrossDeviceComparator(tmp_path).compare([cpu, gpu, npu])
    gate = next(result for result in results if result.metrics["stage"] == "cpu_npu_gpu_accuracy_gate")

    assert gate.status == ResultStatus.FAILED
    assert gate.metrics["failure_kind"] == "npu_exceeds_gpu_baseline"

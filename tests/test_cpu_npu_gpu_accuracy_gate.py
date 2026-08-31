import json

from cruciblex.domain.enums import BackendKind, ResultStatus, TaskKind
from cruciblex.domain.result import ArtifactRef, ExecutionResult, HardwareEvidence
from cruciblex.plugins.comparators import allclose  # noqa: F401 - registers the built-in comparator
from cruciblex.report.cross_compare import CrossDeviceComparator


def _result(tmp_path, backend, values, plan_id, accuracy_policy=None):
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
        metrics={"atol": 0.01, "rtol": 0.0, "accuracy_policy": accuracy_policy or {}},
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


def test_quantized_gate_records_all_required_metrics_and_small_value_count(tmp_path):
    policy = {
        "category": "quantized",
        "small_value_threshold": 0.1,
        "thresholds": {"ae": 0.02, "mare": 1.0, "mere": 1.0, "rmse": 0.02, "small_value_error_count": 0},
    }
    cpu = _result(tmp_path, BackendKind.CPU, [0.01, 1.0], "cpu", policy)
    gpu = _result(tmp_path, BackendKind.GPU, [0.02, 1.01], "gpu", policy)
    npu = _result(tmp_path, BackendKind.NPU, [0.015, 1.005], "npu", policy)

    results = CrossDeviceComparator(tmp_path).compare([cpu, gpu, npu])
    gate = next(result for result in results if result.metrics["stage"] == "cpu_npu_gpu_accuracy_gate")

    assert gate.status == ResultStatus.PASSED
    assert {"ae", "mare", "mere", "rmse", "small_value_error_count"} <= set(gate.metrics["npu_metrics"])


def test_small_value_error_count_uses_top_level_max_error_count(tmp_path):
    policy = {"category": "floating", "small_value_threshold": 0.1, "max_error_count": 0}
    cpu = _result(tmp_path, BackendKind.CPU, [0.01, 1.0], "cpu", policy)
    gpu = _result(tmp_path, BackendKind.GPU, [0.021, 1.0], "gpu", policy)
    npu = _result(tmp_path, BackendKind.NPU, [0.021, 1.0], "npu", policy)

    results = CrossDeviceComparator(tmp_path).compare([cpu, gpu, npu])
    gate = next(result for result in results if result.metrics["stage"] == "cpu_npu_gpu_accuracy_gate")

    assert gate.status == ResultStatus.FAILED
    assert "small_value_error_count:threshold" in gate.metrics["failed_metrics"]


def test_integer_gate_requires_bitwise_match(tmp_path):
    policy = {"category": "integer"}
    cpu = _result(tmp_path, BackendKind.CPU, [1, 2], "cpu", policy)
    gpu = _result(tmp_path, BackendKind.GPU, [1, 2], "gpu", policy)
    npu = _result(tmp_path, BackendKind.NPU, [1, 3], "npu", policy)

    results = CrossDeviceComparator(tmp_path).compare([cpu, gpu, npu])
    gate = next(result for result in results if result.metrics["stage"] == "cpu_npu_gpu_accuracy_gate")

    assert gate.status == ResultStatus.FAILED
    assert "bitwise_match" in gate.metrics["failed_metrics"]

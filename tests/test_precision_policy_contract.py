import json

import numpy as np
import pytest
from pydantic import ValidationError

from cruciblex.domain.case import AccuracyPolicySpec
from cruciblex.domain.enums import BackendKind, ResultStatus, TaskKind
from cruciblex.domain.result import ArtifactRef, ExecutionResult
from cruciblex.plugins.comparators import allclose  # noqa: F401 - registers the built-in comparator
from cruciblex.report.cross_compare import CrossDeviceComparator


def test_accuracy_policy_rejects_inapplicable_or_conflicting_thresholds():
    with pytest.raises(ValidationError, match="not applicable"):
        AccuracyPolicySpec(category="floating", thresholds={"ae": 0.1})
    with pytest.raises(ValidationError, match="conflicts"):
        AccuracyPolicySpec(thresholds={"small_value_error_count": 1}, max_error_count=0)
    with pytest.raises(ValidationError, match="positive"):
        AccuracyPolicySpec(relative_epsilon=0)


def test_bitwise_match_requires_dtype_and_exact_bytes(tmp_path):
    comparator = CrossDeviceComparator(tmp_path)

    assert comparator._accuracy_metrics(np.array([1], dtype=np.int32), np.array([1], dtype=np.int64), {"category": "integer"}, {"atol": 0})["bitwise_match"] is False
    assert comparator._accuracy_metrics(np.array([-0.0], dtype=np.float32), np.array([0.0], dtype=np.float32), {"category": "non_computational"}, {"atol": 0})["bitwise_match"] is False


def test_non_finite_values_produce_explicit_gate_failure(tmp_path):
    comparator = CrossDeviceComparator(tmp_path)
    gpu = comparator._accuracy_metrics(np.array([1.0]), np.array([1.0]), {"category": "floating"}, {"atol": 0.0})
    npu = comparator._accuracy_metrics(np.array([1.0]), np.array([np.nan]), {"category": "floating"}, {"atol": 0.0})

    passed, failures = comparator._accuracy_gate(False, gpu, npu, {"category": "floating"})

    assert passed is False
    assert "non_finite" in failures

    passed, failures = comparator._accuracy_gate(True, npu, gpu, {"category": "floating"})

    assert passed is False
    assert "gpu_non_finite" in failures


def _result(tmp_path, backend, plan_id, fingerprint):
    output = tmp_path / f"{plan_id}.json"
    output.write_text(json.dumps({"dtype": "float64", "shape": [1], "data": [1.0]}), encoding="utf-8")
    inputs = tmp_path / f"{plan_id}-inputs.json"
    inputs.write_text("{}", encoding="utf-8")
    return ExecutionResult(
        plan_id=plan_id,
        case_id=1,
        case_name="precision",
        node_name=backend.value,
        backend=backend,
        device_id=0,
        task=TaskKind.ACCURACY,
        status=ResultStatus.PASSED,
        metrics={"comparison": "allclose", "atol": 0.0, "rtol": 0.0, "accuracy_policy": {"category": "floating"}},
        artifacts=[
            ArtifactRef(name="inputs", path=inputs, kind="inputs", metadata={"case_fingerprint": fingerprint}),
            ArtifactRef(name="candidate_output", path=output, kind="candidate_output"),
        ],
    )


def test_cross_device_compare_does_not_mix_input_fingerprints(tmp_path):
    cpu = _result(tmp_path, BackendKind.CPU, "cpu", "a")
    gpu = _result(tmp_path, BackendKind.GPU, "gpu", "a")
    npu = _result(tmp_path, BackendKind.NPU, "npu", "b")

    comparisons = CrossDeviceComparator(tmp_path).compare([cpu, gpu, npu])

    assert len(comparisons) == 1
    assert comparisons[0].metrics["stage"] == "cross_device_compare"

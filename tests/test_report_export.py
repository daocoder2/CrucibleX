import csv
import json

from cruciblex.domain import (
    ArtifactRef,
    BackendKind,
    ExecutionResult,
    HardwareEvidence,
    ResultStatus,
    RunManifest,
    SchedulerKind,
    TaskKind,
)
from cruciblex.storage.report_export import REPORT_EXPORT_FIELDS
from cruciblex.storage.results import ResultStore


def test_report_exports_use_stable_schema_projection(tmp_path):
    manifest = RunManifest(
        run_id="run-1",
        case_path="case.yaml",
        node_path="nodes.yaml",
        tasks=[TaskKind.ACCURACY],
        scheduler=SchedulerKind.LOCAL,
        output_root=tmp_path,
        cruciblex_version="test",
        metadata={"manifest_sha256": "manifest-hash"},
    )
    result = ExecutionResult(
        plan_id="plan-1",
        case_id=1,
        case_name="torch.abs",
        node_name="gpu",
        backend=BackendKind.GPU,
        device_id=0,
        task=TaskKind.ACCURACY,
        status=ResultStatus.PASSED,
        metrics={
            "resolved_device": "cuda:0",
            "candidate_executor": "torch",
            "comparison": "allclose",
            "max_abs_diff": 0.1,
            "rmse": 0.05,
            "runtime_policy": {"schema_version": 1, "effective": {"deterministic": True}},
            "manifest_lane": "cpu-contract",
            "manifest_lane_kind": "contract",
            "manifest_case_include": "examples/cases/torch.abs.yaml",
            "manifest_case_index": 7,
            "manifest_runtime": {"allow_invalid_cases": False},
            "aclnn_capability": "tensor_list",
            "aclnn_capability_status": "future_abi",
            "aclnn_capability_reason": "requires ACLNN tensor-list ownership contract",
            "aclnn_capability_decisions": [
                {"capability": "tensor_list", "status": "future_abi", "reason": "requires ACLNN tensor-list ownership contract"},
                {"capability": "tensor", "status": "supported", "reason": ""},
            ],
        },
        artifacts=[ArtifactRef(name="inputs", path=tmp_path / "inputs.json", kind="inputs", metadata={"case_fingerprint": "abc"})],
        evidence=HardwareEvidence(
            backend=BackendKind.GPU,
            device_id=0,
            probe_status="available",
            runtime={"torch": "2.6", "cuda_available": True},
            fingerprint="gpu-fingerprint",
        ),
    )
    store = ResultStore(tmp_path)

    raw_path = store.write_results_jsonl([result])
    jsonl_path = store.write_report_jsonl(manifest, [result])
    csv_path = store.write_report_csv(manifest, [result])

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    row = json.loads(jsonl_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert raw["metrics"] == result.metrics
    expected_fields = (
        "report_schema_version", "run_id", "result_schema_version", "plan_id", "case_id", "case_name", "task", "status", "node_name", "backend", "device_id", "resolved_device", "candidate_executor", "case_fingerprint", "matrix_id", "manifest_lane", "manifest_lane_kind", "manifest_case_include", "manifest_case_index", "manifest_sha256", "source_case_id", "generation_index", "generation_seed", "comparison", "max_abs_diff", "mean_abs_diff", "max_relative_error", "mean_relative_error", "rmse", "matched_ratio", "latency_ms", "throughput_items_per_s", "memory_peak_bytes", "failure_kind", "failure_stage", "error", "aclnn_capability", "aclnn_capability_status", "aclnn_capability_reason", "aclnn_capability_decisions_json", "hardware_probe_status", "hardware_backend", "hardware_device_id", "hardware_fingerprint", "hardware_runtime_json", "runtime_policy_json", "manifest_runtime_json", "metrics_json", "evidence_json", "artifact_count",
    )
    assert REPORT_EXPORT_FIELDS == expected_fields
    assert tuple(row) == expected_fields
    assert row["report_schema_version"] == 1
    assert row["run_id"] == "run-1"
    assert row["case_fingerprint"] == "abc"
    assert row["manifest_lane"] == "cpu-contract"
    assert row["manifest_lane_kind"] == "contract"
    assert row["manifest_case_include"] == "examples/cases/torch.abs.yaml"
    assert row["manifest_case_index"] == 7
    assert row["manifest_sha256"] == "manifest-hash"
    assert row["aclnn_capability"] == "tensor_list"
    assert row["aclnn_capability_status"] == "future_abi"
    assert row["aclnn_capability_reason"] == "requires ACLNN tensor-list ownership contract"
    assert json.loads(row["aclnn_capability_decisions_json"])[0]["capability"] == "tensor_list"
    assert json.loads(row["aclnn_capability_decisions_json"])[1]["status"] == "supported"
    assert row["max_abs_diff"] == 0.1
    assert json.loads(row["runtime_policy_json"])["effective"]["deterministic"] is True
    assert json.loads(row["manifest_runtime_json"])["allow_invalid_cases"] is False
    assert row["resolved_device"] == "cuda:0"
    assert row["candidate_executor"] == "torch"
    assert row["hardware_probe_status"] == "available"
    assert row["hardware_backend"] == "gpu"
    assert row["hardware_device_id"] == 0
    assert row["hardware_fingerprint"] == "gpu-fingerprint"
    assert json.loads(row["hardware_runtime_json"])["cuda_available"] is True
    assert json.loads(row["hardware_runtime_json"])["torch"] == "2.6"
    assert tuple(csv_rows[0]) == expected_fields
    assert csv_rows[0]["rmse"] == "0.05"


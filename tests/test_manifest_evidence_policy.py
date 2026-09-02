from types import SimpleNamespace

from cruciblex.domain import BackendKind, ResultStatus, TaskKind
from cruciblex.runtime.pipeline import ExecutionPipeline


def test_manifest_hardware_evidence_policy_enforces_only_accelerator_hardware_lanes():
    def plan(backend, lane_kind="hardware"):
        return SimpleNamespace(
            plan_id="1:lane:device:0:accuracy",
            case=SimpleNamespace(
                id=1,
                name="torch.abs",
                metadata={
                    "manifest_lane_kind": lane_kind,
                    "manifest_runtime": {
                        "require_real_evidence": True,
                        "require_backend_dtype_source": "device_tensor",
                    },
                },
            ),
            node=SimpleNamespace(display_name="node"),
            task=TaskKind.ACCURACY,
            device=SimpleNamespace(backend=backend, id=0),
        )

    recorder = SimpleNamespace(artifacts=[], payloads=[])
    pipeline = ExecutionPipeline()

    missing_dtype = pipeline._result(plan(BackendKind.GPU), ResultStatus.PASSED, {"gpu_available": True}, recorder)
    accepted = pipeline._result(
        plan(BackendKind.NPU),
        ResultStatus.PASSED,
        {"npu_available": True, "backend_dtype_source": "device_tensor"},
        recorder,
    )
    cpu_contract = pipeline._result(plan(BackendKind.CPU, "contract"), ResultStatus.PASSED, {}, recorder)

    assert missing_dtype.status == ResultStatus.FAILED
    assert missing_dtype.metrics["failure_kind"] == "manifest_evidence_policy"
    assert "dtype source" in missing_dtype.error
    assert accepted.status == ResultStatus.PASSED
    assert cpu_contract.status == ResultStatus.PASSED

from cruciblex.domain.enums import BackendKind, ResultStatus, TaskKind
from cruciblex.domain.result import ExecutionResult
from cruciblex.report.postprocess import ResultPostProcessor


def _failure(case_id: int, shape: list[int]) -> ExecutionResult:
    return ExecutionResult(
        plan_id=f"plan-{case_id}",
        case_id=case_id,
        case_name="torch.add",
        node_name="cpu",
        backend=BackendKind.CPU,
        device_id=0,
        task=TaskKind.RUN,
        status=ResultStatus.ERROR,
        metrics={"failure_kind": "error", "error_type": "RuntimeError", "output_shape": shape},
        error="dynamic detail should not define cluster",
    )


def test_failure_clusters_include_shape_and_exception_family():
    same_shape = [_failure(1, [2, 2]), _failure(2, [2, 2])]
    same_shape[1].error = "another dynamic detail"
    summary = ResultPostProcessor().summarize(same_shape)
    clusters = summary["failure_clusters"]
    assert len(clusters) == 1

    different_shape = ResultPostProcessor().summarize([_failure(1, [2, 2]), _failure(2, [4, 4])])["failure_clusters"]
    assert len(different_shape) == 2
    assert all("torch.add" in cluster["signature"] for cluster in clusters)
    assert all("RuntimeError" in cluster["signature"] for cluster in clusters)

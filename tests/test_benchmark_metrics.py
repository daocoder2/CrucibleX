import numpy as np

from cruciblex.domain import (
    BackendKind,
    CaseSpec,
    DeviceSpec,
    InvocationSpec,
    JobSpec,
    NodeSpec,
    OperatorSpec,
    OracleSpec,
    ParameterKind,
    ParameterSpec,
    ShapeSpec,
    TaskKind,
)
from cruciblex.runtime.executors import EXECUTOR_REGISTRY, BackendExecutor
from cruciblex.runtime.pipeline import ExecutionPipeline
from cruciblex.runtime.planner import ExecutionPlanner


class CountingExecutor(BackendExecutor):
    calls = 0

    def execute(self, request):
        type(self).calls += 1
        return np.asarray(request.inputs[0])


def test_performance_benchmark_emits_repeat_percentile_and_throughput_metrics(tmp_path):
    EXECUTOR_REGISTRY.register("counting-benchmark")(CountingExecutor)
    CountingExecutor.calls = 0
    case = CaseSpec(
        id=801, operator=OperatorSpec(name="benchmark"),
        invocation=InvocationSpec(
            api="benchmark",
            api_type="function",
            executor="counting-benchmark",
            metadata={"benchmark": {"warmup": 2, "repeat": 3, "throughput_items_per_call": 8, "min_time_ms": 0}, "profiler": {"tool": "msprof", "status": "requested"}},
        ),
        metadata={"benchmark_policy": {"warmup": 1, "repeat": 2, "min_time_ms": 1}},
        oracle=OracleSpec(),
        parameters=[ParameterSpec(name="input", kind=ParameterKind.TENSOR, dtypes=["fp32"], shape=ShapeSpec(dims=[2]))],
    )
    node = NodeSpec(name="cpu", devices=[DeviceSpec(id=0, backend=BackendKind.CPU)])
    plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[node], tasks=[TaskKind.PERFORMANCE_DEVICE], artifacts={"output_root": tmp_path}))[0]
    result = ExecutionPipeline().run(plan, inputs=[np.asarray([1.0, 2.0])])
    assert result.status.value == "passed"
    assert CountingExecutor.calls == 5
    assert result.metrics["warmup_count"] == 2
    assert result.metrics["repeat_count"] == 3
    assert result.metrics["min_time_ms"] == 0
    assert result.metrics["sample_count"] == 3
    assert result.metrics["latency_stddev_ms"] >= 0
    assert result.metrics["effective_duration_ms"] >= 0
    assert result.metrics["latency_p50_ms"] >= 0
    assert result.metrics["latency_p99_ms"] >= result.metrics["latency_p50_ms"]
    assert result.metrics["throughput_items_per_s"] > 0
    profiler = next(artifact for artifact in result.artifacts if artifact.name == "profiler")
    assert profiler.kind == "profiler"

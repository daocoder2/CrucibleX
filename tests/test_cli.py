import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from cruciblex import __version__
from cruciblex.cli import app
from cruciblex.domain import (
    ArtifactPayload,
    BackendKind,
    CaseSpec,
    DeviceSpec,
    ExecutionResult,
    ExecutionRole,
    InvocationSpec,
    JobSpec,
    NodeSpec,
    OperatorSpec,
    OracleSpec,
    ParameterKind,
    ParameterSpec,
    ResultStatus,
    RunContext,
    RunManifest,
    SchedulerKind,
    ShapeSpec,
    TaskKind,
    ValueRange,
)
from cruciblex.generation.loader import load_case, load_job
from cruciblex.plugins import load_builtin_plugins, load_plugins
from cruciblex.runtime.executors import ExecutionRequest
from cruciblex.runtime.planner import ExecutionPlanner
from cruciblex.runtime.scheduler import LocalScheduler
from cruciblex.storage.results import ResultStore

load_builtin_plugins()


def tensor_param(size: int = 1) -> ParameterSpec:
    return ParameterSpec(
        name="input",
        kind=ParameterKind.TENSOR,
        dtypes=["fp32"],
        shape=ShapeSpec(dim_count=[1], dim_values=[size]),
        value_range=ValueRange(valid=[[-1, 1]]),
    )


def cpu_node(device_ids=(0,)) -> NodeSpec:
    return NodeSpec(name="cpu", devices=[DeviceSpec(id=device_id, backend=BackendKind.CPU) for device_id in device_ids])


def run_context(
    output_root: str | Path = "cx_output",
    scheduler: SchedulerKind = SchedulerKind.LOCAL,
    plugin_paths: list[Path] | None = None,
) -> RunContext:
    return RunContext(
        run_id="test-run",
        case_path=Path("cases.yaml"),
        node_path=Path("nodes.yaml"),
        tasks=[TaskKind.ACCURACY],
        scheduler=scheduler,
        output_root=Path(output_root).resolve(),
        plugin_paths=plugin_paths or [],
    )


def write_cpu_nodes(path: Path) -> None:
    path.write_text(
        """
nodes:
  - name: cpu
    host: 127.0.0.1
    devices:
      - id: 0
        backend: cpu
""".strip(),
        encoding="utf-8",
    )


def test_version_exists():
    assert __version__ == "0.1.0"


def test_doctor_reports_ray_status():
    runner = CliRunner()
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Ray:" in result.stdout

def test_domain_model_shapes():
    case = CaseSpec(id=1, operator=OperatorSpec(name="torch.abs"), invocation=InvocationSpec(api="torch.abs", api_type="function", executor="function"), parameters=[tensor_param(16)], oracle=OracleSpec(comparison="allclose", reference_executor="function"))
    plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.ACCURACY]))[0]
    assert plan.plan_id == "1:cpu:cpu:0:accuracy"
    assert plan.device.backend == BackendKind.CPU
    assert plan.task == TaskKind.ACCURACY


def test_job_spec_defaults_to_ray_scheduler():
    job = JobSpec(cases=[CaseSpec(id=2, operator=OperatorSpec(name="torch.abs"), invocation=InvocationSpec(api="torch.abs", api_type="function"))], nodes=[cpu_node()])
    assert job.scheduler == SchedulerKind.RAY


def test_load_job_defaults_to_ray_scheduler():
    job = load_job("examples/cases/torch.abs.yaml", "examples/nodes/local.yaml")
    assert job.scheduler == SchedulerKind.RAY


def test_load_job_normalizes_output_root(monkeypatch, tmp_path):
    repo_root = Path(__file__).parents[1]
    monkeypatch.chdir(tmp_path)
    job = load_job(
        repo_root / "examples/cases/torch.abs.yaml",
        repo_root / "examples/nodes/local.yaml",
        output_path="cx_output",
    )
    assert job.artifacts.output_root == (tmp_path / "cx_output").resolve()
    assert job.artifacts.output_root.is_absolute()


def test_planner_respects_allowed_tasks():
    case = CaseSpec(id=1, operator=OperatorSpec(name="torch.abs"), invocation=InvocationSpec(api="torch.abs", api_type="function", executor="function"))
    node = NodeSpec(name="cpu", devices=[DeviceSpec(id=0, backend=BackendKind.CPU)], allowed_tasks={TaskKind.RUN})
    plans = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[node], tasks=[TaskKind.ACCURACY, TaskKind.RUN]))
    assert [plan.task for plan in plans] == [TaskKind.RUN]


def test_loader_and_local_scheduler_run_example(tmp_path):
    job = load_job("examples/cases/torch.abs.yaml", "examples/nodes/local.yaml", scheduler=SchedulerKind.LOCAL, output_path=tmp_path)
    scheduler = LocalScheduler(run_context())
    for plan in ExecutionPlanner().build(job):
        scheduler.submit(plan)
    results = scheduler.collect()
    assert len(results) == 1
    assert results[0].case_name == "torch.abs"
    assert results[0].status == ResultStatus.PASSED
    assert {artifact.name for artifact in results[0].artifacts} == {"inputs", "candidate_output", "reference_output"}
    assert results[0].candidate_role == ExecutionRole.CANDIDATE
    assert results[0].reference_role == ExecutionRole.REFERENCE
    assert all(artifact.path.exists() for artifact in results[0].artifacts)


def test_load_case_supports_new_operator_schema(tmp_path):
    case_file = tmp_path / "case.yaml"
    case_file.write_text(
        """
id: 7
operator:
  name: custom.double
invocation:
  api: custom.double
  api_type: function
  executor: double
oracle:
  comparison: allclose
  reference_executor: double
parameters:
  - name: input
    kind: tensor
    dtypes: [fp32]
    shape:
      dim_count: [1]
      dim_values: [4]
    value_range:
      valid: [[1, 1]]
""".strip(),
        encoding="utf-8",
    )
    case = load_case(case_file)
    assert case.operator.name == "custom.double"
    assert case.invocation.api_type == "function"
    assert case.invocation.executor == "double"
    assert case.parameters[0].kind == ParameterKind.TENSOR
    assert case.oracle.comparison == "allclose"


def test_load_job_requires_wrapped_case_and_node_documents(tmp_path):
    case_file = tmp_path / "case.yaml"
    case_file.write_text(
        """
id: 7
operator:
  name: custom.double
invocation:
  api: custom.double
  api_type: function
  executor: double
oracle:
  comparison: allclose
  reference_executor: double
parameters:
  - name: input
    kind: tensor
    dtypes: [fp32]
    shape:
      dim_count: [1]
      dim_values: [4]
    value_range:
      valid: [[1, 1]]
""".strip(),
        encoding="utf-8",
    )
    node_file = tmp_path / "nodes.yaml"
    node_file.write_text(
        """
name: cpu
host: 127.0.0.1
devices:
  - id: 0
    backend: cpu
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="cases"):
        load_job(case_file, node_file, scheduler=SchedulerKind.LOCAL, output_path=tmp_path)


def test_plugin_executor_is_loadable(tmp_path):
    plugin_file = tmp_path / "double_plugin.py"
    plugin_file.write_text("""
from cruciblex.runtime.executors.base import BackendExecutor, EXECUTOR_REGISTRY, ExecutionRequest

class DoubleExecutor(BackendExecutor):
    def execute(self, request: ExecutionRequest) -> object:
        return request.inputs[0] * 2

EXECUTOR_REGISTRY.register("double")(DoubleExecutor)
""".strip())
    case_file = tmp_path / "case.yaml"
    case_file.write_text("""
cases:
  - id: 7
    operator:
      name: custom.double
    invocation:
      api: custom.double
      api_type: function
      executor: double
    oracle:
      comparison: allclose
      reference_executor: double
    parameters:
      - name: input
        kind: tensor
        dtypes: [fp32]
        shape:
          dim_count: [1]
          dim_values: [4]
        value_range:
          valid: [[1, 1]]
""".strip())
    node_file = tmp_path / "nodes.yaml"
    write_cpu_nodes(node_file)
    load_plugins([plugin_file])
    result = LocalScheduler(run_context()).submit(ExecutionPlanner().build(load_job(case_file, node_file, scheduler=SchedulerKind.LOCAL, output_path=tmp_path))[0])
    output = json.loads(result.artifacts[1].path.read_text(encoding="utf-8"))
    assert result.status == ResultStatus.PASSED
    assert output["data"] == [2.0, 2.0, 2.0, 2.0]
    assert result.artifacts[1].metadata["role"] == "candidate"
    assert result.artifacts[2].metadata["role"] == "reference"


def test_backend_context_metrics_and_ray_resources(tmp_path):
    from cruciblex.runtime.backends import ray_resources_for
    assert ray_resources_for(DeviceSpec(id=0, backend=BackendKind.CPU)).num_cpus == 1.0
    assert ray_resources_for(DeviceSpec(id=1, backend=BackendKind.GPU)).num_gpus == 1
    assert ray_resources_for(DeviceSpec(id=2, backend=BackendKind.NPU)).resources == {"npu": 1.0}
    result = LocalScheduler(run_context()).submit(ExecutionPlanner().build(load_job("examples/cases/torch.abs.yaml", "examples/nodes/local.yaml", scheduler=SchedulerKind.LOCAL, output_path=tmp_path))[0])
    assert result.metrics["backend"] == "cpu"
    assert result.metrics["host"] == "127.0.0.1"
    assert result.metrics["device_id"] == 0


def test_gpu_backend_runtime_prepares_cuda_environment():
    from cruciblex.runtime.backends import GpuBackendRuntime
    from cruciblex.runtime.backends.base import DeviceContext

    context = DeviceContext(
        host="10.0.0.8",
        node_name="gpu-node",
        device=DeviceSpec(id=2, backend=BackendKind.GPU),
        output_root=Path("out"),
    )

    prepared = GpuBackendRuntime().prepare(context)
    assert prepared is context
    assert context.env["CX_BACKEND"] == "gpu"
    assert context.env["CX_DEVICE_ID"] == "2"
    assert context.env["CUDA_VISIBLE_DEVICES"] == "2"
    assert context.env["NVIDIA_VISIBLE_DEVICES"] == "2"


def test_ray_scheduler_actor_key_and_options():
    from cruciblex.runtime.scheduler.ray import ActorKey, RayScheduler
    case = CaseSpec(id=3, operator=OperatorSpec(name="torch.abs"), invocation=InvocationSpec(api="torch.abs", api_type="function", executor="function"))
    gpu_plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[NodeSpec(name="gpu-node", host="10.0.0.8", devices=[DeviceSpec(id=2, backend=BackendKind.GPU)])], tasks=[TaskKind.RUN]))[0]
    npu_plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[NodeSpec(name="npu-node", host="10.0.0.9", devices=[DeviceSpec(id=4, backend=BackendKind.NPU)])], tasks=[TaskKind.RUN]))[0]
    scheduler = RayScheduler(run_context(scheduler=SchedulerKind.RAY, plugin_paths=[Path("plugin.py")]).model_copy(update={"run_id": "run:test"}))
    assert ActorKey.from_plan(gpu_plan).label() == "10.0.0.8:gpu:2"
    assert scheduler._safe_label("run:test") == "run-test"
    assert [str(path) for path in scheduler.context.plugin_paths] == ["plugin.py"]
    assert gpu_plan.device.backend == BackendKind.GPU
    assert npu_plan.device.backend == BackendKind.NPU



def test_ray_placement_matches_host_and_resources():
    from cruciblex.runtime.scheduler.placement import (
        RayClusterSnapshot,
        RayNodeInfo,
        decide_ray_placement,
    )

    case = CaseSpec(
        id=41,
        operator=OperatorSpec(name="torch.abs"),
        invocation=InvocationSpec(api="torch.abs", api_type="function"),
    )
    plan = ExecutionPlanner().build(
        JobSpec(
            cases=[case],
            nodes=[
                NodeSpec(
                    name="gpu-node",
                    host="10.0.0.8",
                    devices=[DeviceSpec(id=0, backend=BackendKind.GPU)],
                )
            ],
            tasks=[TaskKind.RUN],
        )
    )[0]
    snapshot = RayClusterSnapshot(
        [
            RayNodeInfo(
                node_id="node-1",
                address="10.0.0.8",
                hostname="worker-a",
                alive=True,
                resources={"CPU": 8.0, "GPU": 1.0, "node:node-1": 1.0},
            )
        ]
    )

    placement = decide_ray_placement(plan, snapshot)
    assert placement.node.address == "10.0.0.8"
    assert placement.actor_options() == {
        "num_gpus": 1,
        "resources": {"node:node-1": 0.001},
    }


def test_ray_placement_allows_localhost_single_node():
    from cruciblex.runtime.scheduler.placement import (
        RayClusterSnapshot,
        RayNodeInfo,
        decide_ray_placement,
    )

    case = CaseSpec(
        id=42,
        operator=OperatorSpec(name="torch.abs"),
        invocation=InvocationSpec(api="torch.abs", api_type="function"),
    )
    plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.RUN]))[0]
    snapshot = RayClusterSnapshot(
        [RayNodeInfo(node_id="node-1", address="10.1.2.3", hostname="local", alive=True, resources={"CPU": 2.0})]
    )

    placement = decide_ray_placement(plan, snapshot)
    assert placement.node.address == "10.1.2.3"


def test_ray_placement_rejects_missing_resources():
    from cruciblex.runtime.scheduler.placement import (
        RayClusterSnapshot,
        RayNodeInfo,
        RayPlacementError,
        decide_ray_placement,
    )

    case = CaseSpec(
        id=43,
        operator=OperatorSpec(name="torch.abs"),
        invocation=InvocationSpec(api="torch.abs", api_type="function"),
    )
    plan = ExecutionPlanner().build(
        JobSpec(
            cases=[case],
            nodes=[
                NodeSpec(
                    name="npu-node",
                    host="10.0.0.9",
                    devices=[DeviceSpec(id=0, backend=BackendKind.NPU)],
                )
            ],
            tasks=[TaskKind.RUN],
        )
    )[0]
    snapshot = RayClusterSnapshot(
        [RayNodeInfo(node_id="node-1", address="10.0.0.9", hostname="npu", alive=True, resources={"CPU": 8.0})]
    )

    with pytest.raises(RayPlacementError, match="resources cannot satisfy"):
        decide_ray_placement(plan, snapshot)


def test_ray_scheduler_marks_placement_errors_as_skipped():
    from cruciblex.runtime.scheduler.ray import RayScheduler

    class FakeRay:
        def is_initialized(self):
            return True

        def nodes(self):
            return [
                {
                    "NodeID": "node-1",
                    "NodeManagerAddress": "10.0.0.1",
                    "NodeManagerHostname": "other-host",
                    "Alive": True,
                    "Resources": {"CPU": 2.0, "node:node-1": 1.0},
                }
            ]

    case = CaseSpec(id=51, operator=OperatorSpec(name="torch.abs"), invocation=InvocationSpec(api="torch.abs", api_type="function"))
    plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[NodeSpec(name="gpu-node", host="10.0.0.8", devices=[DeviceSpec(id=0, backend=BackendKind.GPU)])], tasks=[TaskKind.RUN]))[0]
    scheduler = RayScheduler(run_context(scheduler=SchedulerKind.RAY))
    scheduler._ray = lambda: FakeRay()

    result = scheduler.submit(plan)
    assert result.status == ResultStatus.SKIPPED
    assert result.metrics["stage"] == "scheduler"
    assert "no alive Ray node" in result.error


def test_ray_scheduler_marks_collect_failures_as_error(monkeypatch):
    from cruciblex.runtime.scheduler.ray import RayScheduler

    class FakeRay:
        def is_initialized(self):
            return True

        def get(self, ref):
            raise RuntimeError("ray crashed")

    case = CaseSpec(id=52, operator=OperatorSpec(name="torch.abs"), invocation=InvocationSpec(api="torch.abs", api_type="function"))
    plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.RUN]))[0]
    scheduler = RayScheduler(run_context(scheduler=SchedulerKind.RAY))
    scheduler._refs = [(plan, object())]
    monkeypatch.setattr(scheduler, "_ray", lambda: FakeRay())

    results = scheduler.collect()
    assert results[0].status == ResultStatus.ERROR
    assert results[0].metrics["stage"] == "scheduler"
    assert "ray crashed" in results[0].error


def test_npu_and_aclnn_backend_runtimes_prepare_device_env():
    from cruciblex.runtime.backends import BACKEND_REGISTRY
    from cruciblex.runtime.backends.base import DeviceContext

    npu_context = DeviceContext(
        host="127.0.0.1",
        node_name="npu",
        device=DeviceSpec(id=2, backend=BackendKind.NPU),
        output_root=Path("out"),
    )
    aclnn_context = DeviceContext(
        host="127.0.0.1",
        node_name="aclnn",
        device=DeviceSpec(id=3, backend=BackendKind.ACLNN),
        output_root=Path("out"),
    )

    prepared_npu = BACKEND_REGISTRY.resolve(BackendKind.NPU).prepare(npu_context)
    prepared_aclnn = BACKEND_REGISTRY.resolve(BackendKind.ACLNN).prepare(aclnn_context)

    assert prepared_npu.env["CX_BACKEND"] == "npu"
    assert prepared_npu.env["ASCEND_DEVICE_ID"] == "2"
    assert prepared_aclnn.env["CX_BACKEND"] == "aclnn"
    assert prepared_aclnn.env["ASCEND_DEVICE_ID"] == "3"


def test_aclnn_executor_calls_importable_runtime_module(tmp_path, monkeypatch):
    from cruciblex.runtime.executors import EXECUTOR_REGISTRY

    runtime_module = tmp_path / "fake_aclnn_runtime.py"
    runtime_module.write_text(
        """
import numpy as np

def abs(input_value):
    return np.abs(input_value)
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    load_builtin_plugins()
    case = CaseSpec(
        id=61,
        operator=OperatorSpec(name="fake_aclnn_runtime.abs"),
        invocation=InvocationSpec(api="fake_aclnn_runtime.abs", api_type="function", executor="aclnn"),
        parameters=[tensor_param(2)],
    )
    executor = EXECUTOR_REGISTRY.resolve("aclnn")
    plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.RUN]))[0]

    output = executor.execute(
        ExecutionRequest(case=case, inputs=[np.array([-2.0, 3.0], dtype=np.float32)], plan=plan)
    )

    assert output.tolist() == [2.0, 3.0]


def test_aclnn_executor_skips_when_runtime_module_is_missing():
    from cruciblex.runtime.executors import EXECUTOR_REGISTRY, ExecutionNotSupportedError

    load_builtin_plugins()
    case = CaseSpec(
        id=62,
        operator=OperatorSpec(name="missing_aclnn_runtime.abs"),
        invocation=InvocationSpec(api="missing_aclnn_runtime.abs", api_type="function", executor="aclnn"),
        parameters=[tensor_param(2)],
    )
    plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.RUN]))[0]

    with pytest.raises(ExecutionNotSupportedError, match="requires importable runtime module"):
        EXECUTOR_REGISTRY.resolve("aclnn").execute(
            ExecutionRequest(case=case, inputs=[np.array([-2.0, 3.0], dtype=np.float32)], plan=plan)
        )

def test_default_device_policy_maps_torch_devices():
    from cruciblex.plugins.executors.policy import DEFAULT_DEVICE_POLICY
    from cruciblex.runtime.backends.base import DeviceContext
    assert DEFAULT_DEVICE_POLICY.torch_device(DeviceContext(host="127.0.0.1", node_name="cpu", device=DeviceSpec(id=0, backend=BackendKind.CPU), output_root=Path("out"))) == "cpu"
    assert DEFAULT_DEVICE_POLICY.torch_device(DeviceContext(host="127.0.0.1", node_name="gpu", device=DeviceSpec(id=2, backend=BackendKind.GPU), output_root=Path("out"))) == "cuda:2"
    assert DEFAULT_DEVICE_POLICY.torch_device(DeviceContext(host="127.0.0.1", node_name="npu", device=DeviceSpec(id=4, backend=BackendKind.NPU), output_root=Path("out"))) == "npu:4"



def test_torch_executor_rejects_unavailable_cuda_device():
    from cruciblex.plugins.executors.torch import TorchFunctionExecutor
    from cruciblex.runtime.executors import ExecutionNotSupportedError

    class FakeCuda:
        def is_available(self):
            return False

    class FakeTorch:
        cuda = FakeCuda()

    with pytest.raises(ExecutionNotSupportedError, match="torch device is unavailable"):
        TorchFunctionExecutor()._ensure_device_available(FakeTorch(), "cuda:0")


def test_torch_executor_rejects_missing_torch_npu(monkeypatch):
    from cruciblex.plugins.executors.torch import TorchFunctionExecutor
    from cruciblex.runtime.executors import ExecutionNotSupportedError

    def missing_torch_npu(name):
        if name == "torch_npu":
            raise ImportError(name)
        return __import__(name)

    monkeypatch.setattr("importlib.import_module", missing_torch_npu)
    with pytest.raises(ExecutionNotSupportedError, match="requires torch_npu"):
        TorchFunctionExecutor()._ensure_device_available(object(), "npu:0")

def test_candidate_and_reference_can_use_distinct_executors(tmp_path):
    plugin_file = tmp_path / "biased_plugin.py"
    plugin_file.write_text("""
from cruciblex.runtime.executors.base import BackendExecutor, EXECUTOR_REGISTRY, ExecutionRequest

class CandidateExecutor(BackendExecutor):
    def execute(self, request: ExecutionRequest) -> object:
        return request.inputs[0] * 2

class ReferenceExecutor(BackendExecutor):
    def execute(self, request: ExecutionRequest) -> object:
        return request.inputs[0] * 3

EXECUTOR_REGISTRY.register("candidate_x2")(CandidateExecutor)
EXECUTOR_REGISTRY.register("reference_x3")(ReferenceExecutor)
""".strip())
    case = CaseSpec(id=11, operator=OperatorSpec(name="custom.scale"), invocation=InvocationSpec(api="custom.scale", api_type="function", executor="candidate_x2"), parameters=[tensor_param(2)], oracle=OracleSpec(comparison="allclose", reference_executor="reference_x3"))
    load_plugins([plugin_file])
    result = LocalScheduler(run_context()).submit(ExecutionPlanner().build(JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.ACCURACY]))[0])
    assert result.status == ResultStatus.FAILED
    assert result.metrics["candidate_role"] == "candidate"
    assert result.metrics["reference_role"] == "reference"
    assert result.metrics["max_abs_diff"] == 1.0


def test_custom_comparator_plugin_controls_accuracy_result(tmp_path):
    plugin_file = tmp_path / "comparator_plugin.py"
    plugin_file.write_text("""
from cruciblex.runtime.compare import COMPARATOR_REGISTRY, Comparator, ComparisonReport, ComparisonRequest
from cruciblex.runtime.executors.base import BackendExecutor, EXECUTOR_REGISTRY, ExecutionRequest

class CandidateExecutor(BackendExecutor):
    def execute(self, request: ExecutionRequest) -> object:
        return request.inputs[0] * 2

class ReferenceExecutor(BackendExecutor):
    def execute(self, request: ExecutionRequest) -> object:
        return request.inputs[0] * 3

class AlwaysPassComparator(Comparator):
    def compare(self, request: ComparisonRequest) -> ComparisonReport:
        return ComparisonReport(passed=True, max_abs_diff=99.0, mean_abs_diff=99.0, detail="forced pass")

EXECUTOR_REGISTRY.register("candidate_x2_custom")(CandidateExecutor)
EXECUTOR_REGISTRY.register("reference_x3_custom")(ReferenceExecutor)
COMPARATOR_REGISTRY.register("always_pass")(AlwaysPassComparator)
""".strip())
    case = CaseSpec(id=12, operator=OperatorSpec(name="custom.scale"), invocation=InvocationSpec(api="custom.scale", api_type="function", executor="candidate_x2_custom"), parameters=[tensor_param(2)], oracle=OracleSpec(comparison="always_pass", reference_executor="reference_x3_custom"))
    load_plugins([plugin_file])
    result = LocalScheduler(run_context()).submit(ExecutionPlanner().build(JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.ACCURACY]))[0])
    assert result.status == ResultStatus.PASSED
    assert result.metrics["comparison"] == "always_pass"
    assert result.metrics["compare_detail"] == "forced pass"


def test_custom_generator_plugin_controls_inputs(tmp_path):
    plugin_file = tmp_path / "generator_plugin.py"
    plugin_file.write_text("""
import numpy as np
from cruciblex.runtime.generation import GENERATOR_REGISTRY, GenerationRequest, InputGenerator

class ConstantInputGenerator(InputGenerator):
    def generate(self, request: GenerationRequest) -> list[object]:
        return [np.full((3,), 5, dtype=np.float32)]

GENERATOR_REGISTRY.register("constant_five")(ConstantInputGenerator)
""".strip())
    case = CaseSpec(id=13, operator=OperatorSpec(name="torch.abs"), invocation=InvocationSpec(api="torch.abs", api_type="function", executor="function"), parameters=[tensor_param()], generator="constant_five", oracle=OracleSpec(comparison="allclose", reference_executor="function"))
    load_plugins([plugin_file])
    result = LocalScheduler(run_context()).submit(ExecutionPlanner().build(JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.ACCURACY], artifacts={"output_root": tmp_path}))[0])
    inputs = json.loads(result.artifacts[0].path.read_text(encoding="utf-8"))
    assert result.status == ResultStatus.PASSED
    assert inputs[0]["data"] == [5.0, 5.0, 5.0]


def test_expected_error_passes_when_candidate_raises_matching_error(tmp_path):
    plugin_file = tmp_path / "error_plugin.py"
    plugin_file.write_text("""
from cruciblex.runtime.executors.base import BackendExecutor, EXECUTOR_REGISTRY, ExecutionRequest

class RaisingExecutor(BackendExecutor):
    def execute(self, request: ExecutionRequest) -> object:
        raise ValueError("invalid shape for operator")

EXECUTOR_REGISTRY.register("raising")(RaisingExecutor)
""".strip())
    case = CaseSpec(id=14, operator=OperatorSpec(name="custom.error"), invocation=InvocationSpec(api="custom.error", api_type="function", executor="raising"), parameters=[tensor_param()], oracle=OracleSpec(expected_error="invalid shape"))
    load_plugins([plugin_file])
    result = LocalScheduler(run_context()).submit(ExecutionPlanner().build(JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.ACCURACY]))[0])
    assert result.status == ResultStatus.PASSED
    assert result.error is None
    assert result.metrics["error_matched"] is True
    assert "ValueError" in result.metrics["actual_error"]


def test_expected_error_fails_when_candidate_error_does_not_match(tmp_path):
    plugin_file = tmp_path / "wrong_error_plugin.py"
    plugin_file.write_text("""
from cruciblex.runtime.executors.base import BackendExecutor, EXECUTOR_REGISTRY, ExecutionRequest

class RaisingExecutor(BackendExecutor):
    def execute(self, request: ExecutionRequest) -> object:
        raise RuntimeError("different failure")

EXECUTOR_REGISTRY.register("wrong_raising")(RaisingExecutor)
""".strip())
    case = CaseSpec(id=15, operator=OperatorSpec(name="custom.error"), invocation=InvocationSpec(api="custom.error", api_type="function", executor="wrong_raising"), parameters=[tensor_param()], oracle=OracleSpec(expected_error="invalid shape"))
    load_plugins([plugin_file])
    result = LocalScheduler(run_context()).submit(ExecutionPlanner().build(JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.ACCURACY]))[0])
    assert result.status == ResultStatus.FAILED
    assert result.metrics["error_matched"] is False
    assert "different failure" in result.error


def test_expected_error_fails_when_candidate_does_not_raise():
    case = CaseSpec(id=16, operator=OperatorSpec(name="torch.abs"), invocation=InvocationSpec(api="torch.abs", api_type="function", executor="function"), parameters=[tensor_param()], oracle=OracleSpec(expected_error="invalid shape"))
    result = LocalScheduler(run_context()).submit(ExecutionPlanner().build(JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.ACCURACY]))[0])
    assert result.status == ResultStatus.FAILED
    assert result.metrics["error_matched"] is False
    assert "expected error was not raised" in result.error


def test_planner_expands_cases_over_execution_slots_in_stable_order():
    first = CaseSpec(id=21, operator=OperatorSpec(name="torch.abs"), invocation=InvocationSpec(api="torch.abs", api_type="function"))
    second = CaseSpec(id=22, operator=OperatorSpec(name="torch.add"), invocation=InvocationSpec(api="torch.add", api_type="function"))
    plans = ExecutionPlanner().build(JobSpec(cases=[first, second], nodes=[cpu_node((0, 1))], tasks=[TaskKind.ACCURACY, TaskKind.RUN]))
    assert [plan.plan_id for plan in plans] == ["21:cpu:cpu:0:accuracy", "21:cpu:cpu:1:accuracy", "21:cpu:cpu:0:run", "21:cpu:cpu:1:run", "22:cpu:cpu:0:accuracy", "22:cpu:cpu:1:accuracy", "22:cpu:cpu:0:run", "22:cpu:cpu:1:run"]



def test_pipeline_can_defer_artifact_persistence(tmp_path):
    from cruciblex.runtime.pipeline import ExecutionPipeline

    job = load_job(
        "examples/cases/torch.abs.yaml",
        "examples/nodes/local.yaml",
        scheduler=SchedulerKind.LOCAL,
        output_path=tmp_path,
    )
    plan = ExecutionPlanner().build(job)[0]
    result = ExecutionPipeline().run(plan, persist_artifacts=False)
    assert result.status == ResultStatus.PASSED
    assert result.artifacts == []
    assert [payload.name for payload in result.artifact_payloads] == [
        "inputs",
        "candidate_output",
        "reference_output",
    ]
    assert not (tmp_path / plan.case.name / plan.plan_id).exists()


def test_ray_scheduler_materializes_worker_artifacts_on_driver(tmp_path, monkeypatch):
    from cruciblex.runtime.scheduler.ray import RayScheduler

    class FakeRay:
        def get(self, ref):
            return ref

    case = CaseSpec(
        id=31,
        operator=OperatorSpec(name="custom.echo"),
        invocation=InvocationSpec(api="custom.echo", api_type="function"),
    )
    plan = ExecutionPlanner().build(
        JobSpec(
            cases=[case],
            nodes=[cpu_node()],
            tasks=[TaskKind.RUN],
            artifacts={"output_root": tmp_path},
        )
    )[0]
    result = ExecutionResult(
        plan_id=plan.plan_id,
        case_id=plan.case.id,
        case_name=plan.case.name,
        node_name=plan.node.display_name,
        backend=plan.device.backend,
        device_id=plan.device.id,
        task=plan.task,
        status=ResultStatus.PASSED,
        artifact_payloads=[ArtifactPayload(name="worker_output", kind="json", data={"ok": True})],
    )
    raw_result = result.model_dump(mode="json")
    raw_result["artifact_payloads"] = [
        payload.model_dump(mode="json") for payload in result.artifact_payloads
    ]
    scheduler = RayScheduler(run_context(output_root=tmp_path, scheduler=SchedulerKind.RAY).model_copy(update={"run_id": "run:test"}))
    scheduler._refs = [(plan, raw_result)]
    monkeypatch.setattr(scheduler, "_ray", lambda: FakeRay())

    collected = scheduler.collect()
    assert collected[0].artifact_payloads == []
    assert collected[0].artifacts[0].name == "worker_output"
    assert collected[0].artifacts[0].path.exists()
    assert json.loads(collected[0].artifacts[0].path.read_text(encoding="utf-8")) == {"ok": True}


def test_result_store_writes_jsonl_summary_and_manifest(tmp_path):
    result = LocalScheduler(run_context()).submit(ExecutionPlanner().build(load_job("examples/cases/torch.abs.yaml", "examples/nodes/local.yaml", scheduler=SchedulerKind.LOCAL, output_path=tmp_path))[0])
    store = ResultStore(tmp_path)
    manifest = RunManifest(case_path=Path("cases.yaml"), node_path=Path("nodes.yaml"), tasks=[TaskKind.ACCURACY], scheduler=SchedulerKind.RAY, output_root=tmp_path, plugin_paths=[Path("plugin.py")], cruciblex_version=__version__, plan_count=3)
    manifest_path = store.write_manifest(manifest)
    results_path = store.write_results_jsonl([result])
    summary_path = store.write_summary_json({"total": 1, "passed": 1, "failed": 0})
    loaded = store.read_manifest()
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert manifest_path.name == "manifest.json"
    assert loaded.run_id == manifest.run_id
    assert loaded.scheduler == SchedulerKind.RAY
    assert rows[0]["status"] == "passed"
    assert json.loads(summary_path.read_text(encoding="utf-8")) == {"total": 1, "passed": 1, "failed": 0}



def test_run_context_projects_to_manifest(tmp_path):
    context = RunContext(
        run_id="run-context-test",
        case_path=Path("cases.yaml"),
        node_path=Path("nodes.yaml"),
        tasks=[TaskKind.ACCURACY],
        scheduler=SchedulerKind.RAY,
        output_root=tmp_path,
        plugin_paths=[Path("plugin.py")],
        metadata={"source": "test"},
    )
    manifest = context.to_manifest(
        cruciblex_version=__version__,
        plan_count=2,
        submitted_count=1,
        skipped_count=1,
    )
    assert manifest.run_id == context.run_id
    assert manifest.output_root == tmp_path
    assert manifest.plugin_paths == [Path("plugin.py")]
    assert manifest.metadata == {"source": "test"}
    assert manifest.plan_count == 2
    assert manifest.submitted_count == 1
    assert manifest.skipped_count == 1

def test_run_manifest_records_result_outputs(tmp_path):
    manifest = RunManifest(case_path=Path("cases.yaml"), node_path=Path("nodes.yaml"), tasks=[TaskKind.RUN], scheduler=SchedulerKind.RAY, output_root=tmp_path, cruciblex_version=__version__, plan_count=1)
    updated = manifest.with_outputs(results_path=tmp_path / "results.jsonl", summary_path=tmp_path / "summary.json")
    assert updated.results_path == tmp_path / "results.jsonl"
    assert updated.summary_path == tmp_path / "summary.json"
    assert updated.run_id == manifest.run_id


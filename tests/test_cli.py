import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from cruciblex import __version__
from cruciblex.cli import app
from cruciblex.domain import (
    ArtifactPayload,
    ArtifactRef,
    BackendKind,
    CaseSpec,
    DeviceSpec,
    ExecutionResult,
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
from cruciblex.report import ReproBundleWriter
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
    ray_address: str | None = None,
) -> RunContext:
    return RunContext(
        run_id="test-run",
        case_path=Path("cases.yaml"),
        node_path=Path("nodes.yaml"),
        tasks=[TaskKind.ACCURACY],
        scheduler=scheduler,
        output_root=Path(output_root).resolve(),
        ray_address=ray_address,
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








def test_import_atb_and_temu_preserve_backend_metadata(tmp_path):
    from cruciblex.generation.loader import load_cases

    atb_output = tmp_path / "atb_cases.yaml"
    atb_result = CliRunner().invoke(
        app,
        [
            "import-atb",
            "--source",
            "examples/imports/atb.add.config.yaml",
            "--output",
            str(atb_output),
            "--case-id",
            "303",
            "--executor",
            "atb",
            "--reference-executor",
            "torch",
        ],
    )
    assert atb_result.exit_code == 0
    atb_case = load_cases(atb_output)[0]
    assert atb_case.operator.name == "atb.add"
    assert atb_case.invocation.executor == "atb"
    assert atb_case.invocation.api_type == "backend"
    assert atb_case.generator == "default"
    assert atb_case.metadata["backend_import"]["source_format"] == "atb"
    assert atb_case.metadata["backend_import"]["plugin_skeleton"]["module"] == "cruciblex.plugins.executors.atb"
    assert atb_case.metadata["provenance"]["converter_version"] == "cruciblex.importers.backend:v1"
    assert atb_case.parameters[0].dtypes == ["fp16"]
    assert atb_case.parameters[0].shape.dims == [4, 4]
    assert "Imported ATB case:" in atb_result.stdout
    assert "Warnings: 1" in atb_result.stdout

    temu_output = tmp_path / "temu_cases.yaml"
    temu_result = CliRunner().invoke(
        app,
        [
            "import-temu",
            "--source",
            "examples/imports/temu.softmax.config.yaml",
            "--output",
            str(temu_output),
            "--case-id",
            "404",
            "--executor",
            "temu",
            "--reference-executor",
            "torch",
        ],
    )
    assert temu_result.exit_code == 0
    temu_case = load_cases(temu_output)[0]
    assert temu_case.operator.name == "temu.softmax"
    assert temu_case.invocation.executor == "temu"
    assert temu_case.metadata["backend_import"]["source_format"] == "temu"
    assert temu_case.metadata["backend_import"]["plugin_skeleton"]["executor_name"] == "temu"
    assert temu_case.metadata["provenance"]["source_format"] == "temu"
    assert temu_case.parameters[0].dtypes == ["fp32"]
    assert temu_case.parameters[0].shape.dims == [1, 3, 8, 8]
    assert "Imported TEMU case:" in temu_result.stdout

def test_import_profile_writes_generation_policy_case(tmp_path):
    from cruciblex.generation.loader import load_cases

    output = tmp_path / "profile_cases.yaml"
    result = CliRunner().invoke(
        app,
        [
            "import-profile",
            "--source",
            "examples/imports/torch.matmul.profile.yaml",
            "--output",
            str(output),
            "--case-id",
            "202",
            "--executor",
            "torch",
            "--reference-executor",
            "torch",
        ],
    )

    assert result.exit_code == 0
    assert output.exists()
    imported = load_cases(output)[0]
    assert imported.id == 202
    assert imported.operator.name == "torch.matmul"
    assert imported.generator == "default"
    assert imported.generation.count == 4
    assert imported.invocation.executor == "torch"
    assert imported.parameters[0].name == "input"
    assert imported.parameters[0].dtypes == ["fp32", "fp16"]
    assert imported.parameters[0].shape.dim_count == [2]
    assert imported.parameters[0].shape.dim_values == [2, 3, 8, 16]
    assert imported.generation.metadata["profile_shapes"]["other"] == [[3, 4], [16, 32]]
    assert imported.generation.metadata["profile_dtypes"]["other"] == ["fp32", "fp16"]
    assert imported.metadata["provenance"]["source_format"] == "profile"
    assert imported.metadata["provenance"]["converter_version"] == "cruciblex.importers.profile:v1"
    assert imported.metadata["provenance"]["sample_count"] == 2
    assert "Imported profile case:" in result.stdout
    assert "Samples: 2" in result.stdout

def test_import_dump_writes_replayable_case_and_inputs(tmp_path):
    from cruciblex.generation.loader import load_cases
    from cruciblex.runtime.generation import GENERATOR_REGISTRY, GenerationRequest

    output = tmp_path / "cases.yaml"
    result = CliRunner().invoke(
        app,
        [
            "import-dump",
            "--source",
            "examples/imports/torch.add.dump.yaml",
            "--output",
            str(output),
            "--case-id",
            "101",
            "--executor",
            "torch",
            "--reference-executor",
            "torch",
        ],
    )

    assert result.exit_code == 0
    snapshot_path = tmp_path / "inputs.json"
    assert output.exists()
    assert snapshot_path.exists()
    imported = load_cases(output)[0]
    assert imported.id == 101
    assert imported.generator == "dump_replay"
    assert imported.operator.name == "torch.add"
    assert imported.invocation.executor == "torch"
    assert imported.parameters[0].dtypes == ["fp32"]
    assert imported.parameters[0].shape.dims == [2, 2]
    assert imported.metadata["provenance"]["source_format"] == "dump"
    assert imported.metadata["provenance"]["converter_version"] == "cruciblex.importers.dump:v1"
    assert imported.metadata["provenance"]["input_snapshot_path"] == str(snapshot_path.resolve())

    generator = GENERATOR_REGISTRY.resolve("dump_replay")
    inputs = generator.generate(GenerationRequest(case=imported, plan=None))
    assert [list(item.shape) for item in inputs] == [[2, 2], [2, 2]]
    assert inputs[0].dtype == np.float32
    assert np.allclose(inputs[0], np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    assert "Imported dump case:" in result.stdout
    assert "Input snapshot:" in result.stdout







def test_doctor_reports_ray_status():
    runner = CliRunner()
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Ray:" in result.stdout




def test_worker_probe_options_request_gpu_resources_for_gpu_nodes():
    from types import SimpleNamespace

    from cruciblex.cli import _worker_probe_options

    options = _worker_probe_options(
        SimpleNamespace(node_resource_key="node:gpu", resources={"CPU": 32.0, "GPU": 8.0})
    )

    assert options["num_gpus"] == 1
    assert options["resources"] == {"node:gpu": 0.001}

def test_worker_runtime_summary_includes_torch_device_probe_fields():
    from cruciblex.cli import _worker_runtime_summary

    summary = _worker_runtime_summary(
        {
            "runtime_probe": {
                "env": {"CUDA_VISIBLE_DEVICES": "0", "ASCEND_DEVICE_ID": "1"},
                "packages": {
                    "torch": {
                        "version": "2.6.0+cu126",
                        "cuda_version": "12.6",
                        "cuda_available": True,
                        "cuda_device_count": 1,
                    },
                    "torch_npu": {"available": False},
                },
            }
        }
    )

    assert "torch=2.6.0+cu126" in summary
    assert "torch_cuda=12.6" in summary
    assert "cuda_available=True" in summary
    assert "cuda_device_count=1" in summary
    assert "torch_npu_available=False" in summary
    assert "CUDA_VISIBLE_DEVICES=0" in summary
    assert "ASCEND_DEVICE_ID=1" in summary

def test_run_writes_log_file(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--case",
            "examples/cases/torch.abs.yaml",
            "--nodes",
            "examples/nodes/local.yaml",
            "--scheduler",
            "local",
            "--output",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    log_path = tmp_path / "run.log"
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "run.start" in log_text
    assert "run.complete" in log_text
    discovery_path = tmp_path / "driver" / "resource_snapshot.json"
    discovered_nodes_path = tmp_path / "driver" / "discovered_nodes.yaml"
    assert discovery_path.exists()
    assert discovered_nodes_path.exists()
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    assert discovery["kind"] == "ray-resource-discovery"
    assert f"Log: {log_path.resolve()}" in result.stdout
    assert "Discovery:" in result.stdout
    assert "Discovered nodes:" in result.stdout


def test_device_actor_can_return_execution_log_payload(tmp_path):
    from cruciblex.runtime.actors.device import DeviceActor

    job = load_job(
        "examples/cases/torch.abs.yaml",
        "examples/nodes/local.yaml",
        scheduler=SchedulerKind.LOCAL,
        output_path=tmp_path,
    )
    plan = ExecutionPlanner().build(job)[0]
    result = DeviceActor(plan.node.host, plan.device.id, persist_artifacts=False, capture_logs=True).run(plan)

    log_payload = result.artifact_payloads[-1]
    assert log_payload.name == "execution_log"
    assert log_payload.kind == "log"
    assert "actor.run.start" in log_payload.data
    assert "pipeline.start" in log_payload.data
    assert log_payload.metadata["plan_id"] == plan.plan_id

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
    assert {artifact.name for artifact in results[0].artifacts} == {"inputs", "candidate_output"}
    assert results[0].candidate_role is None
    assert results[0].reference_role is None
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


def test_load_job_expands_generated_cases_and_persists_output(tmp_path):
    case_file = tmp_path / "case.yaml"
    case_file.write_text(
        """
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
    generator: default
    generation:
      count: 2
      seed: 99
      max_elements: 4
    parameters:
      - name: input
        kind: tensor
        dtypes: [fp32]
        shape:
          dim_count: [2]
          dim_values: [4]
        value_range:
          valid: [[1, 1]]
""".strip(),
        encoding="utf-8",
    )
    node_file = tmp_path / "nodes.yaml"
    node_file.write_text(
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

    job = load_job(case_file, node_file, scheduler=SchedulerKind.LOCAL, output_path=tmp_path)
    payload = json.loads((tmp_path / "generated_cases.json").read_text(encoding="utf-8"))

    assert len(job.cases) == 2
    assert {case.metadata["source_case_id"] for case in job.cases} == {7}
    assert {case.metadata["generation_index"] for case in job.cases} == {0, 1}
    assert all(int(np.prod(case.parameters[0].shape.dims or [1])) <= 4 for case in job.cases)
    assert len(payload["cases"]) == 2


def test_generate_cli_expands_cases_and_writes_outputs(tmp_path):
    from cruciblex import cli as cli_module

    output = tmp_path / "generated"
    result = CliRunner().invoke(
        cli_module.app,
        ["generate", "--case", "examples/cases/torch.abs.generated.yaml", "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    generated_json = json.loads((output / "generated_cases.json").read_text(encoding="utf-8"))
    generated_yaml = yaml.safe_load((output / "generated_cases.yaml").read_text(encoding="utf-8"))
    assert len(generated_json["cases"]) == 3
    assert len(generated_yaml["cases"]) == 3
    assert (output / "generated_cases.json").exists()
    assert (output / "generated_cases.yaml").exists()


def test_load_job_applies_max_bytes_constraints(tmp_path):
    case_file = tmp_path / "case.yaml"
    case_file.write_text(
        """
cases:
  - id: 30
    operator:
      name: custom.max_bytes
    invocation:
      api: custom.max_bytes
      api_type: function
      executor: function
    oracle:
      comparison: allclose
    generator: default
    generation:
      count: 1
      max_bytes: 16
    parameters:
      - name: input
        kind: tensor
        dtypes: [fp32]
        shape:
          dims: [4, 4]
        value_range:
          valid: [[1, 1]]
""".strip(),
        encoding="utf-8",
    )
    node_file = tmp_path / "nodes.yaml"
    node_file.write_text(
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

    job = load_job(case_file, node_file, scheduler=SchedulerKind.LOCAL, output_path=tmp_path)

    assert len(job.cases) == 1
    assert job.cases[0].generation.max_bytes == 16
    assert int(np.prod(job.cases[0].parameters[0].shape.dims or [1])) <= 4
    assert job.cases[0].parameters[0].metadata["max_bytes"] == 16
    assert job.cases[0].parameters[0].metadata["estimated_bytes"] <= 16


def test_load_job_expands_invalid_cases_with_selected_invalid_values(tmp_path):
    case_file = tmp_path / "case.yaml"
    case_file.write_text(
        """
cases:
  - id: 10
    operator:
      name: custom.invalid
    invocation:
      api: custom.invalid
      api_type: function
      executor: function
    oracle:
      comparison: allclose
    generator: default
    generation:
      invalid_count: 2
    parameters:
      - name: input
        kind: tensor
        dtypes: [fp32]
        shape:
          dims: [1]
        value_range:
          valid: [[1, 1]]
          invalid: [[10, 20], [30, 40]]
""".strip(),
        encoding="utf-8",
    )
    node_file = tmp_path / "nodes.yaml"
    node_file.write_text(
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

    job = load_job(case_file, node_file, scheduler=SchedulerKind.LOCAL, output_path=tmp_path)
    payload = json.loads((tmp_path / "generated_cases.json").read_text(encoding="utf-8"))

    assert len(job.cases) == 3
    invalid_cases = [case for case in job.cases if case.metadata.get("expected_invalid")]
    assert [case.metadata["invalid_index"] for case in invalid_cases] == [0, 1]
    assert [case.parameters[0].metadata["selected_invalid_value"] for case in invalid_cases] == [[10, 20], [30, 40]]
    assert sum(1 for case in payload["cases"] if case["metadata"].get("expected_invalid")) == 2


def test_load_job_applies_random_coverage_constraints(tmp_path):
    case_file = tmp_path / "case.yaml"
    case_file.write_text(
        """
cases:
  - id: 12
    operator:
      name: custom.random
    invocation:
      api: custom.random
      api_type: function
      executor: function
    oracle:
      comparison: allclose
    generator: default
    generation:
      count: 3
      seed: 7
      constraints: [random_coverage]
    parameters:
      - name: input
        kind: tensor
        dtypes: [fp32, fp16]
        shape:
          dims: [2, 2]
        value_range:
          valid: [[0, 1], [2, 3], [4, 5]]
        metadata:
          random_coverage: true
          random_dtypes: [fp32, fp16]
          random_shapes:
            - [2, 2]
            - [1, 4]
          random_values:
            - [0, 1]
            - [2, 3]
""".strip(),
        encoding="utf-8",
    )
    node_file = tmp_path / "nodes.yaml"
    node_file.write_text(
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

    job = load_job(case_file, node_file, scheduler=SchedulerKind.LOCAL, output_path=tmp_path)
    selected = [case.parameters[0].metadata.get("selected_random_value") for case in job.cases]
    shapes = [case.parameters[0].shape.dims for case in job.cases]
    dtypes = [case.parameters[0].dtypes[0] for case in job.cases]

    assert len(job.cases) == 3
    assert selected == [[2, 3], [2, 3], [0, 1]]
    assert shapes == [[1, 4], [2, 2], [1, 4]]
    assert dtypes == ["fp32", "fp32", "fp32"]


def test_load_job_applies_boundary_coverage_constraints(tmp_path):
    case_file = tmp_path / "case.yaml"
    case_file.write_text(
        """
cases:
  - id: 9
    operator:
      name: custom.boundary
    invocation:
      api: custom.boundary
      api_type: function
      executor: function
    oracle:
      comparison: allclose
    generator: default
    generation:
      count: 3
      constraints: [boundary_coverage]
    parameters:
      - name: input
        kind: tensor
        dtypes: [fp32]
        shape:
          dims: [1]
        value_range:
          valid: [[1, 1]]
        metadata:
          cycle_on_index: true
          boundary_dtypes: [fp32, fp64]
          boundary_values:
            - [1]
            - [2, 2]
            - [3, 1]
""".strip(),
        encoding="utf-8",
    )
    node_file = tmp_path / "nodes.yaml"
    node_file.write_text(
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

    job = load_job(case_file, node_file, scheduler=SchedulerKind.LOCAL, output_path=tmp_path)
    payload = json.loads((tmp_path / "generated_cases.json").read_text(encoding="utf-8"))

    assert [case.parameters[0].dtypes for case in job.cases] == [["fp32"], ["fp64"], ["fp32"]]
    assert [case.parameters[0].shape.dims for case in job.cases] == [[1], [2, 2], [3, 1]]
    assert len(payload["cases"]) == 3


def test_load_job_applies_linked_parameter_constraints(tmp_path):
    case_file = tmp_path / "case.yaml"
    case_file.write_text(
        """
cases:
  - id: 8
    operator:
      name: custom.linked
    invocation:
      api: custom.linked
      api_type: function
      executor: function
    oracle:
      comparison: allclose
    generator: default
    generation:
      constraints: [linked_parameters]
    parameters:
      - name: lhs
        kind: tensor
        dtypes: [fp16]
        shape:
          dims: [2, 3]
        value_range:
          valid: [[1, 1]]
      - name: rhs
        kind: tensor
        dtypes: [fp32]
        shape:
          dims: [4, 5]
        value_range:
          valid: [[1, 1]]
        metadata:
          same_dtype_as: lhs
          same_shape_as: lhs
""".strip(),
        encoding="utf-8",
    )
    node_file = tmp_path / "nodes.yaml"
    node_file.write_text(
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

    job = load_job(case_file, node_file, scheduler=SchedulerKind.LOCAL, output_path=tmp_path)
    payload = json.loads((tmp_path / "generated_cases.json").read_text(encoding="utf-8"))

    assert len(job.cases) == 1
    case = job.cases[0]
    assert case.parameters[1].dtypes == ["fp16"]
    assert case.parameters[1].shape.dims == [2, 3]
    assert case.parameters[1].metadata["resolved_dtype_from"] == "lhs"
    assert case.parameters[1].metadata["resolved_shape_from"] == "lhs"
    assert len(payload["cases"]) == 1


def test_expected_invalid_success_is_failed(tmp_path):
    job = load_job(
        "examples/cases/torch.abs.invalid.yaml",
        "examples/nodes/local.yaml",
        tasks=[TaskKind.RUN],
        scheduler=SchedulerKind.LOCAL,
        output_path=tmp_path,
    )
    invalid_plan = ExecutionPlanner().build(job)[1]

    result = LocalScheduler(run_context(output_root=tmp_path)).submit(invalid_plan)

    assert result.status == ResultStatus.FAILED
    assert result.metrics["expected_invalid"] is True
    assert result.metrics["invalid_rejected"] is False
    assert result.metrics["failure_kind"] == "failed"
    assert result.metrics["failure_stage"] == "oracle"
    assert "expected invalid case executed successfully" in result.error


def test_expected_invalid_error_is_passed(tmp_path):
    plugin_file = tmp_path / "reject_plugin.py"
    plugin_file.write_text(
        """
from cruciblex.runtime.executors.base import BackendExecutor, EXECUTOR_REGISTRY, ExecutionRequest

class RejectExecutor(BackendExecutor):
    def execute(self, request: ExecutionRequest) -> object:
        raise ValueError("invalid input rejected")

EXECUTOR_REGISTRY.register("reject")(RejectExecutor)
""".strip(),
        encoding="utf-8",
    )
    load_plugins([plugin_file])
    case = CaseSpec(
        id=88,
        operator=OperatorSpec(name="custom.reject"),
        invocation=InvocationSpec(api="custom.reject", api_type="function", executor="reject"),
        parameters=[tensor_param(1)],
        metadata={"expected_invalid": True, "invalid_index": 0, "source_case_id": 88},
    )
    plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.RUN], artifacts={"output_root": tmp_path}))[0]

    result = LocalScheduler(run_context(output_root=tmp_path)).submit(plan)

    assert result.status == ResultStatus.PASSED
    assert result.metrics["expected_invalid"] is True
    assert result.metrics["invalid_rejected"] is True
    assert "invalid input rejected" in result.metrics["actual_error"]


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
    assert len(result.artifacts) == 2
    assert result.metrics["comparison"] == "allclose"


def test_backend_context_metrics_and_ray_resources(tmp_path):
    from cruciblex.runtime.backends import ray_resources_for
    assert ray_resources_for(DeviceSpec(id=0, backend=BackendKind.CPU)).num_cpus == 1.0
    assert ray_resources_for(DeviceSpec(id=1, backend=BackendKind.GPU)).num_gpus == 1
    assert ray_resources_for(DeviceSpec(id=2, backend=BackendKind.NPU)).resources == {"npu": 1.0}
    assert ray_resources_for(DeviceSpec(id=3, backend=BackendKind.DCU)).resources == {"dcu": 1.0}
    assert ray_resources_for(DeviceSpec(id=4, backend=BackendKind.ACLNN)).resources == {"npu": 1.0}
    assert ray_resources_for(DeviceSpec(id=5, backend=BackendKind.ACLNN, resources={"npu": 0.0, "aclnn_runtime": 1.0})).resources == {"npu": 1.0, "aclnn_runtime": 1.0}
    result = LocalScheduler(run_context()).submit(ExecutionPlanner().build(load_job("examples/cases/torch.abs.yaml", "examples/nodes/local.yaml", scheduler=SchedulerKind.LOCAL, output_path=tmp_path))[0])
    assert result.metrics["backend"] == "cpu"
    assert result.metrics["host"] == "127.0.0.1"
    assert result.metrics["device_id"] == 0


def test_gpu_torch_smoke_case_uses_torch_executor(tmp_path):
    job = load_job(
        "examples/cases/torch.add.gpu.yaml",
        "examples/nodes/ray-cpu-gpu-e2e.yaml",
        scheduler=SchedulerKind.RAY,
        output_path=tmp_path,
    )
    case = job.cases[0]
    assert case.invocation.executor == "torch"
    assert case.oracle.reference_executor == "torch"


def test_gpu_image_pins_torch_26_cuda_126():
    image = Path("docker/Dockerfile.gpu").read_text(encoding="utf-8")
    assert "ARG TORCH_VERSION=2.6.0+cu126" in image
    assert "https://download.pytorch.org/whl/cu126" in image


def test_pipeline_records_resolved_devices_for_device_executors():
    from cruciblex.runtime.backends.base import DeviceContext
    from cruciblex.runtime.pipeline import ExecutionPipeline

    pipeline = ExecutionPipeline()
    assert pipeline._context_metrics(
        DeviceContext(host="127.0.0.1", node_name="cpu", device=DeviceSpec(id=0, backend=BackendKind.CPU), output_root=Path("out")),
        "torch",
    )["resolved_device"] == "cpu"
    assert pipeline._context_metrics(
        DeviceContext(host="127.0.0.2", node_name="gpu", device=DeviceSpec(id=2, backend=BackendKind.GPU), output_root=Path("out")),
        "torch",
    )["resolved_device"] == "cuda:2"
    assert pipeline._context_metrics(
        DeviceContext(host="127.0.0.2", node_name="gpu", device=DeviceSpec(id=2, backend=BackendKind.GPU), output_root=Path("out"), env={"CX_DEVICE_INDEX_MODE": "actor_local"}),
        "torch",
    )["resolved_device"] == "cuda:0"
    assert pipeline._context_metrics(
        DeviceContext(host="127.0.0.3", node_name="npu", device=DeviceSpec(id=1, backend=BackendKind.NPU), output_root=Path("out")),
        "torch",
    )["resolved_device"] == "npu:1"
    assert pipeline._context_metrics(
        DeviceContext(host="127.0.0.4", node_name="aclnn", device=DeviceSpec(id=3, backend=BackendKind.ACLNN), output_root=Path("out")),
        "aclnn",
    )["resolved_device"] == "npu:3"
    assert "resolved_device" not in pipeline._context_metrics(
        DeviceContext(host="127.0.0.5", node_name="gpu", device=DeviceSpec(id=0, backend=BackendKind.GPU), output_root=Path("out")),
        "function",
    )


def test_gpu_backend_runtime_prepares_cuda_environment():
    from cruciblex.runtime.backends import DcuBackendRuntime, GpuBackendRuntime
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
    assert context.env["CX_GPU_AVAILABLE"] in {"true", "false"}
    assert context.env["CX_GPU_DEVICE_COUNT"].isdigit()

    dcu_context = DeviceContext(
        host="10.0.0.10",
        node_name="dcu-node",
        device=DeviceSpec(id=5, backend=BackendKind.DCU),
        output_root=Path("out"),
    )
    prepared = DcuBackendRuntime().prepare(dcu_context)
    assert prepared is dcu_context
    assert dcu_context.env["CX_BACKEND"] == "dcu"
    assert dcu_context.env["CX_DEVICE_ID"] == "5"
    assert dcu_context.env["HIP_VISIBLE_DEVICES"] == "5"
    assert dcu_context.env["ROCR_VISIBLE_DEVICES"] == "5"


@pytest.mark.ray
def test_ray_scheduler_actor_key_and_options():
    from cruciblex.runtime.scheduler.ray import ActorKey, RayScheduler
    case = CaseSpec(id=3, operator=OperatorSpec(name="torch.abs"), invocation=InvocationSpec(api="torch.abs", api_type="function", executor="function"))
    gpu_plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[NodeSpec(name="gpu-node", host="10.0.0.8", devices=[DeviceSpec(id=2, backend=BackendKind.GPU)])], tasks=[TaskKind.RUN]))[0]
    npu_plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[NodeSpec(name="npu-node", host="10.0.0.9", devices=[DeviceSpec(id=4, backend=BackendKind.NPU)])], tasks=[TaskKind.RUN]))[0]
    dcu_plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[NodeSpec(name="dcu-node", host="10.0.0.10", devices=[DeviceSpec(id=6, backend=BackendKind.DCU)])], tasks=[TaskKind.RUN]))[0]
    scheduler = RayScheduler(run_context(scheduler=SchedulerKind.RAY, plugin_paths=[Path("plugin.py")]).model_copy(update={"run_id": "run:test"}))
    assert ActorKey.from_plan(gpu_plan).label() == "10.0.0.8:gpu:2"
    assert scheduler._safe_label("run:test") == "run-test"
    assert [str(path) for path in scheduler.context.plugin_paths] == ["plugin.py"]
    assert gpu_plan.device.backend == BackendKind.GPU
    assert npu_plan.device.backend == BackendKind.NPU
    assert dcu_plan.device.backend == BackendKind.DCU



@pytest.mark.ray
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

    ray_node_ip_resource = RayNodeInfo(
        node_id="node-2",
        address="10.0.0.8",
        hostname="worker-a",
        alive=True,
        resources={"CPU": 8.0, "GPU": 1.0, "node:10.0.0.8": 1.0},
    )
    placement = decide_ray_placement(plan, RayClusterSnapshot([ray_node_ip_resource]))
    assert placement.actor_options() == {
        "num_gpus": 1,
        "resources": {"node:10.0.0.8": 0.001},
    }


@pytest.mark.ray
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


@pytest.mark.ray
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


@pytest.mark.ray
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
    assert result.metrics["failure_kind"] == "skip"
    assert result.metrics["failure_stage"] == "placement"
    assert "no alive Ray node" in result.error


@pytest.mark.ray
def test_ray_init_kwargs_disable_uv_runtime_env_for_ray_client(monkeypatch):
    from cruciblex.runtime.scheduler.ray import ray_init_kwargs

    monkeypatch.delenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", raising=False)

    assert ray_init_kwargs("ray://203.0.113.10:10001") == {
        "ignore_reinit_error": True,
        "address": "ray://203.0.113.10:10001",
        "runtime_env": {"working_dir": None},
    }
    assert os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"



@pytest.mark.ray
def test_ray_scheduler_uses_explicit_ray_address(monkeypatch):
    from cruciblex.runtime.scheduler.ray import RayScheduler

    class FakeRay:
        def __init__(self):
            self.init_calls = []

        def is_initialized(self):
            return False

        def init(self, **kwargs):
            self.init_calls.append(kwargs)

        def nodes(self):
            return []

    case = CaseSpec(id=52, operator=OperatorSpec(name="torch.abs"), invocation=InvocationSpec(api="torch.abs", api_type="function"))
    plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.RUN]))[0]
    fake_ray = FakeRay()
    scheduler = RayScheduler(run_context(scheduler=SchedulerKind.RAY, ray_address="203.0.113.10:6379"))
    monkeypatch.setattr(scheduler, "_ray", lambda: fake_ray)

    result = scheduler.submit(plan)
    assert result.status == ResultStatus.SKIPPED
    assert fake_ray.init_calls == [{"ignore_reinit_error": True, "address": "203.0.113.10:6379"}]


@pytest.mark.ray
def test_ray_scheduler_uses_ray_client_runtime_env(monkeypatch):
    from cruciblex.runtime.scheduler.ray import RayScheduler

    class FakeRay:
        def __init__(self):
            self.init_calls = []

        def is_initialized(self):
            return False

        def init(self, **kwargs):
            self.init_calls.append(kwargs)

        def nodes(self):
            return []

    case = CaseSpec(id=53, operator=OperatorSpec(name="torch.abs"), invocation=InvocationSpec(api="torch.abs", api_type="function"))
    plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.RUN]))[0]
    fake_ray = FakeRay()
    scheduler = RayScheduler(run_context(scheduler=SchedulerKind.RAY, ray_address="ray://203.0.113.10:10001"))
    monkeypatch.setattr(scheduler, "_ray", lambda: fake_ray)

    result = scheduler.submit(plan)
    assert result.status == ResultStatus.SKIPPED
    assert fake_ray.init_calls == [{"ignore_reinit_error": True, "address": "ray://203.0.113.10:10001", "runtime_env": {"working_dir": None}}]




@pytest.mark.ray
def test_ray_scheduler_distributes_driver_inputs_with_ray_put(tmp_path, monkeypatch):
    from cruciblex.runtime.scheduler.ray import RayScheduler

    class FakeActorMethod:
        def __init__(self):
            self.calls = []

        def remote(self, raw_plan, inputs):
            self.calls.append((raw_plan, inputs))
            return "result-ref"

    class FakeActor:
        def __init__(self):
            self.run = FakeActorMethod()

    class FakeActorClass:
        def __init__(self, actor):
            self.actor = actor
            self.options_calls = []
            self.remote_calls = []

        def options(self, **kwargs):
            self.options_calls.append(kwargs)
            return self

        def remote(self, *args):
            self.remote_calls.append(args)
            return self.actor

    class FakeRay:
        def __init__(self):
            self.put_calls = []
            self.actor = FakeActor()
            self.actor_cls = FakeActorClass(self.actor)

        def is_initialized(self):
            return True

        def nodes(self):
            return [
                {
                    "NodeID": "node-1",
                    "NodeManagerAddress": "10.1.2.3",
                    "NodeManagerHostname": "local",
                    "Alive": True,
                    "Resources": {"CPU": 2.0, "node:node-1": 1.0},
                }
            ]

        def put(self, value):
            self.put_calls.append(value)
            return "inputs-ref"

        def remote(self, cls):
            return self.actor_cls

    case = CaseSpec(
        id=54,
        operator=OperatorSpec(name="torch.abs"),
        invocation=InvocationSpec(api="torch.abs", api_type="function"),
        parameters=[tensor_param(3)],
    )
    plan = ExecutionPlanner().build(
        JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.RUN], artifacts={"output_root": tmp_path})
    )[0]
    fake_ray = FakeRay()
    scheduler = RayScheduler(run_context(output_root=tmp_path, scheduler=SchedulerKind.RAY))
    monkeypatch.setattr(scheduler, "_ray", lambda: fake_ray)

    ref = scheduler.submit(plan)

    assert ref == "result-ref"
    assert len(fake_ray.put_calls) == 1
    assert fake_ray.actor.run.calls[0][1] == "inputs-ref"
    assert (tmp_path / plan.case.name / "inputs.json").exists()


@pytest.mark.ray
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


@pytest.mark.ray
def test_ray_scheduler_writes_resource_discovery_snapshot(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cruciblex.runtime.scheduler.ray import RayScheduler

    class FakeActorMethod:
        def remote(self, raw_plan, inputs_ref):
            return "result-ref"

    class FakeActor:
        def __init__(self):
            self.run = FakeActorMethod()

    class FakeRay:
        def is_initialized(self):
            return True

        def nodes(self):
            return [
                {
                    "NodeID": "node-1",
                    "NodeManagerAddress": "127.0.0.1",
                    "NodeManagerHostname": "ray-local",
                    "Alive": True,
                    "Resources": {"CPU": 2.0, "node:node-1": 1.0},
                }
            ]

        def put(self, value):
            return "inputs-ref"

    case = CaseSpec(id=61, operator=OperatorSpec(name="torch.abs"), invocation=InvocationSpec(api="torch.abs", api_type="function"))
    plan = ExecutionPlanner().build(
        JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.RUN], artifacts={"output_root": tmp_path})
    )[0]
    scheduler = RayScheduler(run_context(output_root=tmp_path, scheduler=SchedulerKind.RAY))
    scheduler._ray = lambda: FakeRay()
    monkeypatch.setattr(scheduler, "_actor_for", lambda ray, plan, placement: FakeActor())
    monkeypatch.setattr(scheduler._inputs, "materialize", lambda plan: SimpleNamespace(inputs=[], artifacts=[]))

    ref = scheduler.submit(plan)

    assert ref == "result-ref"
    snapshot = json.loads((tmp_path / "driver" / "resource_snapshot.json").read_text(encoding="utf-8"))
    nodes = yaml.safe_load((tmp_path / "driver" / "discovered_nodes.yaml").read_text(encoding="utf-8"))["nodes"]
    assert snapshot["source"]["node_count"] == 1
    assert snapshot["node_templates"][0]["devices"][0]["backend"] == "cpu"
    assert nodes[0]["devices"][0]["backend"] == "cpu"


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










def test_aclnn_add_case_reuses_bridge_signature_metadata():
    from cruciblex.generation.loader import load_cases
    from cruciblex.plugins.executors.aclnn_bridge import op_spec_from_case

    case = load_cases("examples/cases/aclnn.add.npu.yaml")[0]
    spec = op_spec_from_case(case)

    assert case.invocation.executor == "aclnn"
    assert case.invocation.api_type == "aclnn_function"
    assert case.oracle.metadata["reference_api"] == "torch.add"
    assert spec.symbol == "aclnnAdd"
    assert spec.workspace_symbol == "aclnnAddGetWorkspaceSize"
    assert [item.name for item in spec.inputs] == ["input", "other"]
    assert [(item.name, item.kind, item.value) for item in spec.attributes] == [("alpha", "scalar", 1.0)]
    assert [item.name for item in spec.outputs] == ["output"]

def test_aclnn_abs_case_declares_generic_bridge_metadata():
    from cruciblex.generation.loader import load_cases
    from cruciblex.plugins.executors.aclnn_bridge import op_spec_from_case

    case = load_cases("examples/cases/aclnn.abs.npu.yaml")[0]
    spec = op_spec_from_case(case)

    assert case.invocation.executor == "aclnn"
    assert case.invocation.api_type == "aclnn_function"
    assert case.oracle.reference_executor == "torch"
    assert case.oracle.metadata["reference_api"] == "torch.abs"
    assert spec.symbol == "aclnnAbs"
    assert spec.workspace_symbol == "aclnnAbsGetWorkspaceSize"
    assert [item.name for item in spec.inputs] == ["input"]
    assert [item.name for item in spec.outputs] == ["output"]

def test_accuracy_task_runs_reference_executor_and_compares(tmp_path):
    from cruciblex.domain.enums import ExecutionRole
    from cruciblex.runtime.executors import EXECUTOR_REGISTRY, BackendExecutor
    from cruciblex.runtime.pipeline import ExecutionPipeline

    class CandidatePlusOneExecutor(BackendExecutor):
        def execute(self, request):
            return np.asarray(request.inputs[0]) + 1

    class ReferencePlusOneExecutor(BackendExecutor):
        def execute(self, request):
            assert request.role == ExecutionRole.REFERENCE
            assert request.case.invocation.api == "reference.abs"
            return np.asarray(request.inputs[0]) + 1

    EXECUTOR_REGISTRY.register("test_candidate_plus_one")(CandidatePlusOneExecutor)
    EXECUTOR_REGISTRY.register("test_reference_plus_one")(ReferencePlusOneExecutor)
    case = CaseSpec(
        id=65,
        operator=OperatorSpec(name="test.reference.compare"),
        invocation=InvocationSpec(api="candidate.abs", api_type="function", executor="test_candidate_plus_one"),
        oracle=OracleSpec(
            comparison="allclose",
            reference_executor="test_reference_plus_one",
            metadata={"execute_reference": True, "reference_api": "reference.abs", "reference_api_type": "function"},
        ),
        parameters=[tensor_param(2)],
    )
    plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.ACCURACY]))[0]
    plan.artifacts.output_root = tmp_path

    result = ExecutionPipeline().run(plan, inputs=[np.array([-2.0, 3.0], dtype=np.float32)])

    assert result.status == ResultStatus.PASSED
    assert result.metrics["reference_executor"] == "test_reference_plus_one"
    assert result.metrics["compare_detail"] == "allclose passed"
    assert result.metrics["max_abs_diff"] == 0.0
    assert any(artifact.name == "reference_output" for artifact in result.artifacts)



def test_aclnn_runtime_calls_generic_workspace_then_execute():
    from cruciblex.plugins.executors.aclnn_bridge import AclnnArg, AclnnOpSpec, AclnnRuntime

    class FakeNpu:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def synchronize():
            return None

        @staticmethod
        def current_stream():
            return type("Stream", (), {"npu_stream": 1234})()

    class FakeTensor:
        dtype = "torch.float32"
        device = "npu:0"

        def __init__(self, data):
            self._data = np.asarray(data, dtype=np.float32)
            self.shape = self._data.shape

        def dim(self):
            return self._data.ndim

        def stride(self):
            return (2, 1)

        def data_ptr(self):
            return 5678

        def contiguous(self):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._data

    class FakeTorch:
        Tensor = FakeTensor
        npu = FakeNpu

        @staticmethod
        def empty_like(tensor):
            return FakeTensor(np.zeros_like(tensor._data))

        @staticmethod
        def as_tensor(value):
            return FakeTensor(value)

    class FakeFunction:
        def __init__(self, name, library):
            self.name = name
            self.library = library
            self.restype = None
            self.argtypes = None

        def __call__(self, *args):
            self.library.calls.append((self.name, len(args)))
            if self.name == "aclCreateTensor":
                self.library.next_descriptor += 1
                return self.library.next_descriptor
            if self.name == "aclnnAbsGetWorkspaceSize":
                return 0
            if self.name == "aclnnAbs":
                return 0
            if self.name == "aclDestroyTensor":
                return 0
            return 0

    class FakeLibrary:
        def __init__(self):
            self.calls = []
            self.next_descriptor = 100
            for name in ["aclCreateTensor", "aclDestroyTensor", "aclCreateScalar", "aclDestroyScalar", "aclnnAbsGetWorkspaceSize", "aclnnAbs"]:
                setattr(self, name, FakeFunction(name, self))

    class FakeResolver:
        def __init__(self, library):
            self.library = library

        def resolve(self, spec):
            return self.library

    library = FakeLibrary()
    runtime = AclnnRuntime(resolver=FakeResolver(library))
    runtime._torch_npu = lambda: FakeTorch
    spec = AclnnOpSpec(
        op_name="Abs",
        inputs=(AclnnArg(name="input"),),
        outputs=(AclnnArg(name="output", role="output", like="input"),),
    )

    output = runtime.run(spec, [np.asarray([[-1, 2]], dtype=np.float32)])

    assert output.shape == (1, 2)
    assert ("aclnnAbsGetWorkspaceSize", 4) in library.calls
    assert ("aclnnAbs", 4) in library.calls
    assert [name for name, _ in library.calls].count("aclCreateTensor") == 2
    assert [name for name, _ in library.calls].count("aclDestroyTensor") == 2

def test_aclnn_bridge_normalizes_symbols_and_case_metadata():
    from cruciblex.plugins.executors.aclnn_bridge import normalize_aclnn_symbol, op_spec_from_case

    assert normalize_aclnn_symbol("Abs") == "aclnnAbs"
    assert normalize_aclnn_symbol("aclnnAbs") == "aclnnAbs"
    case = CaseSpec(
        id=63,
        operator=OperatorSpec(name="aclnn.Abs"),
        invocation=InvocationSpec(
            api="Abs",
            api_type="aclnn",
            executor="aclnn",
            metadata={
                "aclnn": {
                    "op_name": "Abs",
                    "inputs": [{"name": "input"}],
                    "outputs": [{"name": "output", "like": "input"}],
                }
            },
        ),
        parameters=[tensor_param(2)],
    )

    spec = op_spec_from_case(case)

    assert spec.op_name == "Abs"
    assert spec.symbol == "aclnnAbs"
    assert spec.workspace_symbol == "aclnnAbsGetWorkspaceSize"
    assert [item.name for item in spec.inputs] == ["input"]
    assert [item.name for item in spec.outputs] == ["output"]


def test_aclnn_executor_routes_aclnn_function_api_type_to_adapter(monkeypatch):
    from cruciblex.plugins.executors import aclnn as aclnn_executor
    from cruciblex.plugins.executors.aclnn import AclnnFunctionExecutor

    class FakeAdapter:
        request = None

        def execute(self, request):
            FakeAdapter.request = request
            return np.asarray(request.inputs[0]) + 1

    class FakeRegistry:
        def supports(self, api_type):
            return api_type == "aclnn_function"

        def resolve(self, api_type):
            return FakeAdapter()

    monkeypatch.setattr(aclnn_executor, "ACLNN_ADAPTER_REGISTRY", FakeRegistry())
    executor = AclnnFunctionExecutor()
    case = CaseSpec(
        id=64,
        operator=OperatorSpec(name="aclnn.Abs"),
        invocation=InvocationSpec(api="Abs", api_type="aclnn_function", executor="aclnn"),
        parameters=[tensor_param(2)],
    )
    plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.RUN]))[0]

    output = executor.execute(ExecutionRequest(case=case, inputs=[np.array([1, 2], dtype=np.float32)], plan=plan))

    assert output.tolist() == [2, 3]
    assert FakeAdapter.request.case.invocation.api_type == "aclnn_function"

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
    assert DEFAULT_DEVICE_POLICY.torch_device(DeviceContext(host="127.0.0.1", node_name="aclnn", device=DeviceSpec(id=5, backend=BackendKind.ACLNN), output_root=Path("out"))) == "npu:5"
    assert DEFAULT_DEVICE_POLICY.torch_device(DeviceContext(host="127.0.0.1", node_name="dcu", device=DeviceSpec(id=6, backend=BackendKind.DCU), output_root=Path("out"))) == "cuda:6"
    assert DEFAULT_DEVICE_POLICY.torch_device(DeviceContext(host="127.0.0.1", node_name="gpu", device=DeviceSpec(id=6, backend=BackendKind.GPU), output_root=Path("out"), env={"CX_DEVICE_INDEX_MODE": "actor_local"})) == "cuda:0"
    assert DEFAULT_DEVICE_POLICY.torch_device(DeviceContext(host="127.0.0.1", node_name="npu", device=DeviceSpec(id=4, backend=BackendKind.NPU), output_root=Path("out"), env={"CX_DEVICE_INDEX_MODE": "actor_local"})) == "npu:0"



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

def test_accuracy_execution_records_comparison_metadata_without_worker_compare(tmp_path):
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
    assert result.status == ResultStatus.PASSED
    assert result.metrics["candidate_role"] == "candidate"
    assert result.metrics["comparison"] == "allclose"
    assert "reference_role" not in result.metrics
    assert "max_abs_diff" not in result.metrics


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
    assert "compare_detail" not in result.metrics


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


def test_default_generator_supports_composite_parameter_collections():
    from cruciblex.plugins.generators.default import DefaultInputGenerator
    from cruciblex.runtime.generation import GenerationRequest

    case = CaseSpec(
        id=41,
        operator=OperatorSpec(name="custom.composite"),
        invocation=InvocationSpec(api="custom.composite", api_type="function", executor="function"),
        parameters=[
            ParameterSpec(
                name="inputs",
                kind=ParameterKind.TENSOR_LIST,
                dtypes=["fp32"],
                shape=ShapeSpec(dims=[1]),
                value_range=ValueRange(valid=[[0, 1]]),
                metadata={
                    "list_length": 3,
                    "item_shapes": [[1], [2], [3]],
                    "item_values": [[0, 0], [1, 1], [2, 2]],
                    "item_dtypes": ["fp32", "int32"],
                },
            ),
            ParameterSpec(
                name="axes",
                kind=ParameterKind.SCALAR_TUPLE,
                dtypes=["int64"],
                value_range=ValueRange(valid=[[0, 4]]),
                metadata={"tuple_length": 2, "item_values": [0, 1]},
            ),
            ParameterSpec(
                name="flags",
                kind=ParameterKind.ATTRIBUTE_LIST,
                dtypes=["bool"],
                metadata={"items": [{"value_range": {"valid": [True]}}, {"value_range": {"valid": [False]}}]},
            ),
        ],
    )
    plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.RUN]))[0]

    inputs = DefaultInputGenerator().generate(GenerationRequest(case=case, plan=plan))

    assert [item.shape for item in inputs[0]] == [(1,), (2,), (3,)]
    assert [str(item.dtype) for item in inputs[0]] == ["float32", "int32", "float32"]
    assert [item.tolist() for item in inputs[0]] == [[0.0], [1, 1], [2.0, 2.0, 2.0]]
    assert inputs[1] == (0, 1)
    assert inputs[2] == [True, False]


def test_default_generator_serializes_composite_inputs(tmp_path):
    from cruciblex.runtime.executors.base import EXECUTOR_REGISTRY, BackendExecutor

    class CompositeEchoExecutor(BackendExecutor):
        def execute(self, request: ExecutionRequest) -> object:
            return request.inputs[0]

    EXECUTOR_REGISTRY.register("composite_echo_test")(CompositeEchoExecutor)
    case = CaseSpec(
        id=42,
        operator=OperatorSpec(name="custom.echo"),
        invocation=InvocationSpec(api="custom.echo", api_type="function", executor="composite_echo_test"),
        parameters=[
            ParameterSpec(
                name="inputs",
                kind=ParameterKind.TENSOR_LIST,
                dtypes=["fp32"],
                shape=ShapeSpec(dims=[2]),
                value_range=ValueRange(valid=[[1, 2]]),
                metadata={"list_length": 2},
            )
        ],
    )
    result = LocalScheduler(run_context(tmp_path)).submit(
        ExecutionPlanner().build(
            JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.RUN], artifacts={"output_root": tmp_path})
        )[0]
    )

    serialized = json.loads(result.artifacts[0].path.read_text(encoding="utf-8"))
    assert result.status == ResultStatus.PASSED
    assert serialized[0][0]["shape"] == [2]
    assert serialized[0][1]["data"] == [1.0, 2.0]


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
    assert result.metrics["failure_kind"] == "expected_error_matched"
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
    assert result.metrics["failure_kind"] == "expected_error_mismatch"
    assert "different failure" in result.error


def test_expected_error_fails_when_candidate_does_not_raise():
    case = CaseSpec(id=16, operator=OperatorSpec(name="torch.abs"), invocation=InvocationSpec(api="torch.abs", api_type="function", executor="function"), parameters=[tensor_param()], oracle=OracleSpec(expected_error="invalid shape"))
    result = LocalScheduler(run_context()).submit(ExecutionPlanner().build(JobSpec(cases=[case], nodes=[cpu_node()], tasks=[TaskKind.ACCURACY]))[0])
    assert result.status == ResultStatus.FAILED
    assert result.metrics["error_matched"] is False
    assert result.metrics["failure_kind"] == "expected_error_not_raised"
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
    ]
    assert not (tmp_path / plan.case.name / plan.plan_id).exists()


def test_driver_input_materializer_reuses_case_inputs(tmp_path):
    from cruciblex.runtime.inputs import DriverInputMaterializer

    case = CaseSpec(
        id=30,
        operator=OperatorSpec(name="torch.abs"),
        invocation=InvocationSpec(api="torch.abs", api_type="function"),
        parameters=[tensor_param(4)],
    )
    plans = ExecutionPlanner().build(
        JobSpec(
            cases=[case],
            nodes=[cpu_node((0, 1))],
            tasks=[TaskKind.RUN],
            artifacts={"output_root": tmp_path},
        )
    )
    materializer = DriverInputMaterializer()

    first = materializer.materialize(plans[0])
    second = materializer.materialize(plans[1])

    assert first is second
    assert first.artifacts[0].path == tmp_path / case.name / "inputs.json"
    assert first.artifacts[0].metadata == {
        "role": "input",
        "scope": "case",
        "sources": [{"parameter": "input", "source": "value_range"}],
    }
    assert json.loads(first.artifacts[0].path.read_text(encoding="utf-8"))[0]["shape"] == [4]



@pytest.mark.ray
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
        artifact_payloads=[
            ArtifactPayload(name="worker_output", kind="json", data={"ok": True}),
            ArtifactPayload(name="execution_log", kind="log", data="worker log line\n"),
        ],
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
    assert [artifact.name for artifact in collected[0].artifacts] == ["inputs", "worker_output", "execution_log"]
    assert collected[0].artifacts[0].path == tmp_path / plan.case.name / "inputs.json"
    assert collected[0].artifacts[1].path.exists()
    assert collected[0].artifacts[2].path.name == "execution_log.log"
    assert json.loads(collected[0].artifacts[1].path.read_text(encoding="utf-8")) == {"ok": True}
    assert collected[0].artifacts[2].path.read_text(encoding="utf-8") == "worker log line\n"
    assert (tmp_path / "run.log").read_text(encoding="utf-8") == "worker log line\n"



def test_cross_device_comparator_skips_non_accuracy_tasks(tmp_path):
    from cruciblex.report.cross_compare import CrossDeviceComparator

    ref_path = tmp_path / "torch.add" / "ref" / "candidate_output.json"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(json.dumps([1, 2]), encoding="utf-8")
    cand_path = tmp_path / "torch.add" / "cand" / "candidate_output.json"
    cand_path.parent.mkdir(parents=True, exist_ok=True)
    cand_path.write_text(json.dumps([1, 2]), encoding="utf-8")

    results = [
        ExecutionResult(
            plan_id="ref",
            case_id=1,
            case_name="torch.add",
            node_name="cpu",
            backend=BackendKind.CPU,
            device_id=0,
            task=TaskKind.RUN,
            status=ResultStatus.PASSED,
            artifacts=[ArtifactRef(name="candidate_output", path=ref_path, kind="candidate_output")],
        ),
        ExecutionResult(
            plan_id="cand",
            case_id=1,
            case_name="torch.add",
            node_name="gpu",
            backend=BackendKind.GPU,
            device_id=0,
            task=TaskKind.RUN,
            status=ResultStatus.PASSED,
            artifacts=[ArtifactRef(name="candidate_output", path=cand_path, kind="candidate_output")],
        ),
    ]

    assert CrossDeviceComparator().compare(results) == []



def test_cross_device_comparator_compares_accuracy_tasks(tmp_path):
    from cruciblex.report.cross_compare import CrossDeviceComparator

    ref_path = tmp_path / "torch.add" / "ref" / "candidate_output.json"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(json.dumps([1, 2]), encoding="utf-8")
    cand_path = tmp_path / "torch.add" / "cand" / "candidate_output.json"
    cand_path.parent.mkdir(parents=True, exist_ok=True)
    cand_path.write_text(json.dumps([1, 2]), encoding="utf-8")

    results = [
        ExecutionResult(
            plan_id="ref",
            case_id=1,
            case_name="torch.add",
            node_name="cpu",
            backend=BackendKind.CPU,
            device_id=0,
            task=TaskKind.ACCURACY,
            status=ResultStatus.PASSED,
            metrics={"comparison": "allclose", "atol": 1e-6, "rtol": 1e-6},
            artifacts=[ArtifactRef(name="candidate_output", path=ref_path, kind="candidate_output")],
        ),
        ExecutionResult(
            plan_id="cand",
            case_id=1,
            case_name="torch.add",
            node_name="gpu",
            backend=BackendKind.GPU,
            device_id=0,
            task=TaskKind.ACCURACY,
            status=ResultStatus.PASSED,
            metrics={"comparison": "allclose", "atol": 1e-6, "rtol": 1e-6},
            artifacts=[ArtifactRef(name="candidate_output", path=cand_path, kind="candidate_output")],
        ),
    ]

    comparisons = CrossDeviceComparator(tmp_path).compare(results)
    assert len(comparisons) == 1
    assert comparisons[0].status == ResultStatus.PASSED
    assert comparisons[0].metrics["stage"] == "cross_device_compare"
    assert comparisons[0].artifacts[0].path == tmp_path / "torch.add" / "cross_compare" / "ref__vs__cand.json"
    assert comparisons[0].artifacts[0].path.exists()


def test_result_post_processor_appends_accuracy_comparisons(tmp_path):
    from cruciblex.report import ResultPostProcessor

    ref_path = tmp_path / "torch.add" / "ref" / "candidate_output.json"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(json.dumps([1, 2]), encoding="utf-8")
    cand_path = tmp_path / "torch.add" / "cand" / "candidate_output.json"
    cand_path.parent.mkdir(parents=True, exist_ok=True)
    cand_path.write_text(json.dumps([1, 2]), encoding="utf-8")
    results = [
        ExecutionResult(
            plan_id="ref",
            case_id=1,
            case_name="torch.add",
            node_name="cpu",
            backend=BackendKind.CPU,
            device_id=0,
            task=TaskKind.ACCURACY,
            status=ResultStatus.PASSED,
            artifacts=[ArtifactRef(name="candidate_output", path=ref_path, kind="candidate_output")],
        ),
        ExecutionResult(
            plan_id="cand",
            case_id=1,
            case_name="torch.add",
            node_name="gpu",
            backend=BackendKind.GPU,
            device_id=0,
            task=TaskKind.ACCURACY,
            status=ResultStatus.PASSED,
            artifacts=[ArtifactRef(name="candidate_output", path=cand_path, kind="candidate_output")],
        ),
    ]

    processor = ResultPostProcessor(tmp_path)
    processed = processor.process(results)

    assert processed[:2] == results
    assert len(processed) == 3
    assert processed[2].metrics["stage"] == "cross_device_compare"
    summary = json.loads((tmp_path / "postprocess.json").read_text(encoding="utf-8"))
    assert summary["comparisons"][0]["metrics"]["stage"] == "cross_device_compare"


def test_pipeline_records_performance_and_memory_metrics(tmp_path):
    case = CaseSpec(
        id=90,
        operator=OperatorSpec(name="torch.abs"),
        invocation=InvocationSpec(api="torch.abs", api_type="function", executor="function"),
        parameters=[tensor_param(4)],
    )
    perf_plan = ExecutionPlanner().build(
        JobSpec(
            cases=[case],
            nodes=[
                NodeSpec(
                    name="cpu-perf",
                    devices=[DeviceSpec(id=0, backend=BackendKind.CPU)],
                    allowed_tasks={TaskKind.PERFORMANCE_DEVICE},
                )
            ],
            tasks=[TaskKind.PERFORMANCE_DEVICE],
            artifacts={"output_root": tmp_path / "perf"},
        )
    )[0]
    mem_plan = ExecutionPlanner().build(
        JobSpec(
            cases=[case],
            nodes=[
                NodeSpec(
                    name="cpu-mem",
                    devices=[DeviceSpec(id=0, backend=BackendKind.CPU)],
                    allowed_tasks={TaskKind.MEMORY_DEVICE},
                )
            ],
            tasks=[TaskKind.MEMORY_DEVICE],
            artifacts={"output_root": tmp_path / "mem"},
        )
    )[0]

    perf_result = LocalScheduler(run_context(output_root=tmp_path / "perf")).submit(perf_plan)
    mem_result = LocalScheduler(run_context(output_root=tmp_path / "mem")).submit(mem_plan)

    assert perf_result.status == ResultStatus.PASSED
    assert perf_result.metrics["latency_ms"] >= 0.0
    assert perf_result.metrics["duration_ms"] >= 0.0
    assert mem_result.status == ResultStatus.PASSED
    assert mem_result.metrics["memory_peak_bytes"] >= 0
    assert mem_result.metrics["memory_peak_mb"] >= 0.0

def test_pipeline_records_torch_hardware_memory_metrics(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from cruciblex.runtime.backends.base import DeviceContext
    from cruciblex.runtime.pipeline import ExecutionPipeline

    class FakeCuda:
        def __init__(self):
            self.sync_calls = 0
            self.reset_calls = 0
            self.allocated_values = [100, 300]
            self.reserved_values = [200, 400]

        def is_available(self):
            return True

        def reset_peak_memory_stats(self):
            self.reset_calls += 1

        def synchronize(self):
            self.sync_calls += 1

        def memory_allocated(self):
            return self.allocated_values.pop(0)

        def memory_reserved(self):
            return self.reserved_values.pop(0)

        def max_memory_allocated(self):
            return 512

        def max_memory_reserved(self):
            return 1024

    class FakeExecutor:
        def execute(self, request):
            return np.asarray([1.0], dtype=np.float32)

    fake_cuda = FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    pipeline = ExecutionPipeline()
    monkeypatch.setattr(pipeline, "_resolve_executor", lambda name: FakeExecutor())
    case = CaseSpec(
        id=71,
        operator=OperatorSpec(name="torch.fake"),
        invocation=InvocationSpec(api="torch.fake", api_type="function", executor="torch"),
        parameters=[],
    )
    plan = ExecutionPlanner().build(
        JobSpec(
            cases=[case],
            nodes=[NodeSpec(name="gpu", devices=[DeviceSpec(id=2, backend=BackendKind.GPU)], allowed_tasks={TaskKind.MEMORY_DEVICE})],
            tasks=[TaskKind.MEMORY_DEVICE],
            artifacts={"output_root": tmp_path},
        )
    )[0]
    context = DeviceContext(
        host="127.0.0.1",
        node_name="gpu",
        device=DeviceSpec(id=2, backend=BackendKind.GPU),
        output_root=tmp_path,
    )

    result = pipeline.run(plan, context=context, inputs=[])

    assert result.status == ResultStatus.PASSED
    assert fake_cuda.reset_calls == 1
    assert fake_cuda.sync_calls == 2
    assert result.metrics["hardware_peak_memory_reset"] is True
    assert result.metrics["hardware_sync_before"] is True
    assert result.metrics["hardware_sync_after"] is True
    assert result.metrics["hardware_memory_before_allocated_bytes"] == 100
    assert result.metrics["hardware_memory_after_allocated_bytes"] == 300
    assert result.metrics["hardware_memory_allocated_delta_bytes"] == 200
    assert result.metrics["hardware_memory_peak_bytes"] == 512
    assert result.metrics["hardware_memory_peak_mb"] == 512 / (1024.0 * 1024.0)


def test_pipeline_synchronizes_torch_performance_tasks(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from cruciblex.runtime.backends.base import DeviceContext
    from cruciblex.runtime.pipeline import ExecutionPipeline

    class FakeCuda:
        def __init__(self):
            self.sync_calls = 0

        def is_available(self):
            return True

        def synchronize(self):
            self.sync_calls += 1

    class FakeExecutor:
        def execute(self, request):
            return np.asarray([2.0], dtype=np.float32)

    fake_cuda = FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    pipeline = ExecutionPipeline()
    monkeypatch.setattr(pipeline, "_resolve_executor", lambda name: FakeExecutor())
    case = CaseSpec(
        id=72,
        operator=OperatorSpec(name="torch.fake"),
        invocation=InvocationSpec(api="torch.fake", api_type="function", executor="torch"),
        parameters=[],
    )
    plan = ExecutionPlanner().build(
        JobSpec(
            cases=[case],
            nodes=[NodeSpec(name="gpu", devices=[DeviceSpec(id=0, backend=BackendKind.GPU)], allowed_tasks={TaskKind.PERFORMANCE_DEVICE})],
            tasks=[TaskKind.PERFORMANCE_DEVICE],
            artifacts={"output_root": tmp_path},
        )
    )[0]
    context = DeviceContext(
        host="127.0.0.1",
        node_name="gpu",
        device=DeviceSpec(id=0, backend=BackendKind.GPU),
        output_root=tmp_path,
    )

    result = pipeline.run(plan, context=context, inputs=[])

    assert result.status == ResultStatus.PASSED
    assert fake_cuda.sync_calls == 2
    assert result.metrics["latency_ms"] >= 0.0
    assert result.metrics["hardware_sync_before"] is True
    assert result.metrics["hardware_sync_after"] is True


def test_result_post_processor_writes_task_family_summary(tmp_path):
    from cruciblex.report import ResultPostProcessor

    results = [
        ExecutionResult(
            plan_id="perf",
            case_id=2,
            case_name="torch.matmul",
            node_name="gpu",
            backend=BackendKind.GPU,
            device_id=0,
            task=TaskKind.PERFORMANCE_DEVICE,
            status=ResultStatus.PASSED,
            metrics={"latency_ms": 12.5, "throughput": 88.0},
        ),
        ExecutionResult(
            plan_id="mem",
            case_id=3,
            case_name="torch.matmul",
            node_name="npu",
            backend=BackendKind.NPU,
            device_id=1,
            task=TaskKind.MEMORY_DEVICE,
            status=ResultStatus.PASSED,
            metrics={"memory_peak_bytes": 1024.0, "memory_peak_mb": 1.0},
        ),
    ]

    processor = ResultPostProcessor(tmp_path)
    processed = processor.process(results)
    summary = json.loads((tmp_path / "postprocess.json").read_text(encoding="utf-8"))

    assert processed == results
    assert summary["status_by_task_backend"]["performance_device"]["gpu"]["passed"] == 1
    assert summary["status_by_task_backend"]["memory_device"]["npu"]["passed"] == 1
    assert summary["performance"][0]["metrics"]["latency_ms"] == 12.5
    assert summary["memory"][0]["metrics"]["memory_peak_mb"] == 1.0


def test_result_post_processor_records_invalid_cases(tmp_path):
    from cruciblex.report import ResultPostProcessor

    results = [
        ExecutionResult(
            plan_id="invalid-0",
            case_id=4,
            case_name="torch.abs",
            node_name="cpu",
            backend=BackendKind.CPU,
            device_id=0,
            task=TaskKind.RUN,
            status=ResultStatus.FAILED,
            metrics={
                "expected_invalid": True,
                "invalid_index": 0,
                "source_case_id": 13,
                "compare_detail": "expected invalid case executed successfully",
            },
            error="expected invalid case executed successfully",
        ),
        ExecutionResult(
            plan_id="invalid-1",
            case_id=5,
            case_name="torch.abs",
            node_name="cpu",
            backend=BackendKind.CPU,
            device_id=0,
            task=TaskKind.RUN,
            status=ResultStatus.FAILED,
            metrics={
                "expected_invalid": True,
                "invalid_index": 1,
                "source_case_id": 13,
                "compare_detail": "expected invalid case executed successfully",
            },
            error="expected invalid case executed successfully",
        ),
    ]

    processor = ResultPostProcessor(tmp_path)
    processed = processor.process(results)
    summary = json.loads((tmp_path / "postprocess.json").read_text(encoding="utf-8"))

    assert processed == results
    assert summary["invalid_cases"][0]["metrics"]["expected_invalid"] is True
    assert summary["invalid_cases"][0]["metrics"]["invalid_index"] == 0
    assert summary["status_by_task_backend"]["run"]["cpu"]["failed"] == 2
    assert len(summary["failure_clusters"]) == 1
    assert summary["failure_clusters"][0]["count"] == 2
    assert summary["failure_clusters"][0]["expected_invalid"] is True
    assert summary["failure_clusters"][0]["statuses"] == {"failed": 2}
    assert [case["plan_id"] for case in summary["failure_clusters"][0]["cases"]] == ["invalid-0", "invalid-1"]


def test_repro_bundle_writer_emits_rerun_commands(tmp_path):
    manifest = RunManifest(
        case_path=Path("examples/cases/torch.abs.invalid.yaml"),
        node_path=Path("examples/nodes/local.yaml"),
        tasks=[TaskKind.RUN],
        scheduler=SchedulerKind.LOCAL,
        output_root=tmp_path,
        cruciblex_version=__version__,
        plan_count=3,
    )
    store = ResultStore(tmp_path)
    store.write_manifest(manifest)
    (tmp_path / "postprocess.json").write_text(
        json.dumps(
            {
                "failure_clusters": [
                    {
                        "signature": "run|cpu|failed|True|detail|error",
                        "count": 1,
                        "statuses": {"failed": 1},
                        "task": "run",
                        "backend": "cpu",
                        "expected_invalid": True,
                        "error": "error",
                        "cases": [
                            {
                                "plan_id": "1800000:local-cpu:cpu:0:run",
                                "case_id": 1800000,
                                "case_name": "torch.abs",
                                "task": "run",
                                "backend": "cpu",
                                "status": "failed",
                                "metrics": {
                                    "fuzz_case": True,
                                    "source_case_id": 13,
                                    "generation_index": 0,
                                    "generation_seed": 17,
                                },
                            },
                            {
                                "plan_id": "1800001:local-cpu:cpu:0:run",
                                "case_id": 1800001,
                                "case_name": "torch.abs",
                                "task": "run",
                                "backend": "cpu",
                                "status": "failed",
                                "metrics": {
                                    "fuzz_case": True,
                                    "source_case_id": 13,
                                    "generation_index": 1,
                                    "generation_seed": 17,
                                },
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    bundle_path = ReproBundleWriter(tmp_path).write()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    command = bundle["clusters"][0]["rerun_commands"][0]
    assert bundle["clusters"][0]["cluster_id"] == "cluster-0"
    assert "--plan-id 1800000:local-cpu:cpu:0:run" in command
    assert "--case examples/cases/torch.abs.invalid.yaml" in command
    assert "--scheduler local" in command
    assert bundle["clusters"][0]["reduced_fuzz_cases"][0]["generation_seed"] == 17
    assert bundle["clusters"][0]["reduced_fuzz_cases"][1]["generation_index"] == 1

    minimized = json.loads(ReproBundleWriter(tmp_path).write(name="repro_min.json", minimize=True).read_text(encoding="utf-8"))
    assert minimized["minimized"] is True
    assert minimized["clusters"][0]["count"] == 1
    assert minimized["clusters"][0]["source_count"] == 2
    assert len(minimized["clusters"][0]["cases"]) == 1
    assert len(minimized["clusters"][0]["reduced_fuzz_cases"]) == 1
    assert minimized["clusters"][0]["reduced_fuzz_cases"][0]["generation_seed"] == 17


def test_onboard_cli_generates_loadable_scaffold(tmp_path):
    facts = tmp_path / "facts.yaml"
    facts.write_text(
        """
operator:
  name: torch.abs
  api: torch.abs
  api_type: function
  supported_backends: [cpu, gpu, npu]
parameters:
  - name: input
    kind: tensor
    dtype_families: [fp32]
    shape_rules:
      dims: [4]
    value_range:
      valid: [[-1, 1]]
      invalid: []
oracle:
  comparison: allclose
  tolerance:
    atol: 1.0e-6
    rtol: 1.0e-6
invalid_policy:
  expected_error: null
fuzz:
  count: 4
  seed: 23
  constraints: [random_coverage]
  invalid_count: 1
  max_elements: 4
  max_bytes: 16
  random_dtypes: [fp32, fp16]
  random_shapes: [[4], [2, 2]]
  random_values: [[-1, 1], [2, 3]]
""".strip(),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["onboard", "--facts", str(facts), "--output", str(tmp_path / "scaffold")])

    assert result.exit_code == 0, result.output
    scaffold = tmp_path / "scaffold"
    assert (scaffold / "case.yaml").exists()
    assert (scaffold / "nodes.yaml").exists()
    assert (scaffold / "campaign.yaml").exists()
    assert (scaffold / "hardware_nodes.yaml").exists()
    assert (scaffold / "hardware_campaign.yaml").exists()
    assert (scaffold / "executor_plugin.py").exists()
    assert os.access(scaffold / "validate.sh", os.X_OK)
    assert os.access(scaffold / "hardware_validate.sh", os.X_OK)
    validate_text = (scaffold / "validate.sh").read_text(encoding="utf-8")
    hardware_validate_text = (scaffold / "hardware_validate.sh").read_text(encoding="utf-8")
    assert "cx campaign" in validate_text
    assert "cx report" in validate_text
    assert "hardware_campaign.yaml" in hardware_validate_text
    job = load_job(scaffold / "case.yaml", scaffold / "nodes.yaml", tasks=[TaskKind.RUN], scheduler=SchedulerKind.LOCAL)
    fuzz_job = load_job(scaffold / "fuzz_case.yaml", scaffold / "nodes.yaml", tasks=[TaskKind.FUZZ], scheduler=SchedulerKind.LOCAL)
    assert job.cases[0].operator.name == "torch.abs"
    assert job.cases[0].parameters[0].shape.dims == [4]
    assert job.nodes[0].devices[0].backend == BackendKind.CPU
    assert fuzz_job.cases[0].generation.count == 4
    assert fuzz_job.cases[0].generation.seed == 23
    assert fuzz_job.cases[0].generation.invalid_count == 1
    assert fuzz_job.cases[0].generation.max_elements == 4
    assert fuzz_job.cases[0].generation.max_bytes == 16
    assert fuzz_job.cases[0].parameters[0].metadata["random_coverage"] is True
    assert fuzz_job.cases[0].parameters[0].metadata["random_shapes"] == [[4], [2, 2]]
    assert fuzz_job.cases[0].parameters[0].metadata["random_values"] == [[-1, 1], [2, 3]]
    hardware_nodes = yaml.safe_load((scaffold / "hardware_nodes.yaml").read_text(encoding="utf-8"))
    hardware_campaign = yaml.safe_load((scaffold / "hardware_campaign.yaml").read_text(encoding="utf-8"))
    assert [node["devices"][0]["backend"] for node in hardware_nodes["nodes"]] == ["gpu", "npu"]
    assert {run["scheduler"] for run in hardware_campaign["runs"]} == {"ray"}


def test_coverage_cli_builds_backend_campaign(tmp_path, monkeypatch):
    from cruciblex import cli as cli_module

    discovery = {
        "kind": "ray-resource-discovery",
        "source": {
            "available": True,
            "initialized": True,
            "ray_address": None,
            "init_error": None,
            "node_count": 4,
            "alive_node_count": 4,
        },
        "nodes": [],
        "backend_counts": {"aclnn": 1, "cpu": 1, "gpu": 1, "npu": 1},
        "capabilities": [],
        "node_templates": [
            {
                "name": "ray-cpu",
                "host": "127.0.0.1",
                "role": "candidate",
                "allowed_tasks": ["accuracy", "run", "fuzz"],
                "devices": [{"id": 0, "backend": "cpu"}],
                "labels": ["backend:cpu", "host:127.0.0.1", "ray:node-1"],
            },
            {
                "name": "ray-gpu",
                "host": "127.0.0.2",
                "role": "candidate",
                "allowed_tasks": [
                    "accuracy",
                    "run",
                    "performance_device",
                    "performance_device_pta",
                    "performance_e2e",
                    "memory_device",
                ],
                "devices": [{"id": 0, "backend": "gpu"}],
                "labels": ["backend:gpu", "host:127.0.0.2", "ray:node-2"],
            },
            {
                "name": "ray-npu",
                "host": "127.0.0.3",
                "role": "candidate",
                "allowed_tasks": ["accuracy", "run", "performance_device", "memory_device"],
                "devices": [{"id": 0, "backend": "npu", "resources": {"npu": 1.0}}],
                "labels": ["backend:npu", "host:127.0.0.3", "ray:node-3"],
            },
            {
                "name": "ray-aclnn",
                "host": "127.0.0.4",
                "role": "candidate",
                "allowed_tasks": ["accuracy", "run", "performance_device", "memory_device"],
                "devices": [{"id": 0, "backend": "aclnn", "resources": {"npu": 1.0}}],
                "labels": ["backend:aclnn", "host:127.0.0.4", "ray:node-4"],
            },
        ],
    }
    calls: list[dict[str, str]] = []
    monkeypatch.setattr(cli_module, "discover_runtime_resources", lambda ray_address=None: discovery)
    monkeypatch.setattr(
        cli_module,
        "campaign",
        lambda **kwargs: calls.append({"campaign_file": str(kwargs["campaign_file"]), "output": str(kwargs["output"])}) or None,
    )

    result = CliRunner().invoke(
        cli_module.app,
        ["coverage", "--case", "examples/cases/torch.abs.yaml", "--output", str(tmp_path / "coverage"), "--run"],
    )

    assert result.exit_code == 0, result.output
    coverage_root = tmp_path / "coverage"
    batch_root = coverage_root / "driver" / "coverage" / "torch.abs"
    campaign_path = batch_root / "coverage_campaign.yaml"
    assert campaign_path.exists()
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    assert [run["scheduler"] for run in campaign["runs"]] == ["ray", "ray", "ray", "ray"]
    assert [run["task"] for run in campaign["runs"]] == [
        ["accuracy", "run"],
        ["accuracy", "run", "performance_device", "performance_device_pta", "performance_e2e", "memory_device"],
        ["accuracy", "run", "performance_device", "memory_device"],
        ["accuracy", "run", "performance_device", "memory_device"],
    ]
    driver_files = sorted(path.name for path in (coverage_root / "driver").glob("*_nodes.yaml"))
    assert driver_files == ["discovered_nodes.yaml"]
    backend_files = sorted(path.name for path in batch_root.glob("*_nodes.yaml"))
    assert backend_files == ["aclnn_nodes.yaml", "cpu_nodes.yaml", "gpu_nodes.yaml", "npu_nodes.yaml"]
    assert calls == [{"campaign_file": str(campaign_path), "output": str(batch_root / "campaign-output")}]



def test_driver_clean_prunes_old_coverage_batches(tmp_path):
    from cruciblex import cli as cli_module

    coverage_root = tmp_path / "driver" / "coverage"
    old_batch = coverage_root / "old-case"
    new_batch = coverage_root / "new-case"
    old_batch.mkdir(parents=True)
    new_batch.mkdir(parents=True)
    (old_batch / "coverage_campaign.yaml").write_text("runs: []\n", encoding="utf-8")
    (new_batch / "coverage_campaign.yaml").write_text("runs: []\n", encoding="utf-8")
    old_time = time.time() - 1000
    new_time = time.time()
    os.utime(old_batch, (old_time, old_time))
    os.utime(new_batch, (new_time, new_time))

    result = CliRunner().invoke(cli_module.app, ["driver-clean", "--output", str(tmp_path), "--keep", "1"])

    assert result.exit_code == 0, result.output
    assert not old_batch.exists()
    assert new_batch.exists()


def test_campaign_cli_runs_batch_and_writes_summary(tmp_path):
    campaign_file = tmp_path / "campaign.yaml"
    campaign_file.write_text(
        """
runs:
  - name: fuzz-smoke
    case: examples/cases/torch.abs.fuzz.yaml
    nodes: examples/nodes/local.yaml
    task: fuzz
    scheduler: local
    output: {}
""".format(tmp_path / "batch-run").strip(),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["campaign", "--campaign", str(campaign_file), "--output", str(tmp_path / "campaign-output")])

    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "campaign-output" / "campaign_summary.json").read_text(encoding="utf-8"))
    assert [row["name"] for row in summary["runs"]] == ["fuzz-smoke"]
    assert summary["runs"][0]["total"] == 3
    assert summary["runs"][0]["passed"] == 3
    assert summary["runs"][0]["fuzz_cases"] == 3
    assert summary["totals"] == {
        "runs": 1,
        "total": 3,
        "passed": 3,
        "failed": 0,
        "failure_clusters": 0,
        "fuzz_cases": 3,
    }
    assert summary["status_by_task_backend"]["fuzz"]["cpu"]["passed"] == 3
    assert (tmp_path / "batch-run" / "postprocess.json").exists()

    report = CliRunner().invoke(app, ["report", "--output", str(tmp_path / "campaign-output")])

    assert report.exit_code == 0, report.output
    report_text = (tmp_path / "campaign-output" / "campaign_report.md").read_text(encoding="utf-8")
    assert "# CrucibleX Campaign Report" in report_text
    assert "- fuzz_cases: 3" in report_text
    assert "  cpu: passed=3" in report_text
    assert (tmp_path / "campaign-output" / "results.csv").exists()


def test_repro_cli_writes_bundle(tmp_path):
    manifest = RunManifest(
        case_path=Path("examples/cases/torch.abs.invalid.yaml"),
        node_path=Path("examples/nodes/local.yaml"),
        tasks=[TaskKind.RUN],
        scheduler=SchedulerKind.LOCAL,
        output_root=tmp_path,
        cruciblex_version=__version__,
        plan_count=3,
    )
    ResultStore(tmp_path).write_manifest(manifest)
    (tmp_path / "postprocess.json").write_text(
        json.dumps(
            {
                "failure_clusters": [
                    {
                        "signature": "sig",
                        "count": 1,
                        "statuses": {"failed": 1},
                        "task": "run",
                        "backend": "cpu",
                        "expected_invalid": True,
                        "error": "boom",
                        "compare_detail": "detail",
                        "cases": [
                            {
                                "plan_id": "1800000:local-cpu:cpu:0:run",
                                "case_id": 1800000,
                                "case_name": "torch.abs",
                                "task": "run",
                                "backend": "cpu",
                                "status": "failed",
                                "metrics": {
                                    "fuzz_case": True,
                                    "source_case_id": 13,
                                    "generation_index": 0,
                                    "generation_seed": 17,
                                },
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["repro", "--output", str(tmp_path), "--script"])

    assert result.exit_code == 0
    assert (tmp_path / "repro_bundle.json").exists()
    assert (tmp_path / "cluster-0.rerun.sh").exists()
    bundle = json.loads((tmp_path / "repro_bundle.json").read_text(encoding="utf-8"))
    assert "--plan-id 1800000:local-cpu:cpu:0:run" in bundle["clusters"][0]["rerun_commands"][0]
    assert bundle["clusters"][0]["reduced_fuzz_cases"][0]["generation_seed"] == 17
    assert "#!/usr/bin/env bash" in (tmp_path / "cluster-0.rerun.sh").read_text(encoding="utf-8")


def test_repro_cli_replays_semantic_reduction_candidates(tmp_path):
    manifest = RunManifest(
        case_path=Path("cases.yaml"),
        node_path=Path("nodes.yaml"),
        tasks=[TaskKind.RUN],
        scheduler=SchedulerKind.LOCAL,
        output_root=tmp_path,
        cruciblex_version=__version__,
        plan_count=1,
    )
    ResultStore(tmp_path).write_manifest(manifest)
    (tmp_path / "generated_cases.json").write_text(
        json.dumps({
            "cases": [{
                "id": 1,
                "parameters": [{
                    "shape_rules": {"dims": [2, 3]},
                    "dtypes": ["fp16"],
                    "value_range": {"valid": [[-1, 1]]},
                }],
            }],
        }),
        encoding="utf-8",
    )
    (tmp_path / "postprocess.json").write_text(
        json.dumps({
            "failure_clusters": [{
                "signature": "replay",
                "cases": [{"plan_id": "1:local-cpu:cpu:0:run", "case_id": 1}],
            }],
        }),
        encoding="utf-8",
    )
    command = 'python -c "import pathlib,sys; assert pathlib.Path(sys.argv[1]).exists(); sys.exit(1)" {case}'

    result = CliRunner().invoke(app, [
        "repro", "--output", str(tmp_path), "--minimize", "--replay-command", command,
    ])

    assert result.exit_code == 0
    reduction = tmp_path / "repro" / "replay" / "semantic_reduction.yaml"
    assert reduction.exists()
    assert "shape_rules" in reduction.read_text(encoding="utf-8")


def test_repro_cli_filters_single_cluster(tmp_path):
    manifest = RunManifest(
        case_path=Path("examples/cases/torch.abs.invalid.yaml"),
        node_path=Path("examples/nodes/local.yaml"),
        tasks=[TaskKind.RUN],
        scheduler=SchedulerKind.LOCAL,
        output_root=tmp_path,
        cruciblex_version=__version__,
        plan_count=3,
    )
    ResultStore(tmp_path).write_manifest(manifest)
    (tmp_path / "postprocess.json").write_text(
        json.dumps(
            {
                "failure_clusters": [
                    {
                        "signature": "sig-a",
                        "count": 1,
                        "statuses": {"failed": 1},
                        "task": "run",
                        "backend": "cpu",
                        "expected_invalid": True,
                        "error": "boom-a",
                        "compare_detail": "detail-a",
                        "cases": [{"plan_id": "1800000:local-cpu:cpu:0:run", "case_id": 1800000}],
                    },
                    {
                        "signature": "sig-b",
                        "count": 1,
                        "statuses": {"failed": 1},
                        "task": "run",
                        "backend": "cpu",
                        "expected_invalid": True,
                        "error": "boom-b",
                        "compare_detail": "detail-b",
                        "cases": [{"plan_id": "1800001:local-cpu:cpu:0:run", "case_id": 1800001}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["repro", "--output", str(tmp_path), "--cluster-id", "cluster-1"] )

    assert result.exit_code == 0
    bundle = json.loads((tmp_path / "repro_bundle.json").read_text(encoding="utf-8"))
    assert [cluster["cluster_id"] for cluster in bundle["clusters"]] == ["cluster-1"]


def test_run_filters_by_plan_id(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--case",
            "examples/cases/torch.abs.invalid.yaml",
            "--nodes",
            "examples/nodes/local.yaml",
            "--task",
            "run",
            "--scheduler",
            "local",
            "--plan-id",
            "13:local-cpu:cpu:0:run",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["plan_id"] for row in rows] == ["13:local-cpu:cpu:0:run"]


def test_fuzz_task_marks_fuzz_case_in_results(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--case",
            "examples/cases/torch.abs.fuzz.yaml",
            "--nodes",
            "examples/nodes/local.yaml",
            "--task",
            "fuzz",
            "--scheduler",
            "local",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((tmp_path / "postprocess.json").read_text(encoding="utf-8"))
    assert all(row["metrics"]["fuzz_case"] is True for row in rows)
    assert len(summary["fuzz_cases"]) == 3
    assert summary["status_by_task_backend"]["fuzz"]["cpu"]["passed"] == 3

    report_result = CliRunner().invoke(app, ["report", "--output", str(tmp_path)])
    report_text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert report_result.exit_code == 0
    assert "## Fuzz Cases" in report_text
    assert "generation_seed: 17" in report_text
    assert "source_case_id: 21" in report_text


def test_markdown_report_includes_failure_clusters(tmp_path):
    from cruciblex.report import MarkdownReportWriter

    result = ExecutionResult(
        plan_id="invalid-0",
        case_id=4,
        case_name="torch.abs",
        node_name="cpu",
        backend=BackendKind.CPU,
        device_id=0,
        task=TaskKind.RUN,
        status=ResultStatus.FAILED,
        metrics={"compare_detail": "expected invalid case executed successfully"},
        error="expected invalid case executed successfully",
    )
    manifest = RunManifest(
        case_path=Path("cases.yaml"),
        node_path=Path("nodes.yaml"),
        tasks=[TaskKind.RUN],
        scheduler=SchedulerKind.LOCAL,
        output_root=tmp_path,
        cruciblex_version=__version__,
        plan_count=1,
    )
    store = ResultStore(tmp_path)
    store.write_manifest(manifest)
    store.write_results_jsonl([result])
    store.write_summary_json({"total": 1, "passed": 0, "failed": 1})
    (tmp_path / "postprocess.json").write_text(
        json.dumps(
            {
                "failure_clusters": [
                    {
                        "signature": "run|cpu|failed|True|expected invalid case executed successfully|expected invalid case executed successfully",
                        "count": 1,
                        "task": "run",
                        "backend": "cpu",
                        "error": "expected invalid case executed successfully",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report_path = MarkdownReportWriter(tmp_path).write()
    content = report_path.read_text(encoding="utf-8")

    assert "## Failure Clusters" in content
    assert "## Hardware Evidence" in content
    assert "count: 1" in content
    assert "expected invalid case executed successfully" in content


def test_result_store_writes_jsonl_summary_and_manifest(tmp_path):
    result = LocalScheduler(run_context()).submit(ExecutionPlanner().build(load_job("examples/cases/torch.abs.yaml", "examples/nodes/local.yaml", scheduler=SchedulerKind.LOCAL, output_path=tmp_path))[0])
    store = ResultStore(tmp_path)
    manifest = RunManifest(case_path=Path("cases.yaml"), node_path=Path("nodes.yaml"), tasks=[TaskKind.ACCURACY], scheduler=SchedulerKind.RAY, output_root=tmp_path, plugin_paths=[Path("plugin.py")], cruciblex_version=__version__, plan_count=3)
    manifest_path = store.write_manifest(manifest)
    results_path = store.write_results_jsonl([result])
    results_csv_path = store.write_results_csv([result])
    summary_path = store.write_summary_json({"total": 1, "passed": 1, "failed": 0})
    loaded = store.read_manifest()
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    csv_rows = list(csv.DictReader(results_csv_path.read_text(encoding="utf-8").splitlines()))
    assert manifest_path.name == "manifest.json"
    assert loaded.run_id == manifest.run_id
    assert loaded.scheduler == SchedulerKind.RAY
    assert loaded.manifest_schema_version == 1
    assert rows[0]["result_schema_version"] == 1
    assert rows[0]["status"] == "passed"
    assert csv_rows[0]["result_schema_version"] == "1"
    assert csv_rows[0]["status"] == "passed"
    assert csv_rows[0]["plan_id"] == result.plan_id
    assert json.loads(summary_path.read_text(encoding="utf-8")) == {"total": 1, "passed": 1, "failed": 0}


def test_result_store_accepts_legacy_rows_and_rejects_future_schema(tmp_path):
    store = ResultStore(tmp_path)
    legacy = {
        "plan_id": "legacy:cpu:0:run",
        "case_id": 1,
        "case_name": "legacy",
        "node_name": "cpu",
        "backend": "cpu",
        "device_id": 0,
        "task": "run",
        "status": "passed",
    }
    (tmp_path / "results.jsonl").write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    assert store.read_results_jsonl()[0].result_schema_version == 1

    future = dict(legacy, result_schema_version=2)
    (tmp_path / "results.jsonl").write_text(json.dumps(future) + "\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        store.read_results_jsonl()


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
    assert manifest.manifest_schema_version == 1
    assert manifest.run_id == context.run_id
    assert manifest.output_root == tmp_path
    assert manifest.ray_address is None
    assert manifest.plugin_paths == [Path("plugin.py")]
    assert manifest.metadata == {"source": "test"}
    assert manifest.plan_count == 2
    assert manifest.submitted_count == 1
    assert manifest.skipped_count == 1

def test_run_manifest_records_result_outputs(tmp_path):
    manifest = RunManifest(case_path=Path("cases.yaml"), node_path=Path("nodes.yaml"), tasks=[TaskKind.RUN], scheduler=SchedulerKind.RAY, output_root=tmp_path, ray_address="203.0.113.10:6379", cruciblex_version=__version__, plan_count=1)
    updated = manifest.with_outputs(results_path=tmp_path / "results.jsonl", summary_path=tmp_path / "summary.json")
    assert updated.results_path == tmp_path / "results.jsonl"
    assert updated.summary_path == tmp_path / "summary.json"
    assert updated.run_id == manifest.run_id


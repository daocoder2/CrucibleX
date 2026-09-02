from pathlib import Path

import pytest
from pydantic import ValidationError

from cruciblex import __version__
from cruciblex.cli import _manifest_run_metadata
from cruciblex.domain.enums import SchedulerKind, TaskKind
from cruciblex.domain.manifest import MANIFEST_V1_LANE_KINDS, MANIFEST_V1_TOP_LEVEL_FIELDS
from cruciblex.domain.run import RunManifest
from cruciblex.generation.loader import load_job_from_manifest, load_manifest
from cruciblex.report.repro import ReproBundleWriter
from cruciblex.runtime.planner import ExecutionPlanner


def _write_case(path: Path, operator: str = "torch.abs") -> None:
    path.write_text(
        f"""cases:
  - id: 1
    operator: {{name: {operator}}}
    invocation: {{api: {operator}, api_type: function, executor: torch}}
    parameters:
      - {{name: input, kind: tensor, dtypes: [fp32], shape: {{dims: [2]}}}}
""",
        encoding="utf-8",
    )


def _write_gpu_nodes(path: Path) -> None:
    path.write_text(
        """nodes:
  - name: gpu
    devices:
      - id: 0
        backend: gpu
""",
        encoding="utf-8",
    )


def test_manifest_backend_filter_uses_lane_backends_before_planning(tmp_path):
    case_path = tmp_path / "case.yaml"
    node_path = tmp_path / "nodes.yaml"
    manifest_path = tmp_path / "manifest.yaml"
    _write_case(case_path)
    _write_gpu_nodes(node_path)
    manifest_path.write_text(
        f"""version: 1
kind: manifest
filters:
  include_backends: [gpu]
lanes:
  - name: gpu-lane
    backends: [gpu]
    cases:
      - include: {case_path}
""",
        encoding="utf-8",
    )

    job = load_job_from_manifest(manifest_path, node_path, tasks=[TaskKind.ACCURACY], scheduler=SchedulerKind.LOCAL, output_path=tmp_path / "out")
    plans = ExecutionPlanner().build(job)

    assert len(job.cases) == 1
    assert [plan.device.backend.value for plan in plans] == ["gpu"]



def test_checked_in_manifest_includes_are_independent_of_current_directory(tmp_path, monkeypatch):
    repo_root = Path(__file__).parents[1]
    monkeypatch.chdir(tmp_path)

    job = load_job_from_manifest(
        repo_root / "examples/manifests/local-smoke-manifest.yaml",
        repo_root / "examples/nodes/local.yaml",
        tasks=[TaskKind.ACCURACY],
        scheduler=SchedulerKind.LOCAL,
        output_path=tmp_path / "out",
    )

    assert len(job.cases) == 1
    assert job.cases[0].metadata["manifest_case_include"] == str((repo_root / "examples/cases/torch.abs.yaml").resolve())


def test_manifest_provenance_hashes_every_declared_include_before_filters(tmp_path):
    first_case = tmp_path / "first.yaml"
    second_case = tmp_path / "second.yaml"
    node_path = tmp_path / "nodes.yaml"
    manifest_path = tmp_path / "manifest.yaml"
    _write_case(first_case, "torch.abs")
    _write_case(second_case, "torch.neg")
    _write_gpu_nodes(node_path)
    manifest_path.write_text(
        f"""version: 1
kind: manifest
filters:
  include_operators: [torch.abs]
lanes:
  - name: contract
    backends: [gpu]
    cases:
      - include: {first_case}
      - include: {second_case}
""",
        encoding="utf-8",
    )

    job = load_job_from_manifest(manifest_path, node_path, tasks=[TaskKind.ACCURACY], scheduler=SchedulerKind.LOCAL, output_path=tmp_path / "out")
    metadata = _manifest_run_metadata(manifest_path, job, ExecutionPlanner().build(job))

    assert {item["path"] for item in metadata["manifest_includes"]} == {str(first_case), str(second_case)}



def test_manifest_v1_canonical_contract_freezes_top_level_fields_and_lane_kinds():
    repo_root = Path(__file__).parents[1]
    manifest = load_manifest(repo_root / "examples/manifests/operator-boundary-campaign.yaml")

    assert tuple(manifest.model_dump()) == MANIFEST_V1_TOP_LEVEL_FIELDS
    assert {lane.kind for lane in manifest.lanes} == {"contract", "hardware"}
    assert set(MANIFEST_V1_LANE_KINDS) == {"contract", "hardware", "preflight_blocked"}


def test_operator_boundary_campaign_preserves_contract_and_evidence_lanes(tmp_path):
    repo_root = Path(__file__).parents[1]
    job = load_job_from_manifest(
        repo_root / "examples/manifests/operator-boundary-campaign.yaml",
        repo_root / "examples/nodes/local.yaml",
        tasks=[TaskKind.ACCURACY],
        scheduler=SchedulerKind.LOCAL,
        output_path=tmp_path / "out",
    )
    plans = ExecutionPlanner().build(job)

    assert len(job.cases) == 20
    assert sum(bool(case.metadata.get("expected_invalid")) for case in job.cases) == 5
    assert {case.metadata["manifest_lane"] for case in job.cases} == {"cpu-contract", "gpu-legal-evidence", "npu-legal-evidence"}
    assert len(plans) == 11
    assert {plan.case.metadata["manifest_lane"] for plan in plans} == {"cpu-contract"}
    assert {plan.device.backend.value for plan in plans} == {"cpu"}


def test_complex_norm_manifest_selects_only_legal_generated_cases(tmp_path):
    repo_root = Path(__file__).parents[1]
    job = load_job_from_manifest(
        repo_root / "examples/manifests/complex-norm-evidence.yaml",
        repo_root / "examples/nodes/local.yaml",
        tasks=[TaskKind.ACCURACY],
        scheduler=SchedulerKind.LOCAL,
        output_path=tmp_path / "out",
    )
    plans = ExecutionPlanner().build(job)

    assert len(job.cases) == 12
    assert all(not case.metadata.get("expected_invalid") for case in job.cases)
    assert {case.metadata["manifest_lane"] for case in job.cases} == {"cpu-legal-contract", "gpu-legal-evidence", "npu-legal-evidence"}
    assert {case.metadata["manifest_case_index"] for case in job.cases} == {0, 1, 2, 3}
    assert len(plans) == 4
    assert {plan.case.name for plan in plans} == {"torch.group_norm", "torch.instance_norm", "torch.layer_norm"}


def test_manifest_schema_rejects_empty_case_include(tmp_path):
    manifest_path = tmp_path / "empty-include.yaml"
    manifest_path.write_text(
        """version: 1
kind: manifest
lanes:
  - name: broken
    cases:
      - include: ""
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="must not be empty"):
        load_manifest(manifest_path)


def test_manifest_schema_rejects_unknown_lane_kind(tmp_path):
    case_path = tmp_path / "case.yaml"
    _write_case(case_path)
    manifest_path = tmp_path / "unknown-lane-kind.yaml"
    manifest_path.write_text(
        f"""version: 1
kind: manifest
lanes:
  - name: exploratory
    kind: exploratory
    cases:
      - include: {case_path}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Input should be 'contract', 'hardware' or 'preflight_blocked'"):
        load_manifest(manifest_path)


def test_complex_operator_evidence_manifest_selects_legal_gpu_and_npu_plans(tmp_path):
    repo_root = Path(__file__).parents[1]
    manifest = repo_root / "examples/manifests/complex-operator-evidence.yaml"

    gpu_job = load_job_from_manifest(manifest, repo_root / "examples/nodes/local-gpu.yaml", tasks=[TaskKind.ACCURACY], scheduler=SchedulerKind.LOCAL, output_path=tmp_path / "gpu")
    npu_job = load_job_from_manifest(manifest, repo_root / "examples/nodes/local-npu.yaml", tasks=[TaskKind.ACCURACY], scheduler=SchedulerKind.LOCAL, output_path=tmp_path / "npu")

    assert len(gpu_job.cases) == len(npu_job.cases) == 14
    assert all(not case.metadata.get("expected_invalid") for case in (*gpu_job.cases, *npu_job.cases))
    gpu_plans = ExecutionPlanner().build(gpu_job)
    npu_plans = ExecutionPlanner().build(npu_job)
    assert {plan.case.name for plan in gpu_plans} == {"torch.conv2d", "torch.layer_norm", "torch.scaled_dot_product_attention"}
    assert {plan.case.name for plan in npu_plans} == {"torch.conv2d", "torch.layer_norm", "torch.scaled_dot_product_attention"}
    assert {plan.case.id for plan in gpu_plans} == {301, 302, 303, 918, 919, 920, 923}
    assert {plan.case.id for plan in npu_plans} == {301, 302, 303, 918, 919, 920, 923}


def test_aclnn_supported_evidence_manifest_has_five_unique_npu_hardware_plans(tmp_path):
    repo_root = Path(__file__).parents[1]
    job = load_job_from_manifest(
        repo_root / "examples/manifests/aclnn-supported-evidence.yaml",
        repo_root / "examples/nodes/local-npu.yaml",
        tasks=[TaskKind.ACCURACY],
        scheduler=SchedulerKind.LOCAL,
        output_path=tmp_path / "out",
    )
    plans = ExecutionPlanner().build(job)

    assert len(job.cases) == 5
    assert len(plans) == 5
    assert {plan.device.backend.value for plan in plans} == {"npu"}
    assert len({plan.plan_id for plan in plans}) == 5
    assert {case.metadata["manifest_lane"] for case in job.cases} == {"npu-aclnn-supported"}


def test_manifest_repro_command_uses_manifest_source(tmp_path):
    manifest = RunManifest(
        case_path=Path("examples/manifests/operator-suite.yaml"),
        node_path=Path("examples/nodes/local.yaml"),
        tasks=[TaskKind.ACCURACY],
        scheduler=SchedulerKind.LOCAL,
        output_root=tmp_path,
        cruciblex_version=__version__,
        metadata={"manifest": "examples/manifests/operator-suite.yaml"},
    )

    command = ReproBundleWriter(tmp_path)._rerun_command(manifest, "1:gpu:0:accuracy", 0)

    assert "--manifest examples/manifests/operator-suite.yaml" in command
    assert "--case" not in command
    assert "--plan-id 1:gpu:0:accuracy" in command

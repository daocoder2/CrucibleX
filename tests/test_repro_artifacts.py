import json
from pathlib import Path

from cruciblex.domain import (
    CaseSpec,
    InvocationSpec,
    OperatorSpec,
    OracleSpec,
    ParameterKind,
    ParameterSpec,
    RunManifest,
    SchedulerKind,
    ShapeSpec,
    TaskKind,
    ValueRange,
)
from cruciblex.generation.loader import load_case
from cruciblex.report.repro import ReproBundleWriter
from cruciblex.storage.results import ResultStore


def test_repro_writer_materializes_standalone_case_and_inputs(tmp_path):
    source_case = CaseSpec(
        id=13,
        operator=OperatorSpec(name="torch.abs"),
        invocation=InvocationSpec(api="torch.abs", api_type="function", executor="function"),
        oracle=OracleSpec(),
        parameters=[ParameterSpec(name="input", kind=ParameterKind.TENSOR, dtypes=["fp32"], shape=ShapeSpec(dims=[2]), value_range=ValueRange(valid=[[-1, 1]]))],
    )
    generated = source_case.model_copy(update={"id": 1300000, "metadata": {"source_case_id": 13, "generation_index": 0, "generation_seed": 17}})
    (tmp_path / "generated_cases.json").write_text(json.dumps({"cases": [generated.model_dump(mode="json")]}, default=str), encoding="utf-8")
    inputs = tmp_path / "source-inputs.json"
    inputs.write_text(json.dumps({"inputs": [[-1.0, 1.0]]}), encoding="utf-8")
    plan_id = "1300000:local-cpu:cpu:0:run"
    result = {
        "plan_id": plan_id, "case_id": 1300000, "case_name": "torch.abs", "node_name": "local-cpu",
        "backend": "cpu", "device_id": 0, "task": "run", "status": "failed",
        "metrics": {"fuzz_case": True, "source_case_id": 13, "generation_index": 0, "generation_seed": 17},
        "artifacts": [{"name": "inputs", "path": str(inputs), "kind": "inputs", "metadata": {}}], "error": "controlled failure",
    }
    ResultStore(tmp_path).write_results_jsonl([])
    (tmp_path / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")
    manifest = RunManifest(run_id="run-test", case_path="examples/cases/test.yaml", node_path="examples/nodes/local.yaml", tasks=[TaskKind.RUN], scheduler=SchedulerKind.LOCAL, output_root=tmp_path, cruciblex_version="test")
    ResultStore(tmp_path).write_manifest(manifest)
    (tmp_path / "postprocess.json").write_text(json.dumps({"failure_clusters": [{"signature": "run|cpu|failed|error|False||controlled failure", "count": 1, "statuses": {"failed": 1}, "task": "run", "backend": "cpu", "error": "controlled failure", "cases": [{"plan_id": plan_id, "case_id": 1300000, "case_name": "torch.abs", "metrics": result["metrics"]}]}]}), encoding="utf-8")

    bundle = json.loads(ReproBundleWriter(tmp_path).write(minimize=True).read_text(encoding="utf-8"))
    artifacts = bundle["clusters"][0]["artifacts"]
    root = Path(artifacts["root"])
    case_path = root / "minimized_case.yaml"
    input_path = root / "inputs.json"
    assert artifacts["minimized_case"] == str(case_path)
    assert artifacts["inputs"] == str(input_path)
    assert load_case(case_path).id == 1300000
    assert json.loads(input_path.read_text(encoding="utf-8"))["inputs"] == [[-1.0, 1.0]]
    assert "--plan-id" in (root / "repro.sh").read_text(encoding="utf-8")

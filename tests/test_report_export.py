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
            "comparison": "allclose",
            "max_abs_diff": 0.1,
            "rmse": 0.05,
            "runtime_policy": {"schema_version": 1, "effective": {"deterministic": True}},
        },
        artifacts=[ArtifactRef(name="inputs", path=tmp_path / "inputs.json", kind="inputs", metadata={"case_fingerprint": "abc"})],
        evidence=HardwareEvidence(backend=BackendKind.GPU, device_id=0, probe_status="available", runtime={"torch": "2.6"}),
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
    assert list(row) == REPORT_EXPORT_FIELDS
    assert row["report_schema_version"] == 1
    assert row["run_id"] == "run-1"
    assert row["case_fingerprint"] == "abc"
    assert row["max_abs_diff"] == 0.1
    assert json.loads(row["runtime_policy_json"])["effective"]["deterministic"] is True
    assert row["hardware_probe_status"] == "available"
    assert list(csv_rows[0]) == REPORT_EXPORT_FIELDS
    assert csv_rows[0]["rmse"] == "0.05"


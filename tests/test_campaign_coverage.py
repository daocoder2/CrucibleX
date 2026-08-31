import json

from typer.testing import CliRunner

from cruciblex.cli import app


def test_campaign_coverage_aggregates_run_output_roots(tmp_path):
    campaign = tmp_path / "campaign"
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for root, operator, backend in ((first, "torch.relu", "cpu"), (second, "torch.relu", "gpu")):
        (root / "results.jsonl").write_text(json.dumps({
            "plan_id": f"{backend}-p", "case_id": 1, "case_name": operator, "node_name": backend,
            "backend": backend, "device_id": 0, "task": "accuracy", "status": "passed",
            "metrics": {}, "artifacts": [], "error": None,
        }) + "\n", encoding="utf-8")
    campaign.mkdir()
    (campaign / "campaign_summary.json").write_text(json.dumps({"runs": [{"output": str(first)}, {"output": str(second)}]}), encoding="utf-8")
    policy = tmp_path / "policy.yaml"
    policy.write_text("required:\n  backend: [cpu, gpu]\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["campaign-coverage", "--output", str(campaign), "--policy", str(policy)])

    assert result.exit_code == 0
    report = json.loads((campaign / "campaign_coverage.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["campaign_outputs"] == [str(first), str(second)]

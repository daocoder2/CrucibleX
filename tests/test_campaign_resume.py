import json
from pathlib import Path

from cruciblex import cli


def test_campaign_auto_resume_and_explicit_resume(tmp_path, monkeypatch):
    campaign_file = tmp_path / "campaign.yaml"
    campaign_file.write_text(
        "runs:\n  - name: one\n    case: case.yaml\n    nodes: nodes.yaml\n    task: run\n", encoding="utf-8"
    )
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "results.jsonl").write_text("{}\n", encoding="utf-8")
        (output / "summary.json").write_text(json.dumps({"total": 1, "passed": 1, "failed": 0}), encoding="utf-8")
        (output / "postprocess.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cli, "run", fake_run)
    output = tmp_path / "output"
    cli.campaign(campaign_file=campaign_file, output=output)
    assert calls[0]["resume_from"] is None

    cli.campaign(campaign_file=campaign_file, output=output)
    assert calls[1]["resume_from"] == output / "one"

    campaign_file.write_text(
        "runs:\n  - name: one\n    case: case.yaml\n    nodes: nodes.yaml\n    task: run\n    resume_from: saved\n    retry_failed: true\n", encoding="utf-8"
    )
    cli.campaign(campaign_file=campaign_file, output=output)
    assert calls[2]["resume_from"] == Path("saved")
    assert calls[2]["retry_failed"] is True

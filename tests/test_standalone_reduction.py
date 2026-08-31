import json
from pathlib import Path

from typer.testing import CliRunner

from cruciblex.cli import app


def test_reduce_cli_writes_standalone_case_and_attempt_evidence(tmp_path):
    case = tmp_path / "case.yaml"
    case.write_text(
        "id: 1\nparameters:\n  - shape_rules:\n      dims: [2, 3]\n    dtypes: [fp16]\n    value_range:\n      valid: [[-1, 1]]\n",
        encoding="utf-8",
    )
    output = tmp_path / "reduced"
    result = CliRunner().invoke(app, [
        "reduce", "--case", str(case), "--output", str(output),
        "--replay-command", "python -c \"import pathlib,sys; assert pathlib.Path(sys.argv[1]).exists(); sys.exit(1)\" {case}",
    ])

    assert result.exit_code == 0
    assert (output / "reduced_case.yaml").exists()
    attempts = [json.loads(line) for line in (output / "replay_attempts.jsonl").read_text().splitlines()]
    assert attempts and all(item["exit_code"] == 1 for item in attempts)
    summary = json.loads((output / "reduction_summary.json").read_text())
    assert summary["status"] == "reduced"
    assert Path(output / "rerun.sh").stat().st_mode & 0o111

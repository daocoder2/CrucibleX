import json
import os
import subprocess
from pathlib import Path

RUNNER = Path(__file__).parents[1] / "scripts" / "run-npu-msprof.sh"


def _fake_tools(tmp_path: Path, fail: bool) -> Path:
    tools = tmp_path / "bin"
    tools.mkdir(parents=True)
    (tools / "cx").write_text("#!/usr/bin/env bash\nset -e\nout=\"\"; while (($#)); do [[ $1 == --output ]] && { out=$2; shift 2; continue; }; shift; done\nmkdir -p \"$out\"; echo '{}' > \"$out/results.jsonl\"\n", encoding="utf-8")
    behavior = "exit 7" if fail else "mkdir -p \"$out/PROF_test/device_0/sqlite\"; touch \"$out/PROF_test/device_0/sqlite/time.db\"; \"$app\""
    (tools / "msprof").write_text("#!/usr/bin/env bash\nset -e\nout=\"\"; app=\"\"; for arg in \"$@\"; do [[ $arg == --output=* ]] && out=${arg#*=}; [[ $arg == --application=* ]] && app=${arg#*=}; done\n" + behavior + "\n", encoding="utf-8")
    for path in tools.iterdir():
        path.chmod(0o755)
    return tools


def test_msprof_runner_records_capture_and_nonblocking_failure(tmp_path):
    for fail, expected in ((False, "captured"), (True, "failed")):
        tool_dir = _fake_tools(tmp_path / expected, fail)
        output = tmp_path / (expected + "-output")
        env = {**os.environ, "PATH": f"{tool_dir}:{os.environ['PATH']}"}
        result = subprocess.run(["bash", str(RUNNER), "case.yaml", "nodes.yaml", str(output)], env=env, capture_output=True, text=True, check=False)
        assert result.returncode == 0
        manifest = json.loads((output / "profiler_manifest.json").read_text())
        assert manifest["status"] == expected
        assert manifest["exit_code"] == (7 if fail else 0)
        assert (output / "profiler.stdout.log").exists()
        assert (output / "profiler.stderr.log").exists()

import os
import subprocess
from pathlib import Path

GATE = Path(__file__).parents[1] / "scripts" / "operator_contract_hardware_gate.sh"


def _fake_cx(tmp_path: Path, include_evidence: bool) -> Path:
    tools = tmp_path / "bin"
    tools.mkdir(parents=True)
    payload = '{"metrics":{"input_dtype_contracts":{"input":{}},"backend_dtype_source":"device_tensor"},"evidence":{"kind":"npu"}}' if include_evidence else '{"metrics":{"input_dtype_contracts":{"input":{}},"backend_dtype_source":"device_tensor"}}'
    script = f'''#!/usr/bin/env bash
out=""
while (($#)); do
  if [[ $1 == --output ]]; then out=$2; shift 2; continue; fi
  shift
done
mkdir -p "$out"
printf '%s\n' '{{"passed": 1, "failed": 0, "error": 0}}' > "$out/summary.json"
printf '%s\n' '{payload}' > "$out/results.jsonl"
'''
    executable = tools / "cx"
    executable.write_text(script, encoding="utf-8")
    executable.chmod(0o755)
    return tools


def test_operator_contract_hardware_gate_requires_dtype_and_hardware_evidence(tmp_path):
    for include_evidence, expected in ((True, 0), (False, 1)):
        tools = _fake_cx(tmp_path / str(include_evidence), include_evidence)
        env = {
            **os.environ,
            "PATH": f"{tools}:{os.environ['PATH']}",
            "CASE_REDUCE": "reduce.yaml",
            "CASE_SORT": "sort.yaml",
            "CASE_INDEX": "index.yaml",
            "CASE_MATMUL": "matmul.yaml",
            "OUTPUT_ROOT": str(tmp_path / "output" / str(include_evidence)),
        }
        result = subprocess.run(["bash", str(GATE)], env=env, capture_output=True, text=True, check=False)
        assert result.returncode == expected

#!/usr/bin/env bash
set -euo pipefail

GPU_ARCHIVE="${GPU_ARCHIVE:?请设置 GPU_ARCHIVE}"
NPU_ARCHIVE="${NPU_ARCHIVE:?请设置 NPU_ARCHIVE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-cx_output/cpu-gpu-npu-accuracy}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../src${PYTHONPATH:+:${PYTHONPATH}}"

rm -rf "$OUTPUT_ROOT" "$OUTPUT_ROOT.gpu" "$OUTPUT_ROOT.npu"
mkdir -p "$OUTPUT_ROOT.gpu" "$OUTPUT_ROOT.npu"
tar -xzf "$GPU_ARCHIVE" -C "$OUTPUT_ROOT.gpu"
tar -xzf "$NPU_ARCHIVE" -C "$OUTPUT_ROOT.npu"

OUTPUT_ROOT="$OUTPUT_ROOT" GPU_ROOT="$OUTPUT_ROOT.gpu" NPU_ROOT="$OUTPUT_ROOT.npu" uv run --no-sync python - <<'PY'
import json
import os
from pathlib import Path
from cruciblex.domain.result import ExecutionResult
from cruciblex.plugins.comparators import allclose  # noqa: F401 注册内置比较器
from cruciblex.report.cross_compare import CrossDeviceComparator

output_root = Path(os.environ["OUTPUT_ROOT"])
roots = {"gpu": Path(os.environ["GPU_ROOT"]), "npu": Path(os.environ["NPU_ROOT"])}

def load(name):
    results = []
    for line in (roots[name] / "results.jsonl").read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        if raw.get("metrics", {}).get("stage"):
            continue
        result = ExecutionResult.model_validate(raw)
        artifacts = []
        for artifact in result.artifacts:
            relative = Path(artifact.path).relative_to("/out")
            artifacts.append(artifact.model_copy(update={"path": roots[name] / relative}))
        results.append(result.model_copy(update={"artifacts": artifacts}))
    return results

gpu_results = load("gpu")
npu_results = load("npu")
cpu = next(item for item in gpu_results if item.backend.value == "cpu")
gpu = next(item for item in gpu_results if item.backend.value == "gpu")
npu = next(item for item in npu_results if item.backend.value == "npu")
input_refs = [next(item for item in result.artifacts if item.kind == "inputs").metadata.get("case_fingerprint") for result in (cpu, gpu, npu)]
if len(set(input_refs)) != 1:
    raise SystemExit(f"三侧 case_fingerprint 不一致: {input_refs}")
comparisons = CrossDeviceComparator(output_root).compare([cpu, gpu, npu])
all_results = [cpu, gpu, npu, *comparisons]
(output_root / "results.jsonl").parent.mkdir(parents=True, exist_ok=True)
(output_root / "results.jsonl").write_text("\n".join(item.model_dump_json() for item in all_results) + "\n", encoding="utf-8")
gate = next((item for item in comparisons if item.metrics.get("stage") == "cpu_npu_gpu_accuracy_gate"), None)
if gate is None:
    raise SystemExit("未生成 CPU/GPU/NPU 三侧精度 gate")
print(json.dumps({"status": gate.status.value, "metrics": gate.metrics, "artifact": str(gate.artifacts[0].path), "case_fingerprint": input_refs[0]}, ensure_ascii=False))
if gate.status.value != "passed":
    raise SystemExit(1)
PY

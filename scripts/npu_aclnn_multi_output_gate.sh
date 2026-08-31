#!/usr/bin/env bash
set -euo pipefail

NODE_PATH="${NODE_PATH:-examples/nodes/local-npu.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-cx_output/aclnn-multi-output-e2e}"

for operator in sort max_dim; do
  cx run \
    --case "examples/cases/aclnn.${operator}.npu.yaml" \
    --nodes "${NODE_PATH}" \
    --scheduler local \
    --task accuracy \
    --output "${OUTPUT_ROOT}/${operator}"
done

python - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for operator in ("sort", "max_dim"):
    summary = json.loads((root / operator / "summary.json").read_text(encoding="utf-8"))
    if summary.get("failed", 0) or summary.get("error", 0) or not summary.get("passed", 0):
        raise SystemExit(f"ACLNN {operator} gate did not pass")
PY

echo "[npu-aclnn-multi-output] passed: ${OUTPUT_ROOT}"

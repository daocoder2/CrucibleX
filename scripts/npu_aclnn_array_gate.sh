#!/usr/bin/env bash
set -euo pipefail

NODE_PATH="${NODE_PATH:-examples/nodes/local-npu.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-cx_output/aclnn-array-e2e}"

cx run \
  --case examples/cases/aclnn.mean.npu.yaml \
  --nodes "${NODE_PATH}" \
  --scheduler local \
  --task accuracy \
  --output "${OUTPUT_ROOT}/mean"

python - "${OUTPUT_ROOT}/mean/summary.json" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
if summary.get("failed", 0) or summary.get("error", 0) or not summary.get("passed", 0):
    raise SystemExit("ACLNN Mean array gate did not pass")
PY

echo "[npu-aclnn-array] passed: ${OUTPUT_ROOT}"

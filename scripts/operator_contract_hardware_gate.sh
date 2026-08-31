#!/usr/bin/env bash
set -euo pipefail

NODE_PATH="${NODE_PATH:-examples/nodes/local-npu.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-cx_output/operator-contract-hardware-gate}"
SCHEDULER="${SCHEDULER:-local}"

# Provide only lanes backed by a supported Torch or ACLNN case in this environment.
CASE_REDUCE="${CASE_REDUCE:?set CASE_REDUCE to a reduce Case path}"
CASE_SORT="${CASE_SORT:?set CASE_SORT to a sort/topk Case path}"
CASE_INDEX="${CASE_INDEX:?set CASE_INDEX to an index/select/gather/scatter Case path}"
CASE_MATMUL="${CASE_MATMUL:?set CASE_MATMUL to a matmul/bmm Case path}"

run_lane() {
  local name="$1"
  local case_path="$2"
  local output="${OUTPUT_ROOT}/${name}"
  cx run \
    --case "${case_path}" \
    --nodes "${NODE_PATH}" \
    --scheduler "${SCHEDULER}" \
    --task accuracy \
    --output "${output}"
  python - "${output}/summary.json" "${output}/results.jsonl" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
if summary.get("failed", 0) or summary.get("error", 0):
    raise SystemExit("hardware lane did not pass")
records = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8") if line.strip()]
if not records:
    raise SystemExit("missing result records")
metrics = records[-1].get("metrics", {})
for key in ("input_dtype_contracts", "backend_dtype_source"): 
    if not metrics.get(key):
        raise SystemExit(f"missing {key}")
evidence = records[-1].get("evidence", {})
if not evidence.get("kind"):
    raise SystemExit("missing hardware evidence")
PY
}

run_lane reduce "${CASE_REDUCE}"
run_lane sort "${CASE_SORT}"
run_lane index "${CASE_INDEX}"
run_lane matmul "${CASE_MATMUL}"

echo "[operator-contract-hardware] passed: ${OUTPUT_ROOT}"

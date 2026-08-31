#!/usr/bin/env bash
set -euo pipefail

NODE_PATH="${NODE_PATH:-examples/nodes/local-npu.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-cx_output/hardware-dtype-layout-gate}"
SCHEDULER="${SCHEDULER:-local}"

# Set each lane to a checked-in or externally supplied Case document.
CASE_BF16="${CASE_BF16:?set CASE_BF16 to a bf16 Case path}"
CASE_LAYOUT="${CASE_LAYOUT:?set CASE_LAYOUT to a storage-offset/layout Case path}"
CASE_STRIDE="${CASE_STRIDE:?set CASE_STRIDE to a stride Case path}"
CASE_SPECIAL="${CASE_SPECIAL:?set CASE_SPECIAL to a NaN/Inf/subnormal Case path}"

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
  python - "${output}/results.jsonl" <<'PY'
import json
import sys
records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
if not records:
    raise SystemExit("missing gate result")
metrics = records[-1].get("metrics", {})
if not metrics.get("input_dtype_contracts"):
    raise SystemExit("missing input dtype contract evidence")
if not metrics.get("backend_dtype_source"):
    raise SystemExit("missing backend device dtype evidence")
PY
}

run_lane bf16 "${CASE_BF16}"
run_lane layout "${CASE_LAYOUT}"
run_lane stride "${CASE_STRIDE}"
run_lane special "${CASE_SPECIAL}"

echo "[hardware-dtype-layout] passed: ${OUTPUT_ROOT}"

#!/usr/bin/env bash
set -euo pipefail

CASE_PATH="${CASE_PATH:-examples/cases/torch.add.gpu.yaml}"
NODE_PATH="${NODE_PATH:-examples/nodes/ray-cpu-gpu-e2e.yaml}"
RAY_ADDRESS="${RAY_ADDRESS:?set RAY_ADDRESS=ray://<head-ip>:10001}"
OUTPUT_ROOT="${OUTPUT_ROOT:-cx_output/gpu-hardware-gate}"
COVERAGE_POLICY="${COVERAGE_POLICY:-scripts/gpu_hardware_gate_policy.yaml}"

run_task() {
  local task="$1"
  local output="${OUTPUT_ROOT}/${task}"
  echo "[gpu-gate] ${task}"
  env RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run cx run \
    --case "${CASE_PATH}" \
    --nodes "${NODE_PATH}" \
    --scheduler ray \
    --ray-address "${RAY_ADDRESS}" \
    --task "${task}" \
    --output "${output}"
}

run_task accuracy
run_task performance_device
run_task memory_device

echo "[gpu-gate] coverage"
env RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run cx coverage-report \
  --output "${OUTPUT_ROOT}/accuracy" \
  --input "${OUTPUT_ROOT}/performance_device" \
  --input "${OUTPUT_ROOT}/memory_device" \
  --policy "${COVERAGE_POLICY}" \
  --report coverage.json
if ! grep -q '"kind": "gpu_evidence"' "${OUTPUT_ROOT}/accuracy/results.jsonl"; then
  echo "[gpu-gate] missing gpu_evidence artifact" >&2
  exit 1
fi

echo "[gpu-gate] passed: ${OUTPUT_ROOT}"

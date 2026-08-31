#!/usr/bin/env bash
set -euo pipefail

NODE_PATH="${NODE_PATH:-examples/nodes/ray-cpu-gpu-e2e.yaml}"
RAY_ADDRESS="${RAY_ADDRESS:?set RAY_ADDRESS=ray://<head-ip>:10001}"
OUTPUT_ROOT="${OUTPUT_ROOT:-cx_output/operator-breadth-gate}"
POLICY_PATH="${POLICY_PATH:-scripts/operator_breadth_policy.yaml}"

run_case() {
  local name="$1"
  local case_path="examples/cases/torch.${name}.hardware.yaml"
  local output="${OUTPUT_ROOT}/${name}"
  env RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run cx run \
    --case "${case_path}" \
    --nodes "${NODE_PATH}" \
    --scheduler ray \
    --ray-address "${RAY_ADDRESS}" \
    --task accuracy \
    --output "${output}"
  grep -q '"kind": "gpu_evidence"' "${output}/results.jsonl"
}

mkdir -p "${OUTPUT_ROOT}"
run_case relu
run_case softmax
run_case matmul

env RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run cx coverage-report \
  --output "${OUTPUT_ROOT}/relu" \
  --input "${OUTPUT_ROOT}/softmax" \
  --input "${OUTPUT_ROOT}/matmul" \
  --policy "${POLICY_PATH}" \
  --report coverage.json

echo "[operator-breadth] passed: ${OUTPUT_ROOT}"

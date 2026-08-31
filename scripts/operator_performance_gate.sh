#!/usr/bin/env bash
set -euo pipefail

NODE_PATH="${NODE_PATH:-examples/nodes/ray-cpu-gpu-e2e.yaml}"
RAY_ADDRESS="${RAY_ADDRESS:?set RAY_ADDRESS=ray://<head-ip>:10001}"
OUTPUT_ROOT="${OUTPUT_ROOT:-cx_output/operator-performance-gate}"
POLICY_PATH="${POLICY_PATH:-scripts/operator_performance_policy.yaml}"

run_case() {
  local name="$1"
  local task="$2"
  env RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run cx run \
    --case "examples/cases/torch.${name}.hardware.yaml" \
    --nodes "${NODE_PATH}" \
    --scheduler ray \
    --ray-address "${RAY_ADDRESS}" \
    --task "${task}" \
    --output "${OUTPUT_ROOT}/${name}/${task}"
}

for name in relu softmax matmul; do
  run_case "${name}" performance_device
  run_case "${name}" memory_device
done

env RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run cx coverage-report \
  --output "${OUTPUT_ROOT}/relu/performance_device" \
  --input "${OUTPUT_ROOT}/softmax/performance_device" \
  --input "${OUTPUT_ROOT}/matmul/performance_device" \
  --input "${OUTPUT_ROOT}/relu/memory_device" \
  --input "${OUTPUT_ROOT}/softmax/memory_device" \
  --input "${OUTPUT_ROOT}/matmul/memory_device" \
  --policy "${POLICY_PATH}" \
  --report performance_coverage.json

echo "[operator-performance] passed: ${OUTPUT_ROOT}"

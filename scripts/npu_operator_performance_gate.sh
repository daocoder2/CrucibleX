#!/usr/bin/env bash
set -euo pipefail

NODE_PATH="${NODE_PATH:-examples/nodes/local-npu.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-cx_output/npu-operator-performance}"
POLICY_PATH="${POLICY_PATH:-scripts/npu_operator_performance_policy.yaml}"

run_case() {
  local name="$1"
  local task="$2"
  cx run \
    --case "examples/cases/torch.${name}.hardware.yaml" \
    --nodes "${NODE_PATH}" \
    --scheduler local \
    --task "${task}" \
    --output "${OUTPUT_ROOT}/${name}/${task}"
}

for name in relu softmax matmul; do
  run_case "${name}" performance_device
  run_case "${name}" memory_device
done

cx coverage-report \
  --output "${OUTPUT_ROOT}/relu/performance_device" \
  --input "${OUTPUT_ROOT}/softmax/performance_device" \
  --input "${OUTPUT_ROOT}/matmul/performance_device" \
  --input "${OUTPUT_ROOT}/relu/memory_device" \
  --input "${OUTPUT_ROOT}/softmax/memory_device" \
  --input "${OUTPUT_ROOT}/matmul/memory_device" \
  --policy "${POLICY_PATH}" \
  --report performance_coverage.json

echo "[npu-operator-performance] passed: ${OUTPUT_ROOT}"

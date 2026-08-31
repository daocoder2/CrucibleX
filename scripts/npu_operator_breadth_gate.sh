#!/usr/bin/env bash
set -euo pipefail

NODE_PATH="${NODE_PATH:-examples/nodes/local-npu.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-cx_output/npu-operator-breadth}"
POLICY_PATH="${POLICY_PATH:-scripts/npu_operator_breadth_policy.yaml}"

run_case() {
  local name="$1"
  local task="$2"
  local case_path="examples/cases/torch.${name}.hardware.yaml"
  local output="${OUTPUT_ROOT}/${name}/${task}"
  cx run \
    --case "${case_path}" \
    --nodes "${NODE_PATH}" \
    --scheduler local \
    --task "${task}" \
    --output "${output}"
}

for name in relu softmax matmul; do
  run_case "${name}" accuracy
done

cx coverage-report \
  --output "${OUTPUT_ROOT}/relu/accuracy" \
  --input "${OUTPUT_ROOT}/softmax/accuracy" \
  --input "${OUTPUT_ROOT}/matmul/accuracy" \
  --policy "${POLICY_PATH}" \
  --report coverage.json

echo "[npu-operator-breadth] passed: ${OUTPUT_ROOT}"

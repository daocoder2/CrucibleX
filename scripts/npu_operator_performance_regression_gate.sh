#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="${BASELINE_ROOT:-baselines/npu-operator-performance}"
POLICY_PATH="${POLICY_PATH:-scripts/npu_performance_gate_policy.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-cx_output/npu-operator-performance-candidate}"

OUTPUT_ROOT="${OUTPUT_ROOT}" scripts/npu_operator_performance_gate.sh

: > "${OUTPUT_ROOT}/results.jsonl"
for operator in relu softmax matmul; do
  for task in performance_device memory_device; do
    cat "${OUTPUT_ROOT}/${operator}/${task}/results.jsonl" >> "${OUTPUT_ROOT}/results.jsonl"
  done
done

cx performance-gate \
  --baseline "${BASELINE_ROOT}" \
  --candidate "${OUTPUT_ROOT}" \
  --policy "${POLICY_PATH}" \
  --output "${OUTPUT_ROOT}/performance_gate.json"

echo "[npu-operator-performance-regression] passed: ${OUTPUT_ROOT}/performance_gate.json"

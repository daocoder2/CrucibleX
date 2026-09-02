#!/usr/bin/env bash
set -euo pipefail

MANIFEST_PATH="${MANIFEST_PATH:-examples/manifests/operator-boundary-campaign.yaml}"
NODE_PATH="${NODE_PATH:-examples/nodes/local.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-cx_output/operator-boundary-gate}"
TASK="${TASK:-accuracy}"

cx manifest validate "${MANIFEST_PATH}"
cx run \
  --manifest "${MANIFEST_PATH}" \
  --nodes "${NODE_PATH}" \
  --scheduler local \
  --task "${TASK}" \
  --output "${OUTPUT_ROOT}"

python - "${OUTPUT_ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
required = {"manifest.json", "results.jsonl", "report.jsonl", "summary.json", "postprocess.json"}
missing = sorted(name for name in required if not (root / name).is_file())
if missing:
    raise SystemExit(f"operator-boundary gate missing archive files: {missing}")

summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
if summary.get("failed", 0) or summary.get("error", 0) or not summary.get("passed", 0):
    raise SystemExit("operator-boundary gate has failed, error, or zero passed results")

rows = [json.loads(line) for line in (root / "report.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows or any(row.get("status") != "passed" for row in rows):
    raise SystemExit("operator-boundary gate report contains non-passed rows")
hardware_rows = [row for row in rows if row.get("manifest_lane_kind") == "hardware"]
if hardware_rows:
    metrics = [json.loads(row.get("metrics_json", "{}")) for row in hardware_rows]
    if any(item.get("backend_dtype_source") != "device_tensor" for item in metrics):
        raise SystemExit("operator-boundary hardware rows lack device_tensor evidence")

print(f"[operator-boundary-gate] passed: {root}")
PY

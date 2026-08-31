#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 CASE_FILE NODES_FILE OUTPUT_DIR" >&2
  exit 2
fi

case_file=$1
nodes_file=$2
output_dir=$3
trace_dir="${output_dir}/msprof"
run_dir="${output_dir}/run"
mkdir -p "${output_dir}"
wrapper=$(mktemp "${TMPDIR:-/tmp}/cx-msprof-wrapper.XXXXXX")
trap "rm -f ${wrapper}" EXIT

cat > "${wrapper}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec cx run --case "${case_file}" --nodes "${nodes_file}" --task performance_device --scheduler local --output "${run_dir}"
EOF
chmod 755 "${wrapper}"

set +e
msprof --output="${trace_dir}" --application="${wrapper}" >"${output_dir}/profiler.stdout.log" 2>"${output_dir}/profiler.stderr.log"
profile_exit_code=$?
set -e

python - "${output_dir}/profiler_manifest.json" "${trace_dir}" "${run_dir}" "${profile_exit_code}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
trace_dir = Path(sys.argv[2])
run_dir = Path(sys.argv[3])
exit_code = int(sys.argv[4])
trace_files = sorted(str(path.relative_to(trace_dir)) for path in trace_dir.rglob("*") if path.is_file()) if trace_dir.exists() else []
result_path = run_dir / "results.jsonl"
run_succeeded = result_path.exists()
manifest_path.write_text(json.dumps({
    "tool": "msprof",
    "status": "captured" if exit_code == 0 and trace_files else "failed",
    "exit_code": exit_code,
    "trace_dir": str(trace_dir),
    "trace_file_count": len(trace_files),
    "trace_files": trace_files,
    "run_results": str(result_path) if run_succeeded else None,
    "stdout_log": str(manifest_path.parent / "profiler.stdout.log"),
    "stderr_log": str(manifest_path.parent / "profiler.stderr.log"),
}, indent=2), encoding="utf-8")
PY

# The performance run is authoritative. A profiler failure is recorded, not propagated.
exit 0

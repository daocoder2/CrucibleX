#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/workspace/src:${PYTHONPATH:-}
exec cx run --case /workspace/examples/cases/torch.abs.npu.yaml --nodes /workspace/examples/nodes/local-npu.yaml --task performance_device --scheduler local --output /tmp/cx-msprof-inner

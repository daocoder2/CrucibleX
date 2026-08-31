# Ray-First Rollout

1. Build and push the Ray images from the repository root.

```bash
SSH_USER=<user> SSH_OPTS="-i <key-path> -o BatchMode=yes" docker/deploy-ray.sh build
SSH_USER=<user> SSH_OPTS="-i <key-path> -o BatchMode=yes" docker/deploy-ray.sh push
```

2. Deploy the head and worker containers.

```bash
SSH_USER=<user> SSH_OPTS="-i <key-path> -o BatchMode=yes" docker/deploy-ray.sh deploy-all
```

3. Verify the worker image revision with the driver fingerprint probe.

```bash
env RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run cx doctor --ray-address ray://<head-host>:10001
```

4. Run the required Ray E2E gates.

```bash
env RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run cx run --case examples/cases/torch.add.gpu.yaml --nodes examples/nodes/ray-cpu-gpu-e2e.yaml --task accuracy --scheduler ray --ray-address ray://<head-host>:10001 --output cx_output/ray-accuracy-e2e

env RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run cx run --case examples/cases/torch.add.gpu.yaml --nodes examples/nodes/ray-cpu-gpu-e2e.yaml --task performance_device --scheduler ray --ray-address ray://<head-host>:10001 --output cx_output/ray-performance-e2e

env RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run cx run --case examples/cases/torch.add.gpu.yaml --nodes examples/nodes/ray-cpu-gpu-e2e.yaml --task memory_device --scheduler ray --ray-address ray://<head-host>:10001 --output cx_output/ray-memory-e2e
```

5. Check the generated artifacts.

- `results.jsonl`
- `summary.json`
- `postprocess.json`
- `repro_bundle.json` for clustered failure replay
- `repro_bundle.json` with `--minimize` for a smaller rerun set
- `campaign_summary.json` and `campaign_report.md` for batch runs
- for accuracy, the derived comparison artifact under `<case>/cross_compare/`
- for performance, `latency_ms` in the postprocess rows
- for memory, `memory_peak_bytes` and `memory_peak_mb` in the postprocess rows

6. Generate rerun commands from a failure cluster when a run does not behave as expected.

```bash
uv run cx repro --output cx_output/ray-accuracy-e2e --cluster-id cluster-0 --script
uv run cx run --case examples/cases/torch.abs.invalid.yaml --nodes examples/nodes/local.yaml --task run --scheduler local --plan-id 1800000:local-cpu:cpu:0:run --output cx_output/rerun-check
```

7. For NPU, ACLNN, and DCU hardware runs, use node files that expose `npu` or `dcu` resources as appropriate.

- NPU: `npu: 1.0`
- ACLNN: `npu: 1.0`
- DCU: `dcu: 1.0`

See `examples/nodes/ray-accelerators-template.yaml` for a reference layout.

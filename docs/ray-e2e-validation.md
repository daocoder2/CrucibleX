# Ray E2E Validation Runbook

This runbook checks the real Ray execution path on a multi-node cluster.

## Goal

Validate that:

- placement resolves to the intended Ray node
- GPU plans require GPU resources
- driver-side artifact persistence still works
- scheduler failures become structured results instead of crashing the run
- the CLI can complete a real Ray batch end to end

## Minimum Topology

- 1 Ray head node
- 1 Ray worker with CPU only
- 1 Ray worker with GPU, or a second worker that exposes a distinct resource profile you want to validate

For the current required E2E gate, validate CPU and GPU first. NPU, DCU, and ACLNN use the same execution and post-processing contracts, but their hardware E2E runs should wait until matching Ray resources and runtime images are available. For NPU validation, use a worker that exposes the custom `npu` resource. ACLNN also defaults to `npu` because it runs on Ascend/NPU hardware. For DCU validation, use a worker that exposes the custom `dcu` resource. See `examples/nodes/ray-accelerators-template.yaml` for the expected node-file shape.

## Preconditions

- `cx` is installed in the active environment
- every Ray worker can import the project code and any plugins you pass with `--plugin`
- the driver output root is writable
- external drivers use Ray Client through `ray://<head-host>:10001`; native `<head-host>:6379` is for in-cluster/native Ray processes
- Ray Client submissions should pass `runtime_env={"working_dir": None}` so the driver does not upload local source trees
- because source upload is disabled, every worker image or environment must already have the current project revision installed before an E2E run
- generated inputs are persisted once on the driver as `inputs.json`; Ray object store only carries the live execution copy to workers
- node host values in the YAML file match the Ray node manager address or hostname seen by `cx doctor`

## Sanity Check

Run this on the driver before the batch:

```bash
uv run cx doctor --ray-address ray://<head-host>:10001
```

Use native `<head-host>:6379` only when the driver runs inside the Ray cluster runtime environment.

Expected:

- Ray is initialized or available
- the alive Ray nodes list shows the cluster members you expect
- each node prints its address, hostname, and visible resources
- the driver and worker `pipeline_sha256` fingerprints match; a mismatch means the Ray worker image is still running a different project revision

## Example Input Files

Start from the shipped smoke case:

- `examples/cases/torch.abs.yaml`

Create a node file for the real cluster. A minimal shape looks like this:

```yaml
nodes:
  - name: cpu-node
    host: 10.0.0.11
    role: candidate
    allowed_tasks: [accuracy, run]
    devices:
      - id: 0
        backend: cpu

  - name: gpu-node
    host: 10.0.0.12
    role: candidate
    allowed_tasks: [accuracy, run]
    devices:
      - id: 0
        backend: gpu
```

If you want to verify NPU placement, add a node with `backend: npu` and a matching Ray `npu` resource.

## Execute

Run the batch on the driver:

```bash
uv run cx run \
  --case examples/cases/torch.abs.yaml \
  --nodes /path/to/real-ray-nodes.yaml \
  --scheduler ray \
  --ray-address ray://<head-host>:10001 \
  --output cx_output/ray-e2e
```

If you need an extra plugin file, add `--plugin /path/to/plugin.py`.

## Pass Criteria

A run is good when all of these hold:

- `cx run` exits successfully
- the result table shows the expected backend and device labels
- `cx_output/ray-e2e/manifest.json` exists
- `cx_output/ray-e2e/results.jsonl` exists
- `cx_output/ray-e2e/summary.json` exists
- `cx_output/ray-e2e/postprocess.json` exists for non-resume runs
- artifact files are written under the driver output root, not on worker local paths
- a missing host or insufficient resources produces `SKIPPED`, not a crash
- a Ray collection error produces `ERROR`, not a crash

## Failure Triage

If placement fails:

- check `NodeSpec.host` against `cx doctor`
- check the Ray node has the expected `GPU` or `npu` resource
- confirm the node is alive

If actor startup fails:

- confirm the worker can import the project package
- confirm any `--plugin` paths are visible on that worker
- confirm the worker environment has the backend runtime you expect

If artifacts do not appear:

- verify the driver output root is writable
- verify the run completed and `results.jsonl` was written
- check whether the worker returned `artifact_payloads` and the driver materialized them

## Suggested First Runs

1. CPU-only placement smoke.
2. GPU placement smoke.
3. A negative run where one plan targets a host that does not exist in the cluster.
4. A resource-mismatch run where a GPU plan lands on a CPU-only node and is skipped.
5. A full run that produces at least one artifact and one summary file.

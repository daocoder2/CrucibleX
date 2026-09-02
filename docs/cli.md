# CrucibleX CLI Reference

This page summarizes the main commands used for execution, report generation, reproduction, onboarding, and case expansion.

## Shared Output Files

Most commands write to an output directory selected with `--output`.

- `manifest.json`: run metadata and paths
- `results.jsonl`: one execution result per line
- `results.csv`: tabular result export
- `summary.json`: total, passed, and failed counts
- `postprocess.json`: failure clusters, fuzz rows, and summary rows
- `report.md` or `campaign_report.md`: rendered markdown report
- `repro_bundle.json`: rerun commands and reduced failure context


## `cx doctor`

Inspect driver and Ray worker runtime state.

```bash
uv run cx doctor --ray-address ray://<head-host>:10001
```

With a Ray address, worker rows include CrucibleX package identity plus runtime probe fields for Torch version, `torch.version.cuda`, CUDA availability, CUDA device count, Torch NPU availability, and visible device environment variables. GPU evidence must come from the real GPU cluster; NPU evidence must come from the real NPU host or cluster.

## `cx run`

Execute a case on a set of nodes, or pass a manifest for lane-oriented execution. `--task` continues to select the execution task kind; `--manifest` selects a top-level manifest.

```bash
uv run cx run \
  --case examples/cases/torch.abs.yaml \
  --nodes examples/nodes/local.yaml \
  --task run \
  --scheduler local \
  --output cx_output/local-run

uv run cx run \
  --manifest examples/manifests/operator-contract-suite.yaml \
  --nodes examples/nodes/local.yaml \
  --scheduler local \
  --output cx_output/operator-contract-suite

uv run cx run \
  --manifest examples/manifests/local-smoke-manifest.yaml \
  --nodes examples/nodes/local.yaml \
  --scheduler local \
  --output cx_output/local-smoke-manifest
```

Inspect a manifest without executing it:

```bash
uv run cx manifest validate examples/manifests/operator-contract-suite.yaml
uv run cx manifest plan examples/manifests/operator-contract-suite.yaml --nodes examples/nodes/local.yaml
uv run cx manifest plan examples/manifests/operator-contract-suite.yaml --nodes examples/nodes/local.yaml --json
```

Use `--scheduler local` for smoke tests. Ray remains the default path for distributed execution. For comparison cases, `oracle.reference_executor` is only executed when `oracle.metadata.execute_reference` is set. Bare ACLNN function cases use `executor: aclnn` with `api_type: aclnn_function`; op signatures live under `invocation.metadata.aclnn`.

## `cx report`

Render a markdown report from an existing output directory.

```bash
uv run cx report --output cx_output/local-run
```

For a campaign output, the command writes `campaign_report.md` and refreshes `results.csv` when batch rows are available.

## `cx repro`

Create rerun commands and bundle failure context from `postprocess.json`.

```bash
uv run cx repro --output cx_output/local-run --script
```

Useful flags: `--cluster-id` to focus one cluster and `--minimize` to keep one representative case per cluster. Use `--replay-command` with a `{case}` placeholder to replay each candidate; the default `--replay-exit-code 1` preserves a failing command, while another exit code can be selected explicitly.

The bundle includes reduced fuzz case rows when fuzz provenance is available. When replay is enabled, accepted candidates are written to `semantic_reduction.yaml` and replay attempts are recorded in the bundle.


## `cx import-dump`

Convert a concrete operator dump with inputs into replayable CrucibleX case YAML plus an input snapshot.

```bash
uv run cx import-dump \
  --source examples/imports/torch.add.dump.yaml \
  --output cx_output/imported-dump/cases.yaml \
  --executor torch \
  --reference-executor torch
```

The command writes `cases.yaml` and a sibling `inputs.json`. The generated case uses the built-in `dump_replay` generator, so later runs reuse the captured input values instead of sampling fresh values. The case provenance records the source dump path, converter version, lossy fields, and input snapshot path.

## `cx import-profile`

Convert profile-derived dtype and shape samples into standard CrucibleX case YAML using the default generator.

```bash
uv run cx import-profile \
  --source examples/imports/torch.matmul.profile.yaml \
  --output cx_output/imported-profile/cases.yaml \
  --executor torch \
  --reference-executor torch
```

The importer aggregates observed parameter dtypes and shapes into `parameters[*].dtypes`, `shape.dim_count`, and `shape.dim_values`. It also preserves the original sample sets under `generation.metadata.profile_shapes` and `generation.metadata.profile_dtypes` for later policy tuning.

## `cx import-atb` / `cx import-temu`

Convert ATB or Temu backend configs into standard CrucibleX case YAML while preserving backend-specific execution metadata.

```bash
uv run cx import-atb \
  --source examples/imports/atb.add.config.yaml \
  --output cx_output/imported-atb/cases.yaml \
  --executor atb \
  --reference-executor torch

uv run cx import-temu \
  --source examples/imports/temu.softmax.config.yaml \
  --output cx_output/imported-temu/cases.yaml \
  --executor temu \
  --reference-executor torch
```

The generated cases keep backend-specific config under `metadata.backend_import` and include a plugin skeleton hint such as `cruciblex.plugins.executors.atb` or `cruciblex.plugins.executors.temu`. Backend execution still requires a matching executor plugin and real hardware validation.

## `cx onboard`

Generate a starter scaffold from an operator facts file.

```bash
uv run cx onboard \
  --facts examples/operator-onboarding/operator_facts.yaml \
  --output cx_output/onboarding-smoke
```

The scaffold includes `case.yaml`, `fuzz_case.yaml`, `nodes.yaml`, `hardware_nodes.yaml`, `campaign.yaml`, `hardware_campaign.yaml`, `executor_plugin.py`, `validate.sh`, `hardware_validate.sh`, and a local README.

## `cx generate`

Expand a case YAML into concrete generated cases without executing them.

```bash
uv run cx generate \
  --case examples/cases/torch.abs.generated.yaml \
  --output cx_output/generated-smoke
```

This writes `generated_cases.json` and `generated_cases.yaml`.

## Typical flows

- local smoke: `cx run` then `cx report`
- fuzz smoke: `cx run --task fuzz` then `cx report`
- onboarding: `cx onboard`, then `cx run`, then `cx generate`
- reproduction: `cx repro` after a failed run or campaign
- GPU hardware gate: `scripts/gpu_hardware_gate.sh` on a deployed Ray GPU cluster
- coverage matrix: `cx coverage-report --output <accuracy-dir> --input <performance-dir> --input <memory-dir> --policy <policy.yaml>`
- standalone reduction: `cx reduce --case <case.yaml> --replay-command '<command with {case}>' --output <reduced-dir>`
- generate filters: repeat `--include dimension=value` or `--exclude dimension=value` for `operator`, `backend`, `task`, `dtype`, or `tag`.
- operator breadth gate: `scripts/operator_breadth_gate.sh` runs relu/softmax/matmul CPU/GPU accuracy and enforces `scripts/operator_breadth_policy.yaml`.
- NPU operator breadth gate: inside the documented privileged 910B3 container, run `scripts/npu_operator_breadth_gate.sh`; it uses local NPU scheduling and enforces `scripts/npu_operator_breadth_policy.yaml` across operator accuracy.
- performance matrix gates: `scripts/operator_performance_gate.sh` covers CPU/GPU and `scripts/npu_operator_performance_gate.sh` covers NPU for relu/softmax/matmul performance and memory; cases inherit `metadata.benchmark_policy` and invocation-level benchmark settings override it. Metrics include latency stddev, requested/effective duration, and sample count.
- NPU performance regression: `scripts/npu_operator_performance_regression_gate.sh` reruns the matrix, compares it with `baselines/npu-operator-performance/results.jsonl` using `scripts/npu_performance_gate_policy.yaml`, and writes `performance_gate.json`; NPU profiler is explicitly optional while latency/throughput/memory thresholds remain enforced.
- campaign coverage: `cx campaign-coverage --output <campaign-output> --policy <policy.yaml>` reads every run output listed in `campaign_summary.json` and writes `campaign_coverage.json` without manually listing roots.
- ATB/Temu external runtime: imported cases use `metadata.backend_import.config.command`; the command receives a `cruciblex.external-runtime.v1` JSON request on stdin and returns JSON on stdout. Without a command, execution is recorded as unsupported.
- ACLNN signatures: `invocation.metadata.aclnn` supports tensor/scalar arguments, native C ABI `native_int`/`native_bool` attributes, `int_array`/`float_array`/`bool_array`, optional omission, multiple tensor outputs, and declared output `shape: [..]`; unsupported kinds are reported explicitly. Run `scripts/npu_aclnn_multi_output_gate.sh` for real CANN `aclnnSort`/`aclnnMaxDim` and `scripts/npu_aclnn_array_gate.sh` for the real `aclnnMean` `aclIntArray` ABI in the documented privileged 910B3 container.

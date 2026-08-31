# ATK Capability Coverage Roadmap

本文将 ATK 作为已验证的算子测试工作流和资产参考。CrucibleX 是独立的评测框架，目标是在保持 Ray-first 架构更小、更严格、更易验证的前提下，沉淀持久的用户价值。兼容适配器是可选边界，不是产品目标。

## First Principles

- 将 ATK 能力视为用户价值证据，而不是完整性目标或实现蓝图。
- Keep operator facts, generated cases, execution topology, backend adaptation, and reports as separate contracts.
- Use Ray as the default placement and fan-out layer; keep local execution as a debug path.
- Treat CPU, GPU, NPU, ACLNN, and DCU as resource-backed execution targets, not separate product flows.
- Let workers execute one plan and return structured artifacts; let the driver own cross-device compare, aggregation, reports, and repro.
- Translate old ATK assets into the new schema through importers instead of preserving old runtime classes.
- Make generated cases deterministic, bounded, and reproducible by seed.
- Source hardware metrics from backend runtimes, not generic Python process telemetry.

## Current Position

The core trunk is now viable:

- Domain contracts exist for cases, nodes, plans, results, run context, and manifests.
- Local and Ray schedulers are implemented.
- CPU, GPU, NPU, ACLNN, and DCU backend runtime boundaries exist.
- Ray resource mapping exists for CPU, GPU, NPU, ACLNN, and DCU.
- Artifact persistence, resume, repro bundles, CSV, markdown, postprocess summaries, and cross-device compare are implemented.
- GPU has a validated Torch path: torch 2.6.0+cu126 in the GPU image, Ray GPU placement, examples/cases/torch.add.gpu.yaml, and CPU-vs-GPU allclose through Ray.
- NPU has a concrete verification path through the NPU image and torch_npu smoke case, and should be run as a hardware gate whenever matching Ascend workers are available.

The remaining work is breadth: old ATK had many front-door converters, generator variants, backend-specific probes, and operational commands. CrucibleX has the cleaner trunk but needs selective reconstruction of those high-value surfaces.

## Coverage Matrix

| ATK capability area | Current CrucibleX status | Target coverage | Priority |
| --- | --- | --- | --- |
| Run orchestration | Rebuilt with cx run, planning, local/Ray schedulers, result store, resume, and reports | Keep as the primary trunk; add only missing operational controls | P0 |
| CPU/GPU operator validation | CPU and GPU Torch images are explicit; Ray GPU E2E is validated | Use Torch GPU case as the required gate for GPU operator claims | P0 |
| NPU validation | NPU image and smoke case exist; backend maps to Ray custom npu resource | Keep repeatable Ascend hardware gates and record torch/torch_npu/CANN versions | P0 |
| ACLNN validation | Executor imports a runtime module and maps to NPU resource | Build real ACLNN sample modules and hardware E2E gates | P1 |
| DCU validation | Backend and resource mapping exist | Add HIP/torch-dcu or vendor runtime executor and hardware gate | P2 |
| Case schema | Clean YAML/JSON case and node schema | Extend around constraints and metadata, not old class compatibility | P0 |
| Data generation | Deterministic default generator, count/seed, constraints, invalid cases, random/boundary coverage | Rebuild ATK's rich dtype/shape/value/traversal coverage as plugins | P0 |
| Parameter relationships | Linked dtype/shape and simple random/boundary metadata exist | Add shape expressions, broadcasting, rank coupling, optional params, list and tensor-list helpers | P0 |
| Fuzz | Fuzz task, provenance, reports, and repro rows exist | Make fuzz policy expressive enough for operator facts and large campaigns | P1 |
| Performance | Latency rows exist | Add backend synchronization, warmup/repeat policy, percentiles, throughput, and vendor profiler hooks | P1 |
| Memory | Process memory rows exist | Add CUDA/NPU/DCU allocator metrics and peak reset/sync semantics | P1 |
| Reports | JSONL, CSV, summary, postprocess, markdown, failure clusters | Add backend matrices, trend-ready exports, operator coverage tables, and richer report titles | P1 |
| Repro | Repro bundles and per-plan rerun commands exist | Emit minimized case YAML and input snapshots for failing generated/fuzz cases | P1 |
| Operator onboarding | Facts template and scaffold generation exist | Promote facts-to-case automation and hardware promotion commands | P1 |
| API-to-YAML import | Not rebuilt | Translate API docs/specs into CrucibleX facts/cases | P1 |
| Dump/profile/Temu/ATB migration | Not rebuilt | Import old artifacts into normalized facts/cases plus preserved provenance | P1/P2 |
| XRun/Excel flows | Not rebuilt | Provide import/export adapters only if active users still depend on those files | P2 |
| Server/DB/UI | Not rebuilt | Defer until CLI/batch contracts stabilize | P3 |

## P0: Must Cover Next

### 1. Generation Core Parity

Old ATK's main strength was broad operator input-space coverage. CrucibleX needs richer generation before backend coverage will scale.

Implemented:

- Generated JSON and YAML are persisted for expanded cases.
- Deterministic expansion supports count, invalid_count, seeds, max_elements, max_bytes, boundary coverage, random coverage, and linked dtype/shape metadata.
- Default input generation now supports tensor_list, tensor_tuple, scalar_list, scalar_tuple, attribute_list, and attribute_tuple with collection length, item_shapes, item_values, item_dtypes, and explicit items metadata.

Remaining deliverables:

- Shape relationships support declarative `same_rank`, `same_numel`, `broadcastable_with`, `dim_equal`, `divisible_by`, and `transpose_of` metadata through the `shape_relationships` constraint; see `docs/shape-relationships.md`.
- Dtype generation supports `dtype_policy` groups with deterministic selection, allow-list filtering, and backend-specific `backend_allowed` filtering.
- Dtype promotion supports declarative `dtype_promotion.sources` and records the resolved source dtypes in parameter metadata.
- Value generation supports `value_policy` kinds `constant`, `zero`, `one`, `nan`, and `inf` where the dtype permits it, in addition to boundary/random ranges.
- Attribute generation supports `enum_values` and `optional_values` while preserving the positional input contract.
- The checked-in `examples/cases/numpy.add.generated.yaml` case combines broadcasting, dtype filtering, constant/one values, deterministic expansion, and executable CPU add; two generated runs are byte-identical and both expanded plans pass.
- Hardware operator breadth now includes checked-in `torch.relu`, `torch.softmax(dim=1)`, and `torch.matmul` cases; all three passed real Ray CPU/GPU accuracy and cross-device comparison on the unified cluster. The same three cases also passed real NPU accuracy on a validated NPU host using the documented privileged `cx-ray:2.58.0-py311-npu-aarch64` container; the NPU coverage gate passed.

Remaining deliverables:

- Shape work: `rank_range`, `dimension_alias`, and cross-parameter `product_limits` are supported through generation constraints; transpose-like relationships remain a future extension.
- Dtype work: mixed dtype constraints remain; backend-specific filtering and reference dtype promotion are supported by generation constraints.
- Value work: deterministic `uniform`, `normal`, and `sparsity` policies are supported, along with dtype-aware `integer_bounds` and scaled `float_bounds`; richer boundary sets remain a future extension.
- Optional omission/keyword binding and nested structures beyond first-level collections.
- Human-readable generated YAML for every run path, including load_job paths outside cx generate.

Design constraint: implement this as generator and constraint plugins over the current domain model. Do not reintroduce ATK's deep generator class hierarchy.

### 2. Hardware Gates As Product Claims

GPU and NPU claims should always have a command and artifact trail.

Implemented:

- examples/cases/torch.add.gpu.yaml is the GPU Torch smoke gate.
- cx run writes driver/resource_snapshot.json and driver/discovered_nodes.yaml for every run.
- resource_snapshot.json includes driver and Ray worker runtime probes for Python, torch, torch.version.cuda, CUDA availability, torch_npu, and selected device environment variables.
- cx doctor --ray-address prints the same worker runtime summary directly for interactive checks.
- GPU gate docs and tests keep executor: torch explicit, so Ray scheduling checks are not confused with GPU operator validation.
- Hardware validation claims require real devices: GPU and NPU on documented real hardware environments.

Remaining deliverables:

- Add a documented NPU smoke gate with exact image, CANN, torch, torch_npu, device availability, Ray resource, and result artifacts from the real NPU machine.

### 3. Backend Device Mapping Correctness

Ray reindexes visible GPUs inside num_gpus=1 actors. Current GPU E2E uses device id 0, which is safe. Multi-device correctness needs an explicit contract.

Implemented:

- Execution metrics now include candidate_executor for successful runs.
- Device executor paths now record resolved_device for torch/aclnn over CPU, GPU, NPU, ACLNN, and DCU-style device mappings.
- Ray actors now use actor-local device indexing, so num_gpus=1 Torch workers resolve GPU execution to cuda:0 while local execution keeps physical device IDs.
- The same actor-local mode maps NPU/ACLNN execution to npu:0 when enabled.

Remaining deliverables:

- Validate actor-local device indexing on multi-NPU hardware runs after rebuilding matching workers.
- Include resolved device evidence in hardware probe summaries and docs for each hardware gate.

## P1: High-Value ATK Coverage

### 4. Performance And Memory Metrics

Current performance and memory rows include backend-aware timing, allocator evidence, and an external vendor-profiler evidence path for NPU. Full profiler parity across all backends is still out of scope.

Implemented:

- Torch GPU/DCU and NPU/ACLNN performance and memory tasks synchronize the device queue around the timed region when the runtime exposes torch.cuda or torch.npu.
- Torch memory tasks reset peak allocator stats and record allocated, reserved, max allocated, max reserved, allocated delta, and peak memory metrics when available.
- CPU memory tasks keep process-level tracemalloc peak metrics.
- Performance cases support inherited `case.metadata.benchmark_policy` and local `invocation.metadata.benchmark` overrides with warmup, repeat, minimum duration, throughput items per call, mean/min/stddev/p50/p90/p95/p99 latency, effective duration, sample count, and throughput metrics.
- `torch.abs.npu.yaml` validates the benchmark contract on real Ascend NPU hardware: warmup=5, repeat=20, percentile/throughput metrics, synchronization, and allocator evidence are preserved under `cx_output/npu-benchmark-evidence`.
- Case profiler requests emit a standard `profiler` artifact with requested tool/status provenance.
- `cx import-msprof --source <trace-dir> --output <summary.json>` parses stable CANN CSV exports into a normalized summary with device count, top operators, task summaries, source files, and warnings.
- Real Ascend trace import is validated from `cx_output/npu-msprof-evidence/cx-msprof-e2e/msprof`: status=parsed, one device, one operator aggregate, 29 task rows, and no warnings.

Remaining deliverables:

- GPU performance/memory matrix passed for relu, softmax, and matmul on CPU/GPU through `scripts/operator_performance_gate.sh`; NPU matrix passed for the same operators on a validated NPU host through `scripts/npu_operator_performance_gate.sh`.
- NPU baseline is checked in under `baselines/npu-operator-performance/results.jsonl` with sanitized image/CANN provenance; `scripts/npu_operator_performance_regression_gate.sh` reruns the matrix and passed `cx performance-gate` using task-scoped latency/throughput/memory thresholds.
- Performance gate matching includes `case_id`, so generated cases with the same operator/backend/task do not overwrite each other; profiler requirements are explicit per policy.
- NPU memory metrics through CANN/NPU SMI beyond torch.npu allocator hooks.
- Optional vendor profiler artifact collection: PTA, Ascend profiler, DCU profiler.

### 5. Importers For Old Assets

Migration should be a translation boundary. Old assets should become CrucibleX facts, cases, campaigns, or plugin stubs.

Implemented:

- cx import-dump converts concrete dump inputs into a dump_replay case plus sibling inputs.json snapshot.
- cx import-profile converts observed dtype/shape samples into default-generator case policies and preserves raw sample sets in generation metadata.
- cx import-atb and cx import-temu convert backend configs into standard case YAML and preserve plugin skeleton hints in backend_import metadata.
- Imported cases include metadata.provenance with source path, source format, converter version, warnings, lossy fields, and snapshot paths when applicable.
- The dump_replay generator reuses captured inputs instead of sampling fresh values.

Remaining deliverables:

- ATB/Temu executor depth now has registered external-runtime adapters using the `cruciblex.external-runtime.v1` JSON stdin/stdout contract; missing runtime commands produce explicit unsupported evidence. SDK-specific kernel semantics remain pending until a real ATB/Temu runtime is available.

Priority order should now move to real backend executor depth and NPU validation once a separate NPU host is available.

### 6. ACLNN And NPU Depth

The torch_npu path is now validated on a single real NPU host without Ray. Bare ACLNN coverage is still a separate executor depth item.

Implemented:

- `torch.abs` through torch_npu lowering to CANN is validated on real host a validated NPU host with the validated NPU image.
- The NPU runtime probe confirmed `Ascend910B3`, `npu_count=8`, `torch=2.6.0+cpu`, `torch_npu=2.6.0.post5`, and a successful `torch.abs` kernel on `npu:0`.
- Local no-Ray gates passed for `run`, `accuracy`, `performance_device`, and `memory_device` using `examples/cases/torch.abs.npu.yaml` and `examples/nodes/local-npu.yaml`.
- Accuracy tasks can execute `oracle.reference_executor` when `oracle.metadata.execute_reference` is true, and compare candidate output against a real reference output in the same result.
- Bare ACLNN work now has a generic op bridge: op metadata resolves `Abs` to `aclnnAbs` and `aclnnAbsGetWorkspaceSize` without registering a one-off executor.
- The ACLNN bridge implements the first C ABI runner path: torch_npu tensor device pointers, scalar attributes, `aclCreateTensor`, `aclCreateScalar`, `GetWorkspaceSize`, workspace allocation, op launch, stream sync, cleanup, and output copy to numpy.
- `examples/cases/aclnn.abs.npu.yaml` passed real NPU accuracy against a `torch.abs` reference on a validated NPU host; `performance_device` and `memory_device` also passed.
- `examples/cases/aclnn.add.npu.yaml` passed real NPU accuracy against a `torch.add` reference through the same bridge by adding only signature metadata for the second tensor input and `alpha` scalar.

Remaining deliverables:

- ACLNN signature schema v1 now supports int/float/bool arrays, scalar aliases, optional attributes through binding omission, and multiple tensor outputs. Tensor lists and dynamic output counts remain explicitly unsupported until their C ABI ownership contract is defined.
- Add more ACLNN sample ops beyond `Abs` and `Add` to exercise array attributes, optional arguments, and non-unary signatures.
- Deeper CANN/NPU profiler artifacts beyond torch_npu allocator hooks.

### 7. Fuzz And Failure Reduction

Fuzz is present, but it needs stronger generation and reduction before it replaces old ATK fuzz workflows.

Deliverables:

- Facts-driven fuzz policy for per-parameter distributions and cross-parameter constraints.
- Failure reducer that emits minimized standalone case YAML, not only reduced rows in a repro bundle.
- Deduping by failure signature, operator, backend, dtype, shape, and exception family.
- Rerun scripts that target one generated case, one input artifact, or one failed backend pair.

## P2: Compatibility And Operational Breadth

### 8. Campaign And Batch Ergonomics

- `cx generate` now supports repeatable include/exclude selectors for operator, backend, task, dtype, and tag; shape filtering remains a planned extension.
- A repeatable `scripts/operator_breadth_gate.sh` runs the checked-in relu/softmax/matmul cases across CPU/GPU and enforces the combined coverage matrix.
- `cx campaign-coverage` aggregates all run output roots from `campaign_summary.json` and applies one policy across the batch. Each campaign row records plan_count, submitted_count, skipped_count, and resumed_from for auditable resume behavior.
- Sharding for large generated campaigns.
- Matrix summary by operator/backend/task/status.
- Stable output naming for CI and dashboard ingestion.

### 9. XRun, Excel, And Legacy Report Adapters

Do not rebuild these blindly. Add adapters only for files that users still exchange.

- Inventory real user artifacts first.
- Add one converter per confirmed workflow.
- Emit strict loss reports when old fields cannot map cleanly.

### 10. Server Or UI

Old ATK had service/database/UI-adjacent pieces. Rebuild only after CLI contracts settle.

- Keep result schema stable first.
- Add a read-only report viewer before any scheduling server.
- Avoid making a server own execution state until Ray and artifact persistence are mature.

## Acceptance Gates

A capability counts as covered only when all of these are true:

- There is a user-facing command or documented workflow.
- There is a checked-in example or template.
- There is a unit test for schema/planning behavior.
- Hardware claims have a real hardware E2E artifact trail.
- Results are persisted under the normal output contract.
- Failures can be rerun from a repro command or minimized case.

## Recommended Milestones

### Milestone 1: Generator Parity Foundation

Shape expressions, composite parameter kinds, rich dtype/value policies, generated YAML persistence, and one representative operator case that uses broadcasting, attrs, and list-like parameters.

### Milestone 2: Hardware Metrics Correctness

Torch CUDA timing synchronization, CUDA memory metrics, NPU timing/memory probes, resolved device metrics, and backend library versions in results.

### Milestone 3: Import Old ATK Assets

Start with cx import api and cx import dump, preserve provenance and lossy-field warnings, then add profile/ATB/Temu importers after schema stability.

### Milestone 4: ACLNN/NPU/DCU Depth

Real ACLNN multi-output hardware E2E is complete: `examples/cases/aclnn.sort.npu.yaml` calls CANN 8.3 `aclnnSortGetWorkspaceSize` and `aclnnSort` with host-side native bool/int attributes and `[2,4]` fp32/int64 outputs; `examples/cases/aclnn.max_dim.npu.yaml` validates `aclnnMaxDimGetWorkspaceSize` and `aclnnMaxDim` with reduced `[2,1]` fp32/int64 outputs. Both pass through `scripts/npu_aclnn_multi_output_gate.sh` on a validated NPU host. Array descriptor hardware E2E is also complete: `examples/cases/aclnn.mean.npu.yaml` calls CANN `aclnnMeanGetWorkspaceSize` and `aclnnMean` with `aclIntArray dim=[1]`, native `keepDim=false`, native fp32 dtype, and fp32 `[2]` output; it passes through `scripts/npu_aclnn_array_gate.sh`. DCU executor/runtime and backend-specific profiler artifacts remain future work.

## What Not To Copy From ATK

- Do not recreate one CLI command per internal conversion step if one cx import namespace can preserve intent.
- Do not mix backend environment setup into case generation.
- Do not make Ray workers compare cross-device results.
- Do not persist scheduler-specific state as the canonical result format.
- Do not require torch, CANN, ACLNN, or DCU packages as base project dependencies; keep them in images or plugins.

The target is ATK capability coverage with a smaller, stricter core: broad input generation, real hardware validation, clean importer boundaries, and reproducible artifacts.

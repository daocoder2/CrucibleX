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

The remaining work now splits into two parts: harden the ATK ideas already absorbed by CrucibleX, and fill the still-open deep gaps. The cleaner trunk is in place; the remaining work is selective extension, not broad reconstruction.

## ATK Core Module Import Assessment

ATK should be mined for product capability and backend knowledge, not copied as a runtime architecture. The useful migration unit is the behavior contract: generator policy, comparator policy, backend bridge semantics, importer loss reporting, and report fields. Avoid moving ATK's global state, dynamic CLI registration, Celery scheduler, DB ownership, or service runtime into the CrucibleX core.

### Current Stabilization Focus

Recent CrucibleX iterations have already absorbed several first-priority ATK ideas: deterministic matrix profiles, mixed tolerance and richer accuracy metrics, a precision policy contract, and bounded campaign matrix expansion. Treat these as CX-native contracts from this point forward. The next work should stabilize examples, docs, and acceptance tests around them before mining more ATK breadth.

The practical priority shift is:

- Move generation and comparison work from P0 import to contract stabilization: changes that affect `CaseSpec.generation.metadata`, operator facts, or generated artifact evidence stay P0; adding more built-in facts is P1 unless it enables a hardware gate.
- Replace the vague goal of "more facts" with a concrete backlog: reduce, topk/sort, index/mask, reshape/layout, matmul variants, conv, norm, attention, and ACLNN signature families each need named parameter constraints, output-shape evidence, invalid-case generation, and at least one checked-in CPU/GPU/NPU or ACLNN example where the backend is supported.
- Keep backend run modes, report field semantics, and old ATK asset importers as separate P1 tracks; traversal, XRun, Excel, server, and DB remain compatibility or product-layer decisions, not core architecture work.

### Rewrite Into CrucibleX Core Or Plugins

| ATK area | Why it matters | CrucibleX target | Priority |
| --- | --- | --- | --- |
| `atk/case_generator/generator/` | Rich dtype, shape, value, tensor/scalar/attribute, boundary, and multi-parameter input-space coverage is ATK's strongest durable asset | Re-express as generation constraints and generator plugins over `CaseSpec`, not as ATK's generator class hierarchy | P0 |
| `atk/case_generator/generator/data_types/matrix_profiles.py` | Numerical-risk profiles for matrix operators are more valuable than plain random tensors | Already landed in CX as `matrix_profile`; extend the profile library and operator coverage | P1 |
| `atk/case_generator/generator/processor.py` | Shape, size, numel, and cross-input restriction handling captures real operator constraints | CX already has the first declarative constraints; extend rank ranges, dimension aliases, product limits, broadcasting, and storage-shape policies | P1 |
| `atk/case_generator/traversal/` | Batch combination and traversal logic supports operator breadth and fuzz campaigns | CX has already absorbed the campaign matrix direction; keep only the bounded selection behavior that still adds value | P2 |
| `atk/tasks/post_process/` | Mixed tolerance, benchmark comparison, invalid-value checks, and error metrics improve accuracy decisions | Mixed tolerance and core accuracy metrics already exist in CX; keep extending comparator plugins and report postprocess metrics | P1 |
| `atk/tasks/run_modes/` | Ascend-specific run controls such as deterministic algorithms, L2 cache, and memory reuse affect reproducibility and performance evidence | Model as backend/runtime policy metadata on cases, nodes, or execution requests | P1 |
| `atk/tasks/backends/lib_interface/acl_wrapper.py` and ACLNN API helpers | Tensor/scalar/list descriptors, workspace lifecycle, formats, storage shape, and output ownership are hard-won backend knowledge | Use as a reference for `plugins/executors/aclnn_bridge.py` capability increments with hardware gates | P1 |
| `atk/tasks/report/report_title/` | The field inventory reflects user-facing accuracy, performance, memory, and summary expectations | Extract field semantics into CX JSONL/CSV/markdown exports; do not copy the title-factory implementation | P1 |

### Keep As Import Or Export Adapters

| ATK area | CrucibleX treatment |
| --- | --- |
| `api_to_yaml.py` | Add or extend an API/spec importer that produces operator facts and case YAML with provenance. |
| `dump_to_case.py` | Continue extending `cx import-dump` for concrete input snapshots and replay cases. |
| `profile_to_case.py` | Continue extending `cx import-profile` for observed dtype/shape/value samples. |
| `temu_to_case.py` and `migrate_atb_cases.py` | Keep translation at the importer boundary and preserve backend metadata for external-runtime execution. |
| `utils/xrun/` and Excel helpers | Add adapters only for confirmed user workflows, with strict lossy-field reports. |

### Do Not Reintroduce As Core Architecture

| ATK area | Reason |
| --- | --- |
| `atk/tasks/celery_*` | Ray and the local scheduler are the CrucibleX execution planes; adding Celery would split lifecycle ownership. |
| `atk/server/` and `atk/db/` | Service and database ownership should wait until CLI, artifact, and result contracts are stable; start with read-only report viewing if needed. |
| `atk/bin/*` dynamic command structure | Keep user workflows, but express them through stable `cx` command groups instead of one legacy command per conversion step. |
| `atk/common/output_manager.py` and global output state | CrucibleX already owns artifacts through run manifests, result stores, and driver-side persistence. |
| `atk/configs/*` as a full object model | Borrow field semantics, but keep CX's smaller Pydantic domain schema as the canonical contract. |

## Coverage Matrix

| ATK capability area | Current CrucibleX status | Target coverage | Priority |
| --- | --- | --- | --- |
| Run orchestration | Rebuilt with cx run, planning, local/Ray schedulers, result store, resume, and reports | Keep as the primary trunk; add only missing operational controls | P0 |
| Manifest campaign orchestration | Manifest schema, lane/case include expansion, runtime policy projection, filters, indexes, and report fields are implemented | Stabilize as the public batch entry for contract, hardware, and preflight evidence lanes | P0 |
| CPU/GPU operator validation | CPU and GPU Torch images are explicit; Ray GPU E2E is validated | Use Torch GPU case as the required gate for GPU operator claims | P0 |
| NPU validation | NPU image and smoke case exist; backend maps to Ray custom npu resource | Keep repeatable Ascend hardware gates and record torch/torch_npu/CANN versions | P0 |
| ACLNN validation | Executor imports a runtime module and maps to NPU resource | Build real ACLNN sample modules and hardware E2E gates | P1 |
| DCU validation | Backend and resource mapping exist | Add HIP/torch-dcu or vendor runtime executor and hardware gate | P2 |
| Case schema | Clean YAML/JSON case and node schema | Extend around constraints and metadata, not old class compatibility | P0 |
| Data generation | Deterministic default generator, count/seed, constraints, invalid cases, random/boundary coverage | Most of the core is now in CX; mine ATK only for remaining edge cases and richer coverage policies | P1 |
| Parameter relationships | Linked dtype/shape and simple random/boundary metadata exist | Add shape expressions, broadcasting, rank coupling, optional params, list and tensor-list helpers | P0 |
| Fuzz | Fuzz task, provenance, reports, and repro rows exist | Make fuzz policy expressive enough for operator facts and large campaigns | P1 |
| Performance | Latency rows exist | Add backend synchronization, warmup/repeat policy, percentiles, throughput, and vendor profiler hooks | P1 |
| Memory | Process memory rows exist | Add CUDA/NPU/DCU allocator metrics and peak reset/sync semantics | P1 |
| Reports | JSONL, CSV, summary, postprocess, markdown, failure clusters | Keep extending the established report semantics with backend matrices, trend-ready exports, operator coverage tables, and richer report titles | P1 |
| Repro | Repro bundles and per-plan rerun commands exist | Emit minimized case YAML and input snapshots for failing generated/fuzz cases | P1 |
| Operator onboarding | Facts template and scaffold generation exist | Promote facts-to-case automation and hardware promotion commands | P1 |
| API-to-YAML import | Not rebuilt | Translate API docs/specs into CrucibleX facts/cases | P1 |
| Dump/profile/Temu/ATB migration | Importers and external-runtime adapters exist for the main paths | Keep old assets behind import/export adapters with provenance and lossy-field reports | P1/P2 |
| XRun/Excel flows | Not rebuilt | Provide import/export adapters only if active users still depend on those files | P2 |
| Server/DB/UI | Not rebuilt | Do not reintroduce ATK server/DB as core; defer until CLI/batch contracts stabilize | P3 |

## P0: Must Cover Next

### 0. Manifest Public Contract

Manifest is now the top-level batch contract for P0/P1 work. It must remain an orchestration layer, not a replacement for case, operator fact, backend, or evidence schemas.

Implemented:

- `cx run --manifest`, `cx manifest validate`, and `cx manifest plan` load lanes, case includes, runtime policy, filters, and reporting options.
- Manifest lanes distinguish `contract`, `hardware`, and `preflight_blocked` evidence purposes.
- Lane and case include metadata are projected into run artifacts and stable report exports through `manifest_lane`, `manifest_lane_kind`, and `manifest_case_include`.
- Manifest hashes, include hashes, lane indexes, and case indexes provide an auditable link from batch entrypoint to expanded execution plans.

Current P0 boundary work:

- Freeze manifest schema v1 field semantics for `task`, `lanes`, `runtime`, `filters`, and `reporting`.
- Keep explicit lane backends authoritative over case-level inference.
- Keep hardware evidence requirements as reportable metadata and gate policy, not local smoke-test hard failures.
- Maintain `examples/manifests/operator-boundary-campaign.yaml` as the canonical contract/hardware/preflight campaign example.

### 1. Generation Core Parity

Old ATK's main strength was broad operator input-space coverage. CrucibleX now has the core generation mechanism in place: policy libraries, operator facts, contract expansion, invalid mutation, generated artifact persistence, and typed collection generation. Treat generation parity as established for the common tensor-operator families; the active work is boundary completion and hardware-backed acceptance.

Established coverage:

- Generated JSON and YAML are persisted for expanded cases.
- Deterministic expansion supports count, invalid_count, seeds, max_elements, max_bytes, boundary coverage, random coverage, linked dtype/shape metadata, and operator-fact merging.
- Default input generation supports tensor_list, tensor_tuple, scalar_list, scalar_tuple, attribute_list, and attribute_tuple with collection length, item_shapes, item_values, item_dtypes, explicit items metadata, and collection relationships.
- Reduce, sort/topk, index/mask, reshape/layout, and matmul/bmm families have legal expansion, representative invalid cases, output-shape or dtype evidence, and checked-in examples.
- Operator boundary campaign evidence separates CPU contract execution from GPU/NPU legal hardware lanes, so generated expected-invalid samples are not counted as device runtime rejection evidence.

P0 acceptance status:

- Manifest public contract is covered by validate, plan, run, lane/case index, include/hash provenance, and stable report projections. The checked-in operator-boundary CI smoke executes a no-Torch CPU manifest; the canonical Torch boundary campaign remains a source-baked Docker evidence gate.
- ACLNN capability decisions are executable and reportable. The checked-in preflight campaign verifies tensor-list/optional forms remain `future_abi`, while dynamic output, non-ND format, and declared strides remain `preflight_blocked`, before any native library call.
- A supported ACLNN capability needs matching resource lifecycle plus declared mock-lifecycle or NPU-E2E evidence; changing matrix status without that declaration fails the promotion gate.

Current P1 boundary work:

| Family | Current state | Remaining boundary |
| --- | --- | --- |
| conv2d | supported for legal NCHW/OIHW generation, output-shape contract, channel/groups invalid samples, and CPU/GPU/NPU legal evidence | broaden stride/padding/dilation/groups combinations and add more non-default bias/layout variants |
| layer_norm | supported for trailing normalized_shape, affine parameter contracts, invalid mismatch samples, and source-baked CPU/GPU/NPU legal evidence, including rank-4 two-axis `[4,5]` affine normalization | expand dtype/layout combinations and cover more rank variants |
| group_norm / instance_norm | supported for legal generation, channel/group or affine constraints, invalid mismatch samples, and source-baked CPU/GPU/NPU legal evidence through the four-case complex norm campaign, including `use_input_stats=false` with explicit running mean/variance | add more channel/group divisibility negatives and optional affine/running-stat combinations |
| scaled_dot_product_attention | supported for legal Q/K/V, causal, broadcast-mask, and GQA GPU/NPU evidence; dropout-range, mask-broadcast, and causal-plus-mask combinations execute as expected-invalid CPU contract cases | expand dtype/layout combinations and add new backend-specific capability lanes only when a reproducible runtime rejection exists |
| ACLNN signatures | partial: tensor/scalar/native arrays/static multi-output are supported or E2E-covered; unsupported forms are preflight-blocked or future_abi | tensor-list ownership, optional tensor/list ABI, dynamic output counts, non-ND format, and stride/storage-offset ABI remain blocked |

Acceptance stays strict: every promoted family needs one policy-library or fact entry, legal expansion test, invalid expansion test when constrained, checked-in example YAML, and hardware evidence when making CPU/GPU/NPU/ACLNN claims. ACLNN facts remain tied to ABI capability decisions instead of generation assumptions.

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
- Operator boundary campaign now has separated CPU contract, GPU legal-evidence, and NPU legal-evidence lanes with archived manifest/results/report/summary/postprocess artifacts.
- `examples/manifests/torch-indexing-matrix-evidence.yaml` has source-baked GPU and NPU evidence for fixed legal topk, gather, scatter, reshape, and bmm: each lane passed 5/5 with device-tensor dtype evidence and a complete manifest/results/report/summary/postprocess archive. ACLNN evidence is documented separately for ACLNN executor lanes.
- Fixed-input three-side add evidence records CPU/GPU/NPU comparability under a shared case fingerprint.

Remaining deliverables:

- `examples/manifests/complex-operator-evidence.yaml` is the reusable legal-only GPU/NPU entrypoint for baseline and grouped/dilated conv2d, layer_norm, masked attention, causal attention, broadcast-mask attention, and grouped-query attention. Its source-baked GPU and NPU runs each passed 7/7 with device-tensor dtype evidence and complete manifest/results/report/summary/postprocess archives.
- `examples/manifests/attention-gqa-capability-probe.yaml` validates `scaled_dot_product_attention(enable_gqa=True)` on legal grouped-query shapes; its source-baked GPU and NPU probes each passed 1/1 with device-tensor dtype evidence and complete archives.
- Expand hardware evidence matrices for conv/norm/attention and ACLNN non-unary signatures beyond the current legal-lane samples.
- Validate actor-local device indexing on multi-NPU hardware after rebuilding matching workers; single-device NPU evidence does not prove multi-device placement.

### 3. Backend Device Mapping Correctness

Ray reindexes visible GPUs inside num_gpus=1 actors. Current GPU E2E uses device id 0, which is safe. Multi-device correctness needs an explicit contract.

Implemented:

- Execution metrics now include candidate_executor for successful runs.
- Device executor paths now record resolved_device for torch/aclnn over CPU, GPU, NPU, ACLNN, and DCU-style device mappings.
- Ray actors now use actor-local device indexing, so num_gpus=1 Torch workers resolve GPU execution to cuda:0 while local execution keeps physical device IDs.
- The same actor-local mode maps NPU/ACLNN execution to npu:0 when enabled.

Remaining deliverables:

- Include resolved-device evidence in hardware probe summaries, report export rows, and each hardware gate archive.
- Record multi-device placement assumptions next to each gate result: local physical device IDs versus Ray actor-local device IDs.

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

The torch_npu path is now validated on a single real NPU host without Ray. Bare ACLNN coverage now has a bridge, capability matrix, and multiple real E2E samples; the remaining depth is tensor-list ownership, dynamic output counts, non-ND format, stride/storage-offset ABI, and deeper profiler artifacts.

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
- `examples/manifests/aclnn-supported-evidence.yaml` is the reusable NPU evidence entrypoint for Abs, Add, Mean (int array), MaxDim, and Sort (static multi-output). Its source-baked NPU run passed 5/5 with device-tensor dtype evidence and a complete manifest/results/report/summary/postprocess archive.
- Add more ACLNN sample ops beyond `Abs` and `Add` to exercise float/bool array attributes, optional arguments, and non-unary signatures.
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

## Attention Runtime Capability Boundary

A `preflight_blocked` attention lane is reserved for a documented backend capability, not an invalid PyTorch call. A future declaration must identify the backend, the feature (for example a kernel-specific mask, causal, or dropout mode), a stable rejection reason, and either a runtime probe or source-baked device evidence. Until such proof exists, dropout-range, causal-plus-mask, and mask-broadcast failures remain operator contract invalid cases and must not be promoted to a runtime capability verdict.


## What Not To Copy From ATK

- Do not recreate one CLI command per internal conversion step if one cx import namespace can preserve intent.
- Do not mix backend environment setup into case generation.
- Do not make Ray workers compare cross-device results.
- Do not persist scheduler-specific state as the canonical result format.
- Do not require torch, CANN, ACLNN, or DCU packages as base project dependencies; keep them in images or plugins.

The target is ATK capability coverage with a smaller, stricter core: broad input generation, real hardware validation, clean importer boundaries, and reproducible artifacts.

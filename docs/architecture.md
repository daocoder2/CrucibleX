# CrucibleX Architecture

See [evaluation-principles.md](evaluation-principles.md) for the product-level principles and evolution constraints.

CrucibleX is a clean-slate operator testing toolkit. It keeps the useful ATK lessons, but it does not copy ATK data objects or orchestration shape.

## First Principles

CrucibleX models five questions explicitly:

1. What is being tested: `OperatorSpec`, `CaseSpec`, and `ParameterSpec`.
2. Where it can run: `NodeSpec` and `DeviceSpec`.
3. What this job asks for: `JobSpec` and `TaskKind`.
4. What will actually run: `ExecutionPlan`.
5. What happened: `ExecutionResult` and `ArtifactRef`.

## Core Boundaries

- `domain` defines stable entities and must stay independent from Ray, torch, CANN, report formats, and storage engines.
- `generation` owns case design loading, deterministic input materialization, and declarative shape relationships; see [shape-relationships.md](shape-relationships.md).
- `runtime` owns planning, scheduling, device actors, and task executors.
- `storage` persists input/output artifacts, versioned input provenance, result state, resume metadata, and evidence references; see [runtime-provenance.md](runtime-provenance.md).
- `report` converts execution results into summaries and export files.
- `plugins` provides controlled extension points for API execution, data generation, comparison, and run modes.

## Lessons Kept From ATK

- Case generation and execution remain separate.
- Backends, data generators, API executors, and accuracy comparators remain pluggable.
- Inputs, outputs, reports, and resume state remain explicit artifacts instead of transient scheduler objects.
- Device execution is treated as a resource boundary.

## Deliberate Changes From ATK

- Nodes describe resources and environment; jobs describe requested tasks.
- Task support on a node is expressed as `allowed_tasks`, not as the job execution plan.
- Ray is the primary execution and placement layer. It owns device actor lifecycle, placement, and run fan-out; the local scheduler is only a fallback path for lightweight debugging.
- Reports and persistent state do not depend on the scheduler implementation.
- The Ray scheduler is the default execution path. The local scheduler remains available as a narrow debug and smoke fallback behind `--scheduler local`.

## Current Minimal Flow

1. Load cases and nodes from YAML or JSON.
2. Build an execution plan matrix from cases, tasks, nodes, and devices.
3. Generate deterministic input artifacts for each plan.
4. Execute a built-in smoke operator implementation.
5. Compare outputs for accuracy tasks and return structured results.
6. Persist inputs and outputs as JSON artifacts.

The built-in executor is intentionally small. It is only the first smoke path for the architecture; real torch, ACLNN, NPU, and custom plugin execution should enter through executor interfaces instead of changing the domain model.
## Executor Boundary

Operator execution is behind `BackendExecutor` and receives an `ExecutionRequest`. Built-in smoke executors live in `runtime/executors`, while user extensions are loaded explicitly through `--plugin`.

Plugins register normal Python classes with `EXECUTOR_REGISTRY.register(name)`. CrucibleX intentionally avoids implicit repository-wide scans and decorator parsing so plugin loading stays predictable and debuggable.

`InvocationSpec.executor` selects the candidate executor. `OracleSpec.reference_executor` selects the reference executor. They may be the same for smoke tests, but real accuracy testing should use an independent reference path. Executors can inspect the execution request, including the plan, device context, and execution role, without changing domain objects.
## Backend Runtime Boundary

`BackendRuntime` owns device lifecycle, not operator semantics. It prepares a `DeviceContext`, applies backend-specific environment decisions, and cleans up after the pipeline finishes.

`DeviceActor` is the boundary where runtime placement meets execution. It resolves the backend runtime from the plan device, prepares context, runs the pipeline, and performs cleanup.

Ray resource mapping is isolated in `runtime/backends/resources.py`. GPU maps to `num_gpus=1`, NPU maps to a custom `{"npu": 1.0}` resource, and CPU maps to `num_cpus`. The domain model does not depend on Ray.
## Ray Execution Strategy

Ray actors are keyed by host, backend, and device id within a single run. This keeps one actor per device slot during the run and avoids mixing devices inside a single worker. Actors are not detached by default; each CLI run owns its worker lifecycle so code and plugin state do not leak across runs.

Actor construction uses `ray_resources_for(device)` to map device type to runtime resources. The mapping is intentionally thin and lives outside the domain layer. User plugin paths are passed from the driver to each Ray actor and loaded during actor initialization.

Ray placement is explicit. `NodeSpec.host` matches a Ray node by node manager address or hostname; localhost can match the only alive Ray node for single-machine development. Actor options include backend resources plus Ray node affinity when the discovered node exposes a `node:<node-id>` resource. Missing nodes or insufficient resources become scheduler-level `SKIPPED` results. Ray initialization, actor creation, and result collection failures become scheduler-level `ERROR` results.
## Built-in Executor Plugins

Built-in executors live under `plugins/executors`, not under `runtime/executors`. Runtime owns the interface; plugins own concrete operator calls. The candidate/reference role is part of the execution request and surfaces in result artifacts and metadata.

- `numpy` is the required smoke executor and provides deterministic local coverage for early architecture tests.
- `torch` is an optional executor. It is importable without making torch a project dependency; actual execution returns `SKIPPED` when torch is unavailable.

A case selects the candidate path through `invocation.executor` and the reference path through `oracle.reference_executor`.
## Torch Device Policy

The torch executor delegates device selection to `DevicePolicy`. This keeps tensor conversion and API invocation separate from CPU/GPU/NPU placement rules.

`DefaultDevicePolicy` maps CPU to `cpu`, GPU to `cuda:<device_id>`, and NPU to `npu:<device_id>`. Custom torch/NPU behavior should extend the policy or provide a new executor plugin instead of adding branches to the execution pipeline.
## Comparator Plugins

Accuracy comparison is also plugin-driven. Runtime owns the `Comparator` and `ComparisonRequest` interfaces, while concrete comparison policies live under `plugins/comparators` or user plugin files.

`OracleSpec.comparison` resolves through `COMPARATOR_REGISTRY`. The built-in `allclose` comparator is loaded as a required built-in plugin. Custom comparators can return domain-level pass/fail decisions and metrics without changing the execution pipeline.
## Generator Plugins

Input generation is plugin-driven because it is a policy decision. Runtime owns `InputGenerator`, `GenerationRequest`, and `GENERATOR_REGISTRY`; built-in and user generators provide concrete input materialization.

`CaseSpec.generator` resolves through `GENERATOR_REGISTRY`. The built-in `default` generator provides deterministic tensor/scalar/attribute values. Fuzzing, boundary-value generation, seed replay, and recorded sample replay should be implemented as generator plugins without changing the pipeline.

## Plugin Boundary

CrucibleX does not treat every component as a plugin. Plugins are reserved for volatile policies: input generation, candidate/reference execution, comparison, and backend-specific device policies. Stable lifecycle and orchestration remain in core runtime: domain models, loading, planning, scheduling, device actor lifecycle, artifact storage, result reporting, and run-level result persistence.
## Logging

CrucibleX uses the `cruciblex` logger namespace for execution tracing. Core runtime emits lifecycle logs for run start, plan build, submission, device preparation, generation, execution, comparison, cleanup, and run completion.

Logging stays outside plugin contract boundaries. Plugins should expose stable behavior and return structured results; the core owns the trace of what ran, where it ran, and how it completed.
## Exception Oracle

`OracleSpec.expected_error` is evaluated on the candidate execution path. If the candidate raises an exception containing the expected text, the result is `PASSED`. If the candidate raises a different exception or returns successfully, the result is `FAILED`.

Infrastructure-level unsupported execution remains `SKIPPED`; it is not treated as an expected operator failure. This keeps negative operator cases separate from missing executor/backend capability.
## Planning Shape

`ExecutionPlanner` now builds execution slots first and then combines cases with slots. That keeps the logic flatter, makes filtering explicit, and leaves room for future slot policies without deep nested loops.

## Task Semantics

Task meaning is a driver-level contract, not an executor detail. Every device backend may run the same task independently, but the post-processing stage decides how to interpret the results.

| Task family | Execution shape | Driver post-process | Notes |
| --- | --- | --- | --- |
| `accuracy`, `accuracy_load`, `accuracy_dc` | Run candidate and reference plans independently, usually across different backend/device slots | Compare candidate output against the chosen reference backend | This is the only family that should emit cross-device comparison artifacts by default |
| `performance_device`, `performance_device_pta`, `performance_e2e`, `performance_benchmark` | Run each backend/device independently | Aggregate latency, throughput, and benchmark summaries | No numerical compare unless a specific operator policy requires it |
| `memory_device` | Run each backend/device independently | Aggregate memory allocation and reservation metrics | The output is a report, not a compare decision |
| `run` | Run each backend/device independently | Persist outputs and status only | Suitable for smoke, integration, and capability checks |
| `fuzz` | Run selected cases independently or by sampling policy | Cluster failures, retain repro artifacts, and summarize coverage | Compare is not the primary output |

Backend selection is orthogonal to task family. CPU, GPU, NPU, DCU, ACLNN, and other future backends are all just independent execution targets. The scheduler fans them out, and the driver decides which post-process applies. CPU and GPU are the current required E2E validation targets; NPU, DCU, and ACLNN follow the same contract when matching resources are available. GPU maps to Ray `GPU`, NPU maps to custom `npu`, DCU maps to custom `dcu`, and ACLNN defaults to the same `npu` resource because it executes on Ascend/NPU hardware.

`ResultPostProcessor` is the single driver-side entry point for result interpretation. It receives collected execution results after scheduler collection and may append derived results, such as accuracy comparison rows. It also writes the run-level `postprocess.json` summary for performance, memory, and comparison views. The CLI should not contain task-specific post-processing branches.

## Result Persistence

`ResultStore` writes `results.jsonl` and `summary.json` at the run output root. `ResultPostProcessor` writes `postprocess.json` beside them when driver-side post-processing runs. The CLI emits these file paths after a run so local and Ray executions share the same persisted result contract.

Artifacts follow the long-term driver-owned persistence model. Local execution may materialize artifacts directly, but Ray workers return structured `ArtifactPayload` values and the driver converts them into final `ArtifactRef` files under the absolute run output root during result collection. This keeps multi-node execution from depending on each worker local filesystem layout.
## Run Context And Manifest

A run is the whole `cx run` batch, not a single case or execution plan. `RunContext` is the in-memory runtime contract passed across CLI, loading, scheduling, and result collection. It carries run id, case path, node path, tasks, scheduler, plugin paths, output root, and runtime metadata.

`RunManifest` is the persistent projection of that context plus execution counts and output files. The CLI writes `manifest.json` before execution with the planned run context and updates it after execution with `results.jsonl` and `summary.json`. Resume and retry should build on this contract by reloading the manifest, rebuilding plans, and comparing them with loaded results.
## Resume And Retry

`cx run --resume-from <output-root-or-manifest>` loads prior `results.jsonl` and builds a merged run view. Plans with historical terminal results are skipped by default; `--retry-failed` resubmits failed, errored, timed out, and cancelled plans. The final `results.jsonl` remains a complete ordered run view, combining reused historical results with newly executed results.

The default output root is `cx_output`. Run-level files live directly under that root: `manifest.json`, `results.jsonl`, and `summary.json`.
CLI resume behavior is covered end to end: a resumed run writes a new manifest, merged `results.jsonl`, and summary under the selected output root. The previous output root is treated as input state, not modified in place.
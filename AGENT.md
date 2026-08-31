# Agent Guide for CrucibleX

This guide tells coding agents how to work in this repository. Keep it focused on project understanding, behavior constraints, development conventions, operating commands, and verification standards. Environment-specific validation evidence belongs in local ignored notes such as `docker/private-notes.md`, not here.

## Project Understanding

CrucibleX is a clean-slate, Ray-first operator testing toolkit for case generation, device scheduling, execution, comparison, result persistence, and report delivery. It is the successor to the previous ATK codebase, but it should not recreate ATK's legacy object model or compatibility layers unless explicitly required.

Core modules:

- `src/cruciblex/domain/`: stable Pydantic entities, enums, plans, results, nodes, and run metadata.
- `src/cruciblex/generation/`: case loading, case expansion, filtering, and deterministic input generation support.
- `src/cruciblex/runtime/`: planning, scheduling, device actors, executor interfaces, backend runtime, logging, and pipeline orchestration.
- `src/cruciblex/storage/`: artifact recording, run manifests, result JSONL/CSV output, summaries, and resume state.
- `src/cruciblex/report/`: markdown reports, postprocess summaries, coverage, performance gates, repro bundles, and reduction helpers.
- `src/cruciblex/plugins/`: volatile policy extensions for executors, generators, comparators, and backend-specific behavior.
- `examples/`: reusable case, node, import, campaign, and onboarding templates.
- `docs/` and `docker/`: user-facing CLI, architecture, rollout, deployment, and hardware validation documentation.
- `tests/`: unit and integration coverage for CLI, generation, runtime, reporting, importers, and hardware contracts.

Architecture rules:

- Keep `domain` independent from Ray, torch, CANN, storage engines, report formats, and plugin implementations.
- Keep scheduler behavior, artifact persistence, and report rendering independent from each other.
- Keep task meaning as a driver-level contract. Executors run operator calls; postprocessing interprets task-specific results.
- Ray is the primary scheduler and placement layer. The local scheduler is a narrow fallback for smoke tests and debugging.

## Behavior Constraints

Agents may:

- Modify source, tests, examples, docs, and scripts when the user asks for implementation work.
- Add focused abstractions when they reduce real complexity or match an existing repository pattern.
- Add or update tests when behavior changes.
- Update docs when public commands, architecture boundaries, deployment steps, or validation procedures change.
- Use `cx_output/`, `cx_output_resume/`, and `out/` for local runs, but do not commit those generated artifacts.

Agents must not:

- Reintroduce legacy ATK packages or compatibility layers without explicit scope change.
- Move stable orchestration into plugins. Plugins are for policy points, not core lifecycle ownership.
- Make local scheduling the implicit default again.
- Mix generated artifacts with source fixtures or checked-in examples.
- Claim real hardware correctness for GPU/NPU/DCU/ACLNN/ATB/Temu paths without matching evidence in docs or tests.
- Commit environment-specific host credentials, private access details, or transient execution logs in this file.
- Revert unrelated user changes while trying to clean up the worktree.

## 开发语言约定

- 新增或实质性改写的文档、代码注释、示例说明、变更说明和提交信息以中文为主。
- 代码标识符、CLI 参数、协议字段、产品名、第三方 API 名称和必要的专业术语保持英文，避免破坏接口可读性。
- 历史英文文档不因语言约定单独翻译；修改时按变更范围逐步中文化。
- Git 提交使用 Conventional Commit 格式，说明部分使用中文，例如 `feat(runtime): 增加结果契约版本标识`。

## Development Standards

Code style:

- Use Python 3.11 syntax and project-managed dependencies from `pyproject.toml` and `uv.lock`.
- Use absolute imports inside `src/cruciblex`; do not add relative imports.
- Favor explicit data flow, typed Pydantic models, small functions, and narrow interfaces.
- Prefer structured result objects and metrics over ad hoc strings.
- Keep deterministic behavior where generation, selection, resume, and reporting depend on stable output.
- Keep logging under the `cruciblex` logger namespace and treat run tracing as part of runtime design.

Design rules:

- Preserve module boundaries before adding new abstractions.
- Use `RunContext` for run-level in-memory state and `RunManifest` for persisted run metadata.
- Keep Ray artifact persistence driver-owned: workers return artifact payloads and the driver writes final artifact files.
- Map scheduler and infrastructure failures into structured `ExecutionResult` statuses instead of aborting a whole batch when possible.
- Add new plugins only when behavior is truly policy-driven, such as input generation, operator execution, comparison, or backend-specific device policy.

Naming and files:

- Keep the public CLI command as `cx`.
- Keep the default output root as `cx_output`.
- Use clear filenames that match the feature area, for example `report/performance_gate.py` or `runtime/scheduler/ray.py`.
- Keep source code, test fixtures, checked-in baselines, and generated run outputs clearly separated.

## Operating Flow

Install dependencies:

```bash
uv sync
```

Inspect the CLI:

```bash
uv run cx --help
uv run cx doctor
```

Run a local smoke case:

```bash
uv run cx run --case examples/cases/torch.abs.yaml --nodes examples/nodes/local.yaml --scheduler local
```

Generate cases without executing them:

```bash
uv run cx generate --case examples/cases/torch.abs.generated.yaml --output cx_output/generated-smoke
```

Render a report from an existing output directory:

```bash
uv run cx report --output cx_output/local-run
```

Run the full test suite:

```bash
uv run --extra dev pytest
```

Run lint when changing style-sensitive code:

```bash
uv run --extra dev ruff check src tests
```

Use Ray only when the environment is prepared and reachable:

```bash
uv run cx run --case <case.yaml> --nodes <nodes.yaml> --scheduler ray --ray-address <ray-address>
```

## Verification Standards

Before finishing a change, verify the smallest surface that proves the change works:

- For domain/model changes, run the focused model/loader/planner tests plus any serialization tests.
- For CLI changes, run the relevant `CliRunner` tests and at least one representative CLI smoke command when practical.
- For runtime pipeline or scheduler changes, run local scheduler tests and Ray-specific tests when the environment supports Ray.
- For generation changes, verify deterministic output and generated case persistence.
- For report, persistence, coverage, gate, repro, or reduction changes, inspect the generated JSON/CSV/markdown shape and run corresponding tests.
- For plugin changes, test registration, missing-dependency behavior, unsupported execution paths, and one successful smoke path when possible.
- For hardware-specific behavior, document the validated environment and command in `docker/README.md` or a dedicated doc, and keep hardware tests marked appropriately.

Minimum general check before handing off broad changes:

```bash
uv run --extra dev pytest
```

If a meaningful verification step cannot be run locally, state exactly what was not run and why.

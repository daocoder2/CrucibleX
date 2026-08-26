# Agent Guide for CrucibleX

This repository is a clean-slate, Ray-first operator testing toolkit. Future work should strengthen the long-term trunk, not recreate ATK shape or add transitional compatibility layers unless explicitly required.

## Working Rules

- Prefer first-principles design over copying prior ATK code.
- Keep the core domain independent from Ray, torch, storage, and report formats.
- Use absolute imports only inside `src/cruciblex`; do not add relative imports.
- Keep the public CLI command as `cx`.
- Keep the default output root as `cx_output`.
- Treat logging as part of the system design, not an afterthought.
- Make changes that improve the long-term trunk, even if they require removing earlier shortcuts.

## Architecture Boundaries

- `domain` holds stable entities, enums, plans, results, and run metadata.
- `generation` loads cases and materializes inputs.
- `runtime` owns planning, scheduling, device actors, executors, and lifecycle orchestration.
- `storage` persists artifacts, results, summaries, and resume state.
- `report` produces human-readable run output.
- `plugins` is for volatile policy points such as generators, executors, and comparators.

Do not move stable orchestration into plugins. Keep runtime behavior explicit and inspectable.

## Scheduler Policy

- Ray is the primary scheduler and placement layer.
- Keep Ray placement explicit: match configured nodes to discovered Ray nodes before creating actors.
- Local scheduling is a narrow fallback for debugging and smoke tests.
- Use `--scheduler local` in tests or quick local checks when Ray is unnecessary.
- Do not make the local path the implicit default again.

## Coding Standards

- Favor explicit types and explicit data flow.
- Keep file-level scope small and boundaries clear.
- Avoid broad refactors unrelated to the task at hand.
- Prefer structured result objects over ad hoc strings.
- Preserve deterministic behavior where the system already expects it.
- Keep generated artifacts, test fixtures, and source code separate.

## Verification Expectations

Before finishing a change, verify the relevant surface:

- Run unit tests for the area you touched.
- Run the CLI path if the change affects end-to-end behavior.
- Check the generated output when changing reports or persistence.
- Confirm imports remain absolute and the package still loads.

Typical commands:

```bash
uv run --extra dev pytest
uv run cx --help
uv run cx run --case examples/cases/torch.abs.yaml --nodes examples/nodes/local.yaml --scheduler local
```

## Extension Rules

- Add new plugins only when the behavior is truly policy-driven.
- Add new core abstractions only when they reduce real complexity.
- Keep executor, generator, and comparator contracts narrow.
- Keep device policy separate from operator semantics.
- Pass run-level state through `RunContext`; keep `RunManifest` as the persisted projection.
- Keep result persistence and report rendering independent from scheduling.
- Keep Ray artifact persistence driver-owned: workers return payloads, the driver writes final artifact files.
- Map scheduler failures into structured `ExecutionResult` statuses instead of aborting the run.

## Documentation Rules

- Update this file when the development rules change.
- Update `docs/architecture.md` when the system boundary changes.
- Keep README-level messaging aligned with the actual implementation.

## Practical Default

When in doubt:

1. Read the surrounding code and tests first.
2. Preserve the existing boundary if it still makes sense.
3. Make the smallest change that advances the trunk.
4. Verify it with the smallest meaningful test or CLI run.
5. Write down any new rule here if it should govern future work.

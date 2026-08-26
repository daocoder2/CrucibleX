# CrucibleX

CrucibleX is a Ray-first operator testing toolkit for case generation, device scheduling, execution, comparison, and report delivery. It is the clean-slate successor to the previous ATK codebase.

## Layout

- `src/cruciblex/domain/` core entities and enums
- `src/cruciblex/generation/` case and input generation
- `src/cruciblex/runtime/` Ray execution plane, device actors, and local smoke scheduler
- `src/cruciblex/storage/` artifacts, results, and resume state
- `src/cruciblex/report/` summary and export helpers
- `src/cruciblex/plugins/` generator, executor, and comparator plugins
- `tests/` unit and integration tests
- `examples/` minimal node and case examples

## Development

Install and run with uv:

```bash
uv sync
uv run cx --help
uv run cx doctor
uv run cx run --case examples/cases/torch.abs.yaml --nodes examples/nodes/local.yaml
```

Ray is the default execution path. Use `--scheduler local` only for lightweight debugging or smoke tests that should avoid starting Ray.

Run tests with:

```bash
uv run --extra dev pytest
```

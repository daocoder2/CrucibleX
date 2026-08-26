# CrucibleX

CrucibleX（坩埚）是一个 Ray-first 算子测试与验证工具包。它以“熔炼”和“解控”为核心隐喻：一方面把用例生成、设备调度、候选/参考执行、精度对比、结果持久化与报告生成串成可复现的验证流程，持续淬炼 kernel；另一方面解除算子验证中常见的环境绑定、后端差异、调度不透明和结果分散问题，让开发者更快定位缺陷，重新掌控验证过程。

> Forge your kernels, burn the bugs.
>
> 千锤百炼，方见真章.

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

# Manifest 设计与实现说明

本文档描述一个顶层任务清单格式，用来把 CrucibleX 里分散的 case、lane、backend、contract 和 evidence 组织成一次完整评测任务。首版实现支持 `cx run --manifest`、lane/case 展开、lane 级显式 backend 过滤、runtime 展开策略、case filters，以及 lane/case 索引输出。

## 背景

当前仓库把职责拆在多个 YAML 中：

- 单个 operator 的 case 文件，描述输入、binding、oracle 和生成约束。
- generated contract 文件，描述 declarative contract 和 invalid 变体。
- hardware evidence 文件，描述真实 CPU/GPU/NPU/ACLNN 执行样本。
- 文档中的 matrix，只负责索引和边界说明。

这种拆分的优点是边界清楚、复用高、审计容易。缺点是一次完整任务的入口分散，调用方需要自己拼装 case、backend 与运行选项。

## 目标

设计一个 manifest 文件作为“任务入口”，它只负责编排，不承载单个算子的细节。

它应该能够：

- 引用已有 case 文件，而不是复制 case 内容。
- 定义一次任务里要跑哪些 lane。
- 为 lane 指定 backend、筛选条件和默认执行参数。
- 区分 contract、hardware 和 preflight_blocked 三类 lane；generated 是 case 展开来源或样本属性，不作为 lane kind。
- 保持现有 case 文件的单一职责不变。

它不应该：

- 重复表达 shape relationship、dtype policy、operator contract 等 case 细节。
- 把真实硬件 evidence 和声明式 contract 混在同一层里。
- 替代已有 case 文件、oracle 文件或 hardware 证据文件。

## V1 Public Contract

Manifest v1 freezes the top-level fields in this order: `version`, `kind`, `task`, `lanes`, `runtime`, `filters`, and `reporting`. Each section forbids unknown fields. The only lane kinds are `contract`, `hardware`, and `preflight_blocked`.

`cx manifest validate --json` freezes its top-level keys as `manifest`, `task`, `lanes`, `cases`, and `lane_kinds`. Its `cases` count is the selected source-case count before generation expansion. `cx manifest plan --json` freezes its top-level keys as `manifest`, `task`, `cases`, `plans`, and `items`; its `cases` count is the expanded case count. Each plan item freezes `plan_id`, `lane`, `lane_kind`, `case_id`, `case_name`, `backend`, `node`, `device_id`, and `task`. Future additive or breaking changes require a new manifest/report schema version rather than silently changing v1 output.

Every hardware evidence run archives `manifest.json`, `results.jsonl`, `report.jsonl`, `summary.json`, and `postprocess.json`. A plan or local validation does not itself establish a hardware claim; the archive must come from a source-baked device run.


## 建议结构

```yaml
version: 1
kind: manifest

task:
  name: operator-contract-suite
  description: 高级 operator contract 回归与证据收集
  defaults:
    oracle: torch
    generator: default
    tolerance:
      atol: 1.0e-5
      rtol: 1.0e-5

lanes:
  - name: cpu-gpu-legal
    kind: hardware
    backends: [cpu, gpu]
    cases:
      - include: examples/cases/torch.mean.hardware.yaml
      - include: examples/cases/torch.matmul.hardware.yaml
      - include: examples/cases/torch.group-norm.generated.yaml

  - name: npu-legal
    kind: hardware
    backends: [npu]
    cases:
      - include: examples/cases/aclnn.mean.npu.yaml
      - include: examples/cases/torch.gather.npu.yaml
      - include: examples/cases/torch.scatter.npu.yaml

  - name: contract-invalid
    kind: contract
    backends: [cpu]
    cases:
      - include: examples/cases/torch.gather-3d.generated.yaml
      - include: examples/cases/torch.group-norm.generated.yaml

runtime:
  allow_generated_cases: true
  allow_invalid_cases: true
  require_real_evidence: true
  require_backend_dtype_source: device_tensor

filters:
  include_tags: [reduce, matmul, norm, gather, scatter, attention]
  exclude_tags: [acl_runtime_blocker]

reporting:
  output_dir: .artifacts/contracts
  emit_case_index: true
  emit_lane_index: true
```

## 语义分层

### 1. task

`task` 是人类可读的任务描述，只放：

- 名称
- 目标
- 默认 oracle / tolerance / generator

它不参与算子契约计算。

### 2. lanes

`lanes` 是调度单元。每个 lane 描述一类执行目的，例如：

- `hardware`
- `contract`
- `preflight_blocked`

`backends` 是可选字段，但如果 lane 的目的就是跑某些后端，应该显式写出来。对 hardware / evidence lane，推荐把 backend 作为显式约束；当 `backends` 省略时，运行器可以从被引用 case、case tag 或默认策略推断。

lane 里可以指定多个 backend，也可以继续细分为 CPU/GPU/NPU/ACLNN 子 lane。显式声明的 `backends` 优先于推断结果。

### 3. cases

`cases` 只做引用：

- `include` 引用已有 YAML。
- 不在 manifest 中复制 case 内容。
- 运行器在展开时读取 case 再合并 lane defaults。

这能保证 case 文件仍然是单一事实来源。

### 4. runtime

`runtime` 用来声明任务级别开关：

- `allow_generated_cases`：为 `false` 时把每个 case 的生成数量收敛到 1。
- `allow_invalid_cases`：为 `false` 时清空 invalid 展开数量。
- `require_real_evidence`：写入 case metadata，供后续 evidence gate / report 使用。
- `require_backend_dtype_source`：写入 case metadata，供后续 evidence gate / report 使用。

这些是任务策略，不是单个 case 的事实。首版实现已经让 generated / invalid 展开策略生效；evidence 要求先作为 metadata 透传，不在本地 smoke 阶段硬拦。

### 4.1 backend 解析规则

Manifest schema 会拒绝重复的 lane 名、空 lane、空 case include、未知字段和未知 lane kind。lane kind 的 public contract 固定为 `contract`、`hardware`、`preflight_blocked`，避免配置在进入执行阶段后才失败。

`backends` 保持可选，但解析语义应遵循以下优先级：

1. lane 显式声明的 `backends`。
2. case 文件自身的 backend / executor / tag 信息。
3. manifest 的默认策略。

如果 lane 已显式声明 `backends`，运行器不得再用 case 推断去覆盖它；推断只作为缺省补全。

### 5. filters

`filters` 负责从引用集中筛选执行范围。首版复用现有 case selector，支持：

- `include_operators` / `exclude_operators`
- `include_backends` / `exclude_backends`
- `include_tasks` / `exclude_tasks`
- `include_dtypes` / `exclude_dtypes`
- `include_tags` / `exclude_tags`

### 6. reporting

`reporting` 只管结果落点和索引开关，不影响执行语义。首版支持：

- `output_dir`：manifest 索引输出目录；相对路径按 `--output` 目录解析。
- `emit_case_index`：输出 `manifest_case_index.json`。
- `emit_lane_index`：输出 `manifest_lane_index.json`。

## 设计原则

### 1. manifest 只管编排

manifest 不应承载：

- shape relationship
- value policy
- operator fact 细节
- backend-specific evidence 字段

这些继续留在 case、executor 和 evidence 层。

### 2. 任务与样本分离

任务清单是“怎么跑”，case 文件是“跑什么”。

### 3. 证据边界清楚

建议在 report 中按以下层级输出：

- task
- lane
- case
- backend
- evidence

这样 `contract`、`hardware`、`preflight_blocked` lane 不会互相污染；generated 样本继续通过 case metadata 和 artifact provenance 表达。

### 4. 兼容已有文件

manifest 初版应完全基于现有文件工作，不要求重写当前 case 组织。最小目标是“能引用、能展开、能汇总”。

## 执行流程

首版执行器已经支持：

1. 读取 manifest。
2. 展开 lanes。
3. 读取每个 `include` 的 case 文件。
4. 将 lane 名称、lane kind、include 来源、显式 `backends` 和 runtime policy 写入 case metadata。
5. 按 `runtime.allow_generated_cases` / `runtime.allow_invalid_cases` 调整展开数量。
6. 按 `filters` 选择 case。
7. 复用现有 `JobSpec`、`ExecutionPlanner` 和 scheduler。
8. 在 planner 阶段按 `manifest_backends` 过滤执行 backend。
9. 按 `reporting` 输出 lane/case 索引。
10. 在 run `manifest.json` 里记录 manifest sha256、include case sha256 和 plan count 摘要。
11. 在 `report.jsonl` / `report.csv` 中投影 `manifest_lane`、`manifest_lane_kind` 和 `manifest_case_include`。

## 非目标

这个方案不试图：

- 把所有 case 合并成一个巨型 YAML。
- 用 manifest 代替 case contract。
- 改写已有 evidence 语义。
- 重新定义 backend runtime 的 capability 边界。

## 推荐落点

建议放在：

- `docs/manifest-design.md`

当前已配套：

- `examples/manifests/operator-contract-suite.yaml`
- `examples/manifests/local-smoke-manifest.yaml`
- `docs/campaign-matrix.md` 的引用说明
- `docs/report-export-schema.md` 的 manifest 结果字段说明

主入口命令建议统一写成：

```bash
cx run --manifest examples/manifests/operator-contract-suite.yaml
```

需要 CI 消费计划时使用 JSON 输出：

```bash
cx manifest plan examples/manifests/operator-contract-suite.yaml --nodes examples/nodes/local.yaml --json
```

## 结论

这个设计是合理的，但应该作为**顶层任务清单**，不是把所有 case 细节压成一个单文件巨型配置。最稳的路线是：

- 保留现有分文件 case 组织；
- 新增 manifest 只做编排；
- 执行器按 lane 展开 case 引用；
- evidence 与 contract 继续分层。

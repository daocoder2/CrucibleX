# Runtime Policy 契约

CrucibleX 将运行时行为从自由 metadata 提升为版本化的 `runtime_policy` 契约。Case 声明测试意图，Node 声明可支持能力，pipeline 将请求、生效项和不支持项写入 result metrics 的 `runtime_policy` 字段。

## 支持的通用策略

- `deterministic`：请求启用或关闭 Torch 确定性算法；
- `synchronize_timing`：请求在性能/内存采样的关键阶段执行设备同步。

```yaml
# Case
runtime_policy:
  schema_version: 1
  deterministic: true
  synchronize_timing: true

# Node
runtime_policy_capabilities:
  - deterministic
  - synchronize_timing
```

## 生效规则

- 未声明 `runtime_policy` 时，CX 保持已有行为；性能与内存测量仍默认尝试同步设备。
- 已声明策略但 Node 未声明对应 capability 时，策略不会被静默应用，metrics 会写入 `runtime_policy.unsupported`。
- `synchronize_timing: false` 只有在 Node 声明 `synchronize_timing` capability 时才会关闭同步。
- `deterministic` 只有在 Node 声明 `deterministic` capability 且当前 worker 可调用 `torch.use_deterministic_algorithms` 时才会生效；不可用时记录具体不支持原因。

典型 evidence：

```yaml
runtime_policy:
  schema_version: 1
  requested:
    deterministic: true
    synchronize_timing: false
  effective:
    deterministic: true
    synchronize_timing: false
  unsupported: []
```

后端专有的 L2 cache、multi-stream memory reuse 和 allocator 策略暂不进入通用字段。后续只有在至少一个后端具备可验证执行实现及 evidence 后，才会作为受控扩展加入。

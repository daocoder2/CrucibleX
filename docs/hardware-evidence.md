# 硬件证据契约

硬件评测结果必须区分三件事：计划请求的设备、执行器最终解析的设备，以及运行时实际观察到的证据。`ExecutionResult.evidence` 是结果级的规范字段，用于表达这三者，而不是依赖某个 backend 专用 metrics 键。

## 稳定字段

`HardwareEvidence` 当前包含：

- `schema_version`：证据结构版本。
- `backend`、`host`、`node`、`device_id`：计划和放置上下文。
- `resolved_device`：执行器解析后的设备选择，例如 `cuda:0` 或 `npu:0`。
- `probe_status`：`available`、`unavailable` 或 `unknown`。
- `runtime`：可比较的运行时信息，例如 CUDA 版本和设备数量。
- `fingerprint`：backend 产生的稳定证据指纹。
- `artifact_refs`：可供审计的证据 artifact 路径。

## 真实性规则

- `available` 只表示对应 runtime probe 确认可用，不代表算子精度、性能或稳定性已通过。
- `unknown` 表示当前 backend 尚未提供可验证 probe，绝不能在报告中解释为可用。
- NPU 和 ACLNN 的真实硬件结论必须由匹配资源的执行环境、执行结果和保留 artifact 支撑；一次性 Docker gate 也必须保留 driver 挂载与设备映射的验证上下文。
- CSV、Markdown 等输出应投影 `evidence`；旧的 backend 专用 metrics 仅在迁移期保留。

## 滚动升级

Ray worker 与 Driver 允许短暂处于不同的部署版本。若旧 worker 返回的执行结果缺少 `evidence`，Driver 在持久化时只会从该结果的原始 metrics 与 `gpu_evidence` artifact 归一化证据；已有 `evidence` 不会被覆盖，cross-device compare 记录也不会被伪造成设备执行证据。

## 精度度量策略

Case 可以在 `oracle.accuracy_policy` 中声明算子类别和阈值。类别决定适用的指标，不能用一个通用 allclose 结果替代：

```yaml
oracle:
  accuracy_policy:
    category: non_computational | integer | quantized | floating
    thresholds:
      ae: 0.0
      mare: 0.0
      mere: 0.0
      rmse: 0.0
      small_value_error_count: 0
    small_value_threshold: 0.1
```

| 算子类别 | Bitwise Match | AE | MARE | MERE | RMSE | 小值域错误数 |
| --- | --- | --- | --- | --- | --- | --- |
| 非计算类 / 搬迁 / Cast | 适用 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 |
| 整数计算（INT8/INT16/INT32） | 适用 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 |
| 量化计算（低精度/量化 dtype） | 不适用 | 适用 | 适用 | 适用 | 适用 | 适用 |
| 浮点计算（FLOAT16/FLOAT32） | 不适用 | 不适用 | 适用 | 适用 | 适用 | 适用 |

小值域错误数只在 Case 同时声明 `small_value_threshold` 时计算；错误定义为 reference 绝对值不大于该阈值且误差超过 `atol` 的元素数量。高精度 CPU 是共同 reference，GPU 和 NPU 分别计算上述适用指标；NPU 必须满足自身阈值，并且每个适用指标不劣于 GPU。

## 当前覆盖

GPU 已把现有 probe、CUDA 版本、fingerprint 和 `gpu_evidence` artifact 映射到统一结构。NPU 与 ACLNN 会采集 `torch.npu.is_available()`、设备数量、Torch/torch_npu 版本和设备名称，写入 `npu_evidence` artifact；只有 probe 成功时才标记 `available`。CPU 仍输出 `unknown`。

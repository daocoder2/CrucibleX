# Case 调用契约

Case 不只描述输入，还必须明确输入如何映射到算子调用。`invocation.binding` 是稳定的领域契约；执行器通过它得到 `args` 与 `kwargs`，不需要推断参数顺序。

## 绑定模式

- `positional`：未省略的输入按 Case 参数顺序传入。
- `keyword`：输入以参数名传入关键字参数。
- `mixed`：`positional` 指定的连续前缀按位置传入，其余输入按名称传入。

```yaml
invocation:
  api: numpy.sum
  api_type: function
  executor: numpy
  binding:
    mode: keyword
    names: [a, axis]
```

`names` 为空时，框架使用 `parameters` 中的名称。`omit` 可以按名称或索引省略可选输入。`mixed.positional` 必须是从 `0` 开始的连续前缀，且不能包含声明为 `keyword_only` 的参数。

## 输入值

参数可以使用 `values` 声明 exact values。它优先于 `value_policy` 和 `value_range`，tensor values 会按声明 dtype 转换并校验 shape；无法转换或 shape 不匹配时生成失败。Driver inputs artifact 会记录每个参数的来源，例如 `exact_values`、`value_range`、`value_policy` 或 `default`。

## 兼容策略

历史 Case 中的 `invocation.metadata.binding` 仍可读取，但新 Case 必须使用 `invocation.binding`。兼容读取只位于执行请求边界，不能让非类型化 metadata 重新进入领域语义。

## 代表性示例

`examples/cases/numpy.sum.yaml` 演示 reduction 算子：Case 参数名为 `input`，通过 `names: [a, axis]` 映射到 NumPy API 的 `a` 与 `axis`。`examples/cases/numpy.add.broadcast.yaml` 演示 binary broadcast：它以 positional binding 调用 `numpy.add`，并通过 `broadcastable_with` 校验第二个输入的 `[1, 3, 1]` shape。`examples/cases/numpy.mean.yaml` 演示 axis reduction：通过相同 binding 将 `input`/`axis` 映射到 `a`/`axis`，并验证非标量输出。三个示例都使用可重放的输入契约；`numpy.mean` 使用正式的 `values` 字段固定输入矩阵，`numpy.add.broadcast` 使用 shape 关系生成输入。固定 tensor 的 values 必须匹配声明的 shape；未知参数字段会在加载时拒绝，避免输入语义被静默丢弃。NumPy 示例可在本地 CPU 环境重放。`aclnn.mean.npu.yaml` 也使用同一 `values` 字段表达硬件 gate 的固定输入矩阵。

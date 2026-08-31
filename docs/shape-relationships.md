# Shape 关系契约

Shape 关系用于声明多个输入之间必须满足的结构约束。关系在 Case 展开阶段按确定顺序解析，生成器不会依赖运行时设备或随机状态推断关系。

## `transpose_of`

`transpose_of` 让目标参数复用源参数的维度集合，并可通过 `axes` 指定维度排列：

```yaml
parameters:
  - name: input
    kind: tensor
    shape:
      dims: [2, 3, 4]
  - name: transposed
    kind: tensor
    metadata:
      shape_relationship:
        kind: transpose_of
        source: input
        axes: [2, 0, 1]
```

上例将目标 shape 解析为 `[4, 2, 3]`。省略 `axes` 时默认反转源维度。`axes` 必须是源 rank 的完整排列；声明非法时保留目标原始 shape，避免生成器静默制造一个不满足声明的 shape。

## 可复现性

相同 Case、seed 和生成器版本必须得到相同的关系解析结果。关系解析结果会写入参数的 `resolved_shape_relationship` metadata，便于检查生成输入和定位失败。

## 关系边界

Shape 关系只负责生成阶段的形状契约，不替代算子执行器的 API 校验，也不声明某个 backend 一定支持该 shape。执行、比较和硬件能力仍由对应的 Case、Executor 与 evidence 负责。

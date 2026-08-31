# 精度术语参考

这份文档用于统一 CrucibleX 里常见的精度比较术语，避免把“比较方法”“容差参数”“误差指标”混为一谈。

## 一、术语分层

### 1. 比较方法

- `allclose`：近似相等判定方法。它不是误差指标本身，而是一个判断“两个结果是否足够接近”的规则。
- `bitwise match`：按位完全一致判定。它要求结果逐位相同，不允许任何差异。

### 2. 容差参数

- `atol`：绝对容差（absolute tolerance）。
- `rtol`：相对容差（relative tolerance）。

常见判定形式可以写成：

```text
|a - b| <= atol + rtol * |b|
```

其中：

- `atol` 控制允许的绝对偏差；
- `rtol` 控制允许的相对偏差。

### 3. 误差指标

- `AE`：Absolute Error，绝对误差。
- `MARE`：Maximum Absolute Relative Error，最大相对误差。
- `MERE`：Mean Absolute Relative Error，平均相对误差。
- `RMSE`：Root Mean Square Error，均方根误差。
- `small value error count`：小值域错误数，统计落在小值域内且误差超过阈值的元素数量。

## 二、推荐中文说法

### `allclose`

推荐写法：

- 近似相等判定
- 近似比较
- 容差比较

不建议直接把 `allclose` 说成“误差指标”，它更像是一种判定规则。

### `atol` / `rtol`

推荐写法：

- `atol` = 绝对容差
- `rtol` = 相对容差

如果写成“误差”，容易让人误解成具体计算出来的指标。它们更准确地说是比较参数。

### `AE` / `MARE` / `MERE` / `RMSE`

推荐统一称为：

- 精度指标
- 误差度量指标
- 数值误差指标

### `bitwise match`

推荐统一称为：

- 按位一致
- 完全一致
- 位级一致

## 三、在 CrucibleX 里的层级关系

可以按下面方式理解：

- `oracle.accuracy_policy`：定义某个 Case 应该使用哪些精度规则。
- `comparison`：定义采用哪种比较方法，例如 `allclose`。
- `atol` / `rtol`：定义比较容差。
- `AE` / `MARE` / `MERE` / `RMSE` / `small value error count`：定义更细的误差指标。
- `bitwise match`：用于要求完全一致的场景。

## 四、简短对照

| 术语 | 类型 | 含义 |
| --- | --- | --- |
| `allclose` | 比较方法 | 近似相等判定 |
| `bitwise match` | 比较方法 | 按位完全一致判定 |
| `atol` | 容差参数 | 绝对容差 |
| `rtol` | 容差参数 | 相对容差 |
| `AE` | 误差指标 | 绝对误差 |
| `MARE` | 误差指标 | 最大相对误差 |
| `MERE` | 误差指标 | 平均相对误差 |
| `RMSE` | 误差指标 | 均方根误差 |
| `small value error count` | 误差指标 | 小值域错误数 |

## 五、一个实用原则

如果你在文档里描述“结果是否通过”，优先用“比较方法 + 容差 + 误差指标”三个层次分别说明，不要把它们写成同一类术语。

例如可以写成：

- 使用 `allclose` 做基础比较；
- 通过 `atol` / `rtol` 控制容差；
- 再结合 `AE` / `MARE` / `MERE` / `RMSE` / 小值域错误数判断是否满足算子类别要求。

## 六、CrucibleX 的精确定义

为避免同一术语在不同报告中有不同聚合方式，CrucibleX 使用以下固定定义。设 CPU reference 为 `r`，设备 candidate 为 `x`，逐元素绝对差为 `d = abs(r - x)`：

- `AE`：当前实现为最大绝对误差，即 `max(d)`；报告字段使用 `ae`，其语义等同 `max_ae`。
- `MARE`：`max(d / max(abs(r), relative_epsilon))`。
- `MERE`：`mean(d / max(abs(r), relative_epsilon))`。
- `RMSE`：`sqrt(mean(d * d))`。
- `relative_epsilon`：相对误差分母下限，必须为正数；未声明时为 `1e-12`。reference 为零的元素使用该下限，不会除以零。
- `small value error count`：`abs(r) <= small_value_threshold` 且 `d > atol` 的元素数量。

`non_computational` 与 `integer` 类别的按位一致会同时比较 shape、dtype 和数组原始字节序列；因此数值相同但 dtype 不同，以及 `+0.0` 与 `-0.0`，都会被视为不一致。

浮点和量化指标遇到任一侧包含 `NaN` 或 `Inf` 时会记录 `non_finite_count`，并使三侧 gate 失败。

## 七、三侧配对约束

CPU、GPU、NPU 的结果只有在 `case_id`、task 和 inputs artifact 中的 `case_fingerprint` 均相同的 group 内才会比较。不同 fingerprint 的结果不会被混配，也不会生成 CPU/GPU/NPU 三侧 gate。

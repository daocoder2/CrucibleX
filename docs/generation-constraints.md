# 生成约束与矩阵 Profile

CrucibleX 的 Case generation 使用声明式约束和参数级 value policy。约束由 `generation.constraints` 启用，参数关系和值分布写在参数 `metadata` 中；生成器不会迁移或依赖 ATK 的类层级。

## 已有声明式约束

- `shape_relationships`：支持 `same_numel`、`broadcastable_with`、`dim_equal`、`divisible_by`、`rank_range`、`dimension_alias` 与 `transpose_of`。
- `product_limits`、`max_elements`、`max_bytes`：限制维度乘积、元素总数和估算内存。
- `dtype_promotion`：按 Case 规则调整多输入 dtype。
- `value_policy`：启用参数的边界、分布和矩阵 profile 值策略。

## 矩阵 Profile

`matrix_profile` 仅适用于 rank-2 浮点 tensor，使用 `value_policy_seed` 保证确定性。它用于覆盖普通均匀/正态随机值难以触达的数值敏感矩阵场景。

- `well_conditioned`：正交基与受控奇异值构成的满秩矩阵。`condition_number` 默认是 `4.0`，且不得小于 `1.0`。
- `rank_deficient`：两个低秩因子的乘积。`rank` 默认为 `min(shape) - 1`，必须满足 `0 <= rank < min(shape)`。

```yaml
generation:
  seed: 19
  constraints: [value_policy, shape_relationships, max_bytes]
  max_bytes: 1048576
parameters:
  - name: input
    kind: tensor
    dtypes: [fp32]
    shape: {dims: [64, 32]}
    metadata:
      value_policy_seed: 19
      value_policy:
        kind: matrix_profile
        profile: well_conditioned
        condition_number: 3.0
  - name: rank_deficient_input
    kind: tensor
    dtypes: [fp32]
    shape: {dims: [64, 32]}
    metadata:
      value_policy_seed: 19
      value_policy:
        kind: matrix_profile
        profile: rank_deficient
        rank: 8
```

矩阵 profile 与 `exact_values` 不叠加：显式 `values` 始终优先。对标量、整数、rank 不为 2 的 tensor 或未知 profile，生成器会抛出明确错误，不能静默回退为随机数据。

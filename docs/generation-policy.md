# Generation Policy

CrucibleX 通过 `generation.metadata.operator_facts` 驱动通用生成策略。facts 在 `expand_cases` 时被投影到 Parameter metadata，再由统一 constraint 与 Default generator 执行；不需要为每个算子创建 generator。手写 Parameter metadata 的同名字段优先于 facts。

## Operator Facts

```yaml
generation:
  metadata:
    operator_facts:
      schema_version: 1
      parameters:
        lhs:
          dtypes: [fp16, bf16, fp32]
          dtype_policy:
            backend: npu
            backend_allowed:
              npu: [fp16, bf16, fp32]
            backend_denied:
              cpu: [bf16]
          shape_policy:
            rank_range: [2, 4]
            broadcast_group: binary_inputs
          value_policy:
            kind: exponential
            scale: 1.0
            signed: true
        rhs:
          dtypes: [fp16, bf16, fp32]
          shape_policy:
            broadcast_group: binary_inputs
```

facts 自动启用 `operator_facts`、`dtype_policy`、`value_policy`、`shape_relationships` 与 `dtype_promotion` constraints。

## Dtype Policy

- `allowed` / `denied`：全局候选过滤。
- `backend_allowed` / `backend_denied`：按 `backend` 的 allow/deny。
- `groups` 与 `group`：选择 dtype family。
- `dtype_promotion.sources`：由多个输入推导输出 dtype。
- mixed dtype：为不同参数声明不同 groups 或 candidates，并用 `dtype_promotion` 声明结果；不会隐式将所有参数强制为同一 dtype。
- generator 支持 fp16/fp32/fp64、bf16、bool、有符号/无符号整数及 complex64/complex128。NumPy 没有原生 bf16 时，reference 输入使用 fp32 表示，但保留 `bf16` 声明给后端 executor。

## Value Policy

支持 `zero`、`one`、`constant`、`nan`、`inf`、`subnormal`、`integer_bounds`、`float_bounds`、`extreme`、`uniform`、`normal`、`exponential`、`complex_normal`、`sparsity` 和 `matrix_profile`。

`matrix_profile.profile` 支持 `well_conditioned`、`rank_deficient`、`identity`、`diagonal` 与 `symmetric`。特殊浮点值只应用于 floating/complex dtype；不兼容的 dtype 会在约束层记录过滤结果。

## Shape Policy

- 关系约束：`rank_range`、`broadcast_group`、`dimension_alias`、`same_numel`、`dim_equal`、`transpose_of`、`divisible_by`。
- 规模约束：generation `max_elements` / `max_bytes` 与 metadata `product_limits`。
- layout：`storage_shape` 配合 `slice` 创建 storage-backed view；`non_contiguous: true` 创建非连续视图。slice 结果 shape 必须匹配 Parameter 声明 shape。

```yaml
shape_policy:
  storage_shape: [8, 8]
  slice: [[2, 6], [1, 5]]
  non_contiguous: true
```

新增 policy 必须保持确定性：任何随机 value policy 通过 `value_policy_seed` 固化；新增 operator facts 只描述事实和策略，不能引入专用算子生成器。

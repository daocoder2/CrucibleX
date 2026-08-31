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

## Policy Libraries

参数 policy 可通过 `library` 引用内置条目，并以同一个 dict 覆盖库默认值：

```yaml
dtype_policy:
  library: low_precision_floating
  backend_denied:
    cpu: [bf16]
value_policy:
  library: signed_exponential
  scale: 0.5
shape_policy:
  library: offset_stride
  storage_shape: [8, 8]
  storage_offset: 9
  strides: [8, 2]
```

- dtype：`floating`（默认 fp32）、`low_precision_floating`（fp16/bf16）、`complex`、`integer`、`boolean`。
- value：`float_edges`、`integer_edges`、`boolean_edges`、`complex_edges`、`signed_exponential`、`extreme`、`sparse`。edge library 通过 `boundary_set` 保留 NaN、Inf、subnormal、signed zero、整数 min/max 等边界集合。
- shape/layout：`non_contiguous`、`offset_stride`、`sliced_view`。`storage_offset` 与 `strides` 以元素数计，底层转换为 NumPy byte strides；越界时生成直接失败，绝不创建未验证视图。

## Built-in Operator Facts

`torch.add`、`torch.matmul` 与 `torch.softmax` 会自动合并内置 facts。它们只提供参数级 dtype/value/shape 默认策略，且 Case 中显式 metadata 具有更高优先级。

- `torch.add`：floating dtype、binary broadcast group 与 floating edge values；
- `torch.matmul`：floating dtype、rank-2 well-conditioned matrix profile，以及右输入第 0 维别名到左输入第 1 维；
- `torch.softmax`：floating dtype 与 signed exponential values。

也可以通过 `generation.metadata.operator_fact_library` 指定一个或多个库名。当前内置 facts 不自动声明 invocation binding、attributes、outputs 或 backend 支持能力；这些仍属于 Case、executor 与 Node 契约。

## Dtype Evidence

`bf16` 的 reference input 使用 fp32 存储，但先执行 IEEE bfloat16 round-to-nearest-even 量化。input artifact metadata 记录每个参数的 `declared_dtype`、`reference_dtype` 与 `quantization`，例如 `bf16/fp32/bfloat16_rne`。

pipeline 同时记录 `candidate_output_dtype` 与 `reference_output_dtype`。Torch/ACLNN executor 在 device tensor 转 NumPy 前额外记录 `backend_output_dtype`、`backend_output_device` 和 `backend_dtype_source: device_tensor`；前两者不能互相替代。

## Special Value Validation

value policy 会写入 `value_policy_validation`，包含 requested/effective/rejected。不可用于目标 dtype 的 NaN、Inf、subnormal 或 complex-only policy 会保留明确 rejected 原因，generator 随后拒绝生成该输入，避免静默转换为伪合法值。

## Collection Relationships

在 `linked_parameters` constraint 中，collection 参数可声明 `collection_relationship`：

```yaml
collection_relationship:
  kind: same_length_as # 或 same_item_dtype_as / same_item_shape_as
  source: inputs
```

- `same_length_as` 从 source 的 `items` 或 `length` 元数据派生目标长度。
- `same_item_dtype_as` 复用 source 的 `item_dtypes` 或参数 dtype。
- `same_item_shape_as` 复用 source 的 `item_shapes` 或参数 shape。

关系结果会记录在 `resolved_collection_relationship`，并继续由 default generator 使用既有 `length`、`item_dtypes` 与 `item_shapes` metadata 生成实际元素。

## Extended Operator Facts

内置 facts 还覆盖 `torch.sum`、`torch.mean`、`torch.norm`、`torch.sort`、`torch.topk` 和 `torch.index_select`。

- reduce/sort/topk/norm：input 默认 floating、rank 1-4 和确定性 normal value policy。attribute、输出数量和 output dtype/shape 仍由 Case invocation/output contract 声明。
- index-select：input 默认 floating，`index` 收敛为 `int64`。当前没有 source dimension 驱动的 index range constraint，因此 facts 不会声明自动生成的 index 一定可执行；需要 Case exact values 或专用 value relationship。
- conv 与完整 attention 尚不自动启用。它们分别缺 output spatial formula/groups contract 与 QKV/mask/dropout contract；不要将 generic facts 当成已验证的后端能力。

## Extended Collection Relationships

除基础 same-length/dtype/shape 外，还支持：

- `broadcast_items_with`：逐元素计算可广播的 item shape；不可广播 pair 保持未解析。
- `zip_with`：目标 length 复用 source length，并记录 `collection_pairing: zip`。
- `cartesian_with`：目标 length 为 source 与目标原始长度的乘积，并记录 `collection_pairing: cartesian`。
- nested/ragged：使用逐项 `metadata.items` 表达，元素可再次为 collection；不会隐式填充或压平成矩形。

## Hardware Dtype/Layout Gate

`scripts/hardware_dtype_layout_gate.sh` 是 bf16、layout、stride、special-value 四条 lane 的可复用硬件 gate。调用者必须显式设置 `CASE_BF16`、`CASE_LAYOUT`、`CASE_STRIDE`、`CASE_SPECIAL`，并可覆盖 `NODE_PATH`、`SCHEDULER` 与 `OUTPUT_ROOT`。每条 lane 执行 accuracy 后要求 result metrics 同时含 input dtype contract 和 `backend_dtype_source`，因此只适合实际 Torch/ACLNN device-tensor executor 的环境。

该脚本不包含设备地址、镜像、registry 或其他私有运行配置。

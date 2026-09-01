# Operator Hardware Acceptance Matrix

本矩阵定义高级 operator facts 的真实硬件验收边界。只有通过 `cx run` 且结果同时包含 input dtype contract、`backend_dtype_source` 与 hardware evidence 的 lane 才可标记为验收通过。

## Existing Public Cases

| Family | Public case | Backend | Status |
| --- | --- | --- | --- |
| reduce | `examples/cases/aclnn.mean.npu.yaml` | ACLNN/NPU | dim/keepdim/dtype/output shape sample |
| reduce | `examples/cases/torch.mean.hardware.yaml` | Torch CPU/GPU | dim/keepdim and shared-fingerprint output evidence |
| reduce | `examples/cases/aclnn.max_dim.npu.yaml` | ACLNN/NPU | dim/keepdim/multi-output indices sample |
| sort | `examples/cases/aclnn.sort.npu.yaml` | ACLNN/NPU | dim/descending/multi-output sample |
| sort | `examples/cases/torch.sort.hardware.yaml` | Torch CPU/GPU | stable keyword binding and values/indices output evidence |
| matmul | `examples/cases/torch.matmul.hardware.yaml` | Torch hardware | rank-2 matmul sample |
| topk | `examples/cases/torch.topk.npu.yaml` | Torch/NPU | k/dim/largest/sorted device-tensor evidence |
| index-select | `examples/cases/torch.index_select.npu.yaml` | Torch/NPU | int64 index and dim device-tensor evidence |
| bmm | `examples/cases/torch.bmm.npu.yaml` | Torch/NPU | batch matmul device-tensor evidence |
| gather | `examples/cases/torch.gather.npu.yaml` | Torch/NPU | int64 index device-tensor evidence |
| scatter | `examples/cases/torch.scatter.npu.yaml` | Torch/NPU | int64 index and src device-tensor evidence |
| select | `examples/cases/torch.select.npu.yaml` | Torch/NPU | selected-dimension device-tensor evidence |
| bf16 | `examples/cases/torch.bf16.npu.yaml` | Torch/NPU | bfloat16 RNE input contract and device-tensor evidence |
| special values | `examples/cases/torch.special-values.npu.yaml` | Torch/NPU | Inf/-Inf/subnormal device-tensor evidence |
| where | `examples/cases/torch.where.npu.yaml` | Torch/NPU | boolean condition and broadcast value tensors |
| masked_fill | `examples/cases/torch.masked-fill.npu.yaml` | Torch/NPU | boolean mask and scalar fill value |
| transpose | `examples/cases/torch.transpose.npu.yaml` | Torch/NPU | dimension permutation device-tensor evidence |
| reshape | `examples/cases/torch.reshape.npu.yaml` | Torch/NPU | int64 shape tuple and device-tensor evidence |
| group_norm contract | `examples/cases/torch.group-norm.generated.yaml` | generation/contract | groups/channel/affine shape and invalid mutation sample |
| instance_norm contract | `examples/cases/torch.instance-norm.generated.yaml` | generation/contract | channel/affine/eps and invalid mutation sample |
| gather 3D contract | `examples/cases/torch.gather-3d.generated.yaml` | generation/contract | rank/non-index extent predicate and invalid sample |

## Required Gate Lanes

使用 `scripts/operator_contract_hardware_gate.sh` 时，调用方显式提供 `CASE_REDUCE`、`CASE_SORT`、`CASE_INDEX` 与 `CASE_MATMUL`。可选的 `CASE_MASK`、`CASE_RESHAPE`、`CASE_TRANSPOSE` 使用相同检查。gate 对每条 lane 执行 accuracy，要求 `backend_dtype_source=device_tensor`，并接受结果顶层 evidence 或 NPU/GPU metrics evidence。

- `CASE_REDUCE` 可使用 checked-in ACLNN mean 或 max-dim case。
- `CASE_SORT` 可使用 checked-in ACLNN sort case；`torch.topk.npu.yaml` 已独立通过 NPU device-tensor 验收。
- `CASE_INDEX` 可使用已通过 NPU 验收的 `torch.index_select.npu.yaml`；gather、scatter、select 也已分别通过 Torch/NPU 验收。
- `CASE_MATMUL` 可使用 checked-in Torch matmul case；`torch.bmm.npu.yaml` 已独立通过 NPU device-tensor 验收。

## Dtype And Layout Lanes

`scripts/hardware_dtype_layout_gate.sh` 负责 bf16、layout、stride、special value 四条 lane，调用时必须显式传入 case。它验证 input dtype contract 与 device tensor dtype evidence。

目前不应将 ACLNN layout/stride lane 标为端到端通过：schema 会保留 format、stride、storage offset 与 dynamic output 声明，preflight 会明确拒绝 non-ND format、任何显式 stride、非零 storage offset 和 dynamic output；bridge 仍会连续化输入且 descriptor 固定 ND/storage offset 0。ACLNN tensor-list 与 optional 参数也仍被 preflight 拒绝，因为缺少 tensor-list ownership 与 null ABI contract。workspace 本身由 runtime 管理，不是 Case 参数。

Torch bf16 CPU/GPU shared-fingerprint compare 已通过，输入 contract 为 bf16 RNE 且 GPU dtype source 为 device tensor。special value、Torch layout/stride 可以由实际 device tensor case 进入 gate；ACLNN non-contiguous layout/stride 和 dynamic output 仍为 runtime capability blocker。

## Family Coverage Matrix

| Operator family | Parameter constraints | Output evidence contract | Invalid sample | Checked-in example | Backend execution status |
| --- | --- | --- | --- | --- | --- |
| reduce | dim/keepdim/dtype, multi-dim/negative dim and reduction shape | output shape/dtype plus normalized reduced dimensions | dim out of range, duplicate dim | `aclnn.mean.npu.yaml`, `torch.mean.hardware.yaml` | reduce CPU/GPU shared-fingerprint compare passed; ACLNN/NPU evidence recorded |
| topk/sort | dim/k/largest/sorted | values and int64 indices shape/dtype | dim out of range, k exceeds axis | `torch.topk.npu.yaml`, `aclnn.sort.npu.yaml`, `torch.sort.hardware.yaml` | topk and sort CPU/GPU shared-fingerprint compares passed; Torch/NPU and ACLNN/NPU evidence recorded |
| index/mask | int64 index, rank/non-index extent and broadcast mask constraints | selected/gather/scatter/where output shape | index out of range, index rank/extent mismatch, scatter src mismatch, broadcast mismatch | `torch.gather.npu.yaml`, `torch.where.npu.yaml` | index-select, gather, scatter, select and where CPU/GPU shared-fingerprint compares passed; complex index shape predicates covered by contract tests; Torch/NPU evidence recorded |
| reshape/layout | numel preservation, tuple shape, transpose rank, view contiguity | output shape/dtype and layout policy | numel mismatch, invalid dimensions, non-contiguous view | `torch.reshape.npu.yaml`, `torch.transpose.npu.yaml` | reshape/transpose CPU/GPU shared-fingerprint compares passed; view contract rejects declared non-contiguous input; Torch/NPU passed; ACLNN layout/stride remains blocked by bridge semantics |
| matmul/bmm | inner dimension, matmul batch broadcast, bmm exact batch compatibility | normalized batch/output shape and dtype | inner mismatch, broadcast mismatch, bmm batch mismatch | `torch.bmm.npu.yaml` | bmm CPU/GPU shared-fingerprint compare passed; Torch/NPU evidence recorded; matmul CPU/GPU also passed |
| conv/norm/attention | conv channel aliases and raw stride/padding/dilation validation; layer_norm suffix/affine/eps; group_norm channel/groups/affine; QKV batch/head/embed/mask/dropout/causal predicates | formula-derived output shape/dtype plus validity predicates | channel, invalid stride/padding/dilation, normalized-shape, group/channel, affine, head, dropout and causal-mask mismatch | `torch.conv2d.generated.yaml`, `torch.layer-norm.generated.yaml`, `torch.attention.generated.yaml` | legal conv2d/layer_norm/attention CPU/GPU shared-fingerprint compares passed; group_norm is contract-tested only; attention dropout/causal predicates covered by contract tests; invalid variants remain negative validation; runtime support explicitly capability-gated |
| ACLNN signature | native scalar/array ABI, output dtype/shape, lifecycle | parsed signature and output descriptors | unsupported ABI kind/shape | `aclnn.mean.npu.yaml` | ACLNN mean/max-dim/sort NPU evidence; optional/list ABI blockers documented |

复杂算子的 generated 示例会同时产生合法 case、`expected_invalid` case、解析后的 `resolved_operator_contract` 与输出 shape。当前真实执行矩阵：CPU NumPy broadcast 已通过；CPU/GPU Torch mean、matmul、bmm、where、topk、index-select、gather、scatter、select、reshape、transpose、masked_fill、bf16，以及 conv2d、layer_norm、scaled-dot-product-attention 的合法 case 已在同一容器、同一 case fingerprint 下通过；NPU Torch/ACLNN 多条 lane 已通过。CPU/GPU/NPU 三侧聚合仍由 `scripts/cpu_gpu_npu_accuracy_gate.sh` 汇总；ACLNN 作为 NPU-side executor 单独记录，不能替代 CPU/GPU evidence。

## Evidence Rule

通过 gate 只表示该 Case 在指定真实设备环境中产生了可审计 evidence，不自动扩展为所有 shape、dtype 或 ABI 的后端支持声明。

## Local Probe

本地开发环境未安装 Torch/NPU runtime；真实 Torch/NPU lane 已在配置的 NPU 主机设备专用镜像中执行。当前已取得 reduce、sort、topk、index-select、gather、scatter、select、bmm、bf16、special-value、where、masked_fill、reshape 与 transpose 的 NPU device-tensor evidence。CPU NumPy broadcast case 已通过；CPU/GPU Torch mean、matmul、bmm、where、topk、index-select、reshape、transpose、masked_fill 与复杂算子合法 case 均已通过。CPU/GPU/NPU 聚合 gate 还需要实际 GPU/NPU archives，不能用生成层测试替代。

Gate 依赖和设备可见性是两个独立前置条件；两者均满足后，才可把 gate 结果记为真实硬件 evidence。

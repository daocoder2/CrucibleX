# Operator Hardware Acceptance Matrix

本矩阵定义高级 operator facts 的真实硬件验收边界。只有通过 `cx run` 且结果同时包含 input dtype contract、`backend_dtype_source` 与 hardware evidence 的 lane 才可标记为验收通过。

## Existing Public Cases

| Family | Public case | Backend | Status |
| --- | --- | --- | --- |
| reduce | `examples/cases/aclnn.mean.npu.yaml` | ACLNN/NPU | dim/keepdim/dtype/output shape sample |
| reduce | `examples/cases/aclnn.max_dim.npu.yaml` | ACLNN/NPU | dim/keepdim/multi-output indices sample |
| sort | `examples/cases/aclnn.sort.npu.yaml` | ACLNN/NPU | dim/descending/multi-output sample |
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

## Required Gate Lanes

使用 `scripts/operator_contract_hardware_gate.sh` 时，调用方显式提供 `CASE_REDUCE`、`CASE_SORT`、`CASE_INDEX` 与 `CASE_MATMUL`。可选的 `CASE_MASK`、`CASE_RESHAPE`、`CASE_TRANSPOSE` 使用相同检查。gate 对每条 lane 执行 accuracy，要求 `backend_dtype_source=device_tensor`，并接受结果顶层 evidence 或 NPU/GPU metrics evidence。

- `CASE_REDUCE` 可使用 checked-in ACLNN mean 或 max-dim case。
- `CASE_SORT` 可使用 checked-in ACLNN sort case；`torch.topk.npu.yaml` 已独立通过 NPU device-tensor 验收。
- `CASE_INDEX` 可使用已通过 NPU 验收的 `torch.index_select.npu.yaml`；gather、scatter、select 也已分别通过 Torch/NPU 验收。
- `CASE_MATMUL` 可使用 checked-in Torch matmul case；`torch.bmm.npu.yaml` 已独立通过 NPU device-tensor 验收。

## Dtype And Layout Lanes

`scripts/hardware_dtype_layout_gate.sh` 负责 bf16、layout、stride、special value 四条 lane，调用时必须显式传入 case。它验证 input dtype contract 与 device tensor dtype evidence。

目前不应将 ACLNN layout/stride lane 标为端到端通过：bridge 会连续化输入且 descriptor 固定 ND/storage offset 0。ACLNN dynamic output 也尚未支持，因为 output descriptor 在 workspace 查询前必须静态创建。workspace 本身由 runtime 管理，不是 Case 参数。

因此 bf16、special value、Torch layout/stride 可以由实际 device tensor case 进入 gate；ACLNN non-contiguous layout/stride 和 dynamic output 仍为 runtime capability blocker。

## Evidence Rule

通过 gate 只表示该 Case 在指定真实设备环境中产生了可审计 evidence，不自动扩展为所有 shape、dtype 或 ABI 的后端支持声明。

## Local Probe

本地开发环境未安装 Torch/NPU runtime；真实 Torch/NPU lane 已在配置的 NPU 主机设备专用镜像中执行。当前已取得 reduce、sort、topk、index-select、gather、scatter、select、bmm、bf16 与 special-value 的 NPU device-tensor evidence。

Gate 依赖和设备可见性是两个独立前置条件；两者均满足后，才可把 gate 结果记为真实硬件 evidence。

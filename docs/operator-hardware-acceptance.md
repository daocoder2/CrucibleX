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

## Required Gate Lanes

使用 `scripts/operator_contract_hardware_gate.sh` 时，调用方显式提供 `CASE_REDUCE`、`CASE_SORT`、`CASE_INDEX` 与 `CASE_MATMUL`。gate 对每条 lane 执行 accuracy，检查 summary、input dtype contract、device dtype evidence 与 hardware evidence。

- `CASE_REDUCE` 可使用 checked-in ACLNN mean 或 max-dim case。
- `CASE_SORT` 可使用 checked-in ACLNN sort case；`torch.topk.npu.yaml` 已独立通过 NPU device-tensor 验收。
- `CASE_INDEX` 可使用已通过 NPU 验收的 `torch.index_select.npu.yaml`；select/gather/scatter 仍需要各自的公开 case，不能由 index-select 外推。
- `CASE_MATMUL` 可使用 checked-in Torch matmul case；`torch.bmm.npu.yaml` 已独立通过 NPU device-tensor 验收。

## Dtype And Layout Lanes

`scripts/hardware_dtype_layout_gate.sh` 负责 bf16、layout、stride、special value 四条 lane，调用时必须显式传入 case。它验证 input dtype contract 与 device tensor dtype evidence。

目前不应将 ACLNN layout/stride lane 标为端到端通过：bridge 会连续化输入且 descriptor 固定 ND/storage offset 0。ACLNN dynamic output 也尚未支持，因为 output descriptor 在 workspace 查询前必须静态创建。workspace 本身由 runtime 管理，不是 Case 参数。

因此 bf16、special value、Torch layout/stride 可以由实际 device tensor case 进入 gate；ACLNN non-contiguous layout/stride 和 dynamic output 仍为 runtime capability blocker。

## Evidence Rule

通过 gate 只表示该 Case 在指定真实设备环境中产生了可审计 evidence，不自动扩展为所有 shape、dtype 或 ABI 的后端支持声明。

## Local Probe

本次开发环境探测到公开可见的 NVIDIA GPU，但项目 `uv` 环境未安装 `torch` 或 `torch_npu`，因此未执行任何 Torch/ACLNN 实机 lane。这不是 operator failure：运行 gate 前必须在目标设备环境安装与 case/executor 匹配的 runtime，并使用对应 Node document。

Gate 依赖和设备可见性是两个独立前置条件；两者均满足后，才可把 gate 结果记为真实硬件 evidence。

# Operator Boundary Campaign

examples/manifests/operator-boundary-campaign.yaml 是基础 operator contract matrix 的 manifest 入口。它把 contract 覆盖与 hardware evidence 分开，因此生成的非法样本不会被表述为 device runtime failure evidence。

## Lanes

- cpu-contract 覆盖 reduce、sort/topk、gather、view/layout 与 batched matmul broadcast。generated fixtures 同时生成合法 case 和声明式非法变体。
- gpu-legal-evidence 只包含合法的 reduction、sort 与 matmul fixtures。
- npu-legal-evidence 只包含合法的 topk、gather/scatter、mask、reshape 与 bmm fixtures。

manifest 为 hardware run 要求 device_tensor backend dtype provenance 与 real evidence。GPU/NPU lane 必须在兼容设备环境执行；仅本地 plan 不是 hardware evidence。

## Counts

cx manifest validate 在 generation 前验证 15 个 source cases。展开后 campaign 有 20 个 execution cases：

- 11 个 CPU contract cases，其中 5 个 expected_invalid samples。
- 3 个 GPU legal-evidence cases。
- 6 个 NPU legal-evidence cases。

这些是 manifest case selection 与 generation 结果，不等同于任意节点配置下的 plan 数。使用 `examples/nodes/local.yaml` 做本地 planning 时，只会生成 11 个 CPU contract plans；GPU/NPU hardware lanes 需要显式 GPU/NPU/ACLNN 节点才会进入 plan。`cx manifest plan --json` 的 public contract 是同时报告 selected `cases`、实际可调度 `plans`，并在每个 item 上投影 `lane` 与 `lane_kind`。

非法样本包括 topk 非法 dimension 和 k、gather 非法 index、non-contiguous view，以及 matmul batch-broadcast mismatch。它们是 contract metadata；除非真实 runtime execution 记录对应 result，否则不得当作 runtime rejection evidence。

## Commands

    cx manifest validate examples/manifests/operator-boundary-campaign.yaml
    cx manifest plan examples/manifests/operator-boundary-campaign.yaml \
      --nodes examples/nodes/local.yaml --scheduler local --json

只在提供选定 backend 与 device-tensor evidence 的环境运行 hardware lanes。run manifest 记录 manifest hash 和全部 include hashes；report export 保留 lane、include 与 manifest runtime policy projection。

## CPU Evidence

本 campaign 已在 source-baked root CPU Docker image 中执行 cpu-contract lane。运行使用一次性容器与 /out artifact root，没有 runtime-mount source tree。

- 11 个 CPU accuracy plans 全部 passed，其中包括 6 个 legal cases 与 5 个 expected-invalid cases。
- expected-invalid samples 覆盖 topk 非法 dimension 与 k、gather 越界 index、non-contiguous torch.Tensor.view，以及 matmul batch-broadcast mismatch；每项都经真实 Torch runtime 执行后按 expected-invalid lifecycle 记录为 passed。
- 审计 archive 包含 /out/manifest.json、results.jsonl、report.jsonl、summary.json 与 postprocess.json；summary 为 total=11、passed=11、failed=0。

CPU result 表示此组 contract 的真实 CPU runtime 行为，但不替代 GPU/NPU hardware evidence。

## NPU Evidence

本 campaign 已在 source-baked NPU Docker image 中执行 npu-legal-evidence lane。运行使用一次性容器与 /out artifact root，没有 runtime-mount source tree。

- manifest 展开 20 个 execution cases，NPU node 只计划 6 个 npu-legal-evidence accuracy plans。
- torch.topk、torch.gather、torch.scatter、torch.masked_fill、torch.reshape 与 torch.bmm 全部 passed。
- 每条 report record 都包含 hardware_probe_status=available、backend_dtype_source=device_tensor，以及 manifest_runtime_json 中的 require_real_evidence=true 和 require_backend_dtype_source=device_tensor。
- 审计 archive 包含 /out/manifest.json、results.jsonl、report.jsonl、summary.json 与 postprocess.json；summary 为 total=6、passed=6、failed=0。

这项 evidence 只覆盖 NPU legal lane。CPU contract 和 GPU legal lane 仍遵守各自的 runtime/evidence 边界；generated expected_invalid cases 仍不是 runtime rejection evidence。

## GPU Evidence

本 campaign 也已在 source-baked GPU Docker image 中执行 gpu-legal-evidence lane。运行使用一次性 --gpus all container 与 /out artifact root，没有 runtime-mount source tree。

- NPU-independent GPU lane 只计划 3 个 accuracy plans：torch.mean、torch.sort 与 torch.matmul，全部 passed。
- 每条 result 都记录 gpu_available=true、backend_dtype_source=device_tensor，以及相同的 manifest hardware evidence policy。
- 审计 archive 包含 /out/manifest.json、results.jsonl、report.jsonl、summary.json 与 postprocess.json；summary 为 total=3、passed=3、failed=0。

这项 evidence 只覆盖 GPU legal lane。它不替代 CPU contract execution，也不将 generated expected_invalid samples 解释为 GPU runtime rejection evidence。

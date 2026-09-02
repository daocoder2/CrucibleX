# Complex Operator Contracts

## Evidence Scope

The generated contracts for torch.conv2d, torch.group_norm, torch.instance_norm, torch.layer_norm, and torch.scaled_dot_product_attention keep legal generation, declared invalid mutations, and output contracts distinct from accelerator evidence.

- Conv2d, layer norm, and scaled dot product attention already have source-baked GPU/NPU legal evidence recorded by the operator contract campaign.
- examples/manifests/complex-norm-evidence.yaml closes the remaining group norm and instance norm legal path. It expands generated fixtures, filters out expected-invalid variants, and requires device_tensor evidence for GPU/NPU hardware lanes.
- The CPU lane executed torch.group_norm and torch.instance_norm: 2/2 passed.
- The GPU lane executed the same two cases: 2/2 passed, each with gpu_available=true and backend_dtype_source=device_tensor.
- The NPU lane executed the same two cases: 2/2 passed, each with npu_available=true and backend_dtype_source=device_tensor.

The generated invalid variants remain contract samples. They are not claimed as accelerator runtime rejection evidence unless a corresponding runtime result is recorded.

## Status Matrix

| Family | Status | Covered contract | Remaining boundary |
| --- | --- | --- | --- |
| conv2d | supported legal lane | NCHW/OIHW input-weight relation, output spatial shape, groups/channel validation, explicit invalid channel/groups/kernel samples | widen stride/padding/dilation/groups combinations and add non-default bias/layout variants |
| layer_norm | supported legal lane | normalized_shape tied to trailing dimensions, affine parameter shape, output shape/dtype, mismatch invalid samples | expand dtype/layout/rank combinations |
| group_norm | supported legal lane | channel/group divisibility, weight/bias contract, output shape/dtype, mismatch invalid samples | broaden groups/channel negative matrix and optional affine coverage |
| instance_norm | supported legal lane | functional invocation with optional running stats disabled, affine parameter shape, output shape/dtype, mismatch invalid samples | add running-stat and optional affine combinations |
| scaled_dot_product_attention | partial legal lane | Q/K/V batch-head-embed relation, mask shape relation, output shape/dtype, head mismatch invalid sample | causal/dropout/mask combinations and backend-specific unsupported paths need more lanes |

## Instance Norm Invocation

torch.instance_norm requires running statistics and runtime-control arguments in the validated runtime version. The generated fixture supplies running_mean, running_var, weight, bias, use_input_stats, momentum, eps, and cudnn_enabled through typed keyword binding. Positional binding would incorrectly map affine tensors to running-statistics arguments.

## ACLNN Boundary

ACLNN ABI coverage is independent of these Torch execution results. Static ND tensors and static multi-output operations remain supported only within documented lifecycle evidence. Tensor lists and optional forms remain future ABI; dynamic output, non-ND format, and explicit storage/stride forms remain preflight-blocked. See docs/aclnn-capability-matrix.md.

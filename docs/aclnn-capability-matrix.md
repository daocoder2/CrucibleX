# ACLNN Capability Matrix

ACLNN bridge 以 ABI 参数 kind 为能力边界。每个 Case 的 `invocation.metadata.aclnn` 会在动态库解析前进行 preflight；未覆盖 ABI 被显式拒绝，不会进入不明所有权的 native 调用。

## 已支持

| ABI kind | 资源生命周期 | 验证状态 |
| --- | --- | --- |
| `tensor` | `aclCreateTensor` / `aclDestroyTensor` | 单输出、多输出 NPU E2E |
| `scalar` | `aclCreateScalar` / `aclDestroyScalar` | mock 生命周期覆盖 |
| `native_int` | caller-owned | Sort/MaxDim NPU E2E |
| `native_bool` | caller-owned | Sort/MaxDim NPU E2E |
| `int_array` | `aclCreateIntArray` / `aclDestroyIntArray` | Mean NPU E2E |
| `float_array` | `aclCreateFloatArray` / `aclDestroyFloatArray` | mock 生命周期覆盖 |
| `bool_array` | `aclCreateBoolArray` / `aclDestroyBoolArray` | mock 生命周期覆盖 |

## 明确不支持

| ABI kind | 拒绝原因 |
| --- | --- |
| `tensor_list` | 需要确认 CANN 版本对应的 tensor-list 创建、析构与 keepalive 所有权契约 |
| `optional_tensor` | 需要确认空 tensor 指针在 GetWorkspaceSize 和执行符号中的 ABI 语义 |
| `optional_scalar` | 需要确认空 scalar 指针在 GetWorkspaceSize 和执行符号中的 ABI 语义 |

新增 ABI 的完成标准：更新 `ACLNN_CAPABILITY_MATRIX`、实现资源创建与逆序析构、增加 mock 生命周期测试，并在目标 CANN/NPU 组合上新增独立 E2E gate。

# ACLNN Capability Matrix

ACLNN bridge uses ABI argument kinds and tensor-layout declarations as explicit capability boundaries. Each case's `invocation.metadata.aclnn` is validated before dynamic-library resolution. A rejected declaration never reaches a native call with unknown ownership.

The executable decision surfaces are `aclnn_capability_decision(arg)` and `aclnn_spec_capability_decisions(spec)`. They return `capability`, `status`, and `reason`; the spec form additionally emits the `multi_output_static` verdict. `AclnnRuntime.validate_capabilities` consumes the same argument decision, so report/preflight callers do not need to parse exception text. Status values are stable: `supported`, `preflight_blocked`, `future_abi`, and `unsupported` for unknown kinds.

A preflight failure is exported as `aclnn_capability`, `aclnn_capability_status`, and `aclnn_capability_reason` in result metrics and the stable report schema. The first blocking decision is also retained as the head element of `aclnn_capability_decisions_json`, which preserves the full spec-level decision list for audit. A capability may be promoted to `supported` only when the executable promotion gate has a matching lifecycle declaration and at least one `mock_lifecycle` or `npu_e2e` evidence declaration.

## Supported

| Capability | Resource lifecycle | Evidence boundary |
| --- | --- | --- |
| `tensor` with static `ND`, zero offset, and implicit contiguous layout | `aclCreateTensor` / `aclDestroyTensor` | NPU E2E |
| Static multi-output tensors | one tensor descriptor lifecycle per output | Sort and MaxDim NPU E2E |
| `scalar` | `aclCreateScalar` / `aclDestroyScalar` | mock lifecycle |
| `native_int`, `native_bool` | caller-owned | Sort/MaxDim NPU E2E |
| `int_array` | `aclCreateIntArray` / `aclDestroyIntArray` | Mean NPU E2E |
| `float_array`, `bool_array` | matching create/destroy API | mock lifecycle |

## Preflight Blocked

These declarations are parser-preserved and rejected before native symbol resolution. They remain blocked until a descriptor and memory-lifecycle contract is proven for the target CANN ABI.

| Declaration | Current rejection boundary |
| --- | --- |
| Dynamic output | dynamic allocation/descriptor lifecycle |
| Non-`ND` format | format-specific tensor descriptor contract |
| Non-zero storage offset or explicit strides | storage, offset, stride, and keepalive contract |

## Future ABI

These ABI forms have no established ownership/null contract. They are explicitly rejected and must not be implemented by forwarding Python objects or guessed null pointers.

| ABI kind | Missing proof |
| --- | --- |
| `tensor_list` | CANN-version-specific list creation, destruction, and tensor keepalive ownership |
| `optional_tensor_list` | null list semantics in both workspace and execute symbols |
| `optional_tensor` | null tensor semantics in both workspace and execute symbols |
| `optional_scalar` | null scalar semantics in both workspace and execute symbols |

Completion for a future ABI requires: capability-matrix update, concrete resource create/destroy in reverse order, mock lifecycle regression, and a target CANN/NPU E2E gate. Parser preservation or preflight rejection alone is not runtime support.

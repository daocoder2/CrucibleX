from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from cruciblex.runtime.executors import ExecutionNotSupportedError, ExecutionRequest

ACL_FORMAT_ND = 2

ACLNN_CAPABILITY_MATRIX = {
    "tensor": {"status": "supported", "lifecycle": "aclCreateTensor/aclDestroyTensor"},
    "scalar": {"status": "supported", "lifecycle": "aclCreateScalar/aclDestroyScalar"},
    "native_int": {"status": "supported", "lifecycle": "caller_owned"},
    "native_bool": {"status": "supported", "lifecycle": "caller_owned"},
    "int_array": {"status": "supported", "lifecycle": "aclCreateIntArray/aclDestroyIntArray"},
    "float_array": {"status": "supported", "lifecycle": "aclCreateFloatArray/aclDestroyFloatArray"},
    "bool_array": {"status": "supported", "lifecycle": "aclCreateBoolArray/aclDestroyBoolArray"},
    "tensor_list": {"status": "unsupported", "reason": "requires ACLNN tensor-list ownership contract"},
    "optional_tensor": {"status": "unsupported", "reason": "requires null tensor ABI contract"},
    "optional_scalar": {"status": "unsupported", "reason": "requires null scalar ABI contract"},
}
_TORCH_DTYPE_TO_ACL = {
    "torch.float32": 0,
    "torch.float16": 1,
    "torch.int8": 2,
    "torch.int32": 3,
    "torch.uint8": 4,
    "torch.int16": 6,
    "torch.int64": 9,
    "torch.bool": 12,
}
_DTYPE_NAME_TO_ACL = {
    "float": 0,
    "float32": 0,
    "fp32": 0,
    "float16": 1,
    "fp16": 1,
    "int8": 2,
    "int32": 3,
    "uint8": 4,
    "int16": 6,
    "int64": 9,
    "bool": 12,
}
_ACL_TO_CTYPES_SCALAR = {
    0: ctypes.c_float,
    1: ctypes.c_uint16,
    2: ctypes.c_int8,
    3: ctypes.c_int32,
    4: ctypes.c_uint8,
    6: ctypes.c_int16,
    9: ctypes.c_int64,
    12: ctypes.c_bool,
}


@dataclass(frozen=True, slots=True)
class AclnnArg:
    name: str
    kind: str = "tensor"
    role: str = "input"
    like: str | None = None
    value: Any = None
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    strides: tuple[int, ...] | None = None
    storage_offset: int = 0
    format: str = "ND"
    dynamic: bool = False
    optional: bool = False
    keyword_only: bool = False


AclnnTensorArg = AclnnArg


@dataclass(frozen=True, slots=True)
class AclnnOpSpec:
    op_name: str
    inputs: tuple[AclnnArg, ...]
    outputs: tuple[AclnnArg, ...]
    attributes: tuple[AclnnArg, ...] = ()

    @property
    def symbol(self) -> str:
        return normalize_aclnn_symbol(self.op_name)

    @property
    def workspace_symbol(self) -> str:
        return f"{self.symbol}GetWorkspaceSize"

    @property
    def call_args(self) -> tuple[AclnnArg, ...]:
        return (*self.inputs, *self.attributes, *self.outputs)


def normalize_aclnn_symbol(name: str) -> str:
    text = name.strip()
    if not text:
        raise ExecutionNotSupportedError("ACLNN op name is empty")
    if text.startswith("aclnn"):
        return text
    return f"aclnn{text[0].upper()}{text[1:]}"


def _arg_from_mapping(item: dict[str, Any], index: int, *, role: str) -> AclnnArg:
    return AclnnArg(
        name=str(item.get("name", f"{role}_{index}")),
        kind=str(item.get("kind", "tensor")),
        role=str(item.get("role", role)),
        like=str(item.get("like")) if item.get("like") is not None else None,
        value=item.get("value"),
        dtype=str(item.get("dtype")) if item.get("dtype") is not None else None,
        shape=_output_shape(item.get("shape")),
        strides=_output_strides(item.get("strides")),
        storage_offset=int(item.get("storage_offset", 0)),
        format=str(item.get("format", "ND")),
        dynamic=bool(item.get("dynamic", False)),
        optional=bool(item.get("optional", False)),
        keyword_only=bool(item.get("keyword_only", False)),
    )


def _output_strides(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, int) or item <= 0 for item in value):
        raise ExecutionNotSupportedError("ACLNN strides must be a list of positive integers")
    return tuple(value)


def _output_shape(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(dim, int) or dim < 0 for dim in value):
        raise ExecutionNotSupportedError("ACLNN output shape must be a list of non-negative integers")
    return tuple(value)


def op_spec_from_case(case: Any) -> AclnnOpSpec:
    metadata = case.invocation.metadata.get("aclnn", {}) if isinstance(case.invocation.metadata, dict) else {}
    op_name = str(metadata.get("op_name") or metadata.get("op") or case.invocation.api)
    raw_inputs = metadata.get("inputs") or [
        {"name": parameter.name or f"input_{index}", "kind": "tensor"}
        for index, parameter in enumerate(case.parameters)
    ]
    raw_outputs = metadata.get("outputs") or [
        {"name": "output", "kind": "tensor", "like": raw_inputs[0].get("name", "input") if raw_inputs else None}
    ]
    raw_attributes = metadata.get("attributes") or []
    return AclnnOpSpec(
        op_name=op_name,
        inputs=tuple(_arg_from_mapping(item, index, role="input") for index, item in enumerate(raw_inputs)),
        attributes=tuple(
            _arg_from_mapping(item, index, role="attribute") for index, item in enumerate(raw_attributes)
        ),
        outputs=tuple(_arg_from_mapping(item, index, role="output") for index, item in enumerate(raw_outputs)),
    )


class AclnnLibraryResolver:
    def __init__(self, search_roots: list[str | Path] | None = None) -> None:
        self.search_roots = [Path(root) for root in (search_roots or self._default_roots())]

    def resolve(self, spec: AclnnOpSpec) -> ctypes.CDLL:
        symbol = spec.workspace_symbol
        for library_path in self._candidate_libraries():
            try:
                library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            if hasattr(library, symbol) and hasattr(library, spec.symbol):
                return library
        raise ExecutionNotSupportedError(
            f"ACLNN op symbols not found: {spec.workspace_symbol}, {spec.symbol} under {self.search_roots}"
        )

    def _candidate_libraries(self) -> list[Path]:
        candidates: list[Path] = []
        for root in self.search_roots:
            if root.is_file() and root.name.startswith("libopapi.so"):
                candidates.append(root)
                continue
            if root.exists():
                candidates.extend(root.glob("**/libopapi.so*"))
        return sorted({path.resolve() for path in candidates})

    def _default_roots(self) -> list[str]:
        roots = []
        ascend_home = os.environ.get("ASCEND_HOME_PATH")
        if ascend_home:
            roots.append(ascend_home)
        roots.extend([
            "/usr/local/Ascend/ascend-toolkit/latest",
            "/usr/local/Ascend/ascend-toolkit",
        ])
        return roots


@dataclass(slots=True)
class _AclHandlePackage:
    handle: int
    destroy_name: str
    keepalive: Any = None


class AclnnRuntime:
    def __init__(self, resolver: AclnnLibraryResolver | None = None) -> None:
        self.resolver = resolver or AclnnLibraryResolver()

    def run(self, spec: AclnnOpSpec, inputs: list[object]) -> object:
        self.validate_capabilities(spec)
        if not spec.inputs or not spec.outputs:
            raise ExecutionNotSupportedError("ACLNN bridge currently requires tensor inputs and tensor outputs")
        torch = self._torch_npu()
        library = self.resolver.resolve(spec)
        self._configure_library(library)
        device_inputs = self._prepare_inputs(torch, spec, inputs)
        device_outputs = self._create_outputs(torch, spec, device_inputs)
        packages = self._marshal_call_args(library, spec, device_inputs, device_outputs)
        workspace = None
        try:
            workspace_size = ctypes.c_uint64(0)
            executor = ctypes.c_void_p()
            workspace_func = getattr(library, spec.workspace_symbol)
            workspace_args = [
                ctypes.c_void_p(package.handle) if isinstance(package, _AclHandlePackage) else package
                for package in packages
            ]
            ret = workspace_func(*workspace_args, ctypes.byref(workspace_size), ctypes.byref(executor))
            self._check_status(ret, spec.workspace_symbol)
            if workspace_size.value:
                import acl

                workspace, malloc_ret = acl.rt.malloc(workspace_size.value, 0)
                self._check_status(malloc_ret, "acl.rt.malloc")
            stream = ctypes.c_void_p(int(torch.npu.current_stream().npu_stream))
            run_func = getattr(library, spec.symbol)
            ret = run_func(ctypes.c_void_p(workspace or 0), workspace_size.value, executor, stream)
            self._check_status(ret, spec.symbol)
            torch.npu.synchronize()
            self.last_execution_evidence = {
                "backend_output_dtype": [str(output.dtype) for output in device_outputs],
                "backend_output_device": [str(output.device) for output in device_outputs],
                "backend_dtype_source": "device_tensor",
            }
            outputs = [output.detach().cpu().numpy() for output in device_outputs]
            return outputs[0] if len(outputs) == 1 else outputs
        finally:
            for package in reversed(packages):
                if isinstance(package, _AclHandlePackage) and package.handle:
                    getattr(library, package.destroy_name)(ctypes.c_void_p(package.handle))
            if workspace:
                import acl

                acl.rt.free(workspace)

    def validate_capabilities(self, spec: AclnnOpSpec) -> None:
        for arg in spec.call_args:
            capability = ACLNN_CAPABILITY_MATRIX.get(arg.kind)
            if capability is None:
                raise ExecutionNotSupportedError(f"unsupported ACLNN argument kind: {arg.kind}")
            if capability["status"] != "supported":
                raise ExecutionNotSupportedError(
                    f"unsupported ACLNN argument kind: {arg.kind}; {capability.get('reason', 'no capability')}"
                )
            if arg.kind == "tensor":
                if arg.format != "ND":
                    raise ExecutionNotSupportedError("ACLNN bridge supports only ND tensor format")
                if arg.storage_offset != 0:
                    raise ExecutionNotSupportedError("ACLNN bridge does not support non-zero tensor storage_offset")
                if arg.strides is not None:
                    raise ExecutionNotSupportedError("ACLNN bridge does not preserve declared tensor strides")
        for output in spec.outputs:
            if output.kind != "tensor":
                raise ExecutionNotSupportedError(f"unsupported ACLNN output kind: {output.kind}")
            if output.dynamic:
                raise ExecutionNotSupportedError("ACLNN bridge does not support dynamic output allocation")

    def _torch_npu(self):
        try:
            import torch
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise ExecutionNotSupportedError("ACLNN bridge requires torch and torch_npu") from exc
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise ExecutionNotSupportedError("ACLNN bridge requires an available torch_npu device")
        return torch

    def _configure_library(self, library: ctypes.CDLL) -> None:
        library.aclCreateTensor.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        library.aclCreateTensor.restype = ctypes.c_void_p
        library.aclDestroyTensor.argtypes = [ctypes.c_void_p]
        library.aclDestroyTensor.restype = ctypes.c_int32
        library.aclCreateScalar.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        library.aclCreateScalar.restype = ctypes.c_void_p
        library.aclDestroyScalar.argtypes = [ctypes.c_void_p]
        library.aclDestroyScalar.restype = ctypes.c_int32
        for name in ("aclCreateIntArray", "aclCreateFloatArray", "aclCreateBoolArray"):
            if hasattr(library, name):
                getattr(library, name).restype = ctypes.c_void_p
        if hasattr(library, "aclDestroyIntArray"):
            library.aclDestroyIntArray.argtypes = [ctypes.c_void_p]
            library.aclDestroyIntArray.restype = ctypes.c_int32
        if hasattr(library, "aclDestroyFloatArray"):
            library.aclDestroyFloatArray.argtypes = [ctypes.c_void_p]
            library.aclDestroyFloatArray.restype = ctypes.c_int32
        if hasattr(library, "aclDestroyBoolArray"):
            library.aclDestroyBoolArray.argtypes = [ctypes.c_void_p]
            library.aclDestroyBoolArray.restype = ctypes.c_int32

    def _prepare_inputs(self, torch: Any, spec: AclnnOpSpec, values: list[object]) -> dict[str, Any]:
        if len(values) < len(spec.inputs):
            raise ExecutionNotSupportedError(
                f"ACLNN op {spec.op_name} expects {len(spec.inputs)} inputs, got {len(values)}"
            )
        return {arg.name: self._to_npu_tensor(torch, values[index]) for index, arg in enumerate(spec.inputs)}

    def _to_npu_tensor(self, torch: Any, value: object) -> Any:
        tensor = value if self._is_torch_tensor(torch, value) else torch.as_tensor(np.asarray(value))
        if str(tensor.device).split(":", 1)[0] != "npu":
            tensor = tensor.to("npu:0")
        return tensor.contiguous()

    def _create_outputs(self, torch: Any, spec: AclnnOpSpec, inputs: dict[str, Any]) -> list[Any]:
        outputs: list[Any] = []
        for output in spec.outputs:
            if output.kind != "tensor":
                raise ExecutionNotSupportedError(f"unsupported ACLNN output kind: {output.kind}")
            source = inputs.get(output.like) if output.like is not None else next(iter(inputs.values()))
            if source is None:
                raise ExecutionNotSupportedError(f"ACLNN output {output.name} references unknown input {output.like}")
            dtype = self._output_dtype(torch, output.dtype) if output.dtype else source.dtype
            if output.shape is not None:
                outputs.append(torch.empty(output.shape, device=source.device, dtype=dtype))
            elif output.dtype:
                outputs.append(torch.empty_like(source, dtype=dtype))
            else:
                outputs.append(torch.empty_like(source))
        return outputs

    def _output_dtype(self, torch: Any, dtype_name: str) -> Any:
        aliases = {
            "fp16": "float16",
            "fp32": "float32",
            "fp64": "float64",
            "int32": "int32",
            "int64": "int64",
            "bool": "bool",
        }
        attribute = aliases.get(dtype_name.lower())
        if attribute is None or not hasattr(torch, attribute):
            raise ExecutionNotSupportedError(f"unsupported ACLNN output dtype: {dtype_name}")
        return getattr(torch, attribute)

    def _marshal_call_args(
        self,
        library: ctypes.CDLL,
        spec: AclnnOpSpec,
        inputs: dict[str, Any],
        outputs: list[Any],
    ) -> list[_AclHandlePackage]:
        output_by_name = {arg.name: outputs[index] for index, arg in enumerate(spec.outputs)}
        packages: list[_AclHandlePackage] = []
        for arg in spec.call_args:
            if arg.kind == "tensor":
                tensor = output_by_name[arg.name] if arg.role == "output" else inputs[arg.name]
                packages.append(self._create_tensor_package(library, tensor))
            elif arg.kind in {"scalar", "int", "float", "bool"}:
                packages.append(self._create_scalar_package(library, arg))
            elif arg.kind == "native_int":
                packages.append(ctypes.c_int64(int(arg.value)))
            elif arg.kind == "native_bool":
                packages.append(ctypes.c_bool(bool(arg.value)))
            elif arg.kind in {"int_array", "float_array", "bool_array"}:
                packages.append(self._create_array_package(library, arg))
            elif arg.kind in {"tensor_list", "optional_tensor", "optional_scalar"}:
                raise ExecutionNotSupportedError(f"unsupported ACLNN argument kind: {arg.kind}")
            else:
                raise ExecutionNotSupportedError(f"unsupported ACLNN argument kind: {arg.kind}")
        return packages

    def _create_tensor_package(self, library: ctypes.CDLL, tensor: Any) -> _AclHandlePackage:
        acl_dtype = self._acl_dtype(tensor)
        shape = [int(dim) for dim in tensor.shape]
        strides = [int(dim) for dim in tensor.stride()]
        view_dims = (ctypes.c_int64 * len(shape))(*shape)
        view_strides = (ctypes.c_int64 * len(strides))(*strides)
        storage_dims = (ctypes.c_int64 * len(shape))(*shape)
        descriptor = library.aclCreateTensor(
            view_dims,
            len(shape),
            acl_dtype,
            view_strides,
            0,
            ACL_FORMAT_ND,
            storage_dims,
            len(shape),
            ctypes.c_void_p(int(tensor.data_ptr())),
        )
        if not descriptor:
            raise ExecutionNotSupportedError("aclCreateTensor returned null")
        return _AclHandlePackage(handle=int(descriptor), destroy_name="aclDestroyTensor", keepalive=tensor)

    def _create_scalar_package(self, library: ctypes.CDLL, arg: AclnnArg) -> _AclHandlePackage:
        acl_dtype = _DTYPE_NAME_TO_ACL.get((arg.dtype or "fp32").lower())
        if acl_dtype is None or acl_dtype not in _ACL_TO_CTYPES_SCALAR:
            raise ExecutionNotSupportedError(f"unsupported ACLNN scalar dtype: {arg.dtype}")
        value = _ACL_TO_CTYPES_SCALAR[acl_dtype](arg.value if arg.value is not None else 1)
        scalar = library.aclCreateScalar(ctypes.byref(value), acl_dtype)
        if not scalar:
            raise ExecutionNotSupportedError("aclCreateScalar returned null")
        return _AclHandlePackage(handle=int(scalar), destroy_name="aclDestroyScalar", keepalive=value)

    def _create_array_package(self, library: ctypes.CDLL, arg: AclnnArg) -> _AclHandlePackage:
        values = arg.value if isinstance(arg.value, (list, tuple)) else []
        if arg.kind == "int_array":
            value_type, create_name, destroy_name = ctypes.c_int64, "aclCreateIntArray", "aclDestroyIntArray"
        elif arg.kind == "float_array":
            value_type, create_name, destroy_name = ctypes.c_double, "aclCreateFloatArray", "aclDestroyFloatArray"
        else:
            value_type, create_name, destroy_name = ctypes.c_bool, "aclCreateBoolArray", "aclDestroyBoolArray"
        if not hasattr(library, create_name) or not hasattr(library, destroy_name):
            raise ExecutionNotSupportedError(f"ACLNN runtime lacks {create_name}/{destroy_name}")
        buffer = (value_type * len(values))(*values)
        descriptor = getattr(library, create_name)(buffer, len(values))
        if not descriptor:
            raise ExecutionNotSupportedError(f"{create_name} returned null")
        return _AclHandlePackage(handle=int(descriptor), destroy_name=destroy_name, keepalive=buffer)

    def _acl_dtype(self, tensor: Any) -> int:
        dtype_name = str(tensor.dtype)
        try:
            return _TORCH_DTYPE_TO_ACL[dtype_name]
        except KeyError as exc:
            raise ExecutionNotSupportedError(f"unsupported ACLNN tensor dtype: {dtype_name}") from exc

    def _is_torch_tensor(self, torch: Any, value: object) -> bool:
        return isinstance(value, torch.Tensor)

    def _check_status(self, status: int, operation: str) -> None:
        if int(status) != 0:
            raise RuntimeError(f"{operation} failed with ACLNN status {status}")


class AclnnAdapter(Protocol):
    def execute(self, request: ExecutionRequest) -> object: ...


class AclnnFunctionAdapter:
    def __init__(self, runtime: AclnnRuntime | None = None) -> None:
        self.runtime = runtime or AclnnRuntime()

    def execute(self, request: ExecutionRequest) -> object:
        spec = op_spec_from_case(request.case)
        binding = request.case.invocation.metadata.get("binding", {})
        omitted = binding.get("omit", []) if isinstance(binding, dict) else []
        omitted_names = {str(item) for item in omitted if isinstance(item, str)}
        omitted_indexes = {item for item in omitted if isinstance(item, int)}
        selected_inputs = []
        selected_values = []
        for index, (argument, value) in enumerate(zip(spec.inputs, request.inputs, strict=False)):
            if argument.name in omitted_names or index in omitted_indexes:
                if not argument.optional:
                    raise ValueError(f"cannot omit required ACLNN argument: {argument.name}")
                continue
            selected_inputs.append(argument)
            selected_values.append(value)
        selected_attributes = tuple(
            argument for argument in spec.attributes if argument.name not in omitted_names
        )
        spec = AclnnOpSpec(
            op_name=spec.op_name,
            inputs=tuple(selected_inputs),
            attributes=selected_attributes,
            outputs=spec.outputs,
        )
        return self.runtime.run(spec, selected_values)


class AclnnAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, type[AclnnFunctionAdapter]] = {}

    def register(self, api_type: str, adapter: type[AclnnFunctionAdapter]) -> None:
        self._adapters[api_type] = adapter

    def resolve(self, api_type: str) -> AclnnFunctionAdapter:
        try:
            return self._adapters[api_type]()
        except KeyError as exc:
            raise ExecutionNotSupportedError(f"unsupported ACLNN api_type: {api_type}") from exc

    def supports(self, api_type: str) -> bool:
        return api_type in self._adapters


ACLNN_ADAPTER_REGISTRY = AclnnAdapterRegistry()
ACLNN_ADAPTER_REGISTRY.register("aclnn_function", AclnnFunctionAdapter)
ACLNN_ADAPTER_REGISTRY.register("aclnn", AclnnFunctionAdapter)


def output_like_first_input(inputs: list[object]) -> np.ndarray:
    if not inputs:
        raise ExecutionNotSupportedError("ACLNN tensor op requires at least one input")
    return np.empty_like(np.asarray(inputs[0]))

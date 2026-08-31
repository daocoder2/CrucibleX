from types import SimpleNamespace

import pytest

from cruciblex.plugins.executors.aclnn_bridge import (
    ACLNN_CAPABILITY_MATRIX,
    AclnnArg,
    AclnnOpSpec,
    AclnnRuntime,
    op_spec_from_case,
)
from cruciblex.runtime.executors.base import ExecutionNotSupportedError


def _case(metadata):
    return SimpleNamespace(
        invocation=SimpleNamespace(api="aclnnSoftmax", metadata={"aclnn": metadata}),
        parameters=[],
    )


def test_op_schema_supports_array_attributes_and_multiple_outputs():
    spec = op_spec_from_case(_case({
        "op_name": "Softmax",
        "inputs": [{"name": "input", "kind": "tensor"}],
        "attributes": [
            {"name": "axes", "kind": "int_array", "value": [1, 2]},
            {"name": "keepdim", "kind": "bool", "value": False, "optional": True},
        ],
        "outputs": [
            {"name": "output", "kind": "tensor", "like": "input"},
            {"name": "aux", "kind": "tensor", "like": "input"},
        ],
    }))
    assert [argument.kind for argument in spec.attributes] == ["int_array", "bool"]
    assert [argument.value for argument in spec.attributes] == [[1, 2], False]
    assert len(spec.outputs) == 2


def test_output_dtype_is_preserved_in_schema():
    spec = op_spec_from_case(_case({
        "op_name": "Sort",
        "inputs": [{"name": "input", "kind": "tensor"}],
        "outputs": [
            {"name": "values", "kind": "tensor", "like": "input", "dtype": "fp32"},
            {"name": "indices", "kind": "tensor", "like": "input", "dtype": "int64"},
        ],
    }))
    assert [argument.dtype for argument in spec.outputs] == ["fp32", "int64"]


def test_output_shape_is_parsed_and_validated():
    spec = op_spec_from_case(_case({
        "op_name": "MaxDim",
        "inputs": [{"name": "input", "kind": "tensor"}],
        "outputs": [{"name": "values", "kind": "tensor", "like": "input", "shape": [2, 1]}],
    }))
    assert spec.outputs[0].shape == (2, 1)

    with pytest.raises(ExecutionNotSupportedError, match="output shape"):
        op_spec_from_case(_case({
            "op_name": "MaxDim",
            "inputs": [{"name": "input", "kind": "tensor"}],
            "outputs": [{"name": "values", "kind": "tensor", "shape": [-1]}],
        }))


def test_schema_keeps_native_abi_attribute_kinds():
    spec = op_spec_from_case(_case({
        "op_name": "Sort",
        "inputs": [{"name": "input", "kind": "tensor"}],
        "attributes": [{"name": "dim", "kind": "native_int", "value": 1}],
        "outputs": [{"name": "output", "kind": "tensor", "like": "input"}],
    }))
    assert spec.attributes[0].kind == "native_int"


def test_int_array_attribute_is_preserved_for_real_array_abi():
    spec = op_spec_from_case(_case({
        "op_name": "Mean",
        "inputs": [{"name": "input", "kind": "tensor"}],
        "attributes": [{"name": "dim", "kind": "int_array", "value": [1]}],
        "outputs": [{"name": "output", "kind": "tensor", "like": "input", "shape": [2]}],
    }))
    assert spec.attributes[0].kind == "int_array"
    assert spec.attributes[0].value == [1]


def test_array_marshaling_requires_runtime_symbols():
    runtime = AclnnRuntime()
    library = SimpleNamespace()
    with pytest.raises(ExecutionNotSupportedError, match="aclCreateIntArray"):
        runtime._create_array_package(library, SimpleNamespace(kind="int_array", value=[1, 2], dtype=None))



def test_capability_matrix_documents_supported_lifecycle_and_native_abi_boundaries():
    assert ACLNN_CAPABILITY_MATRIX["int_array"]["lifecycle"] == "aclCreateIntArray/aclDestroyIntArray"
    assert ACLNN_CAPABILITY_MATRIX["tensor_list"]["status"] == "unsupported"

    runtime = AclnnRuntime()
    supported = AclnnOpSpec(
        op_name="Mean",
        inputs=(AclnnArg(name="input"),),
        attributes=(AclnnArg(name="dim", kind="int_array", value=[1]),),
        outputs=(AclnnArg(name="output", role="output"),),
    )
    unsupported = AclnnOpSpec(
        op_name="Example",
        inputs=(AclnnArg(name="inputs", kind="tensor_list"),),
        outputs=(AclnnArg(name="output", role="output"),),
    )

    runtime.validate_capabilities(supported)
    with pytest.raises(ExecutionNotSupportedError, match="tensor-list ownership"):
        runtime.validate_capabilities(unsupported)

from types import SimpleNamespace

import pytest

from cruciblex.domain.manifest import HARDWARE_EVIDENCE_ARCHIVE_FILES
from cruciblex.plugins.executors.aclnn_bridge import (
    ACLNN_CAPABILITY_EVIDENCE,
    ACLNN_CAPABILITY_MATRIX,
    AclnnArg,
    AclnnCapabilityError,
    AclnnOpSpec,
    AclnnRuntime,
    aclnn_capability_decision,
    aclnn_spec_capability_decisions,
    op_spec_from_case,
    validate_aclnn_capability_promotions,
)
from cruciblex.runtime.executors.base import ExecutionNotSupportedError
from cruciblex.runtime.pipeline import ExecutionPipeline


def _case(metadata):
    return SimpleNamespace(
        invocation=SimpleNamespace(api="aclnnSoftmax", metadata={"aclnn": metadata}),
        parameters=[],
    )


def test_supported_capability_promotion_gate_requires_matching_lifecycle_and_evidence():
    validate_aclnn_capability_promotions()

    capabilities = {"new_abi": {"status": "supported", "lifecycle": "aclCreateNew/aclDestroyNew"}}
    with pytest.raises(ValueError, match="missing lifecycle evidence"):
        validate_aclnn_capability_promotions(capabilities, {})

    evidence = {"new_abi": {"lifecycle": "other", "evidence": ("mock_lifecycle",)}}
    with pytest.raises(ValueError, match="mismatched lifecycle evidence"):
        validate_aclnn_capability_promotions(capabilities, evidence)

    evidence["new_abi"] = {"lifecycle": "aclCreateNew/aclDestroyNew", "evidence": ("npu_e2e",)}
    with pytest.raises(ValueError, match="missing NPU evidence manifest"):
        validate_aclnn_capability_promotions(capabilities, evidence)

    evidence["new_abi"] = {"lifecycle": "aclCreateNew/aclDestroyNew", "evidence": ("npu_e2e",), "manifest": "examples/manifests/aclnn-supported-evidence.yaml", "case_ids": (999,)}
    with pytest.raises(ValueError, match="Manifest v1 evidence archive contract"):
        validate_aclnn_capability_promotions(capabilities, evidence)

    evidence["new_abi"]["archive_files"] = HARDWARE_EVIDENCE_ARCHIVE_FILES
    validate_aclnn_capability_promotions(capabilities, evidence)


def test_capability_preflight_error_carries_structured_verdict():
    spec = AclnnOpSpec(
        op_name="Example",
        inputs=(AclnnArg(name="inputs", kind="tensor_list"),),
        outputs=(AclnnArg(name="output", role="output"),),
    )

    with pytest.raises(AclnnCapabilityError) as raised:
        AclnnRuntime().validate_capabilities(spec)

    assert raised.value.capability_decision == {
        "capability": "tensor_list",
        "status": "future_abi",
        "reason": "unsupported ACLNN argument kind: tensor_list; requires ACLNN tensor-list ownership contract",
    }
    assert raised.value.capability_decisions[0] == raised.value.capability_decision
    assert raised.value.capability_decisions[1] == {"capability": "tensor", "status": "supported", "reason": ""}


def test_pipeline_projects_only_structured_acl_capability_errors():
    error = AclnnCapabilityError({
        "capability": "dynamic_output",
        "status": "preflight_blocked",
        "reason": "ACLNN bridge does not support dynamic output allocation",
    })

    assert ExecutionPipeline()._capability_error_metrics(error) == {
        "aclnn_capability": "dynamic_output",
        "aclnn_capability_status": "preflight_blocked",
        "aclnn_capability_reason": "ACLNN bridge does not support dynamic output allocation",
        "aclnn_capability_decisions": [{
            "capability": "dynamic_output",
            "status": "preflight_blocked",
            "reason": "ACLNN bridge does not support dynamic output allocation",
        }],
    }
    assert ExecutionPipeline()._capability_error_metrics(ExecutionNotSupportedError("ordinary error")) == {}


def test_capability_matrix_distinguishes_supported_blocked_and_future_abi():
    supported = {name for name, value in ACLNN_CAPABILITY_MATRIX.items() if value["status"] == "supported"}
    assert supported == set(ACLNN_CAPABILITY_EVIDENCE)
    assert ACLNN_CAPABILITY_MATRIX["tensor"]["status"] == "supported"
    assert ACLNN_CAPABILITY_MATRIX["multi_output_static"]["status"] == "supported"
    assert ACLNN_CAPABILITY_MATRIX["dynamic_output"]["status"] == "preflight_blocked"
    assert ACLNN_CAPABILITY_MATRIX["non_nd_format"]["status"] == "preflight_blocked"
    assert ACLNN_CAPABILITY_MATRIX["non_contiguous_storage"]["status"] == "preflight_blocked"
    assert ACLNN_CAPABILITY_MATRIX["tensor_list"]["status"] == "future_abi"
    assert ACLNN_CAPABILITY_MATRIX["optional_tensor_list"]["status"] == "future_abi"
    assert ACLNN_CAPABILITY_MATRIX["optional_tensor"]["status"] == "future_abi"



@pytest.mark.parametrize(
    ("arg", "capability", "status"),
    [
        (AclnnArg(name="input"), "tensor", "supported"),
        (AclnnArg(name="output", role="output"), "tensor", "supported"),
        (AclnnArg(name="output", role="output", dynamic=True), "dynamic_output", "preflight_blocked"),
        (AclnnArg(name="input", format="FRACTAL_NZ"), "non_nd_format", "preflight_blocked"),
        (AclnnArg(name="input", storage_offset=1), "non_contiguous_storage", "preflight_blocked"),
        (AclnnArg(name="input", strides=(4, 1)), "non_contiguous_storage", "preflight_blocked"),
        (AclnnArg(name="inputs", kind="tensor_list"), "tensor_list", "future_abi"),
        (AclnnArg(name="inputs", kind="optional_tensor_list"), "optional_tensor_list", "future_abi"),
        (AclnnArg(name="input", kind="optional_tensor"), "optional_tensor", "future_abi"),
        (AclnnArg(name="value", kind="optional_scalar"), "optional_scalar", "future_abi"),
    ],
)
def test_capability_decision_exposes_executable_acl_abi_boundary(arg, capability, status):
    decision = aclnn_capability_decision(arg)

    assert decision["capability"] == capability
    assert decision["status"] == status
    if status != "supported":
        assert decision["reason"]


def test_spec_capability_decisions_distinguish_static_and_dynamic_multi_output():
    static_spec = AclnnOpSpec(
        op_name="Sort",
        inputs=(AclnnArg(name="input"),),
        outputs=(AclnnArg(name="values", role="output"), AclnnArg(name="indices", role="output")),
    )
    dynamic_spec = AclnnOpSpec(
        op_name="Example",
        inputs=(AclnnArg(name="input"),),
        outputs=(AclnnArg(name="output", role="output", dynamic=True), AclnnArg(name="aux", role="output")),
    )

    assert aclnn_spec_capability_decisions(static_spec)[-1] == {
        "capability": "multi_output_static", "status": "supported", "reason": ""
    }
    assert aclnn_spec_capability_decisions(dynamic_spec)[-1] == {
        "capability": "dynamic_output", "status": "preflight_blocked",
        "reason": "ACLNN bridge does not support dynamic output allocation",
    }


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
    assert ACLNN_CAPABILITY_MATRIX["tensor_list"]["status"] == "future_abi"

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


def test_preflight_rejects_storage_offset_independently():
    spec = AclnnOpSpec(
        op_name="Offset",
        inputs=(AclnnArg(name="input", storage_offset=1),),
        outputs=(AclnnArg(name="output", role="output"),),
    )

    with pytest.raises(ExecutionNotSupportedError, match="non-zero tensor storage_offset"):
        AclnnRuntime().validate_capabilities(spec)


def test_schema_preserves_tensor_list_and_optional_declarations_before_preflight():
    spec = op_spec_from_case(_case({
        "inputs": [
            {"name": "inputs", "kind": "tensor_list", "optional": True},
            {"name": "mask", "kind": "optional_tensor", "optional": True},
        ],
        "attributes": [{"name": "scale", "kind": "optional_scalar", "optional": True, "value": None}],
        "outputs": [{"name": "output", "kind": "tensor", "like": "inputs"}],
    }))

    assert [(argument.kind, argument.optional) for argument in spec.inputs] == [("tensor_list", True), ("optional_tensor", True)]
    assert spec.attributes[0].kind == "optional_scalar"
    assert spec.attributes[0].optional is True
    for index, message in [(0, "tensor-list ownership"), (1, "null tensor ABI contract")]:
        with pytest.raises(ExecutionNotSupportedError, match=message):
            AclnnRuntime().validate_capabilities(AclnnOpSpec(op_name="Example", inputs=(spec.inputs[index],), outputs=spec.outputs))
    with pytest.raises(ExecutionNotSupportedError, match="null scalar ABI contract"):
        AclnnRuntime().validate_capabilities(AclnnOpSpec(op_name="Example", inputs=(AclnnArg(name="input"),), attributes=spec.attributes, outputs=spec.outputs))


def test_schema_preserves_layout_declarations_and_preflight_blocks_unsupported_abi():
    spec = op_spec_from_case(_case({
        "inputs": [{"name": "input", "kind": "tensor", "format": "FRACTAL_NZ", "strides": [4, 1], "storage_offset": 2}],
        "outputs": [{"name": "output", "kind": "tensor", "like": "input", "dynamic": True}],
    }))

    assert spec.inputs[0].format == "FRACTAL_NZ"
    assert spec.inputs[0].strides == (4, 1)
    assert spec.inputs[0].storage_offset == 2
    assert spec.outputs[0].dynamic is True
    with pytest.raises(ExecutionNotSupportedError, match="only ND tensor format"):
        AclnnRuntime().validate_capabilities(spec)

    dynamic = AclnnOpSpec(
        op_name="Example",
        inputs=(AclnnArg(name="input"),),
        outputs=(AclnnArg(name="output", role="output", dynamic=True),),
    )
    with pytest.raises(ExecutionNotSupportedError, match="dynamic output allocation"):
        AclnnRuntime().validate_capabilities(dynamic)

    strided = AclnnOpSpec(
        op_name="Example",
        inputs=(AclnnArg(name="input", strides=(4, 1)),),
        outputs=(AclnnArg(name="output", role="output"),),
    )
    with pytest.raises(ExecutionNotSupportedError, match="declared tensor strides"):
        AclnnRuntime().validate_capabilities(strided)

@pytest.mark.parametrize(
    ("output", "message"),
    [
        (AclnnArg(name="output", role="output", format="FRACTAL_NZ"), "only ND tensor format"),
        (AclnnArg(name="output", role="output", storage_offset=1), "non-zero tensor storage_offset"),
        (AclnnArg(name="output", role="output", strides=(4, 1)), "declared tensor strides"),
    ],
)
def test_output_layout_declarations_are_preserved_and_preflight_blocked(output, message):
    spec = AclnnOpSpec(
        op_name="Example",
        inputs=(AclnnArg(name="input"),),
        outputs=(output,),
    )

    with pytest.raises(ExecutionNotSupportedError, match=message):
        AclnnRuntime().validate_capabilities(spec)

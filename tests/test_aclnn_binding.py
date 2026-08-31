import pytest

from cruciblex.domain import CaseSpec, InvocationSpec, OperatorSpec, ParameterKind, ParameterSpec
from cruciblex.plugins.executors.aclnn_bridge import AclnnFunctionAdapter
from cruciblex.runtime.executors.base import ExecutionRequest


class CaptureRuntime:
    def run(self, spec, inputs):
        self.spec = spec
        self.inputs = inputs
        return "captured"


def _request(optional=False):
    case = CaseSpec(
        id=1,
        operator=OperatorSpec(name="aclnn.Fake"),
        invocation=InvocationSpec(
            api="aclnnFake",
            api_type="aclnn_function",
            metadata={
                "aclnn": {
                    "inputs": [
                        {"name": "input", "kind": "tensor"},
                        {"name": "bias", "kind": "tensor", "optional": optional},
                    ],
                    "attributes": [{"name": "alpha", "kind": "scalar", "optional": optional}],
                }
            },
        ),
        parameters=[
            ParameterSpec(name="input", kind=ParameterKind.TENSOR),
            ParameterSpec(name="bias", kind=ParameterKind.TENSOR, required=not optional),
        ],
    )
    plan = None
    return ExecutionRequest(case=case, inputs=["input-value", "bias-value"], plan=plan)


def test_optional_aclnn_arguments_are_removed_from_spec_and_values():
    runtime = CaptureRuntime()
    request = _request(optional=True)
    request.case.invocation.metadata["binding"] = {"omit": ["bias", "alpha"]}

    assert AclnnFunctionAdapter(runtime).execute(request) == "captured"
    assert [argument.name for argument in runtime.spec.inputs] == ["input"]
    assert [argument.name for argument in runtime.spec.attributes] == []
    assert runtime.inputs == ["input-value"]


def test_required_aclnn_argument_cannot_be_omitted():
    runtime = CaptureRuntime()
    request = _request(optional=False)
    request.case.invocation.metadata["binding"] = {"omit": ["bias"]}

    with pytest.raises(ValueError, match="required ACLNN argument"):
        AclnnFunctionAdapter(runtime).execute(request)

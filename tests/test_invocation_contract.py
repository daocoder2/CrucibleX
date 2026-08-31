import numpy as np

from cruciblex.domain import (
    CaseSpec,
    InvocationBindingSpec,
    InvocationSpec,
    OperatorSpec,
    ParameterKind,
    ParameterSpec,
)
from cruciblex.domain.enums import ExecutionRole
from cruciblex.generation.expand import expand_cases
from cruciblex.generation.loader import load_cases
from cruciblex.plugins.executors.numpy import NumpyFunctionExecutor
from cruciblex.runtime.executors.base import ExecutionRequest


def _request(binding: InvocationBindingSpec | None = None, metadata: dict | None = None) -> ExecutionRequest:
    case = CaseSpec(
        id=1,
        operator=OperatorSpec(name="numpy.sum"),
        invocation=InvocationSpec(
            api="numpy.sum",
            api_type="function",
            executor="numpy",
            binding=binding,
            metadata=metadata or {},
        ),
        parameters=[
            ParameterSpec(name="input", kind=ParameterKind.TENSOR),
            ParameterSpec(name="axis", kind=ParameterKind.ATTRIBUTE, required=False),
        ],
    )
    return ExecutionRequest(case=case, inputs=[], plan=None, role=ExecutionRole.CANDIDATE)


def test_typed_binding_is_preferred_over_legacy_metadata():
    request = _request(
        binding=InvocationBindingSpec(mode="keyword", names=["a", "axis"]),
        metadata={"binding": {"mode": "keyword", "omit": ["axis"]}},
    )

    args, kwargs = request.call_arguments([1, 2])

    assert args == []
    assert kwargs == {"a": 1, "axis": 2}


def test_legacy_binding_metadata_remains_readable():
    args, kwargs = _request(metadata={"binding": {"mode": "mixed", "positional": [0]}}).call_arguments([1, 2])

    assert args == [1]
    assert kwargs == {"axis": 2}


def test_numpy_executor_applies_typed_keyword_binding():
    request = _request(binding=InvocationBindingSpec(mode="keyword", names=["a", "axis"]))
    request = ExecutionRequest(
        case=request.case,
        inputs=[np.asarray([[1, 2], [3, 4]]), 1],
        plan=None,
        role=ExecutionRole.CANDIDATE,
    )

    output = NumpyFunctionExecutor().execute(request)

    np.testing.assert_array_equal(output, np.asarray([3, 7]))


def test_reduction_example_declares_typed_keyword_binding():
    case = load_cases("examples/cases/numpy.sum.yaml")[0]

    assert case.invocation.binding == InvocationBindingSpec(mode="keyword", names=["a", "axis"])


def test_mean_example_declares_binding_and_executes_numpy_reduction():
    case = expand_cases(load_cases("examples/cases/numpy.mean.yaml"))[0]
    request = ExecutionRequest(
        case=case,
        inputs=[np.asarray([[3, 1, 4, 2], [8, 6, 7, 5]], dtype=np.float32), 1],
        plan=None,
        role=ExecutionRole.CANDIDATE,
    )

    output = NumpyFunctionExecutor().execute(request)

    assert case.invocation.binding == InvocationBindingSpec(mode="keyword", names=["a", "axis"])
    np.testing.assert_array_equal(output, np.asarray([2.5, 6.5], dtype=np.float32))


def test_broadcast_example_resolves_shapes_and_executes_numpy_add():
    case = expand_cases(load_cases("examples/cases/numpy.add.broadcast.yaml"))[0]
    request = ExecutionRequest(
        case=case,
        inputs=[np.ones((2, 3, 4), dtype=np.float32), np.full((1, 3, 1), 2, dtype=np.float32)],
        plan=None,
        role=ExecutionRole.CANDIDATE,
    )

    output = NumpyFunctionExecutor().execute(request)

    assert [parameter.shape.dims for parameter in case.parameters] == [[2, 3, 4], [1, 3, 1]]
    assert case.parameters[1].metadata["resolved_shape_relationship"] == "broadcastable_with"
    np.testing.assert_array_equal(output, np.full((2, 3, 4), 3, dtype=np.float32))

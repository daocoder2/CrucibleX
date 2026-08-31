from cruciblex.domain import (
    CaseSpec,
    InvocationBindingSpec,
    InvocationSpec,
    OperatorSpec,
    ParameterKind,
    ParameterSpec,
)
from cruciblex.domain.enums import ExecutionRole
from cruciblex.runtime.executors.base import ExecutionRequest


def _request(metadata):
    case = CaseSpec(id=1, operator=OperatorSpec(name="x"), invocation=InvocationSpec(api="x", api_type="function", metadata=metadata), parameters=[ParameterSpec(name="input", kind=ParameterKind.SCALAR), ParameterSpec(name="dim", kind=ParameterKind.ATTRIBUTE, required=False)])
    return ExecutionRequest(case=case, inputs=[1, 2], plan=None, role=ExecutionRole.CANDIDATE)


def test_typed_binding_is_preferred_over_legacy_metadata():
    request = _request({"binding": {"mode": "keyword", "omit": ["dim"]}})
    request.case.invocation.binding = InvocationBindingSpec(mode="keyword", names=["input", "dim"])
    args, kwargs = request.call_arguments([1, 2])
    assert args == []
    assert kwargs == {"input": 1, "dim": 2}


def test_default_binding_remains_positional():
    args, kwargs = _request({}).call_arguments([1, 2])
    assert args == [1, 2]
    assert kwargs == {}


def test_keyword_binding_and_explicit_omission():
    args, kwargs = _request({"binding": {"mode": "keyword", "omit": ["dim"]}}).call_arguments([1, 2])
    assert args == []
    assert kwargs == {"input": 1}


def test_mixed_binding_keeps_selected_positional_arguments():
    args, kwargs = _request({"binding": {"mode": "mixed", "positional": [0]}}).call_arguments([1, 2])
    assert args == [1]
    assert kwargs == {"dim": 2}


def test_binding_rejects_missing_parameter_names():
    request = _request({"binding": {"mode": "keyword", "names": ["input"]}})
    try:
        request.call_arguments([1, 2])
    except ValueError as exc:
        assert "cover every input" in str(exc)
    else:
        raise AssertionError("expected binding validation error")


def test_keyword_only_parameter_cannot_be_positional():
    request = _request({"binding": {"mode": "mixed", "positional": [0, 1]}})
    request.case.parameters[1].metadata["keyword_only"] = True
    try:
        request.call_arguments([1, 2])
    except ValueError as exc:
        assert "keyword-only" in str(exc)
    else:
        raise AssertionError("expected keyword-only validation error")

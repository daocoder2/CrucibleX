import sys
from types import SimpleNamespace

from cruciblex.domain import (
    BackendKind,
    CaseSpec,
    DeviceSpec,
    InvocationSpec,
    JobSpec,
    NodeSpec,
    OperatorSpec,
    RuntimePolicySpec,
    TaskKind,
)
from cruciblex.runtime.backends.base import DeviceContext
from cruciblex.runtime.pipeline import ExecutionPipeline
from cruciblex.runtime.planner import ExecutionPlanner


def _plan(policy, capabilities=()):
    case = CaseSpec(
        id=1,
        operator=OperatorSpec(name="identity"),
        invocation=InvocationSpec(api="identity", api_type="function"),
        runtime_policy=policy,
    )
    node = NodeSpec(
        name="node",
        runtime_policy_capabilities=set(capabilities),
        devices=[DeviceSpec(id=0, backend=BackendKind.CPU)],
    )
    plan = ExecutionPlanner().build(JobSpec(cases=[case], nodes=[node], tasks=[TaskKind.RUN]))[0]
    return plan, DeviceContext.from_node(node, plan.device, plan.artifacts.output_root)


def test_runtime_policy_preserves_default_synchronization_without_request():
    plan, context = _plan(RuntimePolicySpec())

    evidence, synchronize = ExecutionPipeline()._apply_runtime_policy(plan, context)

    assert evidence == {}
    assert synchronize is True


def test_runtime_policy_applies_declared_synchronization_capability():
    plan, context = _plan(RuntimePolicySpec(synchronize_timing=False), ["synchronize_timing"])

    evidence, synchronize = ExecutionPipeline()._apply_runtime_policy(plan, context)

    assert synchronize is False
    assert evidence["requested"] == {"synchronize_timing": False}
    assert evidence["effective"] == {"synchronize_timing": False}
    assert evidence["unsupported"] == []


def test_runtime_policy_records_unsupported_request_without_changing_default():
    plan, context = _plan(RuntimePolicySpec(synchronize_timing=False))

    evidence, synchronize = ExecutionPipeline()._apply_runtime_policy(plan, context)

    assert synchronize is True
    assert evidence["unsupported"] == ["synchronize_timing"]


def test_runtime_policy_applies_deterministic_with_torch_runtime(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(use_deterministic_algorithms=calls.append))
    plan, context = _plan(RuntimePolicySpec(deterministic=True), ["deterministic"])

    evidence, _ = ExecutionPipeline()._apply_runtime_policy(plan, context)

    assert calls == [True]
    assert evidence["effective"] == {"deterministic": True}

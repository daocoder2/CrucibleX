from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cruciblex.domain import (
    CaseSpec,
    DeviceSpec,
    InvocationSpec,
    OperatorSpec,
    ParameterKind,
    ParameterSpec,
)
from cruciblex.domain.enums import BackendKind
from cruciblex.plugins.executors.torch import TorchFunctionExecutor
from cruciblex.runtime.backends.base import DeviceContext
from cruciblex.runtime.executors.base import ExecutionRequest

pytestmark = pytest.mark.gpu


class NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def to(self, device):
        self.device = device
        return self


def test_torch_executor_uses_gpu_device_and_keyword_binding(monkeypatch, tmp_path):
    calls = {}

    def target(input, dim):
        calls["input"] = input
        calls["dim"] = dim
        return np.asarray(input.value) + dim

    fake_cuda = SimpleNamespace(is_available=lambda: True)
    fake_torch = SimpleNamespace(
        cuda=fake_cuda,
        Tensor=FakeTensor,
        from_numpy=lambda value: FakeTensor(value),
        no_grad=lambda: NoGrad(),
        test=target,
    )
    monkeypatch.setattr("importlib.import_module", lambda name: fake_torch if name == "torch" else (_ for _ in ()).throw(ImportError(name)))
    case = CaseSpec(
        id=1,
        operator=OperatorSpec(name="test"),
        invocation=InvocationSpec(api="torch.test", api_type="function", metadata={"binding": {"mode": "keyword"}}),
        parameters=[ParameterSpec(name="input", kind=ParameterKind.TENSOR), ParameterSpec(name="dim", kind=ParameterKind.SCALAR)],
    )
    context = DeviceContext(host="host", node_name="gpu", device=DeviceSpec(id=1, backend=BackendKind.GPU), output_root=Path(tmp_path))
    request = ExecutionRequest(case=case, inputs=[np.array([2]), 3], plan=None, context=context)

    result = TorchFunctionExecutor().execute(request)
    assert result.tolist() == [5]
    assert calls["input"].device == "cuda:1"
    assert calls["dim"] == 3


def test_torch_executor_copies_non_writable_numpy_input(monkeypatch, tmp_path):
    observed = {}

    def target(input):
        observed["writeable"] = input.value.flags.writeable
        return input

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        Tensor=FakeTensor,
        from_numpy=lambda value: FakeTensor(value),
        no_grad=lambda: NoGrad(),
        test=target,
    )
    monkeypatch.setattr("importlib.import_module", lambda name: fake_torch if name == "torch" else (_ for _ in ()).throw(ImportError(name)))
    case = CaseSpec(
        id=2,
        operator=OperatorSpec(name="test"),
        invocation=InvocationSpec(api="torch.test", api_type="function"),
        parameters=[ParameterSpec(name="input", kind=ParameterKind.TENSOR)],
    )
    context = DeviceContext(host="host", node_name="gpu", device=DeviceSpec(id=0, backend=BackendKind.GPU), output_root=Path(tmp_path))
    value = np.asarray([2], dtype=np.float32)
    value.flags.writeable = False
    request = ExecutionRequest(case=case, inputs=[value], plan=None, context=context)

    TorchFunctionExecutor().execute(request)
    assert observed["writeable"] is True

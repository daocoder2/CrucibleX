from types import SimpleNamespace

import pytest

from cruciblex.plugins import load_builtin_plugins
from cruciblex.runtime.executors.base import EXECUTOR_REGISTRY, ExecutionNotSupportedError


@pytest.fixture(autouse=True)
def load_plugins():
    load_builtin_plugins()


def _request(config):
    case = SimpleNamespace(
        metadata={"backend_import": {"config": config}},
        invocation=SimpleNamespace(api="atb.add"),
    )
    return SimpleNamespace(case=case, inputs=[SimpleNamespace(tolist=lambda: [1, 2], dtype="float32", shape=(2,))], context=None)


def test_atb_and_temu_are_registered_and_report_missing_runtime():
    assert "atb" in EXECUTOR_REGISTRY.known()
    assert "temu" in EXECUTOR_REGISTRY.known()
    with pytest.raises(ExecutionNotSupportedError, match="atb runtime command is not configured"):
        EXECUTOR_REGISTRY.resolve("atb").execute(_request({}))


def test_external_runtime_json_protocol_round_trip():
    command = "python -c \"import json,sys; p=json.load(sys.stdin); print(json.dumps({'output': {'runtime': p['runtime'], 'operator': p['operator'], 'input_count': len(p['inputs'])}}))\""
    result = EXECUTOR_REGISTRY.resolve("temu").execute(_request({"command": command, "kernel": "softmax"}))
    assert result == {"runtime": "temu", "operator": "atb.add", "input_count": 1}

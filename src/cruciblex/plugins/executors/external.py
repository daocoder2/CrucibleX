from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from cruciblex.runtime.executors.base import (
    BackendExecutor,
    ExecutionNotSupportedError,
    ExecutionRequest,
)


class ExternalRuntimeExecutor(BackendExecutor):
    runtime_name = "external"

    def execute(self, request: ExecutionRequest) -> object:
        imported = request.case.metadata.get("backend_import", {})
        config = imported.get("config", {}) if isinstance(imported, dict) else {}
        command = config.get("command") if isinstance(config, dict) else None
        if not command:
            raise ExecutionNotSupportedError(
                f"{self.runtime_name} runtime command is not configured; "
                f"set metadata.backend_import.config.command for the {self.runtime_name} adapter"
            )
        payload = {
            "protocol": "cruciblex.external-runtime.v1",
            "runtime": self.runtime_name,
            "operator": request.case.invocation.api,
            "inputs": [_json_value(value) for value in request.inputs],
            "config": config,
            "device": request.context.device.id if request.context else None,
        }
        environment = os.environ.copy()
        if request.context is not None:
            environment.update({str(key): str(value) for key, value in request.context.env.items()})
        completed = subprocess.run(
            command,
            shell=True,
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{self.runtime_name} runtime exited {completed.returncode}: {completed.stderr.strip()}")
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{self.runtime_name} runtime returned non-JSON stdout") from exc
        if isinstance(response, dict) and response.get("status") == "unsupported":
            raise ExecutionNotSupportedError(str(response.get("reason", f"{self.runtime_name} runtime unsupported")))
        return response.get("output") if isinstance(response, dict) and "output" in response else response


def _json_value(value: object) -> Any:
    if hasattr(value, "tolist"):
        return {"data": value.tolist(), "dtype": str(getattr(value, "dtype", "")), "shape": list(getattr(value, "shape", ())) }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)

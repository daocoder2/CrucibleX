from __future__ import annotations

from cruciblex.runtime.executors.base import BackendExecutor, EXECUTOR_REGISTRY, ExecutionRequest


class ExampleExecutor(BackendExecutor):
    def execute(self, request: ExecutionRequest) -> object:
        # Replace with backend-specific invocation when the builtin function executor is insufficient.
        return request.inputs[0]


EXECUTOR_REGISTRY.register("example_executor")(ExampleExecutor)

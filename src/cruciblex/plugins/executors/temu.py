from cruciblex.plugins.executors.external import ExternalRuntimeExecutor
from cruciblex.runtime.executors.base import EXECUTOR_REGISTRY


class TemuExecutor(ExternalRuntimeExecutor):
    runtime_name = "temu"


EXECUTOR_REGISTRY.register("temu")(TemuExecutor)

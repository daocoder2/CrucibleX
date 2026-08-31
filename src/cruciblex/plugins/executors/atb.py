from cruciblex.plugins.executors.external import ExternalRuntimeExecutor
from cruciblex.runtime.executors.base import EXECUTOR_REGISTRY


class AtbExecutor(ExternalRuntimeExecutor):
    runtime_name = "atb"


EXECUTOR_REGISTRY.register("atb")(AtbExecutor)

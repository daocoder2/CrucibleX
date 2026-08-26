from __future__ import annotations

from cruciblex.domain.plan import ExecutionPlan
from cruciblex.domain.result import ExecutionResult
from cruciblex.runtime.backends import BACKEND_REGISTRY, DeviceContext
from cruciblex.runtime.logging import bind_event, get_logger
from cruciblex.runtime.pipeline import ExecutionPipeline

logger = get_logger("actor.device")


class DeviceActor:
    """Host plus device scoped execution unit."""

    def __init__(self, host: str, device_id: int, persist_artifacts: bool = True) -> None:
        self.host = host
        self.device_id = device_id
        self.persist_artifacts = persist_artifacts
        self._pipeline = ExecutionPipeline()

    def run(self, plan: ExecutionPlan) -> ExecutionResult:
        logger.info(bind_event("actor.run.start", host=self.host, device=self.device_id, plan=plan.plan_id))
        context = DeviceContext.from_node(plan.node, plan.device, plan.artifacts.output_root)
        runtime = BACKEND_REGISTRY.resolve(plan.device.backend)
        logger.info(bind_event("actor.prepare", backend=plan.device.backend, node=plan.node.display_name))
        prepared_context = runtime.prepare(context)
        try:
            result = self._pipeline.run(plan, prepared_context, persist_artifacts=self.persist_artifacts)
            logger.info(bind_event("actor.run.complete", plan=plan.plan_id, status=result.status))
            return result
        finally:
            logger.info(bind_event("actor.cleanup", backend=plan.device.backend, node=plan.node.display_name))
            runtime.cleanup(prepared_context)

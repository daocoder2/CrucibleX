from __future__ import annotations

from cruciblex.domain.plan import ExecutionPlan
from cruciblex.domain.result import ExecutionResult
from cruciblex.domain.run import RunContext
from cruciblex.runtime.actors.device import DeviceActor
from cruciblex.runtime.inputs import DriverInputMaterializer
from cruciblex.runtime.logging import bind_event, get_logger
from cruciblex.runtime.scheduler.base import Scheduler

logger = get_logger("scheduler.local")


class LocalScheduler(Scheduler):
    def __init__(self, context: RunContext) -> None:
        self.context = context
        self._inputs = DriverInputMaterializer()
        self._results: list[ExecutionResult] = []

    def submit(self, plan: ExecutionPlan) -> ExecutionResult:
        logger.info(
            bind_event(
                "scheduler.submit",
                run=self.context.run_id,
                plan=plan.plan_id,
                backend=plan.device.backend,
                device=plan.device.id,
                task=plan.task,
            )
        )
        bundle = self._inputs.materialize(plan)
        actor = DeviceActor(plan.node.host, plan.device.id)
        result = actor.run(plan, bundle.inputs)
        result.artifacts = [*bundle.artifacts, *result.artifacts]
        self._results.append(result)
        logger.info(bind_event("scheduler.done", run=self.context.run_id, plan=plan.plan_id, status=result.status))
        return result

    def collect(self) -> list[ExecutionResult]:
        return list(self._results)

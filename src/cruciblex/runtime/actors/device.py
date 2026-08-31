from __future__ import annotations

import io
import logging

from cruciblex.domain.plan import ExecutionPlan
from cruciblex.domain.result import ArtifactPayload, ExecutionResult
from cruciblex.runtime.backends import BACKEND_REGISTRY, DeviceContext
from cruciblex.runtime.logging import bind_event, get_logger
from cruciblex.runtime.pipeline import ExecutionPipeline

logger = get_logger("actor.device")


class DeviceActor:
    """Host plus device scoped execution unit."""

    def __init__(
        self,
        host: str,
        device_id: int,
        persist_artifacts: bool = True,
        capture_logs: bool = False,
        device_index_mode: str | None = None,
    ) -> None:
        self.host = host
        self.device_id = device_id
        self.persist_artifacts = persist_artifacts
        self.capture_logs = capture_logs
        self.device_index_mode = device_index_mode
        self._pipeline = ExecutionPipeline()

    def run(self, plan: ExecutionPlan, inputs: list[object] | None = None) -> ExecutionResult:
        if not self.capture_logs:
            return self._run(plan, inputs)

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root_logger = get_logger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)
        try:
            result = self._run(plan, inputs)
        finally:
            root_logger.removeHandler(handler)
            handler.close()

        log_text = stream.getvalue()
        if log_text:
            result.artifact_payloads.append(
                ArtifactPayload(
                    name="execution_log",
                    kind="log",
                    data=log_text,
                    metadata={
                        "host": self.host,
                        "device_id": self.device_id,
                        "plan_id": plan.plan_id,
                    },
                )
            )
        return result

    def _run(self, plan: ExecutionPlan, inputs: list[object] | None = None) -> ExecutionResult:
        logger.info(bind_event("actor.run.start", host=self.host, device=self.device_id, plan=plan.plan_id))
        context = DeviceContext.from_node(plan.node, plan.device, plan.artifacts.output_root)
        if self.device_index_mode is not None:
            context.env["CX_DEVICE_INDEX_MODE"] = self.device_index_mode
        runtime = BACKEND_REGISTRY.resolve(plan.device.backend)
        logger.info(bind_event("actor.prepare", backend=plan.device.backend, node=plan.node.display_name))
        prepared_context = runtime.prepare(context)
        try:
            result = self._pipeline.run(
                plan,
                prepared_context,
                persist_artifacts=self.persist_artifacts,
                inputs=inputs,
            )
            logger.info(bind_event("actor.run.complete", plan=plan.plan_id, status=result.status))
            return result
        finally:
            logger.info(bind_event("actor.cleanup", backend=plan.device.backend, node=plan.node.display_name))
            runtime.cleanup(prepared_context)

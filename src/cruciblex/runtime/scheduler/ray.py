from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cruciblex.domain.enums import ResultStatus
from cruciblex.domain.plan import ExecutionPlan
from cruciblex.domain.result import ExecutionResult
from cruciblex.domain.run import RunContext
from cruciblex.runtime.logging import bind_event, get_logger
from cruciblex.runtime.scheduler.base import Scheduler, scheduler_result
from cruciblex.runtime.scheduler.placement import (
    RayPlacementDecision,
    RayPlacementError,
    decide_ray_placement,
    discover_ray_cluster,
)
from cruciblex.storage.artifacts import ArtifactStore

logger = get_logger("scheduler.ray")


@dataclass(frozen=True, slots=True)
class ActorKey:
    host: str
    backend: str
    device_id: int

    @classmethod
    def from_plan(cls, plan: ExecutionPlan) -> ActorKey:
        return cls(
            host=plan.node.host,
            backend=plan.device.backend.value,
            device_id=plan.device.id,
        )

    def label(self) -> str:
        return f"{self.host}:{self.backend}:{self.device_id}"


class RayScheduler(Scheduler):
    def __init__(self, context: RunContext) -> None:
        self.context = context
        self._refs: list[tuple[ExecutionPlan, object]] = []
        self._actors: dict[ActorKey, object] = {}
        self._results: list[ExecutionResult] = []

    def submit(self, plan: ExecutionPlan):
        try:
            ray = self._ray()
            self._ensure_initialized(ray)
            snapshot = discover_ray_cluster(ray)
            placement = decide_ray_placement(plan, snapshot)
            logger.info(
                bind_event(
                    "scheduler.submit",
                    run=self.context.run_id,
                    plan=plan.plan_id,
                    backend=plan.device.backend,
                    device=plan.device.id,
                    task=plan.task,
                    mode="ray",
                    node=placement.node.address,
                    hostname=placement.node.hostname,
                )
            )
            actor = self._actor_for(ray, plan, placement)
            ref = actor.run.remote(plan.model_dump(mode="json"))
            self._refs.append((plan, ref))
            return ref
        except RayPlacementError as exc:
            result = scheduler_result(plan, ResultStatus.SKIPPED, "placement", detail=str(exc))
            logger.info(bind_event("scheduler.skip", run=self.context.run_id, plan=plan.plan_id, reason=str(exc)))
            self._results.append(result)
            return result
        except Exception as exc:
            detail = f"{exc.__class__.__name__}: {exc}"
            result = scheduler_result(plan, ResultStatus.ERROR, "scheduler", detail=detail)
            logger.exception(bind_event("scheduler.error", run=self.context.run_id, plan=plan.plan_id))
            self._results.append(result)
            return result

    def collect(self) -> list[ExecutionResult]:
        results = list(self._results)
        if not self._refs:
            return results
        ray = self._ray()
        for plan, ref in self._refs:
            try:
                result = ExecutionResult.model_validate(ray.get(ref))
            except Exception as exc:
                detail = f"{exc.__class__.__name__}: {exc}"
                result = scheduler_result(plan, ResultStatus.ERROR, "scheduler", detail=detail)
                logger.exception(bind_event("scheduler.error", run=self.context.run_id, plan=plan.plan_id, stage="collect"))
            else:
                if result.artifact_payloads:
                    store = ArtifactStore(self.context.output_root)
                    result.artifacts = [store.write_payload(plan, payload) for payload in result.artifact_payloads]
                    result.artifact_payloads = []
            results.append(result)
        return results

    def actor_count(self) -> int:
        return len(self._actors)

    def _actor_for(self, ray: Any, plan: ExecutionPlan, placement: RayPlacementDecision):
        key = ActorKey.from_plan(plan)
        actor = self._actors.get(key)
        if actor is not None:
            return actor
        actor_cls = ray.remote(_RayDeviceActor).options(
            name=f"cruciblex-{self._safe_label(self.context.run_id)}-{self._safe_label(key.label())}",
            **placement.actor_options(),
        )
        actor = actor_cls.remote(
            plan.node.host,
            plan.device.id,
            [str(path) for path in self.context.plugin_paths],
        )
        self._actors[key] = actor
        return actor

    def _safe_label(self, value: str) -> str:
        return "".join(character if character.isalnum() else "-" for character in value)

    def _ensure_initialized(self, ray: Any) -> None:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def _ray(self):
        try:
            import ray
        except ImportError as exc:
            raise RuntimeError("RayScheduler requires the ray package") from exc
        return ray


class _RayDeviceActor:
    def __init__(self, host: str, device_id: int, plugin_paths: list[str]) -> None:
        from pathlib import Path

        from cruciblex.plugins import load_builtin_plugins, load_plugins
        from cruciblex.runtime.actors.device import DeviceActor

        load_builtin_plugins()
        load_plugins([Path(path) for path in plugin_paths])
        logger.info(bind_event("ray.actor.init", host=host, device=device_id, plugins=len(plugin_paths)))
        self._delegate = DeviceActor(host, device_id, persist_artifacts=False)

    def run(self, raw_plan: dict) -> dict:
        from cruciblex.domain.plan import ExecutionPlan

        plan = ExecutionPlan.model_validate(raw_plan)
        result = self._delegate.run(plan)
        raw_result = result.model_dump(mode="json")
        raw_result["artifact_payloads"] = [
            payload.model_dump(mode="json") for payload in result.artifact_payloads
        ]
        return raw_result

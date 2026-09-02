from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from cruciblex.domain.enums import ResultStatus
from cruciblex.domain.plan import ExecutionPlan
from cruciblex.domain.result import ExecutionResult


def scheduler_result(
    plan: ExecutionPlan,
    status: ResultStatus,
    reason: str,
    *,
    detail: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExecutionResult:
    failure_kind = "skip" if status == ResultStatus.SKIPPED else "timeout" if status == ResultStatus.TIMEOUT else "error" if status == ResultStatus.ERROR else "scheduler"
    manifest_metrics = {
        key: plan.case.metadata[key]
        for key in ("manifest_lane", "manifest_lane_kind", "manifest_case_include")
        if key in plan.case.metadata
    }
    metrics = {
        "stage": "scheduler",
        "reason": reason,
        "failure_kind": failure_kind,
        "failure_stage": reason,
        "failure_message": detail or reason,
        **manifest_metrics,
        **(metadata or {}),
    }
    return ExecutionResult(
        plan_id=plan.plan_id,
        case_id=plan.case.id,
        case_name=plan.case.name,
        node_name=plan.node.display_name,
        backend=plan.device.backend,
        device_id=plan.device.id,
        task=plan.task,
        status=status,
        metrics=metrics,
        error=detail or reason,
    )


class Scheduler(ABC):
    @abstractmethod
    def submit(self, plan: ExecutionPlan):
        raise NotImplementedError

    @abstractmethod
    def collect(self) -> list[ExecutionResult]:
        raise NotImplementedError

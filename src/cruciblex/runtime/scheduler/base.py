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
    metrics = {"stage": "scheduler", "reason": reason, **(metadata or {})}
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

from __future__ import annotations

from pathlib import Path

from cruciblex.domain.enums import ResultStatus
from cruciblex.domain.plan import ExecutionPlan
from cruciblex.domain.result import ExecutionResult
from cruciblex.storage.results import ResultStore

_RETRYABLE_STATUSES = {
    ResultStatus.FAILED,
    ResultStatus.ERROR,
    ResultStatus.TIMEOUT,
    ResultStatus.CANCELLED,
}


class ResumeState:
    def __init__(self, results: list[ExecutionResult], retry_failed: bool = False) -> None:
        self._results_by_plan = {result.plan_id: result for result in results}
        self._retry_failed = retry_failed

    @classmethod
    def from_path(cls, path: str | Path, retry_failed: bool = False) -> ResumeState:
        root = _output_root(path)
        return cls(ResultStore(root).read_results_jsonl(), retry_failed=retry_failed)

    def should_run(self, plan: ExecutionPlan) -> bool:
        result = self._results_by_plan.get(plan.plan_id)
        if result is None:
            return True
        return self._retry_failed and result.status in _RETRYABLE_STATUSES

    def filter_plans(self, plans: list[ExecutionPlan]) -> list[ExecutionPlan]:
        return [plan for plan in plans if self.should_run(plan)]

    def skipped_count(self, plans: list[ExecutionPlan]) -> int:
        return len(plans) - len(self.filter_plans(plans))

    def merge_results(self, plans: list[ExecutionPlan], new_results: list[ExecutionResult]) -> list[ExecutionResult]:
        new_by_plan = {result.plan_id: result for result in new_results}
        merged: list[ExecutionResult] = []
        for plan in plans:
            result = new_by_plan.get(plan.plan_id) or self._results_by_plan.get(plan.plan_id)
            if result is not None:
                merged.append(result)
        return merged


def _output_root(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_file():
        return resolved.parent
    return resolved

from __future__ import annotations

from typing import Any

import numpy as np

from cruciblex.domain.enums import ExecutionRole, ResultStatus, TaskKind
from cruciblex.domain.plan import ExecutionPlan
from cruciblex.domain.result import ExecutionResult
from cruciblex.runtime.backends.base import DeviceContext
from cruciblex.runtime.compare import COMPARATOR_REGISTRY, ComparisonRequest
from cruciblex.runtime.executors import (
    EXECUTOR_REGISTRY,
    ExecutionNotSupportedError,
    ExecutionRequest,
)
from cruciblex.runtime.generation import GENERATOR_REGISTRY, GenerationRequest
from cruciblex.runtime.logging import bind_event, get_logger
from cruciblex.storage.artifacts import ArtifactRecorder, ArtifactStore

logger = get_logger("pipeline")


class ExecutionPipeline:
    def run(
        self,
        plan: ExecutionPlan,
        context: DeviceContext | None = None,
        persist_artifacts: bool = True,
    ) -> ExecutionResult:
        logger.info(
            bind_event(
                "pipeline.start",
                plan=plan.plan_id,
                case=plan.case.name,
                generator=plan.case.generator,
                comparator=plan.case.oracle.comparison,
            )
        )
        recorder = ArtifactRecorder(
            ArtifactStore(plan.artifacts.output_root),
            plan,
            persist=persist_artifacts,
        )

        try:
            generator = GENERATOR_REGISTRY.resolve(plan.case.generator)
            inputs = generator.generate(GenerationRequest(case=plan.case, plan=plan, context=context))
        except (KeyError, NotImplementedError) as exc:
            logger.info(bind_event("pipeline.skipped", stage="generate", plan=plan.plan_id, reason=str(exc)))
            return self._result(plan, ResultStatus.SKIPPED, {"reason": str(exc)}, recorder, str(exc))
        except Exception as exc:
            logger.exception(bind_event("pipeline.error", stage="generate", plan=plan.plan_id))
            return self._result(plan, ResultStatus.ERROR, {}, recorder, self._exception_text(exc))
        recorder.record_json("inputs", self._serialize_inputs(inputs), "inputs")

        try:
            candidate_executor = self._resolve_executor(plan.case.invocation.executor or plan.case.invocation.api_type)
        except KeyError as exc:
            logger.info(bind_event("pipeline.skipped", stage="candidate_resolve", plan=plan.plan_id, reason=str(exc)))
            return self._result(plan, ResultStatus.SKIPPED, {"reason": str(exc)}, recorder, str(exc))

        candidate_request = ExecutionRequest(
            case=plan.case,
            inputs=inputs,
            plan=plan,
            context=context,
            role=ExecutionRole.CANDIDATE,
        )
        expected_error = plan.case.oracle.expected_error
        try:
            candidate_output = candidate_executor.execute(candidate_request)
        except (ExecutionNotSupportedError, NotImplementedError) as exc:
            logger.info(bind_event("pipeline.skipped", stage="candidate", plan=plan.plan_id, reason=str(exc)))
            return self._result(plan, ResultStatus.SKIPPED, {"reason": str(exc)}, recorder, str(exc))
        except Exception as exc:
            if expected_error:
                return self._expected_error_result(plan, recorder, expected_error, exc)
            logger.exception(bind_event("pipeline.error", stage="candidate", plan=plan.plan_id))
            return self._result(plan, ResultStatus.ERROR, {}, recorder, self._exception_text(exc))

        recorder.record_json(
            "candidate_output",
            self._serialize_value(candidate_output),
            "candidate_output",
            role=ExecutionRole.CANDIDATE.value,
        )
        metrics = {
            "input_count": len(inputs),
            "output_shape": self._shape_of(candidate_output),
        }
        if context is not None:
            metrics.update(
                {
                    "backend": context.backend.value,
                    "host": context.host,
                    "node": context.node_name,
                    "device_id": context.device.id,
                }
            )

        if expected_error:
            logger.info(bind_event("pipeline.failed", stage="expected_error", plan=plan.plan_id, reason="not_raised"))
            metrics.update(
                {
                    "expected_error": expected_error,
                    "error_matched": False,
                    "compare_detail": "expected error was not raised",
                }
            )
            return self._result(
                plan,
                ResultStatus.FAILED,
                metrics,
                recorder,
                f"expected error was not raised: {expected_error}",
            )

        if plan.task == TaskKind.ACCURACY:
            try:
                reference_executor = self._resolve_executor(
                    plan.case.oracle.reference_executor
                    or plan.case.invocation.executor
                    or plan.case.invocation.api_type
                )
            except KeyError as exc:
                logger.info(bind_event("pipeline.skipped", stage="reference_resolve", plan=plan.plan_id, reason=str(exc)))
                return self._result(plan, ResultStatus.SKIPPED, {"reason": str(exc)}, recorder, str(exc))
            try:
                reference_output = reference_executor.execute(
                    ExecutionRequest(
                        case=plan.case,
                        inputs=inputs,
                        plan=plan,
                        context=context,
                        role=ExecutionRole.REFERENCE,
                    )
                )
            except (ExecutionNotSupportedError, KeyError, NotImplementedError) as exc:
                logger.info(bind_event("pipeline.skipped", stage="reference", plan=plan.plan_id, reason=str(exc)))
                return self._result(plan, ResultStatus.SKIPPED, {"reason": str(exc)}, recorder, str(exc))
            except Exception as exc:
                logger.exception(bind_event("pipeline.error", stage="reference", plan=plan.plan_id))
                return self._result(plan, ResultStatus.ERROR, {}, recorder, self._exception_text(exc))
            recorder.record_json(
                "reference_output",
                self._serialize_value(reference_output),
                "reference_output",
                role=ExecutionRole.REFERENCE.value,
            )
            try:
                comparator = COMPARATOR_REGISTRY.resolve(plan.case.oracle.comparison)
            except KeyError as exc:
                logger.info(bind_event("pipeline.skipped", stage="compare", plan=plan.plan_id, reason=str(exc)))
                return self._result(plan, ResultStatus.SKIPPED, {"reason": str(exc)}, recorder, str(exc))
            report = comparator.compare(
                ComparisonRequest(
                    expected=reference_output,
                    actual=candidate_output,
                    tolerance=self._tolerance(plan.case.oracle.tolerance),
                    metadata={"case_id": plan.case.id, "plan_id": plan.plan_id},
                )
            )
            logger.info(
                bind_event(
                    "pipeline.compare",
                    plan=plan.plan_id,
                    comparator=plan.case.oracle.comparison,
                    passed=report.passed,
                )
            )
            metrics.update(
                {
                    "max_abs_diff": report.max_abs_diff,
                    "mean_abs_diff": report.mean_abs_diff,
                    "compare_detail": report.detail,
                    "comparison": plan.case.oracle.comparison,
                    "candidate_role": ExecutionRole.CANDIDATE.value,
                    "reference_role": ExecutionRole.REFERENCE.value,
                }
            )
            status = ResultStatus.PASSED if report.passed else ResultStatus.FAILED
        else:
            status = ResultStatus.PASSED

        logger.info(bind_event("pipeline.complete", plan=plan.plan_id, status=status))
        return self._result(plan, status, metrics, recorder)

    def _expected_error_result(
        self,
        plan: ExecutionPlan,
        recorder: ArtifactRecorder,
        expected_error: str,
        exc: Exception,
    ) -> ExecutionResult:
        actual_error = self._exception_text(exc)
        matched = self._matches_expected_error(expected_error, actual_error)
        status = ResultStatus.PASSED if matched else ResultStatus.FAILED
        logger.info(
            bind_event(
                "pipeline.expected_error",
                plan=plan.plan_id,
                matched=matched,
                expected=expected_error,
                actual=actual_error,
            )
        )
        return self._result(
            plan,
            status,
            {
                "expected_error": expected_error,
                "actual_error": actual_error,
                "error_matched": matched,
                "compare_detail": "expected error matched" if matched else "expected error did not match",
            },
            recorder,
            None if matched else actual_error,
        )

    def _result(
        self,
        plan: ExecutionPlan,
        status: ResultStatus,
        metrics: dict[str, Any],
        recorder: ArtifactRecorder,
        error: str | None = None,
    ) -> ExecutionResult:
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
            artifacts=list(recorder.artifacts),
            artifact_payloads=list(recorder.payloads),
            error=error,
        )

    def _resolve_executor(self, name: str):
        return EXECUTOR_REGISTRY.resolve(name)

    def _exception_text(self, exc: Exception) -> str:
        return f"{exc.__class__.__name__}: {exc}"

    def _matches_expected_error(self, expected_error: str, actual_error: str) -> bool:
        return expected_error in actual_error

    def _serialize_inputs(self, inputs: list[object]) -> list[Any]:
        return [self._serialize_value(item) for item in inputs]

    def _serialize_value(self, value: object) -> Any:
        if isinstance(value, np.ndarray):
            return {"dtype": str(value.dtype), "shape": list(value.shape), "data": value.tolist()}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        return value

    def _shape_of(self, value: object) -> list[int]:
        if isinstance(value, np.ndarray):
            return list(value.shape)
        return []

    def _tolerance(self, tolerance: dict[str, Any]) -> dict[str, float]:
        return {key: float(value) for key, value in tolerance.items()}

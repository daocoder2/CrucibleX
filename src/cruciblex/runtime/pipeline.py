from __future__ import annotations

import json
import time
import tracemalloc
from typing import Any

import numpy as np

from cruciblex.domain.case import InvocationSpec
from cruciblex.domain.enums import BackendKind, ExecutionRole, ResultStatus, TaskKind
from cruciblex.domain.plan import ExecutionPlan
from cruciblex.domain.result import ExecutionResult, HardwareEvidence
from cruciblex.generation.dtypes import dtype_contract
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
    def generate_inputs(
        self,
        plan: ExecutionPlan,
        context: DeviceContext | None = None,
    ) -> list[object]:
        try:
            generator = GENERATOR_REGISTRY.resolve(plan.case.generator)
            inputs = generator.generate(GenerationRequest(case=plan.case, plan=plan, context=context))
        except (KeyError, NotImplementedError) as exc:
            logger.info(bind_event("pipeline.skipped", stage="generate", plan=plan.plan_id, reason=str(exc)))
            raise
        except Exception:
            logger.exception(bind_event("pipeline.error", stage="generate", plan=plan.plan_id))
            raise
        return inputs

    def run(
        self,
        plan: ExecutionPlan,
        context: DeviceContext | None = None,
        persist_artifacts: bool = True,
        inputs: list[object] | None = None,
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
        if context is not None and context.backend in {BackendKind.GPU, BackendKind.NPU, BackendKind.ACLNN}:
            prefix = "GPU" if context.backend == BackendKind.GPU else "NPU"
            kind = "gpu_evidence" if context.backend == BackendKind.GPU else "npu_evidence"
            evidence_json = context.env.get(f"CX_{prefix}_EVIDENCE_JSON")
            if evidence_json:
                try:
                    recorder.record_json(kind, json.loads(evidence_json), kind, "environment")
                except (OSError, ValueError):
                    logger.warning(bind_event("pipeline.evidence_degraded", plan=plan.plan_id))

        if inputs is None:
            try:
                inputs = self.generate_inputs(plan, context=context)
            except (KeyError, NotImplementedError) as exc:
                logger.info(bind_event("pipeline.skipped", stage="generate", plan=plan.plan_id, reason=str(exc)))
                return self._skip_result(plan, recorder, "generate", str(exc))
            except Exception as exc:  # noqa: BLE001 - generation failures map to ExecutionResult.ERROR
                return self._error_result(plan, recorder, "generate", exc)
            recorder.record_json(
                "inputs",
                self._serialize_inputs(inputs),
                "inputs",
                metadata={"dtype_contracts": self._input_dtype_contracts(plan)},
            )

        profiler = plan.case.invocation.metadata.get("profiler")
        if isinstance(profiler, dict):
            recorder.record_json(
                "profiler",
                {"requested": dict(profiler), "status": profiler.get("status", "requested"), "executor": plan.case.invocation.executor or plan.case.invocation.api_type},
                "profiler",
            )
        candidate_executor_name = plan.case.invocation.executor or plan.case.invocation.api_type
        try:
            candidate_executor = self._resolve_executor(candidate_executor_name)
        except KeyError as exc:
            logger.info(bind_event("pipeline.skipped", stage="candidate_resolve", plan=plan.plan_id, reason=str(exc)))
            return self._skip_result(plan, recorder, "candidate_resolve", str(exc))

        candidate_request = ExecutionRequest(
            case=plan.case,
            inputs=inputs,
            plan=plan,
            context=context,
            role=ExecutionRole.CANDIDATE,
        )
        runtime_policy, synchronize_timing = self._apply_runtime_policy(plan, context)
        expected_error = plan.case.oracle.expected_error
        expected_invalid = bool(plan.case.metadata.get("expected_invalid"))
        track_memory = self._is_memory_task(plan.task)
        track_hardware = self._is_performance_task(plan.task) or track_memory
        hardware_metrics: dict[str, Any] = {}
        if track_memory:
            tracemalloc.start()
        if track_hardware:
            hardware_metrics.update(self._reset_hardware_metrics(context, candidate_executor_name))
            hardware_metrics.update(self._hardware_memory_snapshot(context, candidate_executor_name, "before"))
            if synchronize_timing:
                hardware_metrics.update(self._synchronize_device(context, candidate_executor_name, "before"))
        benchmark = self._benchmark_policy(plan)
        if self._is_performance_task(plan.task):
            for _ in range(benchmark["warmup"]):
                candidate_executor.execute(candidate_request)
                if synchronize_timing:
                    self._synchronize_device(context, candidate_executor_name, "warmup")
        started_at = time.perf_counter()
        samples_ms: list[float] = []
        memory_peak_bytes = 0
        try:
            candidate_output = None
            repeat = int(benchmark["repeat"])
            min_time_ms = float(benchmark["min_time_ms"])
            while len(samples_ms) < repeat or (min_time_ms > 0 and (time.perf_counter() - started_at) * 1000.0 < min_time_ms):
                sample_started = time.perf_counter()
                candidate_output = candidate_executor.execute(candidate_request)
                if benchmark["repeat"] > 1 and synchronize_timing:
                    self._synchronize_device(context, candidate_executor_name, "sample")
                samples_ms.append((time.perf_counter() - sample_started) * 1000.0)
            if track_hardware:
                if synchronize_timing:
                    hardware_metrics.update(self._synchronize_device(context, candidate_executor_name, "after"))
                hardware_metrics.update(self._hardware_memory_snapshot(context, candidate_executor_name, "after"))
            if track_memory:
                _, memory_peak_bytes = tracemalloc.get_traced_memory()
        except (ExecutionNotSupportedError, NotImplementedError) as exc:
            logger.info(bind_event("pipeline.skipped", stage="candidate", plan=plan.plan_id, reason=str(exc)))
            return self._skip_result(plan, recorder, "candidate", str(exc))
        except Exception as exc:
            if expected_error:
                return self._expected_error_result(plan, recorder, expected_error, exc)
            if expected_invalid:
                return self._expected_invalid_error_result(plan, recorder, exc)
            logger.exception(bind_event("pipeline.error", stage="candidate", plan=plan.plan_id))
            return self._error_result(plan, recorder, "candidate", exc)
        finally:
            duration_ms = (time.perf_counter() - started_at) * 1000.0
            if track_memory:
                tracemalloc.stop()

        recorder.record_json(
            "candidate_output",
            self._serialize_value(candidate_output),
            "candidate_output",
            role=ExecutionRole.CANDIDATE.value,
            metadata={"dtype": self._dtype_of(candidate_output)},
        )
        metrics = {
            "input_count": len(inputs),
            "output_shape": self._shape_of(candidate_output),
            "input_dtype_contracts": self._input_dtype_contracts(plan),
            "candidate_output_dtype": self._dtype_of(candidate_output),
            "duration_ms": duration_ms,
            "candidate_executor": candidate_executor_name,
            **({"runtime_policy": runtime_policy} if runtime_policy else {}),
        }
        if context is not None:
            metrics.update(self._context_metrics(context, candidate_executor_name))
        executor_evidence = getattr(candidate_executor, "last_execution_evidence", None)
        if isinstance(executor_evidence, dict):
            metrics.update(executor_evidence)
        if hardware_metrics:
            metrics.update(hardware_metrics)

        if expected_error:
            logger.info(bind_event("pipeline.failed", stage="expected_error", plan=plan.plan_id, reason="not_raised"))
            metrics.update(
                {
                    "expected_error": expected_error,
                    "error_matched": False,
                    "failure_kind": "expected_error_not_raised",
                    "compare_detail": "expected error was not raised",
                }
            )
            return self._failed_result(
                plan,
                recorder,
                "oracle",
                metrics,
                f"expected error was not raised: {expected_error}",
            )
        if expected_invalid:
            logger.info(bind_event("pipeline.failed", stage="expected_invalid", plan=plan.plan_id, reason="not_rejected"))
            metrics.update(
                {
                    "invalid_rejected": False,
                    "compare_detail": "expected invalid case executed successfully",
                }
            )
            return self._failed_result(
                plan,
                recorder,
                "oracle",
                metrics,
                "expected invalid case executed successfully",
            )

        if self._is_accuracy_task(plan.task):
            metrics.update(
                {
                    "comparison": plan.case.oracle.comparison,
                    "atol": float(plan.case.oracle.tolerance.get("atol", 1e-6)),
                    "rtol": float(plan.case.oracle.tolerance.get("rtol", 1e-6)),
                    "tolerance": dict(plan.case.oracle.tolerance),
                    "accuracy_policy": plan.case.oracle.accuracy_policy.model_dump(exclude_none=True, exclude_defaults=True),
                    "candidate_role": ExecutionRole.CANDIDATE.value,
                }
            )
            if self._should_execute_reference(plan):
                try:
                    reference_output, reference_executor_name = self._execute_reference(plan, inputs, context)
                    recorder.record_json(
                        "reference_output",
                        self._serialize_value(reference_output),
                        "reference_output",
                        role=ExecutionRole.REFERENCE.value,
                        metadata={"dtype": self._dtype_of(reference_output)},
                    )
                    metrics["reference_output_dtype"] = self._dtype_of(reference_output)
                    report = COMPARATOR_REGISTRY.resolve(plan.case.oracle.comparison).compare(
                        ComparisonRequest(
                            expected=reference_output,
                            actual=candidate_output,
                            tolerance=plan.case.oracle.tolerance,
                            metadata={"reference_executor": reference_executor_name},
                        )
                    )
                    metrics.update(
                        {
                            "reference_executor": reference_executor_name,
                            "reference_role": ExecutionRole.REFERENCE.value,
                            "compare_detail": report.detail,
                            "max_abs_diff": report.max_abs_diff,
                            "mean_abs_diff": report.mean_abs_diff,
                            **report.metrics,
                        }
                    )
                    if not report.passed:
                        return self._failed_result(plan, recorder, "oracle", metrics, report.detail)
                except (ExecutionNotSupportedError, NotImplementedError, KeyError) as exc:
                    logger.info(bind_event("pipeline.skipped", stage="reference", plan=plan.plan_id, reason=str(exc)))
                    return self._skip_result(plan, recorder, "reference", str(exc))
                except Exception as exc:
                    logger.exception(bind_event("pipeline.error", stage="reference", plan=plan.plan_id))
                    return self._error_result(plan, recorder, "reference", exc)
        elif self._is_performance_task(plan.task):
            metrics.update(self._benchmark_metrics(samples_ms, benchmark, duration_ms))
        elif self._is_memory_task(plan.task):
            metrics.update(
                {
                    "memory_peak_bytes": memory_peak_bytes,
                    "memory_peak_mb": memory_peak_bytes / (1024.0 * 1024.0),
                    **self._hardware_memory_deltas(hardware_metrics),
                }
            )
        status = ResultStatus.PASSED

        logger.info(bind_event("pipeline.complete", plan=plan.plan_id, status=status))
        return self._result(plan, status, metrics, recorder)

    def _should_execute_reference(self, plan: ExecutionPlan) -> bool:
        return bool(plan.case.oracle.reference_executor and plan.case.oracle.metadata.get("execute_reference"))

    def _execute_reference(
        self,
        plan: ExecutionPlan,
        inputs: list[object],
        context: DeviceContext | None,
    ) -> tuple[object, str]:
        reference_executor_name = plan.case.oracle.reference_executor
        if not reference_executor_name:
            raise ExecutionNotSupportedError("reference executor is not configured")
        reference_api = plan.case.oracle.metadata.get("reference_api", plan.case.invocation.api)
        reference_api_type = plan.case.oracle.metadata.get("reference_api_type", plan.case.invocation.api_type)
        reference_case = plan.case.model_copy(
            update={
                "invocation": InvocationSpec(
                    api=str(reference_api),
                    api_type=str(reference_api_type),
                    executor=str(reference_executor_name),
                    metadata={"reference_for": plan.case.invocation.api},
                )
            },
            deep=True,
        )
        reference_executor = self._resolve_executor(str(reference_executor_name))
        return (
            reference_executor.execute(
                ExecutionRequest(
                    case=reference_case,
                    inputs=inputs,
                    plan=plan,
                    context=context,
                    role=ExecutionRole.REFERENCE,
                )
            ),
            str(reference_executor_name),
        )

    def _is_accuracy_task(self, task: TaskKind) -> bool:
        return task in {TaskKind.ACCURACY, TaskKind.ACCURACY_LOAD, TaskKind.ACCURACY_DC}

    def _is_performance_task(self, task: TaskKind) -> bool:
        return task in {
            TaskKind.PERFORMANCE_DEVICE,
            TaskKind.PERFORMANCE_DEVICE_PTA,
            TaskKind.PERFORMANCE_E2E,
            TaskKind.PERFORMANCE_BENCHMARK,
        }

    def _is_memory_task(self, task: TaskKind) -> bool:
        return task == TaskKind.MEMORY_DEVICE

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
                "failure_kind": "expected_error_matched" if matched else "expected_error_mismatch",
                "compare_detail": "expected error matched" if matched else "expected error did not match",
            },
            recorder,
            None if matched else actual_error,
        )

    def _expected_invalid_error_result(
        self,
        plan: ExecutionPlan,
        recorder: ArtifactRecorder,
        exc: Exception,
    ) -> ExecutionResult:
        actual_error = self._exception_text(exc)
        logger.info(bind_event("pipeline.expected_invalid", plan=plan.plan_id, actual=actual_error))
        return self._result(
            plan,
            ResultStatus.PASSED,
            {
                "actual_error": actual_error,
                "invalid_rejected": True,
                "failure_kind": "expected_invalid_rejected",
                "compare_detail": "expected invalid case was rejected",
            },
            recorder,
        )

    def _result(
        self,
        plan: ExecutionPlan,
        status: ResultStatus,
        metrics: dict[str, Any],
        recorder: ArtifactRecorder,
        error: str | None = None,
    ) -> ExecutionResult:
        metrics = self._with_task_intent(plan, self._with_generation_intent(plan, metrics))
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
            evidence=self._hardware_evidence(plan, metrics),
            error=error,
        )

    def _skip_result(
        self,
        plan: ExecutionPlan,
        recorder: ArtifactRecorder,
        stage: str,
        reason: str,
    ) -> ExecutionResult:
        return self._result(
            plan,
            ResultStatus.SKIPPED,
            {
                "failure_kind": "gpu_device_unavailable" if plan.device.backend == BackendKind.GPU else "skip",
                "failure_stage": stage,
                "skip_reason": reason,
                "failure_message": reason,
                "compare_detail": reason,
                "backend_availability": "unavailable" if plan.device.backend == BackendKind.GPU else "unknown",
            },
            recorder,
            reason,
        )

    def _error_result(
        self,
        plan: ExecutionPlan,
        recorder: ArtifactRecorder,
        stage: str,
        exc: Exception,
    ) -> ExecutionResult:
        error_text = self._exception_text(exc)
        return self._result(
            plan,
            ResultStatus.ERROR,
            {
                "failure_kind": self._failure_kind(plan, stage, exc),
                "failure_stage": stage,
                "error_type": exc.__class__.__name__,
                "failure_message": error_text,
            },
            recorder,
            error_text,
        )

    def _hardware_evidence(self, plan: ExecutionPlan, metrics: dict[str, Any]) -> HardwareEvidence:
        backend = plan.device.backend
        probe_status = "unknown"
        runtime: dict[str, Any] = {}
        fingerprint = None
        artifact_refs: list[str] = []
        if backend == BackendKind.GPU:
            probe_status = "available" if metrics.get("gpu_available") else "unavailable"
            runtime = {key: metrics[key] for key in ("cuda_version", "gpu_device_count") if metrics.get(key) is not None}
            fingerprint = metrics.get("gpu_evidence_fingerprint")
            if metrics.get("gpu_evidence_path"):
                artifact_refs.append(str(metrics["gpu_evidence_path"]))
        if backend in {BackendKind.NPU, BackendKind.ACLNN} and "npu_available" in metrics:
            probe_status = "available" if metrics.get("npu_available") else "unavailable"
            runtime = {key: metrics[key] for key in ("npu_device_count", "torch_version", "torch_npu_version", "npu_device_name") if metrics.get(key) is not None}
            fingerprint = metrics.get("npu_evidence_fingerprint")
            if metrics.get("npu_evidence_path"):
                artifact_refs.append(str(metrics["npu_evidence_path"]))
        return HardwareEvidence(
            backend=backend,
            host=metrics.get("host"),
            node=metrics.get("node"),
            device_id=metrics.get("device_id"),
            resolved_device=metrics.get("resolved_device"),
            probe_status=probe_status,
            runtime=runtime,
            fingerprint=fingerprint,
            artifact_refs=artifact_refs,
        )

    def _failure_kind(self, plan: ExecutionPlan, stage: str, exc: Exception) -> str:
        if plan.device.backend != BackendKind.GPU:
            return "error"
        message = str(exc).lower()
        if "driver" in message or "cuda" in message and "not initialized" in message:
            return "gpu_driver_error"
        if "device" in message and ("index" in message or "ordinal" in message or "invalid" in message):
            return "gpu_device_index_invalid"
        if stage == "candidate":
            return "gpu_kernel_error"
        return "gpu_runtime_error"

    def _failed_result(
        self,
        plan: ExecutionPlan,
        recorder: ArtifactRecorder,
        stage: str,
        metrics: dict[str, Any],
        error: str,
    ) -> ExecutionResult:
        enriched = {**metrics, "failure_kind": metrics.get("failure_kind", "failed"), "failure_stage": stage}
        return self._result(plan, ResultStatus.FAILED, enriched, recorder, error)

    def _with_task_intent(self, plan: ExecutionPlan, metrics: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(metrics)
        if plan.task == TaskKind.FUZZ:
            enriched["fuzz_case"] = True
            enriched["fuzz_seed"] = plan.case.generation.seed
        return enriched

    def _with_generation_intent(self, plan: ExecutionPlan, metrics: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(metrics)
        if plan.case.metadata.get("expected_invalid"):
            enriched["expected_invalid"] = True
            enriched["invalid_index"] = plan.case.metadata.get("invalid_index")
        if "generation_index" in plan.case.metadata:
            enriched["generation_index"] = plan.case.metadata.get("generation_index")
        if "generation_seed" in plan.case.metadata:
            enriched["generation_seed"] = plan.case.metadata.get("generation_seed")
        if "source_case_id" in plan.case.metadata:
            enriched["source_case_id"] = plan.case.metadata.get("source_case_id")
        return enriched

    def _resolve_executor(self, name: str):
        return EXECUTOR_REGISTRY.resolve(name)

    def _reset_hardware_metrics(self, context: DeviceContext | None, executor_name: str) -> dict[str, Any]:
        torch_accelerator = self._torch_accelerator(context, executor_name)
        if torch_accelerator is None:
            return {}
        reset_peak = getattr(torch_accelerator, "reset_peak_memory_stats", None)
        if not callable(reset_peak):
            return {}
        try:
            reset_peak()
        except Exception as exc:  # noqa: BLE001 - metrics collection must not fail execution
            return {"hardware_metric_error": self._exception_text(exc)}
        return {"hardware_peak_memory_reset": True}

    def _synchronize_device(self, context: DeviceContext | None, executor_name: str, phase: str) -> dict[str, Any]:
        torch_accelerator = self._torch_accelerator(context, executor_name)
        if torch_accelerator is None:
            return {}
        synchronize = getattr(torch_accelerator, "synchronize", None)
        if not callable(synchronize):
            return {}
        try:
            synchronize()
        except Exception as exc:  # noqa: BLE001 - metrics collection must not fail execution
            return {f"hardware_sync_{phase}_error": self._exception_text(exc)}
        return {f"hardware_sync_{phase}": True}

    def _hardware_memory_snapshot(
        self,
        context: DeviceContext | None,
        executor_name: str,
        phase: str,
    ) -> dict[str, Any]:
        torch_accelerator = self._torch_accelerator(context, executor_name)
        if torch_accelerator is None:
            return {}
        metrics: dict[str, Any] = {}
        for metric_name, method_name in {
            "allocated_bytes": "memory_allocated",
            "reserved_bytes": "memory_reserved",
            "max_allocated_bytes": "max_memory_allocated",
            "max_reserved_bytes": "max_memory_reserved",
        }.items():
            method = getattr(torch_accelerator, method_name, None)
            if not callable(method):
                continue
            try:
                value = int(method())
            except Exception as exc:  # noqa: BLE001
                metrics[f"hardware_memory_{phase}_{metric_name}_error"] = self._exception_text(exc)
            else:
                metrics[f"hardware_memory_{phase}_{metric_name}"] = value
        return metrics

    def _hardware_memory_deltas(self, metrics: dict[str, Any]) -> dict[str, Any]:
        before = metrics.get("hardware_memory_before_allocated_bytes")
        after = metrics.get("hardware_memory_after_allocated_bytes")
        deltas: dict[str, Any] = {}
        if isinstance(before, int) and isinstance(after, int):
            deltas["hardware_memory_allocated_delta_bytes"] = after - before
        max_allocated = metrics.get("hardware_memory_after_max_allocated_bytes")
        if isinstance(max_allocated, int):
            deltas["hardware_memory_peak_bytes"] = max_allocated
            deltas["hardware_memory_peak_mb"] = max_allocated / (1024.0 * 1024.0)
        return deltas

    def _apply_runtime_policy(self, plan: ExecutionPlan, context: DeviceContext | None) -> tuple[dict[str, Any], bool]:
        policy = plan.case.runtime_policy
        requested = policy.model_dump(exclude_none=True, exclude_defaults=True)
        requested.pop("schema_version", None)
        if not requested:
            return {}, True
        capabilities = set(context.runtime_policy_capabilities) if context is not None else set()
        evidence: dict[str, Any] = {
            "schema_version": policy.schema_version,
            "requested": requested,
            "effective": {},
            "unsupported": [],
        }
        synchronize_timing = True
        if "synchronize_timing" in requested:
            if "synchronize_timing" not in capabilities:
                evidence["unsupported"].append("synchronize_timing")
            else:
                synchronize_timing = bool(policy.synchronize_timing)
                evidence["effective"]["synchronize_timing"] = synchronize_timing
        if "deterministic" in requested:
            if "deterministic" not in capabilities:
                evidence["unsupported"].append("deterministic")
            else:
                try:
                    import torch

                    torch.use_deterministic_algorithms(bool(policy.deterministic))
                    evidence["effective"]["deterministic"] = bool(policy.deterministic)
                except Exception as exc:  # noqa: BLE001 - runtime availability is backend-dependent
                    evidence["unsupported"].append(f"deterministic:{type(exc).__name__}")
        return evidence, synchronize_timing

    def _torch_accelerator(self, context: DeviceContext | None, executor_name: str) -> Any | None:
        if context is None or executor_name not in {"torch", "aclnn"}:
            return None
        if context.backend.value not in {"gpu", "dcu", "npu", "aclnn"}:
            return None
        try:
            import torch
        except Exception:  # noqa: BLE001 - no torch means no torch allocator metrics
            return None
        if context.backend.value in {"gpu", "dcu"}:
            accelerator = getattr(torch, "cuda", None)
        else:
            accelerator = getattr(torch, "npu", None)
        is_available = getattr(accelerator, "is_available", None) if accelerator is not None else None
        try:
            return accelerator if callable(is_available) and bool(is_available()) else None
        except Exception:  # noqa: BLE001
            return None

    def _context_metrics(self, context: DeviceContext, executor_name: str) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "backend": context.backend.value,
            "host": context.host,
            "node": context.node_name,
            "device_id": context.device.id,
        }
        resolved_device = self._resolved_device(context, executor_name)
        if resolved_device is not None:
            metrics["resolved_device"] = resolved_device
        if context.backend == BackendKind.GPU:
            metrics.update({
                "gpu_available": context.env.get("CX_GPU_AVAILABLE") == "true",
                "gpu_device_count": int(context.env.get("CX_GPU_DEVICE_COUNT", "0")),
                "gpu_hardware_visible": context.env.get("CX_GPU_HARDWARE_VISIBLE") == "true",
                "gpu_hardware_device_count": int(context.env.get("CX_GPU_HARDWARE_DEVICE_COUNT", "0")),
                "cuda_version": context.env.get("CX_CUDA_VERSION"),
                "gpu_evidence_path": context.env.get("CX_GPU_EVIDENCE"),
                "gpu_evidence_storage": context.env.get("CX_GPU_EVIDENCE_STORAGE"),
                "gpu_evidence": context.env.get("CX_GPU_EVIDENCE_JSON"),
                "gpu_evidence_fingerprint": context.env.get("CX_GPU_EVIDENCE_FINGERPRINT"),
            })
        if context.backend in {BackendKind.NPU, BackendKind.ACLNN}:
            metrics.update({
                "npu_available": context.env.get("CX_NPU_AVAILABLE") == "true",
                "npu_device_count": int(context.env.get("CX_NPU_DEVICE_COUNT", "0")),
                "torch_version": context.env.get("CX_NPU_TORCH_VERSION"),
                "torch_npu_version": context.env.get("CX_NPU_TORCH_NPU_VERSION"),
                "npu_device_name": context.env.get("CX_NPU_DEVICE_NAME"),
                "npu_evidence_path": context.env.get("CX_NPU_EVIDENCE"),
                "npu_evidence_storage": context.env.get("CX_NPU_EVIDENCE_STORAGE"),
                "npu_evidence": context.env.get("CX_NPU_EVIDENCE_JSON"),
                "npu_evidence_fingerprint": context.env.get("CX_NPU_EVIDENCE_FINGERPRINT"),
            })
        return metrics

    def _resolved_device(self, context: DeviceContext, executor_name: str) -> str | None:
        if executor_name not in {"torch", "aclnn"}:
            return None
        if context.backend.value == "cpu":
            return "cpu"
        device_index = 0 if context.env.get("CX_DEVICE_INDEX_MODE") == "actor_local" else context.device.id
        if context.backend.value in {"gpu", "dcu"}:
            return f"cuda:{device_index}"
        if context.backend.value in {"npu", "aclnn"}:
            return f"npu:{device_index}"
        return None

    def _exception_text(self, exc: Exception) -> str:
        return f"{exc.__class__.__name__}: {exc}"

    def _matches_expected_error(self, expected_error: str, actual_error: str) -> bool:
        return expected_error in actual_error

    def _benchmark_policy(self, plan: ExecutionPlan) -> dict[str, float | int]:
        inherited = plan.case.metadata.get("benchmark_policy", {})
        local = plan.case.invocation.metadata.get("benchmark", {})
        inherited = inherited if isinstance(inherited, dict) else {}
        local = local if isinstance(local, dict) else {}
        raw = {**inherited, **local}
        return {"warmup": max(0, int(raw.get("warmup", 0))), "repeat": max(1, int(raw.get("repeat", 1))), "throughput_items_per_call": float(raw.get("throughput_items_per_call", 1)), "min_time_ms": max(0.0, float(raw.get("min_time_ms", 0)))}

    def _benchmark_metrics(self, samples_ms: list[float], benchmark: dict[str, float | int], duration_ms: float) -> dict[str, float | int]:
        values = np.asarray(samples_ms or [duration_ms], dtype=float)
        mean = float(np.mean(values))
        items = float(benchmark["throughput_items_per_call"])
        return {"latency_ms": mean, "latency_mean_ms": mean, "latency_min_ms": float(np.min(values)), "latency_stddev_ms": float(np.std(values)), "latency_p50_ms": float(np.percentile(values, 50)), "latency_p90_ms": float(np.percentile(values, 90)), "latency_p95_ms": float(np.percentile(values, 95)), "latency_p99_ms": float(np.percentile(values, 99)), "throughput_items_per_s": items * 1000.0 / mean if mean else 0.0, "warmup_count": int(benchmark["warmup"]), "repeat_count": int(benchmark["repeat"]), "sample_count": len(samples_ms), "min_time_ms": float(benchmark["min_time_ms"]), "effective_duration_ms": duration_ms}

    def _input_dtype_contracts(self, plan: ExecutionPlan) -> list[dict[str, str]]:
        return [dtype_contract(str(parameter.dtypes[0] if parameter.dtypes else "fp32")) for parameter in plan.case.parameters]

    def _dtype_of(self, value: object) -> str | list[str] | None:
        if isinstance(value, np.ndarray):
            return str(value.dtype)
        if isinstance(value, np.generic):
            return str(value.dtype)
        if isinstance(value, (list, tuple)):
            return [dtype for item in value for dtype in [self._dtype_of(item)] if dtype is not None]
        return None

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

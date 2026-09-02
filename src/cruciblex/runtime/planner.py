from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from cruciblex.domain.enums import BackendKind, TaskKind
from cruciblex.domain.node import DeviceSpec, NodeSpec
from cruciblex.domain.plan import ExecutionPlan, JobSpec


@dataclass(frozen=True, slots=True)
class ExecutionSlot:
    task: TaskKind
    node: NodeSpec
    device: DeviceSpec


class ExecutionPlanner:
    def build(self, job: JobSpec) -> list[ExecutionPlan]:
        slots = self._execution_slots(job)
        return [
            self._plan(case, slot, job)
            for case, slot in product(job.cases, slots)
            if self._case_supports_slot(case, slot)
        ]

    def _execution_slots(self, job: JobSpec) -> list[ExecutionSlot]:
        return [
            ExecutionSlot(task=task, node=node, device=device)
            for task in job.tasks
            for node in self._eligible_nodes(job.nodes, task)
            for device in self._eligible_devices(node)
        ]

    def _plan(self, case, slot: ExecutionSlot, job: JobSpec) -> ExecutionPlan:
        return ExecutionPlan(
            case=case,
            node=slot.node,
            device=slot.device,
            task=slot.task,
            artifacts=job.artifacts,
        )

    def _eligible_nodes(self, nodes: list[NodeSpec], task: TaskKind) -> list[NodeSpec]:
        return [node for node in nodes if node.supports(task)]

    def _eligible_devices(self, node: NodeSpec) -> list[DeviceSpec]:
        node_backend = self._node_backend(node)
        return [device for device in node.devices if device.backend == node_backend]

    def _case_supports_slot(self, case, slot: ExecutionSlot) -> bool:
        backends = case.metadata.get("manifest_backends")
        if backends is None:
            return True
        if not isinstance(backends, list):
            raise TypeError("case manifest_backends metadata must be a list")
        return slot.device.backend.value in {str(backend) for backend in backends}

    def _node_backend(self, node: NodeSpec) -> BackendKind:
        backends = {device.backend for device in node.devices}
        if len(backends) != 1:
            raise ValueError(f"node {node.display_name} must not mix device backends")
        return next(iter(backends))

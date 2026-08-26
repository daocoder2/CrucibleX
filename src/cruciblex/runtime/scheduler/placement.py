from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cruciblex.domain.node import NodeSpec
from cruciblex.domain.plan import ExecutionPlan
from cruciblex.runtime.backends.resources import RayResourceSpec, ray_resources_for

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_NODE_RESOURCE_PREFIX = "node:"
_NODE_AFFINITY_FRACTION = 0.001


class RayPlacementError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RayNodeInfo:
    node_id: str
    address: str
    hostname: str
    alive: bool
    resources: dict[str, float]

    @classmethod
    def from_ray_node(cls, raw_node: dict[str, Any]) -> RayNodeInfo:
        return cls(
            node_id=str(raw_node.get("NodeID") or raw_node.get("node_id") or ""),
            address=str(raw_node.get("NodeManagerAddress") or raw_node.get("address") or ""),
            hostname=str(raw_node.get("NodeManagerHostname") or raw_node.get("hostname") or ""),
            alive=bool(raw_node.get("Alive", raw_node.get("alive", True))),
            resources={str(key): float(value) for key, value in raw_node.get("Resources", {}).items()},
        )

    def matches(self, node: NodeSpec) -> bool:
        host = node.host.strip()
        if host in {self.address, self.hostname}:
            return True
        return host in _LOCAL_HOSTS and self.address in _LOCAL_HOSTS

    @property
    def node_resource_key(self) -> str | None:
        key = f"{_NODE_RESOURCE_PREFIX}{self.node_id}"
        return key if key in self.resources else None


@dataclass(frozen=True, slots=True)
class RayClusterSnapshot:
    nodes: list[RayNodeInfo]

    @property
    def alive_nodes(self) -> list[RayNodeInfo]:
        return [node for node in self.nodes if node.alive]


@dataclass(frozen=True, slots=True)
class RayPlacementDecision:
    node: RayNodeInfo
    resources: RayResourceSpec

    def actor_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.resources.num_cpus is not None:
            options["num_cpus"] = self.resources.num_cpus
        if self.resources.num_gpus is not None:
            options["num_gpus"] = self.resources.num_gpus
        custom_resources = dict(self.resources.resources)
        node_resource_key = self.node.node_resource_key
        if node_resource_key is not None:
            custom_resources[node_resource_key] = _NODE_AFFINITY_FRACTION
        if custom_resources:
            options["resources"] = custom_resources
        return options


def discover_ray_cluster(ray: Any) -> RayClusterSnapshot:
    return RayClusterSnapshot([RayNodeInfo.from_ray_node(node) for node in ray.nodes()])


def decide_ray_placement(plan: ExecutionPlan, snapshot: RayClusterSnapshot) -> RayPlacementDecision:
    alive_nodes = snapshot.alive_nodes
    matched_nodes = [node for node in alive_nodes if _matches_plan_node(plan.node, node, alive_nodes)]
    if not matched_nodes:
        raise RayPlacementError(f"no alive Ray node matches configured host: {plan.node.host}")
    resources = ray_resources_for(plan.device)
    for node in matched_nodes:
        if _has_resources(node, resources):
            return RayPlacementDecision(node=node, resources=resources)
    raise RayPlacementError(
        f"Ray node resources cannot satisfy device {plan.device.slot} on host {plan.node.host}"
    )


def _matches_plan_node(plan_node: NodeSpec, ray_node: RayNodeInfo, alive_nodes: list[RayNodeInfo]) -> bool:
    if ray_node.matches(plan_node):
        return True
    return len(alive_nodes) == 1 and plan_node.host in _LOCAL_HOSTS


def _has_resources(node: RayNodeInfo, resources: RayResourceSpec) -> bool:
    if resources.num_cpus is not None and node.resources.get("CPU", 0.0) < resources.num_cpus:
        return False
    if resources.num_gpus is not None and node.resources.get("GPU", 0.0) < resources.num_gpus:
        return False
    return all(node.resources.get(name, 0.0) >= amount for name, amount in resources.resources.items())

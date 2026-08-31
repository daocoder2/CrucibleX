from cruciblex.runtime.scheduler.base import Scheduler
from cruciblex.runtime.scheduler.local import LocalScheduler
from cruciblex.runtime.scheduler.placement import (
    RayClusterSnapshot,
    RayNodeInfo,
    RayPlacementDecision,
    RayPlacementError,
    decide_ray_placement,
    discover_ray_cluster,
)

__all__ = [
    "LocalScheduler",
    "RayClusterSnapshot",
    "RayNodeInfo",
    "RayPlacementDecision",
    "RayPlacementError",
    "RayScheduler",
    "Scheduler",
    "decide_ray_placement",
    "discover_ray_cluster",
    "ray_init_kwargs",
]


def __getattr__(name: str):
    if name in {"RayScheduler", "ray_init_kwargs"}:
        from cruciblex.runtime.scheduler.ray import RayScheduler, ray_init_kwargs

        return {"RayScheduler": RayScheduler, "ray_init_kwargs": ray_init_kwargs}[name]
    raise AttributeError(name)

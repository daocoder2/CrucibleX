from __future__ import annotations

import importlib.util
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from cruciblex.domain.enums import BackendKind, TaskKind
from cruciblex.runtime.scheduler.placement import (
    RayClusterSnapshot,
    RayNodeInfo,
    discover_ray_cluster,
)

_CPU_TASKS = [TaskKind.ACCURACY.value, TaskKind.RUN.value, TaskKind.FUZZ.value]
_ACCELERATOR_TASKS = [
    TaskKind.ACCURACY.value,
    TaskKind.RUN.value,
    TaskKind.PERFORMANCE_DEVICE.value,
    TaskKind.MEMORY_DEVICE.value,
]
_GPU_TASKS = [
    TaskKind.ACCURACY.value,
    TaskKind.RUN.value,
    TaskKind.PERFORMANCE_DEVICE.value,
    TaskKind.PERFORMANCE_DEVICE_PTA.value,
    TaskKind.PERFORMANCE_E2E.value,
    TaskKind.MEMORY_DEVICE.value,
]


def discover_runtime_resources(ray: Any | None = None, ray_address: str | None = None) -> dict[str, Any]:
    runtime = ray
    available = True
    initialized = False
    init_error: str | None = None
    if runtime is None:
        try:
            import ray as runtime
        except ImportError as exc:
            available = False
            init_error = f"{exc.__class__.__name__}: {exc}"
            return _empty_snapshot(ray_address, available=available, initialized=initialized, init_error=init_error)
    try:
        initialized = bool(runtime.is_initialized())
        if not initialized and ray_address is not None:
            runtime.init(**_ray_init_kwargs(ray_address))
            initialized = bool(runtime.is_initialized())
    except Exception as exc:  # noqa: BLE001 - discovery should report the failure, not raise it
        init_error = f"{exc.__class__.__name__}: {exc}"
    snapshot = discover_ray_cluster(runtime) if initialized else RayClusterSnapshot([])
    discovery = build_discovery_snapshot(snapshot, ray_address, available=available, initialized=initialized, init_error=init_error)
    discovery["runtime_probes"] = {
        "driver": _runtime_probe(),
        "workers": _worker_runtime_probes(runtime, snapshot) if initialized else [],
    }
    return discovery


def write_discovery_files(output: str | Path, discovery: dict[str, Any]) -> dict[str, Path]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    snapshot_path = root / "resource_snapshot.json"
    nodes_path = root / "discovered_nodes.yaml"
    snapshot_path.write_text(_json_text(discovery), encoding="utf-8")
    nodes_path.write_text(_nodes_yaml(discovery.get("node_templates", [])), encoding="utf-8")
    return {"snapshot": snapshot_path, "nodes": nodes_path}


def build_discovery_snapshot(
    snapshot: RayClusterSnapshot,
    ray_address: str | None,
    *,
    available: bool,
    initialized: bool,
    init_error: str | None,
) -> dict[str, Any]:
    node_rows = [_node_payload(node) for node in snapshot.nodes]
    node_templates = [template for node in snapshot.alive_nodes for template in _node_templates(node)]
    backend_counts = Counter(template["devices"][0]["backend"] for template in node_templates)
    return {
        "kind": "ray-resource-discovery",
        "source": {
            "available": available,
            "initialized": initialized,
            "ray_address": ray_address,
            "init_error": init_error,
            "node_count": len(snapshot.nodes),
            "alive_node_count": len(snapshot.alive_nodes),
        },
        "nodes": node_rows,
        "node_templates": node_templates,
        "backend_counts": dict(sorted(backend_counts.items())),
        "capabilities": _capabilities(node_templates),
        "runtime_probes": {"driver": _runtime_probe(), "workers": []},
    }


def _empty_snapshot(
    ray_address: str | None,
    *,
    available: bool,
    initialized: bool,
    init_error: str | None,
) -> dict[str, Any]:
    return {
        "kind": "ray-resource-discovery",
        "source": {
            "available": available,
            "initialized": initialized,
            "ray_address": ray_address,
            "init_error": init_error,
            "node_count": 0,
            "alive_node_count": 0,
        },
        "nodes": [],
        "node_templates": [],
        "backend_counts": {},
        "capabilities": [],
        "runtime_probes": {"driver": _runtime_probe(), "workers": []},
    }


def _runtime_probe() -> dict[str, Any]:
    probe: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "env": _selected_env(),
        "packages": {},
    }
    probe["packages"]["torch"] = _torch_probe()
    probe["packages"]["torch_npu"] = _module_probe("torch_npu")
    return probe


def _worker_runtime_probes(runtime: Any, snapshot: RayClusterSnapshot) -> list[dict[str, Any]]:
    if not hasattr(runtime, "remote"):
        return []
    try:
        remote_probe = runtime.remote(_remote_runtime_probe)
    except Exception as exc:  # noqa: BLE001 - discovery should not fail runs
        return [{"error": f"{exc.__class__.__name__}: {exc}"}]

    refs: list[tuple[RayNodeInfo, object]] = []
    for node in snapshot.alive_nodes:
        options: dict[str, Any] = {}
        if node.node_resource_key:
            options["resources"] = {node.node_resource_key: 0.001}
        try:
            ref = remote_probe.options(**options).remote()
        except Exception as exc:  # noqa: BLE001
            refs.append((node, {"error": f"{exc.__class__.__name__}: {exc}"}))
        else:
            refs.append((node, ref))

    workers: list[dict[str, Any]] = []
    for node, ref in refs:
        payload: dict[str, Any] = {"node": _node_payload(node)}
        if isinstance(ref, dict):
            payload["probe"] = ref
        else:
            try:
                payload["probe"] = runtime.get(ref)
            except Exception as exc:  # noqa: BLE001
                payload["probe"] = {"error": f"{exc.__class__.__name__}: {exc}"}
        workers.append(payload)
    return workers


def _remote_runtime_probe() -> dict[str, Any]:
    from cruciblex.runtime.discovery import _runtime_probe

    return _runtime_probe()


def _selected_env() -> dict[str, str]:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "ASCEND_VISIBLE_DEVICES",
        "ASCEND_DEVICE_ID",
        "ASCEND_HOME_PATH",
        "LD_LIBRARY_PATH",
        "CX_DEVICE_INDEX_MODE",
    ]
    return {key: value for key in keys if (value := os.environ.get(key)) is not None}


def _module_probe(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"available": False}
    return {"available": True, "origin": spec.origin}


def _torch_probe() -> dict[str, Any]:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return {"available": False, "gpu_backend": {"available": False, "device_count": 0, "cuda_version": None, "error": "torch is not installed"}}
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        error = f"{exc.__class__.__name__}: {exc}"
        return {"available": True, "import_error": error, "gpu_backend": {"available": False, "device_count": 0, "cuda_version": None, "error": error}}

    probe: dict[str, Any] = {
        "available": True,
        "version": str(getattr(torch, "__version__", "")) or None,
        "cuda_version": str(getattr(getattr(torch, "version", None), "cuda", "")) or None,
        "cuda_available": False,
        "cuda_device_count": 0,
    }
    cuda = getattr(torch, "cuda", None)
    if cuda is not None:
        is_available = getattr(cuda, "is_available", None)
        device_count = getattr(cuda, "device_count", None)
        if callable(is_available):
            probe["cuda_available"] = bool(is_available())
        if callable(device_count):
            probe["cuda_device_count"] = int(device_count())
    npu = getattr(torch, "npu", None)
    if npu is not None:
        is_available = getattr(npu, "is_available", None)
        device_count = getattr(npu, "device_count", None)
        if callable(is_available):
            probe["npu_available"] = bool(is_available())
        if callable(device_count):
            probe["npu_device_count"] = int(device_count())
    probe["gpu_backend"] = {
        "available": probe["cuda_available"],
        "device_count": probe["cuda_device_count"],
        "cuda_version": probe["cuda_version"],
        "error": None,
    }
    return probe


def _node_payload(node: RayNodeInfo) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "address": node.address,
        "hostname": node.hostname,
        "alive": node.alive,
        "resources": dict(sorted(node.resources.items())),
        "backends": [backend.value for backend in _node_backends(node)],
    }


def _node_templates(node: RayNodeInfo) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for backend in _node_backends(node):
        templates.append(
            {
                "name": _node_template_name(node, backend),
                "host": node.address or node.hostname,
                "role": "candidate",
                "allowed_tasks": _allowed_tasks(backend),
                "devices": [
                    {
                        "id": 0,
                        "backend": backend.value,
                        **(_device_resources(backend, node.resources)),
                    }
                ],
                "labels": _template_labels(node, backend),
            }
        )
    return templates


def _node_backends(node: RayNodeInfo) -> list[BackendKind]:
    resources = node.resources
    backends: list[BackendKind] = []
    if resources.get("CPU", 0.0) > 0.0:
        backends.append(BackendKind.CPU)
    if resources.get("GPU", 0.0) > 0.0:
        backends.append(BackendKind.GPU)
    if resources.get("npu", 0.0) > 0.0 or resources.get("ascend", 0.0) > 0.0:
        backends.extend([BackendKind.NPU, BackendKind.ACLNN])
    if resources.get("aclnn", 0.0) > 0.0 or resources.get("acl", 0.0) > 0.0:
        backends.append(BackendKind.ACLNN)
    if resources.get("dcu", 0.0) > 0.0:
        backends.append(BackendKind.DCU)
    if not backends:
        backends.append(BackendKind.CPU)
    unique: list[BackendKind] = []
    for backend in backends:
        if backend not in unique:
            unique.append(backend)
    return unique


def _allowed_tasks(backend: BackendKind) -> list[str]:
    if backend == BackendKind.CPU:
        return list(_CPU_TASKS)
    if backend == BackendKind.GPU:
        return list(_GPU_TASKS)
    if backend in {BackendKind.NPU, BackendKind.ACLNN, BackendKind.DCU}:
        return list(_ACCELERATOR_TASKS)
    return [TaskKind.ACCURACY.value, TaskKind.RUN.value]


def _device_resources(backend: BackendKind, resources: dict[str, float]) -> dict[str, dict[str, float]]:
    if backend in {BackendKind.NPU, BackendKind.ACLNN}:
        return {"resources": {"npu": max(resources.get("npu", 0.0), 1.0)}}
    if backend == BackendKind.DCU:
        return {"resources": {"dcu": max(resources.get("dcu", 0.0), 1.0)}}
    return {}


def _template_labels(node: RayNodeInfo, backend: BackendKind) -> list[str]:
    labels = {f"ray:{node.node_id}"}
    if node.hostname:
        labels.add(f"host:{node.hostname}")
    labels.add(f"backend:{backend.value}")
    return sorted(labels)


def _node_template_name(node: RayNodeInfo, backend: BackendKind) -> str:
    host = node.hostname or node.address or node.node_id or "ray-node"
    safe_host = "".join(char if char.isalnum() or char in "._-" else "-" for char in host)
    return f"{safe_host}-{backend.value}"


def _capabilities(node_templates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(template["devices"][0]["backend"] for template in node_templates)
    return [
        {"backend": backend, "supported": count > 0, "node_count": count}
        for backend, count in sorted(counts.items())
    ]


def _ray_init_kwargs(address: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"ignore_reinit_error": True}
    if address:
        kwargs["address"] = address
        if address.startswith("ray://"):
            kwargs["runtime_env"] = {"working_dir": None}
    return kwargs


def _json_text(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _nodes_yaml(node_templates: list[dict[str, Any]]) -> str:
    import yaml

    return yaml.safe_dump({"nodes": node_templates}, sort_keys=False)

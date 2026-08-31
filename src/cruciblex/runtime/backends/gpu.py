from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from typing import Any

from cruciblex.domain.enums import BackendKind
from cruciblex.runtime.backends.base import BACKEND_REGISTRY, BackendRuntime, DeviceContext


@BACKEND_REGISTRY.register(BackendKind.GPU)
class GpuBackendRuntime(BackendRuntime):
    def prepare(self, context: DeviceContext) -> DeviceContext:
        context.env["CX_BACKEND"] = BackendKind.GPU.value
        context.env["CX_DEVICE_ID"] = str(context.device.id)
        context.env.setdefault("CUDA_VISIBLE_DEVICES", str(context.device.id))
        context.env.setdefault("NVIDIA_VISIBLE_DEVICES", str(context.device.id))
        diagnostics = self.probe()
        context.env["CX_GPU_AVAILABLE"] = str(diagnostics["available"]).lower()
        context.env["CX_GPU_DEVICE_COUNT"] = str(diagnostics["device_count"])
        context.env["CX_GPU_HARDWARE_VISIBLE"] = str(diagnostics.get("hardware_visible", False)).lower()
        context.env["CX_GPU_HARDWARE_DEVICE_COUNT"] = str(diagnostics.get("hardware_device_count", diagnostics["device_count"]))
        if diagnostics["cuda_version"] is not None:
            context.env["CX_CUDA_VERSION"] = str(diagnostics["cuda_version"])
        evidence = {
            "schema_version": 1,
            "backend": BackendKind.GPU.value,
            "host": context.host,
            "node": context.node_name,
            "device_id": context.device.id,
            "device_selector": f"cuda:{context.device.id}",
            "probe_status": "available" if diagnostics["available"] else "unavailable",
            **diagnostics,
            "visible_devices": context.env["CUDA_VISIBLE_DEVICES"],
        }
        evidence["evidence_fingerprint"] = hashlib.sha256(
            json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        try:
            context.output_root.mkdir(parents=True, exist_ok=True)
            evidence_path = context.output_root / "gpu_evidence.json"
            evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
            context.env["CX_GPU_EVIDENCE"] = str(evidence_path)
            context.env["CX_GPU_EVIDENCE_STORAGE"] = "file"
        except OSError:
            context.env["CX_GPU_EVIDENCE"] = "inline_remote"
            context.env["CX_GPU_EVIDENCE_STORAGE"] = "inline_remote"
        context.env["CX_GPU_EVIDENCE_JSON"] = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
        context.env["CX_GPU_EVIDENCE_FINGERPRINT"] = evidence["evidence_fingerprint"]
        return context

    def probe(self) -> dict[str, Any]:
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            hardware = self._nvidia_smi_probe()
            return {"available": False, "device_count": 0, "cuda_version": None, "hardware_visible": hardware["visible"], "hardware_device_count": hardware["device_count"], "error": f"{exc.__class__.__name__}: {exc}"}
        cuda = getattr(torch, "cuda", None)
        version = str(getattr(getattr(torch, "version", None), "cuda", "")) or None
        errors: list[str] = []
        available = False
        count = 0
        is_available = getattr(cuda, "is_available", None) if cuda is not None else None
        device_count = getattr(cuda, "device_count", None) if cuda is not None else None
        if callable(is_available):
            try:
                available = bool(is_available())
            except Exception as exc:  # noqa: BLE001 - probe must return evidence
                errors.append(f"is_available: {exc.__class__.__name__}: {exc}")
        if callable(device_count):
            try:
                count = int(device_count())
            except Exception as exc:  # noqa: BLE001 - probe must return evidence
                errors.append(f"device_count: {exc.__class__.__name__}: {exc}")
        return {"available": available, "device_count": count, "cuda_version": version, "hardware_visible": available, "hardware_device_count": count, "error": "; ".join(errors) or None}

    def _nvidia_smi_probe(self) -> dict[str, Any]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return {"visible": False, "device_count": 0}
        devices = [line for line in result.stdout.splitlines() if line.strip()]
        return {"visible": result.returncode == 0 and bool(devices), "device_count": len(devices)}
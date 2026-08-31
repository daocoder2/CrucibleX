from __future__ import annotations

import hashlib
import importlib
import json
from typing import Any

from cruciblex.domain.enums import BackendKind
from cruciblex.runtime.backends.base import BACKEND_REGISTRY, BackendRuntime, DeviceContext


class _NpuEvidenceRuntime(BackendRuntime):
    def _prepare(self, context: DeviceContext, backend: BackendKind) -> DeviceContext:
        context.env["CX_BACKEND"] = backend.value
        context.env["CX_DEVICE_ID"] = str(context.device.id)
        context.env.setdefault("ASCEND_DEVICE_ID", str(context.device.id))
        diagnostics = self.probe(context.device.id)
        context.env["CX_NPU_AVAILABLE"] = str(diagnostics["available"]).lower()
        context.env["CX_NPU_DEVICE_COUNT"] = str(diagnostics["device_count"])
        for name in ("torch_version", "torch_npu_version", "device_name"):
            if diagnostics.get(name) is not None:
                context.env[f"CX_NPU_{name.upper()}"] = str(diagnostics[name])
        evidence = {
            "schema_version": 1,
            "backend": backend.value,
            "host": context.host,
            "node": context.node_name,
            "device_id": context.device.id,
            "device_selector": f"npu:{context.device.id}",
            "probe_status": "available" if diagnostics["available"] else "unavailable",
            **diagnostics,
        }
        evidence["evidence_fingerprint"] = hashlib.sha256(
            json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        try:
            context.output_root.mkdir(parents=True, exist_ok=True)
            evidence_path = context.output_root / "npu_evidence.json"
            evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
            context.env["CX_NPU_EVIDENCE"] = str(evidence_path)
            context.env["CX_NPU_EVIDENCE_STORAGE"] = "file"
        except OSError:
            context.env["CX_NPU_EVIDENCE"] = "inline_remote"
            context.env["CX_NPU_EVIDENCE_STORAGE"] = "inline_remote"
        context.env["CX_NPU_EVIDENCE_JSON"] = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
        context.env["CX_NPU_EVIDENCE_FINGERPRINT"] = evidence["evidence_fingerprint"]
        return context

    def probe(self, device_id: int) -> dict[str, Any]:
        try:
            torch = importlib.import_module("torch")
            torch_npu = importlib.import_module("torch_npu")
        except ImportError as exc:
            return {"available": False, "device_count": 0, "torch_version": None, "torch_npu_version": None, "device_name": None, "error": f"{exc.__class__.__name__}: {exc}"}
        npu = getattr(torch, "npu", None)
        errors: list[str] = []
        available = False
        count = 0
        device_name = None
        is_available = getattr(npu, "is_available", None) if npu is not None else None
        device_count = getattr(npu, "device_count", None) if npu is not None else None
        get_device_name = getattr(npu, "get_device_name", None) if npu is not None else None
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
        if available and callable(get_device_name):
            try:
                device_name = str(get_device_name(device_id))
            except Exception as exc:  # noqa: BLE001 - probe must return evidence
                errors.append(f"get_device_name: {exc.__class__.__name__}: {exc}")
        return {
            "available": available,
            "device_count": count,
            "torch_version": str(getattr(torch, "__version__", "")) or None,
            "torch_npu_version": str(getattr(torch_npu, "__version__", "")) or None,
            "device_name": device_name,
            "error": "; ".join(errors) or None,
        }


@BACKEND_REGISTRY.register(BackendKind.NPU)
class NpuBackendRuntime(_NpuEvidenceRuntime):
    def prepare(self, context: DeviceContext) -> DeviceContext:
        return self._prepare(context, BackendKind.NPU)


@BACKEND_REGISTRY.register(BackendKind.ACLNN)
class AclnnBackendRuntime(_NpuEvidenceRuntime):
    def prepare(self, context: DeviceContext) -> DeviceContext:
        return self._prepare(context, BackendKind.ACLNN)

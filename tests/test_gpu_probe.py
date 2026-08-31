import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cruciblex.domain.enums import BackendKind
from cruciblex.domain.node import DeviceSpec
from cruciblex.runtime.backends import gpu
from cruciblex.runtime.backends.base import DeviceContext

pytestmark = pytest.mark.gpu


def test_gpu_probe_reports_cuda_probe_errors(monkeypatch):
    class BrokenCuda:
        def is_available(self):
            raise RuntimeError("driver missing")

        def device_count(self):
            raise RuntimeError("driver unavailable")

    monkeypatch.setattr(gpu.importlib, "import_module", lambda name: SimpleNamespace(cuda=BrokenCuda(), version=SimpleNamespace(cuda="12.1")))
    result = gpu.GpuBackendRuntime().probe()
    assert result["available"] is False
    assert result["device_count"] == 0
    assert result["cuda_version"] == "12.1"
    assert "is_available: RuntimeError" in result["error"]
    assert "device_count: RuntimeError" in result["error"]


def test_gpu_evidence_fingerprint_is_written_to_context(tmp_path, monkeypatch):
    fake_cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    monkeypatch.setattr(gpu.importlib, "import_module", lambda name: SimpleNamespace(cuda=fake_cuda, version=SimpleNamespace(cuda=None)))
    context = DeviceContext(
        host="host",
        node_name="gpu-node",
        device=DeviceSpec(id=2, backend=BackendKind.GPU),
        output_root=Path(tmp_path),
    )
    gpu.GpuBackendRuntime().prepare(context)
    evidence = json.loads((tmp_path / "gpu_evidence.json").read_text(encoding="utf-8"))
    assert evidence["schema_version"] == 1
    assert evidence["device_selector"] == "cuda:2"
    assert evidence["visible_devices"] == "2"
    assert evidence["probe_status"] == "unavailable"
    assert len(evidence["evidence_fingerprint"]) == 64
    assert context.env["CX_GPU_EVIDENCE_FINGERPRINT"] == evidence["evidence_fingerprint"]

    from cruciblex.runtime.pipeline import ExecutionPipeline

    metrics = ExecutionPipeline()._context_metrics(context, "torch")
    assert metrics["gpu_evidence_path"] == context.env["CX_GPU_EVIDENCE"]
    assert metrics["gpu_evidence_fingerprint"] == evidence["evidence_fingerprint"]


def test_gpu_probe_records_physical_devices_without_torch(monkeypatch):
    monkeypatch.setattr(gpu.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError("torch missing")))
    monkeypatch.setattr(
        gpu.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="NVIDIA A800\nNVIDIA A800\n"),
    )
    result = gpu.GpuBackendRuntime().probe()
    assert result["available"] is False
    assert result["device_count"] == 0
    assert result["hardware_visible"] is True
    assert result["hardware_device_count"] == 2
    assert "torch missing" in result["error"]


def test_gpu_failure_kind_distinguishes_driver_device_and_kernel_errors():
    from cruciblex.runtime.pipeline import ExecutionPipeline

    pipeline = ExecutionPipeline()
    plan = SimpleNamespace(device=SimpleNamespace(backend=BackendKind.GPU))
    assert pipeline._failure_kind(plan, "candidate", RuntimeError("CUDA kernel failed")) == "gpu_kernel_error"
    assert pipeline._failure_kind(plan, "candidate", RuntimeError("CUDA driver not initialized")) == "gpu_driver_error"
    assert pipeline._failure_kind(plan, "candidate", RuntimeError("invalid device ordinal")) == "gpu_device_index_invalid"

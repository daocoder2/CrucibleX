import json
from pathlib import Path
from types import SimpleNamespace

from cruciblex.domain.enums import BackendKind
from cruciblex.domain.node import DeviceSpec
from cruciblex.runtime.backends import npu
from cruciblex.runtime.backends.base import DeviceContext
from cruciblex.runtime.pipeline import ExecutionPipeline


def test_npu_probe_writes_available_evidence(monkeypatch, tmp_path):
    fake_npu = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 8,
        get_device_name=lambda device_id: f"Ascend-{device_id}",
    )
    modules = {
        "torch": SimpleNamespace(npu=fake_npu, __version__="2.6.0"),
        "torch_npu": SimpleNamespace(__version__="2.6.0.post5"),
    }
    monkeypatch.setattr(npu.importlib, "import_module", modules.__getitem__)
    context = DeviceContext(
        host="worker",
        node_name="npu-node",
        device=DeviceSpec(id=2, backend=BackendKind.ACLNN),
        output_root=Path(tmp_path),
    )

    npu.AclnnBackendRuntime().prepare(context)

    evidence = json.loads((tmp_path / "npu_evidence.json").read_text(encoding="utf-8"))
    assert evidence["probe_status"] == "available"
    assert evidence["device_count"] == 8
    assert evidence["device_name"] == "Ascend-2"
    assert len(evidence["evidence_fingerprint"]) == 64
    metrics = ExecutionPipeline()._context_metrics(context, "aclnn")
    assert metrics["npu_available"] is True
    assert metrics["npu_evidence_fingerprint"] == evidence["evidence_fingerprint"]


def test_npu_probe_marks_missing_runtime_unavailable(monkeypatch):
    monkeypatch.setattr(npu.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError("missing")))

    probe = npu.NpuBackendRuntime().probe(0)

    assert probe["available"] is False
    assert probe["device_count"] == 0
    assert "ImportError: missing" in probe["error"]

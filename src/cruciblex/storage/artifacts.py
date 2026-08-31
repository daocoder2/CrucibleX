from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cruciblex.domain.plan import ExecutionPlan
from cruciblex.domain.result import ArtifactPayload, ArtifactRef


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def ensure(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def plan_root(self, plan: ExecutionPlan) -> Path:
        return self.ensure() / plan.case.name / plan.plan_id

    def write_json(self, path: Path, data: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def payload_json(
        self,
        name: str,
        data: Any,
        kind: str,
        role: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactPayload:
        metadata = {**(metadata or {}), **({"role": role} if role else {})}
        return ArtifactPayload(name=name, kind=kind, data=data, metadata=metadata)

    def write_payload(self, plan: ExecutionPlan, payload: ArtifactPayload) -> ArtifactRef:
        path = self.plan_root(plan) / self._payload_filename(payload)
        if payload.kind == "log" and isinstance(payload.data, str):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload.data, encoding="utf-8")
        else:
            self.write_json(path, payload.data)
        return ArtifactRef(
            name=payload.name,
            path=path,
            kind=payload.kind,
            metadata=payload.metadata,
        )

    def _payload_filename(self, payload: ArtifactPayload) -> str:
        suffix = "log" if payload.kind == "log" else "json"
        return f"{payload.name}.{suffix}"

    def record_json(
        self,
        plan: ExecutionPlan,
        name: str,
        data: Any,
        kind: str,
        role: str | None = None,
    ) -> ArtifactRef:
        return self.write_payload(plan, self.payload_json(name, data, kind, role))


class ArtifactRecorder:
    def __init__(self, store: ArtifactStore, plan: ExecutionPlan, persist: bool) -> None:
        self.store = store
        self.plan = plan
        self.persist = persist
        self.artifacts: list[ArtifactRef] = []
        self.payloads: list[ArtifactPayload] = []

    def record_json(
        self,
        name: str,
        data: Any,
        kind: str,
        role: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = self.store.payload_json(name, data, kind, role, metadata)
        if self.persist:
            self.artifacts.append(self.store.write_payload(self.plan, payload))
        else:
            self.payloads.append(payload)

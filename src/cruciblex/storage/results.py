from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cruciblex.domain.result import ExecutionResult
from cruciblex.domain.run import RunManifest


class ResultStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def ensure(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def manifest_path(self, name: str = "manifest.json") -> Path:
        return self.ensure() / name

    def write_manifest(self, manifest: RunManifest, name: str = "manifest.json") -> Path:
        path = self.manifest_path(name)
        path.write_text(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def read_manifest(self, name: str = "manifest.json") -> RunManifest:
        path = self.manifest_path(name)
        return RunManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def write_results_jsonl(self, results: list[ExecutionResult], name: str = "results.jsonl") -> Path:
        path = self.ensure() / name
        lines = [json.dumps(result.model_dump(mode="json"), ensure_ascii=False) for result in results]
        content = "\n".join(lines)
        if content:
            content += "\n"
        path.write_text(content, encoding="utf-8")
        return path

    def read_results_jsonl(self, name: str = "results.jsonl") -> list[ExecutionResult]:
        path = self.ensure() / name
        if not path.exists():
            return []
        return [ExecutionResult.model_validate(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def write_summary_json(self, summary: dict[str, Any], name: str = "summary.json") -> Path:
        path = self.ensure() / name
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

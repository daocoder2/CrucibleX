from __future__ import annotations

import json
import shlex
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from cruciblex.domain.run import RunManifest
from cruciblex.report.reduction import reduce_with_predicate, semantic_reduction_candidates
from cruciblex.storage.results import ResultStore


class ReproBundleWriter:
    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root)
        self._store = ResultStore(self.output_root)

    def write(
        self,
        name: str = "repro_bundle.json",
        cluster_id: str | None = None,
        minimize: bool = False,
        replay_predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> Path:
        manifest = self._store.read_manifest()
        postprocess = self._read_postprocess()
        clusters = postprocess.get("failure_clusters", [])
        payload_clusters = [self._cluster_payload(manifest, cluster, index, minimize=minimize) for index, cluster in enumerate(clusters)]
        if cluster_id is not None:
            payload_clusters = [cluster for cluster in payload_clusters if cluster["cluster_id"] == cluster_id]
        payload = {
            "run_id": manifest.run_id,
            "source_output_root": str(manifest.output_root),
            "minimized": minimize,
            "clusters": payload_clusters,
        }
        for cluster in payload_clusters:
            cluster["artifacts"] = self._write_cluster_artifacts(manifest, cluster, replay_predicate)
        path = self.output_root / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _write_cluster_artifacts(
        self,
        manifest: RunManifest,
        cluster: dict[str, Any],
        replay_predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, str]:
        root = self.output_root / "repro" / _slug(str(cluster["signature"]))
        root.mkdir(parents=True, exist_ok=True)
        artifacts = {"root": str(root)}
        failure = root / "failure.json"
        failure.write_text(json.dumps(cluster, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts["failure"] = str(failure)
        representative = next(iter(cluster.get("cases", [])), {})
        case_id = representative.get("case_id")
        case: dict[str, Any] | None = None
        generated = self.output_root / "generated_cases.json"
        if generated.exists():
            payload = json.loads(generated.read_text(encoding="utf-8"))
            case = next((item for item in payload.get("cases", []) if item.get("id") == case_id), None)
            if case is not None:
                candidates = semantic_reduction_candidates(case)
                if replay_predicate is not None:
                    reduced_case, attempts = reduce_with_predicate(case, replay_predicate)
                    if attempts:
                        reduced_path = root / "semantic_reduction.yaml"
                        reduced_path.write_text(yaml.safe_dump(reduced_case, sort_keys=False), encoding="utf-8")
                        artifacts["semantic_reduction"] = str(reduced_path)
                        artifacts["semantic_reduction_attempts"] = json.dumps(attempts, sort_keys=True)
                    else:
                        artifacts["semantic_reduction"] = "already_minimal"
                elif candidates:
                    reduced_path = root / "semantic_reduction_candidate.yaml"
                    reduced_path.write_text(yaml.safe_dump(candidates[0], sort_keys=False), encoding="utf-8")
                    artifacts["semantic_reduction_candidate"] = str(reduced_path)
                    artifacts["semantic_reduction_replay"] = "required"
                else:
                    artifacts["semantic_reduction_candidate"] = "no_reduction_candidate"
            else:
                artifacts["minimized_case"] = "missing_case_id"
        else:
            artifacts["minimized_case"] = "missing_generated_cases"
        plan_id = representative.get("plan_id")
        result = next((item for item in self._store.read_results_jsonl() if item.plan_id == plan_id), None)
        input_ref = next((item for item in result.artifacts if item.name == "inputs"), None) if result else None
        input_path: Path | None = None
        if input_ref and input_ref.path.exists():
            input_path = root / "inputs.json"
            shutil.copy2(input_ref.path, input_path)
            artifacts["inputs"] = str(input_path)
        else:
            artifacts["inputs"] = "missing_inputs"
        if case is not None and input_path is not None:
            replay_case = self._with_fixed_inputs(case, input_path)
            path = root / "minimized_case.yaml"
            path.write_text(yaml.safe_dump(replay_case, sort_keys=False), encoding="utf-8")
            artifacts["minimized_case"] = str(path)
            script = root / "repro.sh"
            script.write_text(self._standalone_rerun_script(manifest, path, root), encoding="utf-8")
            script.chmod(0o755)
            artifacts["repro_script"] = str(script)
        else:
            artifacts["minimized_case"] = artifacts.get("minimized_case", "missing_case_or_inputs")
            artifacts["repro_script"] = "missing_case_or_inputs"
        return artifacts

    def _read_postprocess(self) -> dict[str, Any]:
        path = self.output_root / "postprocess.json"
        if not path.exists():
            return {"failure_clusters": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def _cluster_payload(self, manifest: RunManifest, cluster: dict[str, Any], index: int, minimize: bool = False) -> dict[str, Any]:
        source_cases = list(cluster.get("cases", []))
        cases = source_cases[:1] if minimize and source_cases else source_cases
        plan_ids = [str(case["plan_id"]) for case in cases if case.get("plan_id")]
        shell_commands = [self._rerun_command(manifest, plan_id, index) for plan_id in plan_ids]
        reduced_fuzz_cases = self._reduced_fuzz_cases(cases)
        return {
            "cluster_id": f"cluster-{index}",
            "signature": cluster.get("signature"),
            "count": len(cases),
            "source_count": len(source_cases),
            "minimized": minimize,
            "statuses": cluster.get("statuses", {}),
            "task": cluster.get("task"),
            "backend": cluster.get("backend"),
            "expected_invalid": cluster.get("expected_invalid", False),
            "error": cluster.get("error"),
            "compare_detail": cluster.get("compare_detail"),
            "cases": cases,
            "reduced_fuzz_cases": reduced_fuzz_cases,
            "rerun_commands": shell_commands,
            "rerun_script": self._rerun_script(shell_commands),
        }

    def _reduced_fuzz_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reduced: list[dict[str, Any]] = []
        for case in cases:
            metrics = case.get("metrics") or {}
            fuzz_case = case.get("fuzz_case") or metrics.get("fuzz_case")
            if not (fuzz_case or "generation_seed" in metrics or "source_case_id" in metrics or "generation_index" in metrics):
                continue
            reduced.append(
                {
                    "plan_id": case.get("plan_id"),
                    "case_id": case.get("case_id"),
                    "case_name": case.get("case_name"),
                    "task": case.get("task"),
                    "backend": case.get("backend"),
                    "status": case.get("status"),
                    "source_case_id": metrics.get("source_case_id"),
                    "generation_index": metrics.get("generation_index"),
                    "generation_seed": metrics.get("generation_seed"),
                    "invalid_index": metrics.get("invalid_index"),
                    "failure_kind": case.get("failure_kind"),
                    "failure_stage": case.get("failure_stage"),
                    "reduction": {
                        "strategy": "metadata_only",
                        "preserves": [
                            "source_case_id",
                            "generation_seed",
                            "generation_index",
                            "invalid_index",
                            "failure_kind",
                            "failure_stage",
                        ],
                    },
                }
            )
        return reduced

    def _with_fixed_inputs(self, case: dict[str, Any], input_path: Path) -> dict[str, Any]:
        replay_case = dict(case)
        generation = dict(replay_case.get("generation") or {})
        generation["count"] = 1
        generation_metadata = dict(generation.get("metadata") or {})
        generation_metadata["input_snapshot_path"] = str(input_path)
        generation["metadata"] = generation_metadata
        replay_case["generation"] = generation
        replay_case["generator"] = "dump_replay"
        metadata = dict(replay_case.get("metadata") or {})
        metadata["repro"] = {
            "kind": "fixed_input_replay",
            "source_case_id": case.get("id"),
            "input_snapshot": str(input_path),
        }
        replay_case["metadata"] = metadata
        return replay_case

    def _standalone_rerun_script(self, manifest: RunManifest, case_path: Path, root: Path) -> str:
        parts = ["cx", "run", "--case", str(case_path), "--nodes", str(manifest.node_path)]
        for task in manifest.tasks:
            parts.extend(["--task", task.value])
        parts.extend(["--scheduler", manifest.scheduler.value])
        if manifest.ray_address:
            parts.extend(["--ray-address", manifest.ray_address])
        for plugin in manifest.plugin_paths:
            parts.extend(["--plugin", str(plugin)])
        parts.extend(["--output", str(root / "rerun-output")])
        command = " ".join(shlex.quote(part) for part in parts)
        return self._rerun_script([command])

    def _rerun_script(self, commands: list[str]) -> str:
        if not commands:
            return "#!/usr/bin/env bash\nset -euo pipefail\n"
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        lines.extend(commands)
        return "\n".join(lines) + "\n"

    def _rerun_command(self, manifest: RunManifest, plan_id: str, cluster_index: int) -> str:
        parts = [
            "cx",
            "run",
            "--case",
            str(manifest.case_path),
            "--nodes",
            str(manifest.node_path),
        ]
        for task in manifest.tasks:
            parts.extend(["--task", task.value])
        parts.extend(["--scheduler", manifest.scheduler.value])
        if manifest.ray_address:
            parts.extend(["--ray-address", manifest.ray_address])
        for plugin in manifest.plugin_paths:
            parts.extend(["--plugin", str(plugin)])
        parts.extend(["--plan-id", plan_id])
        parts.extend(["--output", str(manifest.output_root / "repro" / f"cluster-{cluster_index}" / _slug(plan_id))])
        return " ".join(shlex.quote(part) for part in parts)


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)

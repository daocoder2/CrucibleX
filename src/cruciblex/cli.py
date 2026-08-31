from __future__ import annotations

import csv
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

from cruciblex import __version__
from cruciblex.campaign import expand_campaign_payload, select_campaign_shard
from cruciblex.domain.enums import BackendKind, SchedulerKind, TaskKind
from cruciblex.domain.run import RunContext
from cruciblex.generation.expand import expand_cases, persist_generated_cases
from cruciblex.generation.filtering import filter_cases
from cruciblex.generation.loader import load_cases, load_job_from_context
from cruciblex.importers import (
    import_backend_case,
    import_dump_case,
    import_msprof_summary,
    import_profile_case,
    load_dump_inputs,
    write_imported_backend,
    write_imported_dump,
    write_imported_profile,
)
from cruciblex.plugins import load_builtin_plugins, load_plugins
from cruciblex.report import (
    MarkdownReportWriter,
    ReproBundleWriter,
    ResultPostProcessor,
    evaluate_coverage_policy,
    evaluate_performance_gate,
    load_gate_policy,
    summarize,
    summarize_coverage,
    write_performance_gate,
)
from cruciblex.report.standalone_reduction import reduce_case_file
from cruciblex.runtime.compatibility import evaluate_runtime_compatibility
from cruciblex.runtime.discovery import discover_runtime_resources, write_discovery_files
from cruciblex.runtime.logging import bind_event, configure_run_logging, get_logger
from cruciblex.runtime.planner import ExecutionPlanner
from cruciblex.runtime.resume import ResumeState
from cruciblex.runtime.scheduler import LocalScheduler, RayScheduler, ray_init_kwargs
from cruciblex.storage.results import ResultStore

app = typer.Typer(add_completion=False, help="CrucibleX command line interface.")
console = Console()
logger = get_logger("cli")


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(__version__)


@app.command()
def doctor(
    ray_address: Annotated[str | None, typer.Option("--ray-address", envvar="RAY_ADDRESS")] = None,
) -> None:
    """Print Ray and project diagnostics."""
    try:
        import ray
    except ImportError:
        ray_status = "unavailable"
        ray_nodes = []
    else:
        if not ray.is_initialized() and ray_address:
            ray.init(**ray_init_kwargs(ray_address))
        ray_status = "initialized" if ray.is_initialized() else "available"
        if ray.is_initialized():
            from cruciblex.runtime.scheduler.placement import discover_ray_cluster

            ray_nodes = discover_ray_cluster(ray).alive_nodes
            worker_packages = _probe_ray_worker_packages(ray, ray_nodes)
        else:
            ray_nodes = []
            worker_packages = []

    package_info = _local_package_info()
    console.print("CrucibleX is initialized.")
    console.print(f"Project root: {Path.cwd()}")
    console.print(
        f"Driver package: version={package_info['version']} "
        f"pipeline_sha256={package_info['pipeline_sha256']} path={package_info['pipeline_path']}"
    )
    if ray_address is not None:
        console.print(f"Ray address: {ray_address}")
    console.print(f"Ray: {ray_status}")
    if ray_status != "unavailable":
        console.print(f"Ray nodes: {len(ray_nodes)}")
        for node in ray_nodes:
            resources = ",".join(sorted(node.resources))
            console.print(f"- {node.address} {node.hostname} resources={resources}")
        if worker_packages:
            console.print("Worker packages:")
            for package in worker_packages:
                console.print(
                    f"- {package['address']} {package['hostname']} "
                    f"version={package['version']} pipeline_sha256={package['pipeline_sha256']} "
                    f"path={package['pipeline_path']} {_worker_runtime_summary(package)}"
                )


@app.command()
def discover(
    ray_address: Annotated[str | None, typer.Option("--ray-address", envvar="RAY_ADDRESS")] = None,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("cx_output/discovery"),
) -> None:
    """Discover Ray resources and write driver artifacts."""
    output = output.resolve()
    discovery = discover_runtime_resources(ray_address=ray_address)
    artifacts = write_discovery_files(output / "driver", discovery)
    source = discovery.get("source", {})
    console.print(f"Discovery: {artifacts['snapshot']}")
    console.print(f"Nodes: {source.get('node_count', 0)} alive={source.get('alive_node_count', 0)}")
    backend_counts = discovery.get("backend_counts", {})
    if backend_counts:
        console.print(f"Backends: {', '.join(f'{name}={count}' for name, count in backend_counts.items())}")


@app.command()
def coverage(
    case: Annotated[Path, typer.Option("--case", "-c")] = Path("examples/cases/torch.abs.yaml"),
    ray_address: Annotated[str | None, typer.Option("--ray-address", envvar="RAY_ADDRESS")] = None,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("cx_output/coverage"),
    run: Annotated[bool, typer.Option("--run")] = False,
) -> None:
    """Build a backend coverage campaign from discovered resources."""
    output = output.resolve()
    discovery = discover_runtime_resources(ray_address=ray_address)
    write_discovery_files(output / "driver", discovery)
    node_templates = discovery.get("node_templates", [])
    if not node_templates:
        raise typer.BadParameter("resource discovery did not find any backend-capable nodes")
    grouped = _group_templates_by_backend(node_templates)
    coverage_root = output / "driver" / "coverage" / case.stem
    coverage_root.mkdir(parents=True, exist_ok=True)
    campaign_runs: list[dict[str, Any]] = []
    for backend, templates in grouped.items():
        node_path = coverage_root / f"{backend}_nodes.yaml"
        node_path.write_text(yaml.safe_dump({"nodes": templates}, sort_keys=False), encoding="utf-8")
        campaign_runs.append(
            {
                "name": f"{case.stem}-{backend}",
                "case": str(case),
                "nodes": str(node_path),
                "task": _coverage_tasks(backend),
                "scheduler": SchedulerKind.RAY.value,
                "ray_address": ray_address,
                "output": str(coverage_root / "campaign-output" / backend),
            }
        )
    campaign_path = coverage_root / "coverage_campaign.yaml"
    campaign_path.write_text(yaml.safe_dump({"runs": campaign_runs}, sort_keys=False), encoding="utf-8")
    console.print(f"Coverage campaign: {campaign_path}")
    console.print(f"Coverage backends: {', '.join(grouped)}")
    if run:
        campaign(campaign_file=campaign_path, output=coverage_root / "campaign-output")


@app.command()
def driver_clean(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("cx_output"),
    keep: Annotated[int, typer.Option("--keep")] = 3,
) -> None:
    """Prune old driver-side coverage batches."""
    output = output.resolve()
    coverage_root = output / "driver" / "coverage"
    if not coverage_root.exists():
        console.print("Driver coverage: nothing to prune")
        return
    batches = [path for path in coverage_root.iterdir() if path.is_dir()]
    batches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    removed: list[Path] = []
    for path in batches[keep:]:
        shutil.rmtree(path)
        removed.append(path)
    console.print(f"Driver coverage pruned: {len(removed)}")
    for path in removed:
        console.print(f"- {path}")


def _group_templates_by_backend(node_templates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for template in node_templates:
        devices = template.get("devices") or []
        if not devices:
            continue
        backend = str(devices[0].get("backend") or "cpu")
        grouped.setdefault(backend, []).append(template)
    return grouped


def _coverage_tasks(backend: str) -> list[str]:
    if backend == BackendKind.CPU.value:
        return [TaskKind.ACCURACY.value, TaskKind.RUN.value]
    if backend == BackendKind.GPU.value:
        return [
            TaskKind.ACCURACY.value,
            TaskKind.RUN.value,
            TaskKind.PERFORMANCE_DEVICE.value,
            TaskKind.PERFORMANCE_DEVICE_PTA.value,
            TaskKind.PERFORMANCE_E2E.value,
            TaskKind.MEMORY_DEVICE.value,
        ]
    if backend in {BackendKind.NPU.value, BackendKind.ACLNN.value, BackendKind.DCU.value}:
        return [
            TaskKind.ACCURACY.value,
            TaskKind.RUN.value,
            TaskKind.PERFORMANCE_DEVICE.value,
            TaskKind.MEMORY_DEVICE.value,
        ]
    return [TaskKind.ACCURACY.value, TaskKind.RUN.value]


def _local_package_info() -> dict[str, str]:
    import cruciblex
    import cruciblex.runtime.pipeline as pipeline_module

    pipeline_path = Path(pipeline_module.__file__ or "")
    return {
        "version": cruciblex.__version__,
        "pipeline_path": str(pipeline_path),
        "pipeline_sha256": _sha256_file(pipeline_path),
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "missing"


def _probe_ray_worker_packages(ray: Any, ray_nodes: list[Any]) -> list[dict[str, Any]]:
    if not ray_nodes:
        return []

    @ray.remote(num_cpus=0)
    def package_probe() -> dict[str, Any]:
        import hashlib
        import importlib.metadata
        import importlib.util
        import os
        import platform
        import sys
        from pathlib import Path

        def module_probe(name: str) -> dict[str, Any]:
            spec = importlib.util.find_spec(name)
            if spec is None:
                return {"available": False}
            return {"available": True, "origin": spec.origin}

        def torch_probe() -> dict[str, Any]:
            spec = importlib.util.find_spec("torch")
            if spec is None:
                return {"available": False}
            try:
                import torch
            except Exception as exc:  # noqa: BLE001
                return {"available": True, "import_error": f"{exc.__class__.__name__}: {exc}"}
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
            return probe

        pipeline_spec = importlib.util.find_spec("cruciblex.runtime.pipeline")
        pipeline_path = Path(pipeline_spec.origin) if pipeline_spec is not None and pipeline_spec.origin else Path("")
        pipeline_sha256 = (
            hashlib.sha256(pipeline_path.read_bytes()).hexdigest()[:16]
            if pipeline_path.exists()
            else "missing"
        )
        try:
            version = importlib.metadata.version("cruciblex")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        env_keys = [
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "ASCEND_VISIBLE_DEVICES",
            "ASCEND_DEVICE_ID",
            "ASCEND_HOME_PATH",
            "LD_LIBRARY_PATH",
            "CX_DEVICE_INDEX_MODE",
        ]
        runtime_probe = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
            "env": {key: value for key in env_keys if (value := os.environ.get(key)) is not None},
            "packages": {
                "torch": torch_probe(),
                "torch_npu": module_probe("torch_npu"),
            },
        }
        return {
            "version": version,
            "pipeline_path": str(pipeline_path),
            "pipeline_sha256": pipeline_sha256,
            "runtime_probe": runtime_probe,
        }

    probes: list[tuple[Any, Any]] = []
    for node in ray_nodes:
        probes.append((node, package_probe.options(**_worker_probe_options(node)).remote()))

    rows: list[dict[str, Any]] = []
    for node, ref in probes:
        try:
            result = ray.get(ref)
            rows.append(
                {
                    "address": str(getattr(node, "address", "")),
                    "hostname": str(getattr(node, "hostname", "")),
                    "version": str(result.get("version", "unknown")),
                    "pipeline_path": str(result.get("pipeline_path", "unknown")),
                    "pipeline_sha256": str(result.get("pipeline_sha256", "unknown")),
                    "runtime_probe": result.get("runtime_probe", {}),
                }
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics should not fail doctor
            rows.append(
                {
                    "address": str(getattr(node, "address", "")),
                    "hostname": str(getattr(node, "hostname", "")),
                    "version": "error",
                    "pipeline_path": f"{exc.__class__.__name__}: {exc}",
                    "pipeline_sha256": "error",
                    "runtime_probe": {"error": f"{exc.__class__.__name__}: {exc}"},
                }
            )
    return rows


def _worker_probe_options(node: Any) -> dict[str, Any]:
    options: dict[str, Any] = {}
    node_resource_key = getattr(node, "node_resource_key", None)
    node_resources = getattr(node, "resources", {}) if hasattr(node, "resources") else {}
    if isinstance(node_resources, dict) and node_resources.get("GPU", 0.0) > 0.0:
        options["num_gpus"] = 1
    if node_resource_key is not None:
        options["resources"] = {node_resource_key: 0.001}
    return options


def _worker_runtime_summary(package: dict[str, Any]) -> str:
    runtime_probe = package.get("runtime_probe")
    if not isinstance(runtime_probe, dict):
        return "runtime_probe=missing"
    if "error" in runtime_probe:
        return f"runtime_probe_error={runtime_probe['error']}"
    packages = runtime_probe.get("packages", {}) if isinstance(runtime_probe.get("packages"), dict) else {}
    torch_probe = packages.get("torch", {}) if isinstance(packages.get("torch"), dict) else {}
    torch_npu_probe = packages.get("torch_npu", {}) if isinstance(packages.get("torch_npu"), dict) else {}
    env = runtime_probe.get("env", {}) if isinstance(runtime_probe.get("env"), dict) else {}
    visible_env = ",".join(f"{key}={env[key]}" for key in sorted(env) if key.endswith("VISIBLE_DEVICES") or key == "ASCEND_DEVICE_ID")
    return (
        f"torch={torch_probe.get('version', 'unavailable')} "
        f"torch_cuda={torch_probe.get('cuda_version')} "
        f"cuda_available={torch_probe.get('cuda_available')} "
        f"cuda_device_count={torch_probe.get('cuda_device_count')} "
        f"torch_npu_available={torch_npu_probe.get('available')} "
        f"visible_devices={visible_env or 'unset'}"
    )



def _import_backend_command(
    *,
    source: Path,
    output: Path,
    source_format: str,
    case_id: int | None,
    executor: str | None,
    reference_executor: str | None,
) -> None:
    output = output.resolve()
    imported = import_backend_case(
        source,
        source_format=source_format,
        case_id=case_id,
        executor=executor,
        reference_executor=reference_executor,
    )
    output_path = write_imported_backend(output, imported)
    provenance = imported.metadata.get("provenance", {})
    console.print(f"Imported {source_format.upper()} case: {output_path}")
    console.print(f"Operator: {imported.operator.name}")
    console.print(f"Warnings: {len(provenance.get('warnings') or [])}")


@app.command("import-atb")
def import_atb(
    source: Annotated[Path, typer.Option("--source", "-s")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("cx_output/imported-atb/cases.yaml"),
    case_id: Annotated[int | None, typer.Option("--case-id")] = None,
    executor: Annotated[str | None, typer.Option("--executor")] = None,
    reference_executor: Annotated[str | None, typer.Option("--reference-executor")] = None,
) -> None:
    """Import an ATB config into CrucibleX case YAML with backend metadata."""
    _import_backend_command(
        source=source,
        output=output,
        source_format="atb",
        case_id=case_id,
        executor=executor,
        reference_executor=reference_executor,
    )


@app.command("import-temu")
def import_temu(
    source: Annotated[Path, typer.Option("--source", "-s")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("cx_output/imported-temu/cases.yaml"),
    case_id: Annotated[int | None, typer.Option("--case-id")] = None,
    executor: Annotated[str | None, typer.Option("--executor")] = None,
    reference_executor: Annotated[str | None, typer.Option("--reference-executor")] = None,
) -> None:
    """Import a Temu config into CrucibleX case YAML with backend metadata."""
    _import_backend_command(
        source=source,
        output=output,
        source_format="temu",
        case_id=case_id,
        executor=executor,
        reference_executor=reference_executor,
    )

@app.command("import-dump")
def import_dump(
    source: Annotated[Path, typer.Option("--source", "-s")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("cx_output/imported-dump/cases.yaml"),
    case_id: Annotated[int | None, typer.Option("--case-id")] = None,
    executor: Annotated[str | None, typer.Option("--executor")] = None,
    reference_executor: Annotated[str | None, typer.Option("--reference-executor")] = None,
) -> None:
    """Import a dump with concrete inputs into replayable CrucibleX case YAML."""
    output = output.resolve()
    imported, snapshot_path = import_dump_case(
        source,
        output=output,
        case_id=case_id,
        executor=executor,
        reference_executor=reference_executor,
    )
    written = write_imported_dump(output, imported, snapshot_path, load_dump_inputs(source))
    console.print(f"Imported dump case: {written['case']}")
    console.print(f"Input snapshot: {written['inputs']}")
    console.print(f"Operator: {imported.operator.name}")


@app.command("import-msprof")
def import_msprof(
    source: Annotated[Path, typer.Option("--source", "-s")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("cx_output/msprof-summary.json"),
) -> None:
    """Parse CANN msprof CSV exports into a normalized summary artifact."""
    output = output.resolve()
    summary = import_msprof_summary(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"msprof summary: {output}")
    console.print(f"Status: {summary['status']} devices={summary['device_count']}")
    console.print(f"Operators: {len(summary['top_operators'])} tasks={len(summary['task_summary'])}")
    if summary["warnings"]:
        console.print(f"Warnings: {len(summary['warnings'])}")


@app.command("performance-gate")
def performance_gate(
    baseline: Annotated[Path, typer.Option("--baseline")],
    candidate: Annotated[Path, typer.Option("--candidate")],
    policy: Annotated[Path, typer.Option("--policy")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("cx_output/performance_gate.json"),
) -> None:
    """Compare performance, memory, and profiler metrics for CI gating."""
    result = evaluate_performance_gate(baseline, candidate, load_gate_policy(policy))
    output = write_performance_gate(result, output.resolve())
    console.print(f"Performance gate: {result['status']}")
    console.print(f"Gate result: {output}")
    if result["regressions"]:
        raise typer.Exit(code=1)
    if result["insufficient_data"]:
        raise typer.Exit(code=1)


@app.command("import-profile")
def import_profile(
    source: Annotated[Path, typer.Option("--source", "-s")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("cx_output/imported-profile/cases.yaml"),
    case_id: Annotated[int | None, typer.Option("--case-id")] = None,
    executor: Annotated[str | None, typer.Option("--executor")] = None,
    reference_executor: Annotated[str | None, typer.Option("--reference-executor")] = None,
) -> None:
    """Import profile-derived shape and dtype samples into CrucibleX case YAML."""
    output = output.resolve()
    imported = import_profile_case(
        source,
        case_id=case_id,
        executor=executor,
        reference_executor=reference_executor,
    )
    output_path = write_imported_profile(output, imported)
    provenance = imported.metadata.get("provenance", {})
    console.print(f"Imported profile case: {output_path}")
    console.print(f"Operator: {imported.operator.name}")
    console.print(f"Samples: {provenance.get('sample_count', 0)}")


@app.command()
def generate(
    case: Annotated[Path, typer.Option("--case", "-c")] = Path("examples/cases/torch.abs.generated.yaml"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("cx_output/generated"),
    include: Annotated[list[str] | None, typer.Option("--include", help="dimension=value; repeatable")] = None,
    exclude: Annotated[list[str] | None, typer.Option("--exclude", help="dimension=value; repeatable")] = None,
) -> None:
    """Expand a case file and persist the generated set."""
    output = output.resolve()
    generated = expand_cases(load_cases(case))
    generated = filter_cases(generated, include=_parse_case_selectors(include), exclude=_parse_case_selectors(exclude))
    generated_json = persist_generated_cases(generated, output)
    generated_yaml = output / "generated_cases.yaml"
    generated_yaml.write_text(
        yaml.safe_dump({"cases": [case.model_dump(mode="json") for case in generated]}, sort_keys=False),
        encoding="utf-8",
    )
    console.print(f"Generated cases: {generated_json}")
    console.print(f"Generated cases YAML: {generated_yaml}")


@app.command()
def report(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("cx_output"),
) -> None:
    """Render a markdown report from a completed run."""
    output = output.resolve()
    campaign_summary_path = output / "campaign_summary.json"
    if campaign_summary_path.exists():
        report_path = _write_campaign_report(output, campaign_summary_path)
    else:
        report_path = MarkdownReportWriter(output).write()
    results_csv_path = _write_results_csv(output)
    console.print(f"Report: {report_path}")
    if results_csv_path is not None:
        console.print(f"Results CSV: {results_csv_path}")


@app.command("reduce")
def reduce(
    case: Annotated[Path, typer.Option("--case", "-c")],
    replay_command: Annotated[str, typer.Option("--replay-command")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    replay_exit_code: Annotated[int, typer.Option("--replay-exit-code")] = 1,
) -> None:
    """Reduce one case while preserving a replay predicate."""
    summary = reduce_case_file(case.resolve(), output.resolve(), replay_command, replay_exit_code)
    console.print(f"Reduced case: {summary['reduced_case']}")
    console.print(f"Attempts: {summary['attempt_count']} accepted={summary['accepted_count']}")


@app.command("coverage-report")
def coverage_report(
    output: Annotated[Path, typer.Option("--output", "-o")],
    inputs: Annotated[list[Path] | None, typer.Option("--input")] = None,
    policy: Annotated[Path | None, typer.Option("--policy")] = None,
    report: Annotated[Path, typer.Option("--report")] = Path("coverage.json"),
) -> None:
    """Summarize result coverage and enforce an optional policy."""
    output = output.resolve()
    result_paths = [output, *(path.resolve() for path in (inputs or []))]
    rows = [
        row.model_dump(mode="json") if hasattr(row, "model_dump") else row
        for result_path in result_paths
        for row in ResultStore(result_path).read_results_jsonl()
    ]
    policy_data = yaml.safe_load(policy.read_text(encoding="utf-8")) if policy else {}
    result = evaluate_coverage_policy(summarize_coverage(rows), policy_data or {})
    report_path = output / report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"Coverage report: {report_path}")
    console.print(f"Coverage status: {result['status']}")
    if result["missing"]:
        raise typer.Exit(code=1)


@app.command("campaign-coverage")
def campaign_coverage(
    output: Annotated[Path, typer.Option("--output", "-o")],
    policy: Annotated[Path | None, typer.Option("--policy")] = None,
    report: Annotated[Path, typer.Option("--report")] = Path("campaign_coverage.json"),
) -> None:
    """Aggregate coverage from every output root in a campaign summary."""
    output = output.resolve()
    summary_path = output / "campaign_summary.json"
    if not summary_path.exists():
        raise typer.BadParameter(f"campaign summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result_paths = [Path(row["output"]) for row in summary.get("runs", []) if row.get("output")]
    rows = [
        row.model_dump(mode="json") if hasattr(row, "model_dump") else row
        for result_path in result_paths
        for row in ResultStore(result_path).read_results_jsonl()
    ]
    policy_data = yaml.safe_load(policy.read_text(encoding="utf-8")) if policy else {}
    result = evaluate_coverage_policy(summarize_coverage(rows), policy_data or {})
    result["campaign_outputs"] = [str(path) for path in result_paths]
    report_path = output / report
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"Campaign coverage report: {report_path}")
    console.print(f"Coverage status: {result['status']}")
    if result["missing"]:
        raise typer.Exit(code=1)


def _parse_case_selectors(values: list[str] | None) -> dict[str, set[str]]:
    selectors: dict[str, set[str]] = {}
    for value in values or []:
        if "=" not in value:
            raise typer.BadParameter(f"selector must use dimension=value: {value}")
        dimension, selected = value.split("=", 1)
        if dimension not in {"operator", "backend", "task", "dtype", "tag"} or not selected:
            raise typer.BadParameter(f"unsupported selector: {value}")
        selectors.setdefault(dimension, set()).add(selected)
    return selectors


def _write_results_csv(output: Path) -> Path | None:
    store = ResultStore(output)
    campaign_summary_path = output / "campaign_summary.json"
    if campaign_summary_path.exists():
        summary = json.loads(campaign_summary_path.read_text(encoding="utf-8"))
        rows = [
            {
                "name": row.get("name", ""),
                "output": row.get("output", ""),
                "total": str(row.get("total", 0)),
                "passed": str(row.get("passed", 0)),
                "failed": str(row.get("failed", 0)),
                "fuzz_cases": str(row.get("fuzz_cases", 0)),
                "failure_clusters": str(row.get("failure_clusters", 0)),
            }
            for row in summary.get("runs", [])
        ]
        if not rows:
            return None
        path = output / "results.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["name", "output", "total", "passed", "failed", "fuzz_cases", "failure_clusters"],
            )
            writer.writeheader()
            writer.writerows(rows)
        return path
    results_path = output / "results.jsonl"
    if not results_path.exists():
        return None
    results = store.read_results_jsonl()
    if not results:
        return None
    return store.write_results_csv(results)


def _write_campaign_report(output: Path, summary_path: Path) -> Path:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lines = ["# CrucibleX Campaign Report", "", "## Summary"]
    totals = summary.get("totals", {})
    lines.extend(
        [
            f"- runs: {totals.get('runs', len(summary.get('runs', [])))}",
            f"- total: {totals.get('total', 0)}",
            f"- passed: {totals.get('passed', 0)}",
            f"- failed: {totals.get('failed', 0)}",
            f"- fuzz_cases: {totals.get('fuzz_cases', 0)}",
            f"- failure_clusters: {totals.get('failure_clusters', 0)}",
            "",
            "## Status By Task And Backend",
        ]
    )
    status_by_task_backend = summary.get("status_by_task_backend", {})
    if not status_by_task_backend:
        lines.append("- none")
    else:
        for task, by_backend in status_by_task_backend.items():
            lines.append(f"- {task}")
            for backend, counts in by_backend.items():
                rendered_counts = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
                lines.append(f"  {backend}: {rendered_counts}")
    lines.extend(["", "## Runs"])
    for row in summary.get("runs", []):
        lines.extend(
            [
                f"- {row.get('name', '')}",
                f"  output: {row.get('output', '')}",
                f"  total: {row.get('total', 0)}",
                f"  passed: {row.get('passed', 0)}",
                f"  failed: {row.get('failed', 0)}",
                f"  fuzz_cases: {row.get('fuzz_cases', 0)}",
                f"  failure_clusters: {row.get('failure_clusters', 0)}",
            ]
        )
    report_path = output / "campaign_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


@app.command()
def repro(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("cx_output"),
    cluster_id: Annotated[str | None, typer.Option("--cluster-id")] = None,
    script: Annotated[bool, typer.Option("--script")] = False,
    minimize: Annotated[bool, typer.Option("--minimize")] = False,
    replay_command: Annotated[str | None, typer.Option("--replay-command")] = None,
    replay_exit_code: Annotated[int, typer.Option("--replay-exit-code")] = 1,
) -> None:
    """Write repro bundles for clustered failures."""
    if replay_command and not minimize:
        raise typer.BadParameter("--replay-command requires --minimize")
    output = output.resolve()
    replay_candidate = output / "repro" / "replay_candidate.yaml"

    def replay_predicate(candidate: dict[str, Any]) -> bool:
        replay_candidate.parent.mkdir(parents=True, exist_ok=True)
        replay_candidate.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
        command = replay_command.replace("{case}", shlex.quote(str(replay_candidate)))
        completed = subprocess.run(command, shell=True, check=False, capture_output=True, text=True)
        return completed.returncode == replay_exit_code

    bundle_path = ReproBundleWriter(output).write(
        cluster_id=cluster_id,
        minimize=minimize,
        replay_predicate=replay_predicate if replay_command else None,
    )
    console.print(f"Repro bundle: {bundle_path}")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    for cluster in bundle.get("clusters", []):
        console.print(f"{cluster.get('cluster_id')}: {cluster.get('count', 0)} case(s)")
        for command in cluster.get("rerun_commands", []):
            console.print(command)
        if script:
            script_path = output / f"{cluster.get('cluster_id')}.rerun.sh"
            script_path.write_text(cluster.get("rerun_script", ""), encoding="utf-8")
            script_path.chmod(0o755)
            console.print(f"Rerun script: {script_path}")


@app.command()
def onboard(
    facts: Annotated[Path, typer.Option("--facts", "-f")] = Path("examples/operator-onboarding/operator_facts.yaml"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("cx_output/onboarding-scaffold"),
) -> None:
    """Generate operator onboarding scaffold files from facts YAML."""
    output = output.resolve()
    payload = yaml.safe_load(facts.read_text(encoding="utf-8")) or {}
    scaffold = _onboarding_scaffold(payload, output)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, content in scaffold.items():
        path = output / name
        path.write_text(content, encoding="utf-8")
        written.append(path)
    for path in written:
        if path.name in {"validate.sh", "hardware_validate.sh"}:
            path.chmod(0o755)
        console.print(f"Onboarding: {path}")


def _onboarding_scaffold(payload: dict[str, Any], output: Path) -> dict[str, str]:
    operator = payload.get("operator") or {}
    oracle = payload.get("oracle") or {}
    params = payload.get("parameters") or []
    name = str(operator.get("name") or operator.get("api") or "custom.operator")
    api = str(operator.get("api") or name)
    api_type = str(operator.get("api_type") or "function")
    executor = str(operator.get("executor") or "function")
    supported_backends = _supported_backends(operator)
    case_doc = _case_doc(payload, name, api, api_type, executor, params, oracle, generation={"count": 1, "constraints": []})
    fuzz_case_doc = _case_doc(
        payload,
        name,
        api,
        api_type,
        executor,
        params,
        oracle,
        generation=_fuzz_generation(payload),
        fuzz=True,
    )
    nodes_doc = {
        "nodes": [
            {
                "name": "local-cpu",
                "host": "127.0.0.1",
                "role": "candidate",
                "allowed_tasks": ["accuracy", "run", "fuzz"],
                "devices": [{"id": 0, "backend": "cpu"}],
            }
        ]
    }
    campaign_doc = _local_campaign_doc(name, output)
    hardware_nodes_doc = _hardware_nodes_doc(supported_backends)
    hardware_campaign_doc = _hardware_campaign_doc(name, output, supported_backends)
    return {
        "case.yaml": yaml.safe_dump(case_doc, sort_keys=False),
        "fuzz_case.yaml": yaml.safe_dump(fuzz_case_doc, sort_keys=False),
        "nodes.yaml": yaml.safe_dump(nodes_doc, sort_keys=False),
        "hardware_nodes.yaml": yaml.safe_dump(hardware_nodes_doc, sort_keys=False),
        "campaign.yaml": yaml.safe_dump(campaign_doc, sort_keys=False),
        "hardware_campaign.yaml": yaml.safe_dump(hardware_campaign_doc, sort_keys=False),
        "executor_plugin.py": _executor_template(name),
        "validate.sh": _onboarding_validate_script(output),
        "hardware_validate.sh": _hardware_validate_script(output),
        "README.md": _onboarding_readme(name, output),
    }


def _supported_backends(operator: dict[str, Any]) -> list[str]:
    backends = [str(backend) for backend in operator.get("supported_backends", ["cpu"])]
    return backends or ["cpu"]


def _local_campaign_doc(name: str, output: Path) -> dict[str, Any]:
    return {
        "runs": [
            {
                "name": f"{_slug(name)}-run",
                "case": str(output / "case.yaml"),
                "nodes": str(output / "nodes.yaml"),
                "task": "run",
                "scheduler": "local",
                "output": str(output / "run-smoke"),
            },
            {
                "name": f"{_slug(name)}-fuzz",
                "case": str(output / "fuzz_case.yaml"),
                "nodes": str(output / "nodes.yaml"),
                "task": "fuzz",
                "scheduler": "local",
                "output": str(output / "fuzz-smoke"),
            },
        ]
    }


def _hardware_nodes_doc(backends: list[str]) -> dict[str, Any]:
    hardware_backends = [backend for backend in backends if backend != "cpu"]
    return {
        "nodes": [
            {
                "name": f"ray-{backend}",
                "host": "127.0.0.1",
                "role": "candidate",
                "allowed_tasks": ["accuracy", "run", "fuzz"],
                "devices": [{"id": 0, "backend": backend}],
            }
            for backend in hardware_backends
        ]
        or [
            {
                "name": "ray-cpu",
                "host": "127.0.0.1",
                "role": "candidate",
                "allowed_tasks": ["accuracy", "run", "fuzz"],
                "devices": [{"id": 0, "backend": "cpu"}],
            }
        ]
    }


def _hardware_campaign_doc(name: str, output: Path, backends: list[str]) -> dict[str, Any]:
    scheduler = "ray" if any(backend != "cpu" for backend in backends) else "local"
    return {
        "runs": [
            {
                "name": f"{_slug(name)}-hardware-run",
                "case": str(output / "case.yaml"),
                "nodes": str(output / "hardware_nodes.yaml"),
                "task": "run",
                "scheduler": scheduler,
                "output": str(output / "hardware-run"),
            },
            {
                "name": f"{_slug(name)}-hardware-fuzz",
                "case": str(output / "fuzz_case.yaml"),
                "nodes": str(output / "hardware_nodes.yaml"),
                "task": "fuzz",
                "scheduler": scheduler,
                "output": str(output / "hardware-fuzz"),
            },
        ]
    }


def _fuzz_generation(payload: dict[str, Any]) -> dict[str, Any]:
    fuzz = payload.get("fuzz") or {}
    constraints = list(fuzz.get("constraints", ["random_coverage"]))
    if "random_coverage" not in constraints:
        constraints.append("random_coverage")
    generation = {
        "count": int(fuzz.get("count", 3)),
        "seed": int(fuzz.get("seed", 0)),
        "constraints": constraints,
    }
    if fuzz.get("invalid_count") is not None:
        generation["invalid_count"] = int(fuzz.get("invalid_count", 0))
    if fuzz.get("max_elements") is not None:
        generation["max_elements"] = int(fuzz.get("max_elements"))
    if fuzz.get("max_bytes") is not None:
        generation["max_bytes"] = int(fuzz.get("max_bytes"))
    return generation


def _case_doc(
    payload: dict[str, Any],
    name: str,
    api: str,
    api_type: str,
    executor: str,
    params: list[dict[str, Any]],
    oracle: dict[str, Any],
    generation: dict[str, Any],
    fuzz: bool = False,
) -> dict[str, Any]:
    generation = dict(generation)
    generation_metadata = dict(generation.get("metadata") or {})
    generation_metadata.setdefault(
        "operator_facts",
        {
            "schema_version": 1,
            "parameters": {str(param.get("name", "input")): param for param in params},
        },
    )
    generation["metadata"] = generation_metadata
    case = {
        "id": int(payload.get("case_id", 1000)),
        "operator": {"name": name},
        "invocation": {"api": api, "api_type": api_type, "executor": executor},
        "oracle": {
            "comparison": oracle.get("comparison", "allclose"),
            "expected_error": (payload.get("invalid_policy") or {}).get("expected_error"),
            "tolerance": oracle.get("tolerance", {"atol": 1.0e-6, "rtol": 1.0e-6}),
        },
        "generator": "default",
        "generation": generation,
        "parameters": params,
    }
    if fuzz:
        case["parameters"] = [_fuzz_parameter_from_fact(param, payload.get("fuzz") or {}) for param in params]
    else:
        case["parameters"] = [_parameter_from_fact(param) for param in params]
    return {"cases": [case]}


def _parameter_from_fact(param: dict[str, Any]) -> dict[str, Any]:
    shape_rules = param.get("shape_rules") or {}
    value_range = param.get("value_range") or {}
    shape = {key: shape_rules[key] for key in ("dims", "dim_count", "dim_values", "max_elements") if key in shape_rules}
    if not shape:
        shape = {"dims": [1]}
    metadata = {key: param[key] for key in ("dtype_policy", "value_policy", "shape_policy", "dtype_promotion") if key in param}
    if "relationships" in param:
        metadata["shape_relationship"] = param["relationships"]
    return {
        "name": param.get("name", "input"),
        "kind": param.get("kind", "tensor"),
        "dtypes": param.get("dtypes", param.get("dtype_families", ["fp32"])),
        "shape": shape,
        "value_range": {
            "valid": value_range.get("valid", [[-1, 1]]),
            "invalid": value_range.get("invalid", []),
        },
        "metadata": metadata,
    }


def _fuzz_parameter_from_fact(param: dict[str, Any], global_fuzz: dict[str, Any]) -> dict[str, Any]:
    parameter = _parameter_from_fact(param)
    fuzz = {**global_fuzz, **(param.get("fuzz") or {})}
    metadata = {
        **dict(parameter.get("metadata") or {}),
        "random_coverage": True,
        "random_dtypes": list(fuzz.get("random_dtypes", param.get("dtype_families", ["fp32"]))),
        "random_shapes": list(fuzz.get("random_shapes", [list(param.get("shape_rules", {}).get("dims", [1]))])),
        "random_values": list(fuzz.get("random_values", param.get("value_range", {}).get("valid", [[-1, 1]]))),
    }
    parameter["metadata"] = metadata
    return parameter


def _executor_template(operator_name: str) -> str:
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "import importlib",
            "",
            "from cruciblex.runtime.executors.base import BackendExecutor, EXECUTOR_REGISTRY, ExecutionRequest",
            "",
            "",
            "class OnboardingExecutor(BackendExecutor):",
            "    def execute(self, request: ExecutionRequest) -> object:",
            "        target = _resolve_target(request.case.invocation.api)",
            "        return target(*request.inputs)",
            "",
            "",
            "def _resolve_target(api: str):",
            "    module_name, _, attr_name = api.rpartition('.')",
            "    if not module_name:",
            "        raise ValueError(f'expected dotted API path, got {api!r}')",
            "    module = importlib.import_module(module_name)",
            "    return getattr(module, attr_name)",
            "",
            "",
            f"EXECUTOR_REGISTRY.register(\"{_slug(operator_name)}_executor\")(OnboardingExecutor)",
            "",
        ]
    )


def _onboarding_validate_script(output: Path) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"uv run cx campaign --campaign {output / 'campaign.yaml'} --output {output / 'campaign-output'}",
            f"uv run cx report --output {output / 'campaign-output'}",
            "",
        ]
    )


def _hardware_validate_script(output: Path) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"uv run cx campaign --campaign {output / 'hardware_campaign.yaml'} --output {output / 'hardware-campaign-output'}",
            f"uv run cx report --output {output / 'hardware-campaign-output'}",
            "",
        ]
    )


def _onboarding_readme(operator_name: str, output: Path) -> str:
    return "\n".join(
        [
            f"# {operator_name} Onboarding",
            "",
            "Generated scaffold files:",
            "",
            "- `case.yaml`: initial CrucibleX case",
            "- `fuzz_case.yaml`: seed-driven fuzz case",
            "- `nodes.yaml`: local CPU node file",
            "- `hardware_nodes.yaml`: Ray/hardware node template from supported_backends",
            "- `campaign.yaml`: local smoke campaign",
            "- `hardware_campaign.yaml`: Ray/hardware campaign template",
            "- `executor_plugin.py`: optional custom executor skeleton",
            "- `validate.sh`: local campaign and report smoke",
            "- `hardware_validate.sh`: Ray or accelerator campaign smoke template",
            "",
            "Run the local smoke campaign. Hardware templates are generated for review but are not executed by default:",
            "",
            f"    {output / 'validate.sh'}",
            "",
        ]
    )


@app.command()
def campaign(
    campaign_file: Annotated[Path, typer.Option("--campaign", "-f")] = Path("examples/campaigns/local-fuzz.yaml"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("cx_output/campaign"),
    shard_index: Annotated[int, typer.Option("--shard-index")] = 0,
    shard_count: Annotated[int, typer.Option("--shard-count")] = 1,
) -> None:
    """Run a YAML campaign containing multiple CrucibleX runs."""
    output = output.resolve()
    payload = yaml.safe_load(campaign_file.read_text(encoding="utf-8")) or {}
    try:
        runs = expand_campaign_payload(payload)
        runs = select_campaign_shard(runs, shard_index, shard_count)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc))

    output.mkdir(parents=True, exist_ok=True)
    all_runs = expand_campaign_payload(payload)
    campaign_manifest = {
        "campaign_file": str(campaign_file.resolve()),
        "total_runs": len(all_runs),
        "selected_runs": len(runs),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "matrix_ids": [item.get("matrix_id") for item in runs if item.get("matrix_id")],
        "items": [
            {
                "name": item.get("name"),
                "matrix_id": item.get("matrix_id"),
                "output": str(Path(item.get("output") or output / _slug(str(item.get("name") or "run"))).resolve()),
                "resume_from": item.get("resume_from"),
                "retry_failed": bool(item.get("retry_failed", False)),
            }
            for item in runs
        ],
    }
    (output / "campaign_manifest.json").write_text(json.dumps(campaign_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_rows: list[dict[str, Any]] = []
    for index, item in enumerate(runs):
        if not isinstance(item, dict):
            raise typer.BadParameter("each campaign run must be a mapping")
        name = str(item.get("name") or f"run-{index}")
        run_output = Path(item.get("output") or output / _slug(name)).resolve()
        tasks = [TaskKind(value) for value in _as_list(item.get("tasks", item.get("task", ["accuracy"])))]
        scheduler_kind = SchedulerKind(item.get("scheduler", SchedulerKind.LOCAL.value))
        plugins = [Path(plugin) for plugin in _as_list(item.get("plugins", item.get("plugin", [])))]
        plan_ids = _as_list(item.get("plan_id")) if item.get("plan_id") is not None else None
        run(
            case=Path(item["case"]),
            nodes=Path(item["nodes"]),
            task=tasks,
            scheduler=scheduler_kind,
            ray_address=item.get("ray_address"),
            output=run_output,
            plugin=plugins,
            plan_id=plan_ids,
            resume_from=Path(item["resume_from"]) if item.get("resume_from") else (run_output if (run_output / "results.jsonl").exists() else None),
            retry_failed=bool(item.get("retry_failed", False)),
        )
        summary_path = run_output / "summary.json"
        postprocess_path = run_output / "postprocess.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        postprocess = json.loads(postprocess_path.read_text(encoding="utf-8")) if postprocess_path.exists() else {}
        run_manifest_path = run_output / "manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8")) if run_manifest_path.exists() else {}
        summary_rows.append(
            {
                "name": name,
                "matrix_id": item.get("matrix_id"),
                "output": str(run_output),
                "total": summary.get("total", 0),
                "passed": summary.get("passed", 0),
                "failed": summary.get("failed", 0),
                "failure_clusters": len(postprocess.get("failure_clusters", [])),
                "fuzz_cases": len(postprocess.get("fuzz_cases", [])),
                "plan_count": run_manifest.get("plan_count", 0),
                "submitted_count": run_manifest.get("submitted_count", 0),
                "skipped_count": run_manifest.get("skipped_count", 0),
                "resumed_from": str(run_manifest_path) if item.get("resume_from") or run_manifest.get("skipped_count", 0) else None,
                "status_by_task_backend": postprocess.get("status_by_task_backend", {}),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "campaign_summary.json"
    campaign_summary = {
        "runs": summary_rows,
        "totals": _campaign_totals(summary_rows),
        "status_by_task_backend": _merge_status_by_task_backend(summary_rows),
    }
    summary_path.write_text(json.dumps(campaign_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"Campaign summary: {summary_path}")


def _campaign_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "runs": len(rows),
        "total": sum(int(row.get("total", 0)) for row in rows),
        "passed": sum(int(row.get("passed", 0)) for row in rows),
        "failed": sum(int(row.get("failed", 0)) for row in rows),
        "failure_clusters": sum(int(row.get("failure_clusters", 0)) for row in rows),
        "fuzz_cases": sum(int(row.get("fuzz_cases", 0)) for row in rows),
    }


def _merge_status_by_task_backend(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    merged: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        for task, by_backend in row.get("status_by_task_backend", {}).items():
            task_counts = merged.setdefault(task, {})
            for backend, by_status in by_backend.items():
                backend_counts = task_counts.setdefault(backend, {})
                for status, count in by_status.items():
                    backend_counts[status] = backend_counts.get(status, 0) + int(count)
    return merged


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


@app.command()
def run(
    case: Annotated[Path, typer.Option("--case", "-c")] = Path(
        "examples/cases/torch.abs.yaml"
    ),
    nodes: Annotated[Path, typer.Option("--nodes", "-n")] = Path(
        "examples/nodes/local.yaml"
    ),
    task: Annotated[list[TaskKind] | None, typer.Option("--task", "-t")] = None,
    scheduler: Annotated[SchedulerKind, typer.Option("--scheduler", "-s")] = SchedulerKind.RAY,
    ray_address: Annotated[str | None, typer.Option("--ray-address", envvar="RAY_ADDRESS")] = None,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("cx_output"),
    plugin: Annotated[list[Path] | None, typer.Option("--plugin", "-p")] = None,
    plan_id: Annotated[list[str] | None, typer.Option("--plan-id")] = None,
    resume_from: Annotated[Path | None, typer.Option("--resume-from")] = None,
    retry_failed: Annotated[bool, typer.Option("--retry-failed")] = False,
    version_policy: Annotated[str, typer.Option("--version-policy")] = "warn",
) -> None:
    """Build execution plans and run them with the selected scheduler."""
    output = output.resolve()
    log_path = configure_run_logging(output)
    tasks = task or [TaskKind.ACCURACY]
    plugins = plugin or []

    load_builtin_plugins()
    load_plugins(plugins)
    logger.info(
        bind_event(
            "run.start",
            case=case,
            nodes=nodes,
            tasks=",".join(tasks),
            scheduler=scheduler,
            ray_address=ray_address,
            output=output,
            plugins=len(plugins),
        )
    )
    if version_policy not in {"warn", "strict"}:
        raise typer.BadParameter("version policy must be warn or strict")
    context = RunContext(
        case_path=case,
        node_path=nodes,
        tasks=tasks,
        scheduler=scheduler,
        output_root=output,
        ray_address=ray_address,
        plugin_paths=plugins,
    )
    discovery = discover_runtime_resources(ray_address=ray_address if scheduler == SchedulerKind.RAY else None)
    compatibility = evaluate_runtime_compatibility(discovery)
    discovery["version_compatibility"] = compatibility
    discovery_artifacts = write_discovery_files(output / "driver", discovery)
    context = context.model_copy(update={
        "metadata": {
            "input_schema_version": 1,
            "runtime_compatibility": compatibility,
            "version_policy": version_policy,
            "discovery_snapshot": str(discovery_artifacts["snapshot"]),
        }
    })
    if version_policy == "strict" and compatibility["status"] == "mismatched":
        raise typer.BadParameter("driver and worker pipeline fingerprints do not match")
    job = load_job_from_context(context)
    plans = ExecutionPlanner().build(job)
    resume_state = None
    selected_plans = plans
    skipped_count = 0
    if resume_from is not None:
        resume_state = ResumeState.from_path(resume_from, retry_failed=retry_failed)
        selected_plans = resume_state.filter_plans(plans)
        skipped_count = len(plans) - len(selected_plans)
    if plan_id is not None:
        wanted_plan_ids = set(plan_id)
        selected_plans = [plan for plan in selected_plans if plan.plan_id in wanted_plan_ids]
        skipped_count = len(plans) - len(selected_plans)

    result_store = ResultStore(context.output_root)
    manifest = context.to_manifest(
        cruciblex_version=__version__,
        plan_count=len(plans),
        submitted_count=len(selected_plans),
        skipped_count=skipped_count,
    )
    manifest_path = result_store.write_manifest(manifest)
    logger.info(
        bind_event(
            "run.plans",
            count=len(plans),
            submitted=len(selected_plans),
            skipped=skipped_count,
            manifest_path=manifest_path,
        )
    )
    runtime = (
        RayScheduler(context)
        if scheduler == SchedulerKind.RAY
        else LocalScheduler(context)
    )
    for plan in selected_plans:
        logger.info(
            bind_event(
                "run.submit",
                plan=plan.plan_id,
                backend=plan.device.backend,
                device=plan.device.id,
                task=plan.task,
            )
        )
        runtime.submit(plan)

    new_results = runtime.collect()
    results = resume_state.merge_results(plans, new_results) if resume_state is not None else new_results
    postprocess_path: Path | None = None
    if resume_state is None:
        results = ResultPostProcessor(context.output_root).process(results)
        postprocess_path = context.output_root / "postprocess.json"
    summary = summarize(result.model_dump() for result in results)
    results_path = result_store.write_results_jsonl(results)
    results_csv_path = result_store.write_results_csv(results)
    report_jsonl_path = result_store.write_report_jsonl(manifest, results)
    report_csv_path = result_store.write_report_csv(manifest, results)
    summary_path = result_store.write_summary_json(summary)
    manifest = manifest.with_outputs(results_path=results_path, summary_path=summary_path)
    result_store.write_manifest(manifest)
    logger.info(
        bind_event(
            "run.complete",
            count=len(results),
            passed=summary["passed"],
            results_path=results_path,
            summary_path=summary_path,
        )
    )

    table = Table(title="CrucibleX Run")
    table.add_column("Plan")
    table.add_column("Backend")
    table.add_column("Device")
    table.add_column("Task")
    table.add_column("Status")
    for result in results:
        table.add_row(
            result.plan_id,
            result.backend.value,
            str(result.device_id),
            result.task.value,
            result.status.value,
        )
    console.print(table)
    console.print(f"Manifest: {manifest_path}")
    console.print(f"Results: {results_path}")
    console.print(f"Results CSV: {results_csv_path}")
    console.print(f"Report JSONL: {report_jsonl_path}")
    console.print(f"Report CSV: {report_csv_path}")
    console.print(f"Summary: {summary_path}")
    console.print(f"Log: {log_path}")
    console.print(f"Discovery: {discovery_artifacts['snapshot']}")
    console.print(f"Discovered nodes: {discovery_artifacts['nodes']}")
    if postprocess_path is not None:
        console.print(f"Postprocess: {postprocess_path}")

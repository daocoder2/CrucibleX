from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from cruciblex import __version__
from cruciblex.domain.enums import SchedulerKind, TaskKind
from cruciblex.domain.run import RunContext
from cruciblex.generation.loader import load_job_from_context
from cruciblex.plugins import load_builtin_plugins, load_plugins
from cruciblex.report import MarkdownReportWriter, summarize
from cruciblex.runtime.logging import bind_event, get_logger
from cruciblex.runtime.planner import ExecutionPlanner
from cruciblex.runtime.resume import ResumeState
from cruciblex.runtime.scheduler import LocalScheduler, RayScheduler
from cruciblex.storage.results import ResultStore

app = typer.Typer(add_completion=False, help="CrucibleX command line interface.")
console = Console()
logger = get_logger("cli")


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(__version__)


@app.command()
def doctor() -> None:
    """Print Ray and project diagnostics."""
    try:
        import ray
    except ImportError:
        ray_status = "unavailable"
        ray_nodes = []
    else:
        ray_status = "initialized" if ray.is_initialized() else "available"
        if ray.is_initialized():
            from cruciblex.runtime.scheduler.placement import discover_ray_cluster

            ray_nodes = discover_ray_cluster(ray).alive_nodes
        else:
            ray_nodes = []

    console.print("CrucibleX is initialized.")
    console.print(f"Project root: {Path.cwd()}")
    console.print(f"Ray: {ray_status}")
    if ray_status != "unavailable":
        console.print(f"Ray nodes: {len(ray_nodes)}")
        for node in ray_nodes:
            resources = ",".join(sorted(node.resources))
            console.print(f"- {node.address} {node.hostname} resources={resources}")


@app.command()
def generate() -> None:
    """Placeholder for case generation."""
    console.print("Generation entrypoint is not implemented yet.")


@app.command()
def report(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("cx_output"),
) -> None:
    """Render a markdown report from a completed run."""
    output = output.resolve()
    report_path = MarkdownReportWriter(output).write()
    console.print(f"Report: {report_path}")


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
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("cx_output"),
    plugin: Annotated[list[Path] | None, typer.Option("--plugin", "-p")] = None,
    resume_from: Annotated[Path | None, typer.Option("--resume-from")] = None,
    retry_failed: Annotated[bool, typer.Option("--retry-failed")] = False,
) -> None:
    """Build execution plans and run them with the selected scheduler."""
    output = output.resolve()
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
            output=output,
            plugins=len(plugins),
        )
    )
    context = RunContext(
        case_path=case,
        node_path=nodes,
        tasks=tasks,
        scheduler=scheduler,
        output_root=output,
        plugin_paths=plugins,
    )
    job = load_job_from_context(context)
    plans = ExecutionPlanner().build(job)
    resume_state = None
    selected_plans = plans
    skipped_count = 0
    if resume_from is not None:
        resume_state = ResumeState.from_path(resume_from, retry_failed=retry_failed)
        selected_plans = resume_state.filter_plans(plans)
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
    summary = summarize(result.model_dump() for result in results)
    results_path = result_store.write_results_jsonl(results)
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
    console.print(f"Summary: {summary_path}")

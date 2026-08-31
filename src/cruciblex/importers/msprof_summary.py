from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

CONVERTER_VERSION = "cruciblex.importers.msprof_summary:v1"


def import_msprof_summary(source: str | Path) -> dict[str, Any]:
    root = Path(source)
    if root.is_file():
        root = root.parent
    files = _find_exports(root)
    warnings: list[str] = []
    operators = _read_operator_stats(files.get("op_statistic"), warnings)
    tasks = _read_task_stats(files.get("task_time"), warnings)
    summaries = _read_op_summaries(files.get("op_summary"), warnings)
    return {
        "tool": "msprof",
        "status": "parsed" if files else "failed",
        "converter_version": CONVERTER_VERSION,
        "source": str(root),
        "source_files": {key: str(path) for key, path in files.items()},
        "warnings": warnings,
        "device_count": len({item["device_id"] for item in operators + tasks + summaries}),
        "top_operators": sorted(operators, key=lambda item: item.get("total_time_us", 0), reverse=True),
        "task_summary": tasks,
        "op_summary": summaries,
    }


def write_msprof_summary(source: str | Path, output: str | Path) -> Path:
    summary = import_msprof_summary(source)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def _find_exports(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.rglob("*.csv"):
        name = path.name.lower()
        for kind in ("op_statistic", "op_summary", "task_time"):
            if name.startswith(kind):
                result.setdefault(kind, path)
    return result


def _rows(path: Path | None, warnings: list[str]) -> list[dict[str, str]]:
    if path is None:
        warnings.append("missing export")
        return []
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeError) as exc:
        warnings.append(f"failed to parse {path.name}: {exc}")
        return []


def _number(row: dict[str, str], key: str, warnings: list[str], default: float = 0.0) -> float:
    raw = (row.get(key) or "").strip()
    try:
        return float(raw.replace("\t", "")) if raw else default
    except ValueError:
        warnings.append(f"invalid numeric value for {key}: {raw!r}")
        return default


def _device(row: dict[str, str]) -> int | str:
    raw = (row.get("Device_id") or "").strip()
    try:
        return int(raw)
    except ValueError:
        return raw or "unknown"


def _read_operator_stats(path: Path | None, warnings: list[str]) -> list[dict[str, Any]]:
    rows = _rows(path, warnings)
    return [{"device_id": _device(row), "op_type": row.get("OP Type", ""), "core_type": row.get("Core Type", ""), "count": int(_number(row, "Count", warnings)), "total_time_us": _number(row, "Total Time(us)", warnings), "min_time_us": _number(row, "Min Time(us)", warnings), "avg_time_us": _number(row, "Avg Time(us)", warnings), "max_time_us": _number(row, "Max Time(us)", warnings), "ratio_percent": _number(row, "Ratio(%)", warnings)} for row in rows]


def _read_task_stats(path: Path | None, warnings: list[str]) -> list[dict[str, Any]]:
    rows = _rows(path, warnings)
    return [{"device_id": _device(row), "kernel_name": row.get("kernel_name", ""), "kernel_type": row.get("kernel_type", ""), "count": 1, "total_time_us": _number(row, "task_time(us)", warnings)} for row in rows]


def _read_op_summaries(path: Path | None, warnings: list[str]) -> list[dict[str, Any]]:
    rows = _rows(path, warnings)
    return [{"device_id": _device(row), "op_name": row.get("Op Name", ""), "op_type": row.get("OP Type", ""), "task_duration_us": _number(row, "Task Duration(us)", warnings), "task_wait_time_us": _number(row, "Task Wait Time(us)", warnings), "aiv_time_us": _number(row, "aiv_time(us)", warnings)} for row in rows]

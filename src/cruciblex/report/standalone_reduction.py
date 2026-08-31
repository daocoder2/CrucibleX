from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from cruciblex.report.reduction import reduce_with_predicate


def reduce_case_file(case_path: Path, output: Path, command: str, expected_exit_code: int = 1) -> dict[str, Any]:
    payload = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    case = payload["cases"][0] if isinstance(payload, dict) and isinstance(payload.get("cases"), list) else payload
    if not isinstance(case, dict):
        raise TypeError("case YAML must contain a mapping or a non-empty cases list")
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = output / "replay_candidate.yaml"
    attempts: list[dict[str, Any]] = []

    def predicate(candidate: dict[str, Any]) -> bool:
        candidate_path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
        started = time.perf_counter()
        completed = subprocess.run(
            command.replace("{case}", shlex.quote(str(candidate_path))),
            shell=True,
            check=False,
            capture_output=True,
            text=True,
        )
        accepted = completed.returncode == expected_exit_code
        attempts.append({
            "strategy": candidate.get("reduction", {}).get("strategy"),
            "accepted": accepted,
            "exit_code": completed.returncode,
            "expected_exit_code": expected_exit_code,
            "duration_ms": (time.perf_counter() - started) * 1000,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        })
        return accepted

    reduced, _ = reduce_with_predicate(case, predicate)
    reduced_path = output / "reduced_case.yaml"
    reduced_path.write_text(yaml.safe_dump(reduced, sort_keys=False), encoding="utf-8")
    attempts_path = output / "replay_attempts.jsonl"
    attempts_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in attempts), encoding="utf-8")
    rerun_path = output / "rerun.sh"
    rerun_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command.replace("{case}", shlex.quote(str(reduced_path))) + "\n", encoding="utf-8")
    rerun_path.chmod(0o755)
    summary = {
        "source_case": str(case_path),
        "reduced_case": str(reduced_path),
        "expected_exit_code": expected_exit_code,
        "attempt_count": len(attempts),
        "accepted_count": sum(1 for item in attempts if item["accepted"]),
        "status": "reduced" if attempts else "already_minimal",
    }
    (output / "reduction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

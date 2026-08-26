from __future__ import annotations

from collections.abc import Iterable


def summarize(results: Iterable[dict[str, object]]) -> dict[str, int]:
    total = 0
    passed = 0
    failed = 0
    for result in results:
        total += 1
        status = result.get("status")
        if status == "passed":
            passed += 1
        elif status in {"failed", "error", "timeout"}:
            failed += 1
    return {"total": total, "passed": passed, "failed": failed}

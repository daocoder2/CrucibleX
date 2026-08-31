from __future__ import annotations

from typing import Any


def evaluate_runtime_compatibility(discovery: dict[str, Any]) -> dict[str, object]:
    probes = discovery.get("runtime_probes")
    if not isinstance(probes, dict):
        return {"status": "unavailable", "worker_count": 0, "mismatch_count": 0}
    driver = probes.get("driver")
    workers = probes.get("workers")
    if not isinstance(driver, dict) or not isinstance(workers, list):
        return {"status": "unavailable", "worker_count": 0, "mismatch_count": 0}
    driver_fingerprint = _fingerprint(driver)
    comparable = []
    for worker in workers:
        if not isinstance(worker, dict) or not isinstance(worker.get("probe"), dict):
            continue
        fingerprint = _fingerprint(worker["probe"])
        if fingerprint is not None:
            comparable.append(fingerprint)
    if driver_fingerprint is None or not comparable:
        return {"status": "unavailable", "worker_count": len(workers), "mismatch_count": 0}
    mismatches = sum(fingerprint != driver_fingerprint for fingerprint in comparable)
    return {
        "status": "mismatched" if mismatches else "matched",
        "driver_fingerprint": driver_fingerprint,
        "worker_count": len(workers),
        "comparable_worker_count": len(comparable),
        "mismatch_count": mismatches,
    }


def _fingerprint(probe: dict[str, Any]) -> str | None:
    cruciblex = probe.get("cruciblex")
    if not isinstance(cruciblex, dict):
        return None
    value = cruciblex.get("pipeline_sha256")
    return str(value) if value else None

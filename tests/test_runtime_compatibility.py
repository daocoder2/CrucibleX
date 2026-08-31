from cruciblex.runtime.compatibility import evaluate_runtime_compatibility


def _discovery(driver: str | None, workers: list[str | None]) -> dict:
    return {
        "runtime_probes": {
            "driver": {"cruciblex": {"pipeline_sha256": driver}} if driver else {},
            "workers": [
                {"probe": {"cruciblex": {"pipeline_sha256": worker}}} if worker else {"probe": {}}
                for worker in workers
            ],
        }
    }


def test_runtime_compatibility_reports_matching_worker_fingerprints():
    compatibility = evaluate_runtime_compatibility(_discovery("driver", ["driver", "driver"]))

    assert compatibility == {
        "status": "matched",
        "driver_fingerprint": "driver",
        "worker_count": 2,
        "comparable_worker_count": 2,
        "mismatch_count": 0,
    }


def test_runtime_compatibility_reports_and_counts_mismatches():
    compatibility = evaluate_runtime_compatibility(_discovery("driver", ["driver", "worker"]))

    assert compatibility["status"] == "mismatched"
    assert compatibility["mismatch_count"] == 1
    assert compatibility["comparable_worker_count"] == 2


def test_runtime_compatibility_is_unavailable_without_comparable_probes():
    assert evaluate_runtime_compatibility(_discovery(None, [None])) == {
        "status": "unavailable",
        "worker_count": 1,
        "mismatch_count": 0,
    }

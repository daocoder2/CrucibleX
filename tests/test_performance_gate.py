import json

from cruciblex.report.performance_gate import evaluate_performance_gate


def _run(root, latency, throughput, memory):
    root.mkdir()
    (root / "results.jsonl").write_text(json.dumps({
        "plan_id": "p", "case_id": 1, "case_name": "torch.abs", "node_name": "n",
        "backend": "npu", "device_id": 0, "task": "performance_device", "status": "passed",
        "metrics": {"latency_p95_ms": latency, "throughput_items_per_s": throughput, "hardware_memory_peak_bytes": memory},
        "artifacts": [], "error": None
    }) + "\n")
    (root / "profiler_summary.json").write_text(json.dumps({"top_operators": [{"op_type": "Abs", "total_time_us": latency * 10}]}))


def test_gate_compares_metrics_and_profiler(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _run(baseline, 10, 100, 1000)
    _run(candidate, 10.4, 96, 1050)
    result = evaluate_performance_gate(baseline, candidate, {"metrics": {
        "latency_p95_ms": {"max_regression_percent": 5},
        "throughput_items_per_s": {"min_regression_percent": 5},
        "hardware_memory_peak_bytes": {"max_regression_percent": 10},
    }})
    assert result["status"] == "passed"
    assert result["profiler"]["status"] == "compared"


def test_gate_returns_regressed_for_threshold_breach(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _run(baseline, 10, 100, 1000)
    _run(candidate, 11, 100, 1000)
    result = evaluate_performance_gate(baseline, candidate, {"metrics": {"latency_p95_ms": {"max_regression_percent": 5}}})
    assert result["status"] == "regressed"
    assert result["regressions"] == 1


def test_gate_keeps_distinct_generated_case_ids(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    rows = []
    for case_id, latency in ((1, 10.0), (2, 20.0)):
        rows.append({
            "plan_id": f"p-{case_id}", "case_id": case_id, "case_name": "torch.relu", "node_name": "n",
            "backend": "npu", "device_id": 0, "task": "performance_device", "status": "passed",
            "metrics": {"latency_p95_ms": latency}, "artifacts": [], "error": None,
        })
    payload = "\n".join(json.dumps(row) for row in rows) + "\n"
    (baseline / "results.jsonl").write_text(payload)
    (candidate / "results.jsonl").write_text(payload)
    profiler = json.dumps({"top_operators": []})
    (baseline / "profiler_summary.json").write_text(profiler)
    (candidate / "profiler_summary.json").write_text(profiler)
    result = evaluate_performance_gate(baseline, candidate, {"metrics": {"latency_p95_ms": {"max_regression_percent": 1}}})
    assert result["status"] == "passed"
    assert len(result["metrics"]) == 2


def test_gate_allows_explicitly_optional_profiler(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _run(baseline, 10, 100, 1000)
    _run(candidate, 10, 100, 1000)
    (baseline / "profiler_summary.json").unlink()
    (candidate / "profiler_summary.json").unlink()
    result = evaluate_performance_gate(baseline, candidate, {"metrics": {}, "profiler": {"required": False}})
    assert result["status"] == "passed"
    assert result["profiler"]["status"] == "not_required"


def test_gate_scopes_metrics_to_requested_tasks(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _run(baseline, 10, 100, 1000)
    _run(candidate, 10, 100, 1000)
    result = evaluate_performance_gate(baseline, candidate, {
        "metrics": {"memory_peak_bytes": {"tasks": ["memory_device"]}},
        "profiler": {"required": False},
    })
    assert result["status"] == "passed"
    assert result["metrics"] == []

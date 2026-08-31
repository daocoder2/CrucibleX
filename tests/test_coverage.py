from cruciblex.report.coverage import evaluate_coverage_policy, summarize_coverage


def test_coverage_summary_and_policy_report_missing_dimensions():
    summary = summarize_coverage([
        {"case_name": "torch.add", "backend": "cpu", "task": "accuracy", "metrics": {"dtype": "fp32", "output_shape": [2, 3]}},
        {"case_name": "torch.add", "backend": "gpu", "task": "accuracy", "metrics": {"dtype": "fp32", "output_shape": [2, 3]}},
    ])
    assert summary["dimensions"]["backend"] == {"cpu": 1, "gpu": 1}
    assert summary["dimensions"]["shape"] == {"2x3": 2}
    passed = evaluate_coverage_policy(summary, {"required": {"backend": ["cpu", "gpu"]}})
    assert passed["status"] == "passed"
    failed = evaluate_coverage_policy(summary, {"required_combinations": [{"operator": "torch.add", "backend": "npu", "task": "accuracy"}]})
    assert failed["status"] == "failed"
    assert failed["missing"][0]["combination"]["backend"] == "npu"

import numpy as np
import pytest

from cruciblex.plugins.comparators.allclose import AllCloseComparator
from cruciblex.runtime.compare import ComparisonRequest


def _request(expected, actual, tolerance):
    return ComparisonRequest(expected=expected, actual=actual, tolerance=tolerance, metadata={})


def test_allclose_reports_unified_numeric_metrics():
    report = AllCloseComparator().compare(_request([1.0, 2.0], [1.1, 1.8], {"atol": 0.0, "rtol": 0.2}))

    assert report.passed is True
    assert report.max_abs_diff == pytest.approx(0.2)
    assert report.metrics["max_relative_error"] == pytest.approx(0.1)
    assert report.metrics["mean_relative_error"] == pytest.approx(0.1)
    assert report.metrics["rmse"] == pytest.approx(np.sqrt(0.025))
    assert report.metrics["matched_ratio"] == 1.0


def test_mixed_tolerance_allows_limited_element_mismatches_but_enforces_absolute_cap():
    comparator = AllCloseComparator()
    tolerance = {
        "atol": 0.0,
        "rtol": 0.0,
        "mixed": {"required_matched_ratio": 0.75, "max_abs_error": 0.5},
    }

    passed = comparator.compare(_request([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.3], tolerance))
    failed_ratio = comparator.compare(_request([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.3, 1.3], tolerance))
    failed_cap = comparator.compare(_request([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.6], tolerance))

    assert passed.passed is True
    assert passed.metrics["matched_ratio"] == 0.75
    assert failed_ratio.passed is False
    assert failed_cap.passed is False


def test_allclose_rejects_non_finite_values_and_shape_mismatches():
    comparator = AllCloseComparator()

    non_finite = comparator.compare(_request([1.0, np.nan], [1.0, np.nan], {}))
    mismatch = comparator.compare(_request([1.0], [1.0, 2.0], {}))

    assert non_finite.passed is False
    assert non_finite.metrics["non_finite_count"] == 1
    assert mismatch.passed is False
    assert mismatch.metrics["shape_match"] is False


def test_mixed_tolerance_rejects_invalid_matched_ratio():
    with pytest.raises(ValueError, match="required_matched_ratio"):
        AllCloseComparator().compare(_request([1.0], [1.0], {"mixed": {"required_matched_ratio": 1.1}}))

from __future__ import annotations

import numpy as np

from cruciblex.runtime.compare.base import (
    COMPARATOR_REGISTRY,
    Comparator,
    ComparisonReport,
    ComparisonRequest,
)


class AllCloseComparator(Comparator):
    def compare(self, request: ComparisonRequest) -> ComparisonReport:
        expected_arr = np.asarray(request.expected)
        actual_arr = np.asarray(request.actual)
        diff = np.abs(expected_arr - actual_arr)
        max_abs_diff = float(diff.max()) if diff.size else 0.0
        mean_abs_diff = float(diff.mean()) if diff.size else 0.0
        passed = np.allclose(
            expected_arr,
            actual_arr,
            atol=request.tolerance.get("atol", 1e-6),
            rtol=request.tolerance.get("rtol", 1e-6),
        )
        detail = "allclose passed" if passed else "allclose failed"
        return ComparisonReport(passed=passed, max_abs_diff=max_abs_diff, mean_abs_diff=mean_abs_diff, detail=detail)


COMPARATOR_REGISTRY.register("allclose")(AllCloseComparator)

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
        if expected_arr.shape != actual_arr.shape:
            return ComparisonReport(
                passed=False,
                max_abs_diff=float("inf"),
                mean_abs_diff=float("inf"),
                detail=f"allclose shape mismatch: expected={expected_arr.shape}, actual={actual_arr.shape}",
                metrics={"shape_match": False, "non_finite_count": 0},
            )
        diff = np.abs(expected_arr - actual_arr)
        finite = np.isfinite(expected_arr) & np.isfinite(actual_arr)
        non_finite_count = int(np.count_nonzero(~finite))
        max_abs_diff = float(diff[finite].max()) if finite.any() else 0.0
        mean_abs_diff = float(diff[finite].mean()) if finite.any() else 0.0
        atol = float(request.tolerance.get("atol", 1e-6))
        rtol = float(request.tolerance.get("rtol", 1e-6))
        close_mask = np.isclose(expected_arr, actual_arr, atol=atol, rtol=rtol, equal_nan=False)
        denominator = np.maximum(np.abs(expected_arr[finite]), float(request.tolerance.get("relative_epsilon", 1e-12)))
        relative = diff[finite] / denominator
        metrics: dict[str, float | int | bool] = {
            "shape_match": True,
            "non_finite_count": non_finite_count,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "max_relative_error": float(relative.max()) if relative.size else 0.0,
            "mean_relative_error": float(relative.mean()) if relative.size else 0.0,
            "rmse": float(np.sqrt(np.mean(np.square(diff[finite])))) if finite.any() else 0.0,
            "matched_ratio": float(np.mean(close_mask)) if close_mask.size else 1.0,
        }
        passed = bool(non_finite_count == 0 and np.all(close_mask))
        mixed = request.tolerance.get("mixed")
        if isinstance(mixed, dict):
            required_ratio = float(mixed.get("required_matched_ratio", 1.0))
            max_error = mixed.get("max_abs_error")
            if not 0.0 <= required_ratio <= 1.0:
                raise ValueError("mixed.required_matched_ratio must be between 0 and 1")
            passed = bool(non_finite_count == 0 and metrics["matched_ratio"] >= required_ratio)
            if max_error is not None:
                passed = passed and max_abs_diff <= float(max_error)
            metrics["required_matched_ratio"] = required_ratio
            if max_error is not None:
                metrics["max_abs_error"] = float(max_error)
        detail = "allclose passed" if passed else "allclose failed"
        return ComparisonReport(passed=passed, max_abs_diff=max_abs_diff, mean_abs_diff=mean_abs_diff, detail=detail, metrics=metrics)


COMPARATOR_REGISTRY.register("allclose")(AllCloseComparator)

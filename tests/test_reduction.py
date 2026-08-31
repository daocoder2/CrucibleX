from cruciblex.report.reduction import reduce_with_predicate, semantic_reduction_candidates


def test_reduction_shrinks_shape_and_dtype_without_mutating_source():
    source = {
        "parameters": [{"shape_rules": {"dims": [8, 16]}, "dtypes": ["fp16", "fp32"], "fuzz": {"seed": 7}}],
        "generation": {"seed": 7},
    }
    candidates = semantic_reduction_candidates(source)
    assert len(candidates) == 2
    reduced = candidates[0]
    assert reduced["parameters"][0]["shape_rules"]["dims"] == [1, 1]
    assert reduced["parameters"][0]["dtypes"] == ["fp16", "fp32"]
    assert candidates[1]["parameters"][0]["dtypes"] == ["fp32"]
    assert source["parameters"][0]["shape_rules"]["dims"] == [8, 16]
    assert reduced["reduction"]["requires_replay"] is True


def test_reduction_returns_no_candidate_when_already_minimal():
    assert semantic_reduction_candidates({"parameters": [{"shape_rules": {"dims": [1]}, "dtypes": ["fp32"]}]}) == []


def test_reduction_accepts_only_replayed_failure():
    source = {"parameters": [{"shape_rules": {"dims": [4]}, "dtypes": ["fp16"]}]}
    accepted, attempts = reduce_with_predicate(source, lambda case: case["parameters"][0]["dtypes"] == ["fp32"])
    assert accepted["parameters"][0]["dtypes"] == ["fp32"]
    assert attempts == [
        {"strategy": "shape_to_one", "accepted": False},
        {"strategy": "dtype_to_fp32", "accepted": True},
    ]

    rejected, attempts = reduce_with_predicate(source, lambda _: False)
    assert rejected == source
    assert attempts == [
        {"strategy": "shape_to_one", "accepted": False},
        {"strategy": "dtype_to_fp32", "accepted": False},
    ]

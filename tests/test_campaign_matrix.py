import pytest

from cruciblex.campaign import expand_campaign_payload, select_campaign_shard


def test_campaign_matrix_expands_stable_ids():
    payload = {
        "runs": [{"name": "explicit", "case": "a", "nodes": "n"}],
        "matrix": {
            "base": {"case": "a", "nodes": "n", "task": "performance_device"},
            "dimensions": {"dtype": ["fp16", "fp32"], "shape": [[2], [4]]},
        },
    }
    runs = expand_campaign_payload(payload)
    assert len(runs) == 5
    matrix = runs[1:]
    assert len({run["matrix_id"] for run in matrix}) == 4
    assert [run["matrix_id"] for run in matrix] == [run["matrix_id"] for run in expand_campaign_payload(payload)[1:]]


def test_campaign_shards_cover_matrix_without_overlap():
    runs = expand_campaign_payload({"matrix": {"case": ["a", "b", "c", "d"]}})
    shards = [select_campaign_shard(runs, i, 2) for i in range(2)]
    assert {id(run) for shard in shards for run in shard} == {id(run) for run in runs}
    assert not {id(run) for run in shards[0]} & {id(run) for run in shards[1]}


def test_campaign_matrix_limits_runs_deterministically_and_records_coverage():
    payload = {
        "matrix": {
            "dimensions": {"dtype": ["fp16", "fp32", "fp64"], "shape": [[1], [2], [4]]},
            "max_runs": 4,
            "seed": 11,
        }
    }

    first = expand_campaign_payload(payload)
    second = expand_campaign_payload(payload)

    assert len(first) == 4
    assert [item["matrix_id"] for item in first] == [item["matrix_id"] for item in second]
    assert {item["matrix_total"] for item in first} == {9}
    assert {item["matrix_selected"] for item in first} == {4}


def test_campaign_matrix_rejects_non_positive_max_runs():
    with pytest.raises(ValueError, match="max_runs"):
        expand_campaign_payload({"matrix": {"case": ["a"], "max_runs": 0}})

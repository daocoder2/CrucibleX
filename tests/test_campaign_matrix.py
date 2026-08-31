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

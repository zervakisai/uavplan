from uavbench.scenarios.registry import list_scenarios, list_scenarios_by_track


def test_track_partition_complete_and_disjoint():
    all_ids = set(list_scenarios())
    static_ids = set(list_scenarios_by_track("static"))
    dynamic_ids = set(list_scenarios_by_track("dynamic"))

    assert static_ids
    assert dynamic_ids
    assert static_ids.isdisjoint(dynamic_ids)
    assert static_ids | dynamic_ids == all_ids

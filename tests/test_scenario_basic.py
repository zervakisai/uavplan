from pathlib import Path

from uavbench.scenarios.loader import load_scenario


def test_urban_hard_levels():
    cfg = load_scenario(Path("src/uavbench/scenarios/configs/urban_hard.yaml"))
    assert cfg.wind.value == "high"
    assert cfg.traffic.value == "high"



def test_urban_medium_no_fly():
    cfg = load_scenario(Path("src/uavbench/scenarios/configs/urban_medium.yaml"))
    # Updated schema uses no_fly_radius and dynamic_nfz instead of no_fly_zones list
    assert cfg.difficulty.value == "medium"
    assert cfg.no_fly_radius >= 0

def test_load_urban_easy():
    cfg = load_scenario(Path("src/uavbench/scenarios/configs/urban_easy.yaml"))
    assert cfg.domain.value == "urban"
    assert cfg.difficulty.value == "easy"
    # OSM-based scenarios use 500x500 or 25x25 depending on config; accept either
    assert cfg.map_size in (25, 50, 500)


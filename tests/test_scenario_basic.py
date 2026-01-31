from pathlib import Path

from uavbench.scenarios.loader import load_scenario


def test_urban_hard_levels():
    cfg = load_scenario(Path("src/uavbench/scenarios/configs/urban_hard.yaml"))
    assert cfg.wind.value == "high"
    assert cfg.traffic.value == "high"



def test_urban_medium_no_fly():
    cfg = load_scenario(Path("src/uavbench/scenarios/configs/urban_medium.yaml"))
    assert len(cfg.no_fly_zones) == 2
    z0 = cfg.no_fly_zones[0]
    assert z0.x_min == 10 and z0.y_max == 20

def test_load_urban_easy():
    cfg = load_scenario(Path("src/uavbench/scenarios/configs/urban_easy.yaml"))
    assert cfg.domain.value == "urban"
    assert cfg.difficulty.value == "easy"
    assert cfg.map_size == 50


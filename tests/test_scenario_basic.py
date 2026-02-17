from pathlib import Path

from uavbench.scenarios.loader import load_scenario


def test_gov_civil_protection_hard():
    cfg = load_scenario(Path("src/uavbench/scenarios/configs/gov_civil_protection_hard.yaml"))
    assert cfg.difficulty.value == "hard"
    assert cfg.domain.value == "urban"


def test_gov_maritime_domain_medium():
    cfg = load_scenario(Path("src/uavbench/scenarios/configs/gov_maritime_domain_medium.yaml"))
    assert cfg.difficulty.value == "medium"
    assert cfg.no_fly_radius >= 0


def test_load_gov_civil_protection_easy():
    cfg = load_scenario(Path("src/uavbench/scenarios/configs/gov_civil_protection_easy.yaml"))
    assert cfg.domain.value == "urban"
    assert cfg.difficulty.value == "easy"
    assert cfg.map_size == 64


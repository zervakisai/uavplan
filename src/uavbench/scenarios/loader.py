from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from uavbench.scenarios.schema import ScenarioConfig, Domain, Difficulty, Wind, Traffic


def load_scenario(path: Path) -> ScenarioConfig:
    data: dict[str, Any]
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Required
    name = str(data["name"])
    domain = Domain(str(data["domain"]))
    difficulty = Difficulty(str(data["difficulty"]))

    # Optional enums
    wind = Wind(str(data.get("wind", Wind.NONE.value)))
    traffic = Traffic(str(data.get("traffic", Traffic.NONE.value)))

    # Build config (explicit mapping keeps it stable & V&V friendly)
    cfg = ScenarioConfig(
        name=name,
        domain=domain,
        difficulty=difficulty,
        wind=wind,
        traffic=traffic,
        map_size=int(data.get("map_size", 25)),
        max_altitude=int(data.get("max_altitude", 3)),
        building_density=float(data.get("building_density", 0.30)),
        building_level=int(data.get("building_level", 2)),
        start_altitude=int(data.get("start_altitude", 1)),
        safe_altitude=int(data.get("safe_altitude", 3)),
        min_start_goal_l1=int(data.get("min_start_goal_l1", 10)),
        extra_density_medium=float(data.get("extra_density_medium", 0.10)),
        extra_density_hard=float(data.get("extra_density_hard", 0.20)),
        no_fly_radius=int(data.get("no_fly_radius", 3)),
        downtown_window=int(data.get("downtown_window", 7)),
        spawn_clearance=int(data.get("spawn_clearance", 1)),
        debug=bool(data.get("debug", False)),
        extra=dict(data.get("extra", {})) if "extra" in data else None,
    )

    cfg.validate()
    return cfg

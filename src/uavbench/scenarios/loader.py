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

    # Optional enums (support both 'wind'/'traffic' and 'wind_level'/'traffic_level')
    wind_key = "wind_level" if "wind_level" in data else "wind"
    traffic_key = "traffic_level" if "traffic_level" in data else "traffic"
    wind = Wind(str(data.get(wind_key, Wind.NONE.value)))
    traffic = Traffic(str(data.get(traffic_key, Traffic.NONE.value)))

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
        map_source=str(data.get("map_source", "synthetic")),
        osm_tile_id=data.get("osm_tile_id"),
        enable_fire=bool(data.get("enable_fire", False)),
        enable_traffic=bool(data.get("enable_traffic", False)),
        fire_ignition_points=int(data.get("fire_ignition_points", 3)),
        num_emergency_vehicles=int(data.get("num_emergency_vehicles", 5)),
        wind_direction=float(data.get("wind_direction", 0.0)),
        wind_speed=float(data.get("wind_speed", 0.5)),
        fire_blocks_movement=bool(data.get("fire_blocks_movement", False)),
        traffic_blocks_movement=bool(data.get("traffic_blocks_movement", False)),
        enable_moving_target=bool(data.get("enable_moving_target", False)),
        target_speed=float(data.get("target_speed", 1.0)),
        target_buffer_radius=int(data.get("target_buffer_radius", 8)),
        enable_intruders=bool(data.get("enable_intruders", False)),
        num_intruders=int(data.get("num_intruders", 2)),
        intruder_speed=float(data.get("intruder_speed", 0.5)),
        intruder_spawn_zone=str(data.get("intruder_spawn_zone", "north")),
        enable_dynamic_nfz=bool(data.get("enable_dynamic_nfz", False)),
        num_nfz_zones=int(data.get("num_nfz_zones", 3)),
        nfz_expansion_rate=float(data.get("nfz_expansion_rate", 0.8)),
        nfz_max_radius=int(data.get("nfz_max_radius", 35)),
        terminate_on_collision=bool(data.get("terminate_on_collision", True)),
        debug=bool(data.get("debug", False)),
        extra=dict(data.get("extra", {})) if "extra" in data else None,
    )

    cfg.validate()
    return cfg

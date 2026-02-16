from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]

from uavbench.scenarios.schema import (
    ScenarioConfig,
    Domain,
    Difficulty,
    Wind,
    Traffic,
    MissionType,
    Regime,
    InterdictionReferencePlanner,
)


def load_scenario(path: Path) -> ScenarioConfig:
    data: dict[str, Any]
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Required
    name = str(data["name"])
    domain = Domain(str(data["domain"]))
    difficulty = Difficulty(str(data["difficulty"]))
    mission_type = MissionType(str(data.get("mission_type", MissionType.POINT_TO_POINT.value)))
    regime = Regime(str(data.get("regime", Regime.NATURALISTIC.value)))
    paper_track_raw = str(data.get("paper_track", "dynamic" if regime == Regime.STRESS_TEST else "static")).lower()
    paper_track: Literal["static", "dynamic"] = "dynamic" if paper_track_raw == "dynamic" else "static"

    # Optional enums (support both 'wind'/'traffic' and 'wind_level'/'traffic_level')
    wind_key = "wind_level" if "wind_level" in data else "wind"
    traffic_key = "traffic_level" if "traffic_level" in data else "traffic"
    wind = Wind(str(data.get(wind_key, Wind.NONE.value)))
    traffic = Traffic(str(data.get(traffic_key, Traffic.NONE.value)))

    scenario_id = str(data.get("scenario_id", name))

    # Backward-compat normalization for map fields in YAML scenario configs.
    tile_name_raw = data.get("tile_name")
    tile_name = str(tile_name_raw).strip() if tile_name_raw is not None else None

    if "map_source" in data:
        map_source = str(data["map_source"])
    else:
        map_source = "osm" if (tile_name and not tile_name.lower().startswith("synthetic")) else "synthetic"

    if "osm_tile_id" in data and data.get("osm_tile_id") is not None:
        osm_tile_id = str(data["osm_tile_id"]).strip().lower()
    elif map_source == "osm" and tile_name:
        osm_tile_id = tile_name.lower()
    else:
        osm_tile_id = None

    # Build config (explicit mapping keeps it stable & V&V friendly)
    cfg = ScenarioConfig(
        name=name,
        domain=domain,
        difficulty=difficulty,
        mission_type=mission_type,
        regime=regime,
        paper_track=paper_track,
        wind=wind,
        traffic=traffic,
        map_size=int(data.get("map_size", 25)),
        max_altitude=int(data.get("max_altitude", data.get("altitude_levels", 3))),
        building_density=float(data.get("building_density", 0.30)),
        building_level=int(data.get("building_level", 2)),
        start_altitude=int(data.get("start_altitude", 1)),
        safe_altitude=int(data.get("safe_altitude", 3)),
        min_start_goal_l1=int(data.get("min_start_goal_l1", 10)),
        fixed_start_xy=cast(tuple[int, int], tuple(int(v) for v in data["fixed_start_xy"])) if "fixed_start_xy" in data and data.get("fixed_start_xy") is not None else None,
        fixed_goal_xy=cast(tuple[int, int], tuple(int(v) for v in data["fixed_goal_xy"])) if "fixed_goal_xy" in data and data.get("fixed_goal_xy") is not None else None,
        extra_density_medium=float(data.get("extra_density_medium", 0.10)),
        extra_density_hard=float(data.get("extra_density_hard", 0.20)),
        no_fly_radius=int(data.get("no_fly_radius", 3)),
        downtown_window=int(data.get("downtown_window", 7)),
        spawn_clearance=int(data.get("spawn_clearance", 1)),
        map_source=map_source,
        osm_tile_id=osm_tile_id,
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
        risk_weight_population=float(data.get("risk_weight_population", 0.55)),
        risk_weight_adversarial=float(data.get("risk_weight_adversarial", 0.30)),
        risk_weight_smoke=float(data.get("risk_weight_smoke", 0.15)),
        force_replan_count=int(data.get("force_replan_count", 0)),
        event_t1=int(data["event_t1"]) if data.get("event_t1") is not None else None,
        event_t2=int(data["event_t2"]) if data.get("event_t2") is not None else None,
        emergency_corridor_enabled=bool(data.get("emergency_corridor_enabled", True)),
        interdiction_reference_planner=InterdictionReferencePlanner(
            str(data.get("interdiction_reference_planner", InterdictionReferencePlanner.THETA_STAR.value)).lower()
        ),
        plan_budget_static_ms=float(data.get("plan_budget_static_ms", 50.0)),
        plan_budget_dynamic_ms=float(data.get("plan_budget_dynamic_ms", 20.0)),
        replan_every_steps=int(data.get("replan_every_steps", 2)),
        max_replans_per_episode=int(data.get("max_replans_per_episode", 200)),
        terminate_on_collision=bool(data.get("terminate_on_collision", True)),
        debug=bool(data.get("debug", False)),
        extra=dict(data.get("extra", {})) if "extra" in data else None,
    )

    cfg.validate()
    return cfg

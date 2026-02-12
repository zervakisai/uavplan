from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Domain(str, Enum):
    URBAN = "urban"
    # add later: MOUNTAIN = "mountain", MARITIME = "maritime", ...


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Wind(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Traffic(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    # Identity
    name: str
    domain: Domain
    difficulty: Difficulty

    # Disturbances
    wind: Wind = Wind.NONE
    traffic: Traffic = Traffic.NONE

    # Map / flight levels
    map_size: int = 25
    max_altitude: int = 3

    # City generation (levels-only)
    building_density: float = 0.30
    building_level: int = 2

    # Start/goal policy
    start_altitude: int = 1
    safe_altitude: int = 3
    min_start_goal_l1: int = 10

    # Difficulty knobs
    extra_density_medium: float = 0.10
    extra_density_hard: float = 0.20
    no_fly_radius: int = 3

    # “Downtown” / spawn behavior (optional)
    downtown_window: int = 7
    spawn_clearance: int = 1

    # Map source
    map_source: str = "synthetic"       # "synthetic" or "osm"
    osm_tile_id: str | None = None      # e.g. "downtown", "penteli", "piraeus"

    # Dynamic layers
    enable_fire: bool = False
    enable_traffic: bool = False
    fire_ignition_points: int = 3
    num_emergency_vehicles: int = 5
    wind_direction: float = 0.0         # radians, 0 = North
    wind_speed: float = 0.5             # [0, 1]
    fire_blocks_movement: bool = False  # When True, burning cells reject movement
    traffic_blocks_movement: bool = False  # When True, vehicle buffer zones reject movement

    # Moving target (SAR scenarios)
    enable_moving_target: bool = False
    target_speed: float = 1.0
    target_buffer_radius: int = 8

    # Intruders (border patrol scenarios)
    enable_intruders: bool = False
    num_intruders: int = 2
    intruder_speed: float = 0.5
    intruder_spawn_zone: str = "north"

    # Dynamic NFZ (comms-denied / airspace scenarios)
    enable_dynamic_nfz: bool = False
    num_nfz_zones: int = 3
    nfz_expansion_rate: float = 0.8
    nfz_max_radius: int = 35

    # Collision termination policy (UAV-ON standard)
    terminate_on_collision: bool = True

    # Debug
    debug: bool = False

    # Optional "escape hatch" for future, without breaking schema
    extra: dict[str, Any] | None = None

    def validate(self) -> None:
        """Raise ValueError if config is inconsistent (V&V style early failure)."""
        if self.map_size < 5:
            raise ValueError("map_size must be >= 5")
        if self.max_altitude < 1:
            raise ValueError("max_altitude must be >= 1")
        if not (0.0 <= self.building_density <= 1.0):
            raise ValueError("building_density must be in [0,1]")
        if not (0 <= self.building_level <= self.max_altitude):
            raise ValueError("building_level must be in [0, max_altitude] for levels-only model")
        if not (1 <= self.start_altitude <= self.max_altitude):
            raise ValueError("start_altitude must be in [1, max_altitude]")
        if not (0 <= self.safe_altitude <= self.max_altitude):
            raise ValueError("safe_altitude must be in [0, max_altitude]")
        if self.min_start_goal_l1 < 1:
            raise ValueError("min_start_goal_l1 must be >= 1")
        if self.downtown_window < 3 or self.downtown_window % 2 == 0:
            raise ValueError("downtown_window must be odd and >= 3")
        if self.spawn_clearance < 0:
            raise ValueError("spawn_clearance must be >= 0")
        if self.no_fly_radius < 0:
            raise ValueError("no_fly_radius must be >= 0")
        if self.map_source not in ("synthetic", "osm"):
            raise ValueError("map_source must be 'synthetic' or 'osm'")
        if self.map_source == "osm" and self.osm_tile_id is None:
            raise ValueError("osm_tile_id is required when map_source is 'osm'")
        if self.fire_ignition_points < 1:
            raise ValueError("fire_ignition_points must be >= 1")
        if self.num_emergency_vehicles < 0:
            raise ValueError("num_emergency_vehicles must be >= 0")
        if not (0.0 <= self.wind_speed <= 1.0):
            raise ValueError("wind_speed must be in [0, 1]")
        if self.enable_fire and self.map_source != "osm":
            raise ValueError("enable_fire requires map_source='osm' (needs landuse data)")
        if self.fire_blocks_movement and not self.enable_fire:
            raise ValueError("fire_blocks_movement requires enable_fire=True")
        if self.traffic_blocks_movement and not self.enable_traffic:
            raise ValueError("traffic_blocks_movement requires enable_traffic=True")
        if self.enable_intruders and self.num_intruders < 1:
            raise ValueError("enable_intruders requires num_intruders >= 1")
        if self.enable_intruders and self.intruder_spawn_zone not in ("north", "south", "east", "west"):
            raise ValueError("intruder_spawn_zone must be north/south/east/west")
        if self.enable_dynamic_nfz and self.num_nfz_zones < 1:
            raise ValueError("enable_dynamic_nfz requires num_nfz_zones >= 1")

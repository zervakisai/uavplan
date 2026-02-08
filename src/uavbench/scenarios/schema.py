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

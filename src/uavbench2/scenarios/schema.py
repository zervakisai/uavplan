"""Scenario configuration schema (SC-2).

ScenarioConfig is a frozen dataclass with validation on construction.
All enums get their own class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Domain(str, Enum):
    URBAN = "urban"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class MissionType(str, Enum):
    CIVIL_PROTECTION = "civil_protection"
    MARITIME_DOMAIN = "maritime_domain"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"


class Regime(str, Enum):
    NATURALISTIC = "naturalistic"
    STRESS_TEST = "stress_test"


# ---------------------------------------------------------------------------
# ScenarioConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioConfig:
    """Frozen scenario configuration.  Enforces SC-2.

    Required fields must be supplied at construction.
    Optional fields have safe defaults matching easy/static scenarios.
    """

    # Identity
    name: str
    mission_type: MissionType
    difficulty: Difficulty
    domain: Domain = Domain.URBAN
    regime: Regime = Regime.NATURALISTIC
    paper_track: Literal["static", "dynamic"] = "static"

    # Map
    map_size: int = 500
    map_source: Literal["osm", "synthetic"] = "synthetic"
    osm_tile_id: str | None = None
    building_density: float = 0.18

    # Start / Goal
    fixed_start_xy: tuple[int, int] | None = None
    fixed_goal_xy: tuple[int, int] | None = None
    min_start_goal_l1: int = 100

    # Dynamic layers
    enable_fire: bool = False
    enable_traffic: bool = False
    enable_dynamic_nfz: bool = False
    fire_blocks_movement: bool = False
    traffic_blocks_movement: bool = False
    terminate_on_collision: bool = True

    # Episode limits
    max_episode_steps: int | None = None  # None → 4 * map_size

    # Event timing (dynamic track only)
    event_t1: int | None = None
    event_t2: int | None = None

    # Wind
    wind_speed: float = 0.2

    # Fire
    fire_ignition_points: int = 0

    # Traffic
    num_emergency_vehicles: int = 0

    # NFZ
    num_nfz_zones: int = 0

    # Forced replans
    force_replan_count: int = 0

    # Planning budgets
    plan_budget_static_ms: float = 500.0
    plan_budget_dynamic_ms: float = 200.0

    # Replan config
    replan_every_steps: int = 6
    max_replans_per_episode: int = 2000

    # Comms
    comms_dropout_prob: float = 0.0
    comms_latency_steps: int = 0

    def __post_init__(self) -> None:
        """Validate fields on construction (SC-2)."""
        if self.map_size < 1:
            raise ValueError(f"map_size must be >= 1, got {self.map_size}")
        if not 0.0 <= self.building_density <= 1.0:
            raise ValueError(
                f"building_density must be in [0, 1], got {self.building_density}"
            )
        if self.event_t1 is not None and self.event_t2 is not None:
            if self.event_t1 >= self.event_t2:
                raise ValueError(
                    f"event_t1 ({self.event_t1}) must be < event_t2 ({self.event_t2})"
                )

    @property
    def effective_max_steps(self) -> int:
        """Return max episode steps: explicit or 4 * map_size (EN-4)."""
        if self.max_episode_steps is not None:
            return self.max_episode_steps
        return 4 * self.map_size

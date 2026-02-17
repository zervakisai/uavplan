"""Scenario registry: central catalog of all benchmark scenarios.

This module provides:
- SCENARIO_REGISTRY: dict mapping scenario_id -> metadata (loaded dynamically from YAML)
- list_scenarios(): list all scenario IDs
- get_scenario_metadata(): lookup metadata for a scenario
- Various filter functions: by mission type, regime, difficulty, dynamics, etc.

9 government-ready mission scenarios are loaded dynamically from YAML config files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
from pathlib import Path
import yaml  # type: ignore[import-untyped]

from uavbench.scenarios.schema import MissionType, Regime


@dataclass(frozen=True)
class ScenarioMetadata:
    """Metadata for a scenario (used for organization, filtering, reporting)."""
    scenario_id: str
    mission_type: MissionType
    regime: Regime
    paper_track: Literal["static", "dynamic"]
    tile: Optional[str]
    has_fire: bool
    has_traffic: bool
    difficulty: str
    description: str


def _load_all_scenarios() -> dict[str, ScenarioMetadata]:
    """Dynamically load all scenarios from YAML config files."""
    
    registry = {}
    configs_dir = Path(__file__).parent / "configs"
    
    if not configs_dir.exists():
        raise FileNotFoundError(f"Scenarios directory not found: {configs_dir}")
    
    # Load each YAML file
    for yaml_path in sorted(configs_dir.glob("*.yaml")):
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        
        scenario_id = config.get("scenario_id")
        if not scenario_id:
            continue
        
        # Extract metadata from config
        mission_type_str = config.get("mission_type", "POINT_TO_POINT")
        regime_str = config.get("regime", "NATURALISTIC")
        difficulty = config.get("difficulty", "EASY").upper()
        regime_str_norm = str(regime_str).lower()
        paper_track_raw = str(config.get("paper_track", "dynamic" if regime_str_norm == "stress_test" else "static")).lower()
        paper_track: Literal["static", "dynamic"] = "dynamic" if paper_track_raw == "dynamic" else "static"
        
        try:
            mission_type = MissionType(mission_type_str)
        except ValueError:
            mission_type = MissionType.POINT_TO_POINT
        
        try:
            regime = Regime(regime_str)
        except ValueError:
            regime = Regime.NATURALISTIC
        
        tile = config.get("tile_name")
        if tile and tile.startswith("synthetic"):
            tile = None
        
        enable_fire = config.get("enable_fire", False)
        enable_traffic = config.get("enable_traffic", False)
        
        description = config.get("description", "No description")
        
        registry[scenario_id] = ScenarioMetadata(
            scenario_id=scenario_id,
            mission_type=mission_type,
            regime=regime,
            paper_track=paper_track,
            tile=tile,
            has_fire=enable_fire,
            has_traffic=enable_traffic,
            difficulty=difficulty,
            description=description,
        )
    
    return registry


# Load scenarios on module import
SCENARIO_REGISTRY = _load_all_scenarios()


def list_scenarios() -> list[str]:
    """Get list of all scenario IDs."""
    return sorted(SCENARIO_REGISTRY.keys())


def list_scenarios_by_mission(mission_type: MissionType) -> list[str]:
    """Get scenarios filtered by mission type."""
    return sorted([
        sid for sid, meta in SCENARIO_REGISTRY.items()
        if meta.mission_type == mission_type
    ])


def list_scenarios_by_regime(regime: Regime) -> list[str]:
    """Get scenarios filtered by regime."""
    return sorted([
        sid for sid, meta in SCENARIO_REGISTRY.items()
        if meta.regime == regime
    ])


def list_scenarios_by_track(track: str) -> list[str]:
    """Get scenarios filtered by paper track ("static" or "dynamic")."""
    track = track.strip().lower()
    if track not in {"static", "dynamic"}:
        raise ValueError("track must be 'static' or 'dynamic'")
    return sorted([
        sid for sid, meta in SCENARIO_REGISTRY.items()
        if meta.paper_track == track
    ])


def list_scenarios_by_difficulty(difficulty: str) -> list[str]:
    """Get scenarios filtered by difficulty."""
    difficulty = difficulty.upper()
    return sorted([
        sid for sid, meta in SCENARIO_REGISTRY.items()
        if meta.difficulty.upper() == difficulty
    ])


def list_scenarios_with_fire() -> list[str]:
    """Get scenarios with fire dynamics."""
    return sorted([
        sid for sid, meta in SCENARIO_REGISTRY.items()
        if meta.has_fire
    ])


def list_scenarios_with_traffic() -> list[str]:
    """Get scenarios with traffic dynamics."""
    return sorted([
        sid for sid, meta in SCENARIO_REGISTRY.items()
        if meta.has_traffic
    ])


def list_scenarios_with_dynamics() -> list[str]:
    """Get scenarios with any dynamics (fire or traffic)."""
    return sorted([
        sid for sid, meta in SCENARIO_REGISTRY.items()
        if meta.has_fire or meta.has_traffic
    ])


def list_scenarios_by_tile(tile: str) -> list[str]:
    """Get scenarios for a specific tile."""
    return sorted([
        sid for sid, meta in SCENARIO_REGISTRY.items()
        if meta.tile and meta.tile.lower() == tile.lower()
    ])


def list_scenarios_synthetic() -> list[str]:
    """Get synthetic (non-OSM) scenarios."""
    return sorted([
        sid for sid, meta in SCENARIO_REGISTRY.items()
        if meta.tile is None
    ])


def get_scenario_metadata(scenario_id: str) -> Optional[ScenarioMetadata]:
    """Get metadata for a specific scenario."""
    return SCENARIO_REGISTRY.get(scenario_id)


def print_scenario_registry() -> None:
    """Pretty-print the scenario registry."""
    
    print("\n" + "=" * 140)
    print(f"{'UAVBench Scenario Registry':^140}")
    print(f"{f'({len(SCENARIO_REGISTRY)} scenarios — 3 missions × 3 difficulties)':^140}")
    print("=" * 140)
    print()
    
    # Count statistics
    mission_types = set()
    regimes = set()
    difficulties = set()
    tiles = set()
    fire_count = 0
    traffic_count = 0
    
    for meta in SCENARIO_REGISTRY.values():
        mission_types.add(meta.mission_type.value)
        regimes.add(meta.regime.value)
        difficulties.add(meta.difficulty)
        if meta.tile:
            tiles.add(meta.tile)
        if meta.has_fire:
            fire_count += 1
        if meta.has_traffic:
            traffic_count += 1
    
    # Print scenarios in order
    print(f"{'#':>3} | {'Scenario ID':<55} | {'Mission Type':<25} | {'Difficulty':<8} | {'Regime':<15} | {'Track':<8} | {'Dynamics':<20}")
    print("-" * 140)
    
    for i, scenario_id in enumerate(list_scenarios(), 1):
        meta = SCENARIO_REGISTRY[scenario_id]
        
        dynamics = []
        if meta.has_fire:
            dynamics.append("🔥 Fire")
        if meta.has_traffic:
            dynamics.append("🚗 Traffic")
        dynamics_str = " + ".join(dynamics) if dynamics else "-"
        
        print(
            f"{i:3d} | {scenario_id:<55} | {meta.mission_type.value:<25} | "
            f"{meta.difficulty:<8} | {meta.regime.value:<15} | {meta.paper_track:<8} | {dynamics_str:<20}"
        )
    
    print()
    print("=" * 140)
    print(f"Statistics:")
    print(f"  - Total scenarios: {len(SCENARIO_REGISTRY)}")
    print(f"  - Mission types: {len(mission_types)} → {', '.join(sorted(mission_types))}")
    print(f"  - Regimes: {len(regimes)} → {', '.join(sorted(regimes))}")
    print(f"  - Difficulties: {len(difficulties)} → {', '.join(sorted(difficulties))}")
    print(f"  - Tiles: {len(tiles)} → {', '.join(sorted(tiles))}")
    print(f"  - With fire: {fire_count} scenarios")
    print(f"  - With traffic: {traffic_count} scenarios")
    print(f"  - With dynamics: {fire_count + traffic_count} scenarios")
    print("=" * 140 + "\n")


if __name__ == "__main__":
    print_scenario_registry()

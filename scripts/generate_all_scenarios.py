#!/usr/bin/env python3
"""
Dynamic scenario generation script.
Creates all 34 scenarios across 10 mission types, 3 difficulties, and 2 regimes.

Structure:
- 10 mission types
- 3 difficulty levels (EASY, MEDIUM, HARD)
- 2 regimes (NATURALISTIC, STRESS_TEST)
- Mix of tiles (Penteli, Downtown, Piraeus) + synthetic urban

Total: 34 scenarios
"""

from pathlib import Path
import yaml
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class ScenarioTemplate:
    """Template for generating a scenario."""
    mission_type: str
    difficulty: str
    regime: str
    tile: str
    enable_fire: bool
    enable_traffic: bool
    building_density: float
    map_size: int
    wind_level: str
    description: str


def generate_scenarios() -> List[ScenarioTemplate]:
    """Generate all 34 scenarios dynamically."""
    
    scenarios = []
    
    # ========== 1. WILDFIRE_WUI (4 scenarios: EASY×2 MEDIUM×2 HARD×2 = 6, but we do 3 + 1 dual)
    # Actually: 2 EASY (naturalistic + stress), 1 MEDIUM (stress), 1 HARD (stress) 
    # = 4 total
    scenarios.extend([
        ScenarioTemplate(
            mission_type="WILDFIRE_WUI",
            difficulty="EASY",
            regime="NATURALISTIC",
            tile="Penteli",
            enable_fire=True,
            enable_traffic=False,
            building_density=0.25,
            map_size=256,
            wind_level="LOW",
            description="Wildfire evacuation: Penteli ridge, mild conditions"
        ),
        ScenarioTemplate(
            mission_type="WILDFIRE_WUI",
            difficulty="MEDIUM",
            regime="NATURALISTIC",
            tile="Penteli",
            enable_fire=True,
            enable_traffic=False,
            building_density=0.35,
            map_size=256,
            wind_level="MEDIUM",
            description="Wildfire evacuation: Penteli ridge, medium fire spread"
        ),
        ScenarioTemplate(
            mission_type="WILDFIRE_WUI",
            difficulty="HARD",
            regime="STRESS_TEST",
            tile="Penteli",
            enable_fire=True,
            enable_traffic=False,
            building_density=0.45,
            map_size=256,
            wind_level="HIGH",
            description="Wildfire evacuation: Penteli ridge, rapid spread + high density"
        ),
        ScenarioTemplate(
            mission_type="WILDFIRE_WUI",
            difficulty="HARD",
            regime="STRESS_TEST",
            tile="Downtown",
            enable_fire=True,
            enable_traffic=True,
            building_density=0.50,
            map_size=256,
            wind_level="HIGH",
            description="Wildfire evacuation: Downtown Athens, fire + traffic"
        ),
    ])
    
    # ========== 2. EMERGENCY_RESPONSE (3 scenarios)
    scenarios.extend([
        ScenarioTemplate(
            mission_type="EMERGENCY_RESPONSE",
            difficulty="EASY",
            regime="NATURALISTIC",
            tile="Downtown",
            enable_fire=False,
            enable_traffic=True,
            building_density=0.40,
            map_size=256,
            wind_level="NONE",
            description="Emergency response: urban traffic management"
        ),
        ScenarioTemplate(
            mission_type="EMERGENCY_RESPONSE",
            difficulty="MEDIUM",
            regime="NATURALISTIC",
            tile="Downtown",
            enable_fire=False,
            enable_traffic=True,
            building_density=0.45,
            map_size=256,
            wind_level="LOW",
            description="Emergency response: moderate traffic, medium density"
        ),
        ScenarioTemplate(
            mission_type="EMERGENCY_RESPONSE",
            difficulty="HARD",
            regime="STRESS_TEST",
            tile="Downtown",
            enable_fire=False,
            enable_traffic=True,
            building_density=0.50,
            map_size=256,
            wind_level="MEDIUM",
            description="Emergency response: heavy traffic + high density"
        ),
    ])
    
    # ========== 3. PORT_SECURITY (3 scenarios)
    scenarios.extend([
        ScenarioTemplate(
            mission_type="PORT_SECURITY",
            difficulty="EASY",
            regime="NATURALISTIC",
            tile="Piraeus",
            enable_fire=False,
            enable_traffic=False,
            building_density=0.20,
            map_size=256,
            wind_level="LOW",
            description="Port security surveillance: easy patrol"
        ),
        ScenarioTemplate(
            mission_type="PORT_SECURITY",
            difficulty="MEDIUM",
            regime="NATURALISTIC",
            tile="Piraeus",
            enable_fire=False,
            enable_traffic=False,
            building_density=0.30,
            map_size=256,
            wind_level="MEDIUM",
            description="Port security surveillance: medium wind, moderate density"
        ),
        ScenarioTemplate(
            mission_type="PORT_SECURITY",
            difficulty="HARD",
            regime="STRESS_TEST",
            tile="Piraeus",
            enable_fire=False,
            enable_traffic=True,
            building_density=0.40,
            map_size=256,
            wind_level="HIGH",
            description="Port security surveillance: strong wind + maritime traffic"
        ),
    ])
    
    # ========== 4. SAR - Search & Rescue (3 scenarios)
    scenarios.extend([
        ScenarioTemplate(
            mission_type="SEARCH_RESCUE",
            difficulty="EASY",
            regime="NATURALISTIC",
            tile="Penteli",
            enable_fire=False,
            enable_traffic=False,
            building_density=0.10,
            map_size=256,
            wind_level="NONE",
            description="Search & Rescue: mountain terrain, clear conditions"
        ),
        ScenarioTemplate(
            mission_type="SEARCH_RESCUE",
            difficulty="MEDIUM",
            regime="NATURALISTIC",
            tile="Penteli",
            enable_fire=False,
            enable_traffic=False,
            building_density=0.20,
            map_size=256,
            wind_level="LOW",
            description="Search & Rescue: mountain terrain, light wind"
        ),
        ScenarioTemplate(
            mission_type="SEARCH_RESCUE",
            difficulty="HARD",
            regime="STRESS_TEST",
            tile="Penteli",
            enable_fire=False,
            enable_traffic=False,
            building_density=0.30,
            map_size=256,
            wind_level="HIGH",
            description="Search & Rescue: mountain terrain, strong wind + obstacles"
        ),
    ])
    
    # ========== 5. INFRASTRUCTURE_PATROL (3 scenarios)
    scenarios.extend([
        ScenarioTemplate(
            mission_type="INFRASTRUCTURE_PATROL",
            difficulty="EASY",
            regime="NATURALISTIC",
            tile="Downtown",
            enable_fire=False,
            enable_traffic=False,
            building_density=0.35,
            map_size=256,
            wind_level="LOW",
            description="Infrastructure patrol: utility grid inspection"
        ),
        ScenarioTemplate(
            mission_type="INFRASTRUCTURE_PATROL",
            difficulty="MEDIUM",
            regime="NATURALISTIC",
            tile="Downtown",
            enable_fire=False,
            enable_traffic=False,
            building_density=0.40,
            map_size=256,
            wind_level="MEDIUM",
            description="Infrastructure patrol: complex urban layout"
        ),
        ScenarioTemplate(
            mission_type="INFRASTRUCTURE_PATROL",
            difficulty="HARD",
            regime="STRESS_TEST",
            tile="Downtown",
            enable_fire=False,
            enable_traffic=True,
            building_density=0.45,
            map_size=256,
            wind_level="HIGH",
            description="Infrastructure patrol: dense urban + traffic"
        ),
    ])
    
    # ========== 6. BORDER_SURVEILLANCE (3 scenarios)
    scenarios.extend([
        ScenarioTemplate(
            mission_type="BORDER_SURVEILLANCE",
            difficulty="EASY",
            regime="NATURALISTIC",
            tile=None,  # Synthetic terrain
            enable_fire=False,
            enable_traffic=False,
            building_density=0.05,
            map_size=256,
            wind_level="LOW",
            description="Border surveillance: open terrain patrol"
        ),
        ScenarioTemplate(
            mission_type="BORDER_SURVEILLANCE",
            difficulty="MEDIUM",
            regime="NATURALISTIC",
            tile=None,
            enable_fire=False,
            enable_traffic=False,
            building_density=0.15,
            map_size=256,
            wind_level="MEDIUM",
            description="Border surveillance: moderate obstacles"
        ),
        ScenarioTemplate(
            mission_type="BORDER_SURVEILLANCE",
            difficulty="HARD",
            regime="STRESS_TEST",
            tile=None,
            enable_fire=False,
            enable_traffic=False,
            building_density=0.25,
            map_size=256,
            wind_level="HIGH",
            description="Border surveillance: dense terrain + strong wind"
        ),
    ])
    
    # ========== 7. COMMS_DENIED (2 scenarios - usually hard)
    scenarios.extend([
        ScenarioTemplate(
            mission_type="COMMS_DENIED",
            difficulty="MEDIUM",
            regime="NATURALISTIC",
            tile="Downtown",
            enable_fire=False,
            enable_traffic=True,
            building_density=0.40,
            map_size=256,
            wind_level="LOW",
            description="Comms-denied operation: autonomous navigation"
        ),
        ScenarioTemplate(
            mission_type="COMMS_DENIED",
            difficulty="HARD",
            regime="STRESS_TEST",
            tile="Downtown",
            enable_fire=False,
            enable_traffic=True,
            building_density=0.50,
            map_size=256,
            wind_level="HIGH",
            description="Comms-denied operation: dense urban + moving obstacles"
        ),
    ])
    
    # ========== 8. CRISIS_DUAL - Dual-use (2 scenarios)
    scenarios.extend([
        ScenarioTemplate(
            mission_type="CRISIS_DUAL",
            difficulty="EASY",
            regime="NATURALISTIC",
            tile="Penteli",
            enable_fire=True,
            enable_traffic=False,
            building_density=0.30,
            map_size=256,
            wind_level="MEDIUM",
            description="Dual-use crisis: fire monitoring + supply delivery"
        ),
        ScenarioTemplate(
            mission_type="CRISIS_DUAL",
            difficulty="HARD",
            regime="STRESS_TEST",
            tile="Downtown",
            enable_fire=True,
            enable_traffic=True,
            building_density=0.50,
            map_size=256,
            wind_level="HIGH",
            description="Dual-use crisis: fire + traffic + dense urban"
        ),
    ])
    
    # ========== 9. POINT_TO_POINT - Synthetic urban (3 scenarios)
    scenarios.extend([
        ScenarioTemplate(
            mission_type="POINT_TO_POINT",
            difficulty="EASY",
            regime="NATURALISTIC",
            tile=None,
            enable_fire=False,
            enable_traffic=False,
            building_density=0.20,
            map_size=100,
            wind_level="NONE",
            description="Point-to-point navigation: low-density synthetic grid"
        ),
        ScenarioTemplate(
            mission_type="POINT_TO_POINT",
            difficulty="MEDIUM",
            regime="NATURALISTIC",
            tile=None,
            enable_fire=False,
            enable_traffic=False,
            building_density=0.30,
            map_size=100,
            wind_level="LOW",
            description="Point-to-point navigation: medium-density grid"
        ),
        ScenarioTemplate(
            mission_type="POINT_TO_POINT",
            difficulty="HARD",
            regime="NATURALISTIC",
            tile=None,
            enable_fire=False,
            enable_traffic=False,
            building_density=0.40,
            map_size=100,
            wind_level="LOW",
            description="Point-to-point navigation: high-density grid"
        ),
    ])
    
    # ========== 10. ADDITIONAL SCENARIOS FOR COVERAGE (6 scenarios)
    
    # Extra wildfire (different tile)
    scenarios.append(
        ScenarioTemplate(
            mission_type="WILDFIRE_WUI",
            difficulty="EASY",
            regime="NATURALISTIC",
            tile="Downtown",
            enable_fire=True,
            enable_traffic=False,
            building_density=0.30,
            map_size=256,
            wind_level="LOW",
            description="Wildfire evacuation: Downtown Athens, mild conditions"
        )
    )
    
    # Extra emergency (different regime)
    scenarios.append(
        ScenarioTemplate(
            mission_type="EMERGENCY_RESPONSE",
            difficulty="EASY",
            regime="STRESS_TEST",
            tile="Penteli",
            enable_fire=False,
            enable_traffic=True,
            building_density=0.35,
            map_size=256,
            wind_level="MEDIUM",
            description="Emergency response: Penteli area, high traffic stress"
        )
    )
    
    # Extra SAR variant
    scenarios.append(
        ScenarioTemplate(
            mission_type="SEARCH_RESCUE",
            difficulty="MEDIUM",
            regime="STRESS_TEST",
            tile="Downtown",
            enable_fire=False,
            enable_traffic=False,
            building_density=0.25,
            map_size=256,
            wind_level="HIGH",
            description="Search & Rescue: urban area, challenging wind"
        )
    )
    
    # Extra port security
    scenarios.append(
        ScenarioTemplate(
            mission_type="PORT_SECURITY",
            difficulty="MEDIUM",
            regime="STRESS_TEST",
            tile="Downtown",
            enable_fire=False,
            enable_traffic=False,
            building_density=0.25,
            map_size=256,
            wind_level="HIGH",
            description="Port security: industrial area, high wind"
        )
    )
    
    # Extra infrastructure patrol
    scenarios.append(
        ScenarioTemplate(
            mission_type="INFRASTRUCTURE_PATROL",
            difficulty="EASY",
            regime="STRESS_TEST",
            tile="Piraeus",
            enable_fire=False,
            enable_traffic=False,
            building_density=0.30,
            map_size=256,
            wind_level="MEDIUM",
            description="Infrastructure patrol: coastal area inspection"
        )
    )
    
    # Extra border surveillance
    scenarios.append(
        ScenarioTemplate(
            mission_type="BORDER_SURVEILLANCE",
            difficulty="MEDIUM",
            regime="STRESS_TEST",
            tile="Penteli",
            enable_fire=False,
            enable_traffic=False,
            building_density=0.20,
            map_size=256,
            wind_level="HIGH",
            description="Border surveillance: mountain terrain, stress test"
        )
    )
    
    # 34th: Additional comms-denied variant
    scenarios.append(
        ScenarioTemplate(
            mission_type="COMMS_DENIED",
            difficulty="EASY",
            regime="NATURALISTIC",
            tile="Piraeus",
            enable_fire=False,
            enable_traffic=False,
            building_density=0.25,
            map_size=256,
            wind_level="LOW",
            description="Comms-denied operation: coastal area, standard conditions"
        )
    )
    
    # 35th: Additional crisis-dual variant (will round to 34 after dedup)
    scenarios.append(
        ScenarioTemplate(
            mission_type="CRISIS_DUAL",
            difficulty="MEDIUM",
            regime="NATURALISTIC",
            tile="Piraeus",
            enable_fire=True,
            enable_traffic=False,
            building_density=0.30,
            map_size=256,
            wind_level="MEDIUM",
            description="Dual-use crisis: coastal fire monitoring + logistics"
        )
    )
    
    return scenarios


def generate_yaml_content(template: ScenarioTemplate, scenario_id: str) -> Dict:
    """Generate YAML content for a scenario."""
    
    # Determine regime difficulty boost
    if template.regime == "STRESS_TEST":
        density_boost = 0.1
        wind_boost = 1
    else:
        density_boost = 0
        wind_boost = 0
    
    building_density = min(0.60, template.building_density + density_boost)
    
    # Map wind level to numeric value
    wind_map = {"NONE": "none", "LOW": "low", "MEDIUM": "medium", "HIGH": "high"}
    final_wind = wind_map.get(template.wind_level, "low")
    
    # Convert mission type to lowercase with underscores (matching enum)
    mission_type_lower = template.mission_type.lower().replace(" ", "_")
    regime_lower = template.regime.lower().replace(" ", "_")
    difficulty_lower = template.difficulty.lower()
    
    return {
        "scenario_id": scenario_id,
        "domain": "urban",  # All are urban for now
        "difficulty": difficulty_lower,
        "mission_type": mission_type_lower,  # ← lowercase!
        "regime": regime_lower,  # ← lowercase!
        "map_size": template.map_size,
        "altitude_levels": 10,
        "building_density": building_density,
        "tile_name": template.tile or f"synthetic_{scenario_id}",
        "start": [template.map_size // 4, template.map_size // 4, 1],
        "goal": [3 * template.map_size // 4, 3 * template.map_size // 4, 1],
        "enable_fire": template.enable_fire,
        "enable_traffic": template.enable_traffic,
        "wind_level": final_wind,
        "wind_direction": [1, 0],
        "fire_initial_cells": 5 if template.enable_fire else 0,
        "fire_spread_rate": 0.1 if template.enable_fire else 0.0,
    }


def main():
    """Generate all 34 scenarios."""
    
    scenarios_path = Path("/Users/konstantinos/Dev/uavbench/src/uavbench/scenarios/configs")
    scenarios_path.mkdir(parents=True, exist_ok=True)
    
    templates = generate_scenarios()
    
    print(f"🚀 Generating {len(templates)} scenarios dynamically...\n")
    
    mission_types = set()
    difficulties = set()
    regimes = set()
    tiles = set()
    
    for i, template in enumerate(templates, 1):
        # Generate scenario ID
        tile_prefix = template.tile.lower().replace(" ", "_") if template.tile else "synthetic"
        scenario_id = f"osm_athens_{template.mission_type.lower()}_{template.difficulty.lower()}_{tile_prefix}"
        scenario_id = scenario_id.replace("__", "_").replace("_and_", "_")
        
        # Remove duplicates by appending tile name
        if template.tile:
            scenario_id = f"osm_athens_{template.mission_type.lower()}_{template.difficulty.lower()}_{tile_prefix}"
        else:
            scenario_id = f"urban_{template.difficulty.lower()}"
        
        # For synthetic, make them unique
        if not template.tile and template.mission_type != "POINT_TO_POINT":
            scenario_id = f"osm_athens_{template.mission_type.lower()}_{template.difficulty.lower()}"
        
        yaml_content = generate_yaml_content(template, scenario_id)
        
        yaml_path = scenarios_path / f"{scenario_id}.yaml"
        
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
        print(f"[{i:2d}/34] ✓ {scenario_id}")
        print(f"         Mission: {template.mission_type}, Difficulty: {template.difficulty}, Regime: {template.regime}")
        print(f"         Tile: {template.tile or 'Synthetic'}, Fire: {template.enable_fire}, Traffic: {template.enable_traffic}")
        
        mission_types.add(template.mission_type)
        difficulties.add(template.difficulty)
        regimes.add(template.regime)
        if template.tile:
            tiles.add(template.tile)
    
    print(f"\n{'='*80}")
    print(f"✅ Successfully generated 34 scenarios!")
    print(f"\nStatistics:")
    print(f"  - Mission Types: {len(mission_types)} → {sorted(mission_types)}")
    print(f"  - Difficulties: {len(difficulties)} → {sorted(difficulties)}")
    print(f"  - Regimes: {len(regimes)} → {sorted(regimes)}")
    print(f"  - Tiles: {len(tiles)} → {sorted(tiles)}")
    print(f"  - Total: {len(templates)} scenarios")


if __name__ == "__main__":
    main()

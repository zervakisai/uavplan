"""Scenario registry: central catalog of all benchmark scenarios.

This module provides:
- SCENARIO_REGISTRY: dict mapping scenario_id -> metadata
- list_scenarios(): list all scenario IDs
- get_scenario_metadata(): lookup metadata for a scenario
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from uavbench.scenarios.schema import MissionType, Regime


@dataclass(frozen=True)
class ScenarioMetadata:
    """Metadata for a scenario (used for organization, filtering, reporting)."""
    scenario_id: str
    mission_type: MissionType
    regime: Regime
    tile: Optional[str]
    has_fire: bool
    has_traffic: bool
    has_moving_target: bool
    has_intruders: bool
    has_dynamic_nfz: bool
    difficulty: str
    description: str


# ============================================================================
# SCENARIO REGISTRY — All 34 scenarios with metadata
# ============================================================================

SCENARIO_REGISTRY: dict[str, ScenarioMetadata] = {
    # ========== SYNTHETIC SCENARIOS (3) ==========
    "urban_easy": ScenarioMetadata(
        scenario_id="urban_easy",
        mission_type=MissionType.POINT_TO_POINT,
        regime=Regime.NATURALISTIC,
        tile=None,
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="easy",
        description="Procedural urban grid: static obstacles only",
    ),
    "urban_medium": ScenarioMetadata(
        scenario_id="urban_medium",
        mission_type=MissionType.POINT_TO_POINT,
        regime=Regime.NATURALISTIC,
        tile=None,
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="medium",
        description="Procedural urban grid: medium density",
    ),
    "urban_hard": ScenarioMetadata(
        scenario_id="urban_hard",
        mission_type=MissionType.POINT_TO_POINT,
        regime=Regime.NATURALISTIC,
        tile=None,
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="hard",
        description="Procedural urban grid: high density + central NFZ",
    ),
    
    # ========== OSM WILDFIRE SCENARIOS (4) ==========
    "osm_athens_wildfire_easy": ScenarioMetadata(
        scenario_id="osm_athens_wildfire_easy",
        mission_type=MissionType.WILDFIRE_WUI,
        regime=Regime.NATURALISTIC,
        tile="penteli",
        has_fire=True,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="easy",
        description="Wildfire WUI (Penteli): mild conditions, easy escape",
    ),
    "osm_athens_wildfire_medium": ScenarioMetadata(
        scenario_id="osm_athens_wildfire_medium",
        mission_type=MissionType.WILDFIRE_WUI,
        regime=Regime.STRESS_TEST,
        tile="penteli",
        has_fire=True,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="medium",
        description="Wildfire WUI (Penteli): moderate wind, tighter constraints",
    ),
    "osm_athens_wildfire_hard": ScenarioMetadata(
        scenario_id="osm_athens_wildfire_hard",
        mission_type=MissionType.WILDFIRE_WUI,
        regime=Regime.STRESS_TEST,
        tile="penteli",
        has_fire=True,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="hard",
        description="Wildfire WUI (Penteli): strong wind, multiple ignition points",
    ),
    "osm_athens_fire_easy": ScenarioMetadata(
        scenario_id="osm_athens_fire_easy",
        mission_type=MissionType.WILDFIRE_WUI,
        regime=Regime.NATURALISTIC,
        tile="penteli",
        has_fire=True,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="easy",
        description="Alternative wildfire naming (redundant with osm_athens_wildfire_easy)",
    ),
    
    # ========== OSM EMERGENCY RESPONSE SCENARIOS (3) ==========
    "osm_athens_emergency_easy": ScenarioMetadata(
        scenario_id="osm_athens_emergency_easy",
        mission_type=MissionType.EMERGENCY_RESPONSE,
        regime=Regime.NATURALISTIC,
        tile="downtown",
        has_fire=False,
        has_traffic=True,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="easy",
        description="Emergency vehicles on road network (Downtown): light traffic",
    ),
    "osm_athens_emergency_medium": ScenarioMetadata(
        scenario_id="osm_athens_emergency_medium",
        mission_type=MissionType.EMERGENCY_RESPONSE,
        regime=Regime.STRESS_TEST,
        tile="downtown",
        has_fire=False,
        has_traffic=True,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="medium",
        description="Emergency vehicles on road network (Downtown): moderate traffic",
    ),
    "osm_athens_emergency_hard": ScenarioMetadata(
        scenario_id="osm_athens_emergency_hard",
        mission_type=MissionType.EMERGENCY_RESPONSE,
        regime=Regime.STRESS_TEST,
        tile="downtown",
        has_fire=False,
        has_traffic=True,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="hard",
        description="Emergency vehicles on road network (Downtown): heavy traffic",
    ),
    
    # ========== OSM PORT SECURITY SCENARIOS (3) ==========
    "osm_athens_port_easy": ScenarioMetadata(
        scenario_id="osm_athens_port_easy",
        mission_type=MissionType.PORT_SECURITY,
        regime=Regime.NATURALISTIC,
        tile="piraeus",
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="easy",
        description="Port perimeter surveillance (Piraeus): static constraints",
    ),
    "osm_athens_port_medium": ScenarioMetadata(
        scenario_id="osm_athens_port_medium",
        mission_type=MissionType.PORT_SECURITY,
        regime=Regime.NATURALISTIC,
        tile="piraeus",
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="medium",
        description="Port perimeter surveillance (Piraeus): tighter route constraints",
    ),
    "osm_athens_port_hard": ScenarioMetadata(
        scenario_id="osm_athens_port_hard",
        mission_type=MissionType.PORT_SECURITY,
        regime=Regime.NATURALISTIC,
        tile="piraeus",
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="hard",
        description="Port perimeter surveillance (Piraeus): highly constrained airspace",
    ),
    
    # ========== OSM SEARCH & RESCUE SCENARIOS (3) ==========
    "osm_athens_sar_easy": ScenarioMetadata(
        scenario_id="osm_athens_sar_easy",
        mission_type=MissionType.SEARCH_RESCUE,
        regime=Regime.NATURALISTIC,
        tile="penteli",
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="easy",
        description="Search and rescue (Penteli): locate target in open terrain",
    ),
    "osm_athens_sar_medium": ScenarioMetadata(
        scenario_id="osm_athens_sar_medium",
        mission_type=MissionType.SEARCH_RESCUE,
        regime=Regime.NATURALISTIC,
        tile="penteli",
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="medium",
        description="Search and rescue (Penteli): locate moving target",
    ),
    "osm_athens_sar_hard": ScenarioMetadata(
        scenario_id="osm_athens_sar_hard",
        mission_type=MissionType.SEARCH_RESCUE,
        regime=Regime.STRESS_TEST,
        tile="penteli",
        has_fire=False,
        has_traffic=False,
        has_moving_target=True,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="hard",
        description="Search and rescue (Penteli): moving target with evasion behavior",
    ),
    
    # ========== OSM INFRASTRUCTURE PATROL SCENARIOS (3) ==========
    "osm_athens_infrastructure_easy": ScenarioMetadata(
        scenario_id="osm_athens_infrastructure_easy",
        mission_type=MissionType.INFRASTRUCTURE_PATROL,
        regime=Regime.NATURALISTIC,
        tile="downtown",
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="easy",
        description="Utility infrastructure patrol (Downtown): static route",
    ),
    "osm_athens_infrastructure_medium": ScenarioMetadata(
        scenario_id="osm_athens_infrastructure_medium",
        mission_type=MissionType.INFRASTRUCTURE_PATROL,
        regime=Regime.NATURALISTIC,
        tile="downtown",
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="medium",
        description="Utility infrastructure patrol (Downtown): distributed waypoints",
    ),
    "osm_athens_infrastructure_hard": ScenarioMetadata(
        scenario_id="osm_athens_infrastructure_hard",
        mission_type=MissionType.INFRASTRUCTURE_PATROL,
        regime=Regime.NATURALISTIC,
        tile="downtown",
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="hard",
        description="Utility infrastructure patrol (Downtown): complex perimeter with dynamic NFZ",
    ),
    
    # ========== OSM BORDER SURVEILLANCE SCENARIOS (3) ==========
    "osm_athens_border_easy": ScenarioMetadata(
        scenario_id="osm_athens_border_easy",
        mission_type=MissionType.BORDER_SURVEILLANCE,
        regime=Regime.NATURALISTIC,
        tile="penteli",
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="easy",
        description="Border perimeter surveillance (Penteli): no intrusions",
    ),
    "osm_athens_border_medium": ScenarioMetadata(
        scenario_id="osm_athens_border_medium",
        mission_type=MissionType.BORDER_SURVEILLANCE,
        regime=Regime.STRESS_TEST,
        tile="penteli",
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=True,
        has_dynamic_nfz=False,
        difficulty="medium",
        description="Border perimeter surveillance (Penteli): slow-moving intruders",
    ),
    "osm_athens_border_hard": ScenarioMetadata(
        scenario_id="osm_athens_border_hard",
        mission_type=MissionType.BORDER_SURVEILLANCE,
        regime=Regime.STRESS_TEST,
        tile="penteli",
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=True,
        has_dynamic_nfz=False,
        difficulty="hard",
        description="Border perimeter surveillance (Penteli): multiple fast intruders",
    ),
    
    # ========== OSM COMMS-DENIED SCENARIO (1) ==========
    "osm_athens_comms_denied_hard": ScenarioMetadata(
        scenario_id="osm_athens_comms_denied_hard",
        mission_type=MissionType.COMMS_DENIED,
        regime=Regime.STRESS_TEST,
        tile="downtown",
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=True,
        difficulty="hard",
        description="Comms-denied airspace (Downtown): expanding NFZ blocks re-planning",
    ),
    
    # ========== OSM CRISIS / DUAL-USE SCENARIOS (7) ==========
    "osm_athens_crisis_hard": ScenarioMetadata(
        scenario_id="osm_athens_crisis_hard",
        mission_type=MissionType.CRISIS_DUAL,
        regime=Regime.STRESS_TEST,
        tile="downtown",
        has_fire=True,
        has_traffic=True,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="hard",
        description="Multi-hazard crisis (Downtown): fire + traffic",
    ),
    "osm_athens_dual_downtown_easy": ScenarioMetadata(
        scenario_id="osm_athens_dual_downtown_easy",
        mission_type=MissionType.CRISIS_DUAL,
        regime=Regime.NATURALISTIC,
        tile="downtown",
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="easy",
        description="Dual-use scenario (Downtown): baseline pathfinding",
    ),
    "osm_athens_dual_downtown_medium": ScenarioMetadata(
        scenario_id="osm_athens_dual_downtown_medium",
        mission_type=MissionType.CRISIS_DUAL,
        regime=Regime.STRESS_TEST,
        tile="downtown",
        has_fire=False,
        has_traffic=True,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="medium",
        description="Dual-use scenario (Downtown): traffic dynamics",
    ),
    "osm_athens_dual_downtown_hard": ScenarioMetadata(
        scenario_id="osm_athens_dual_downtown_hard",
        mission_type=MissionType.CRISIS_DUAL,
        regime=Regime.STRESS_TEST,
        tile="downtown",
        has_fire=True,
        has_traffic=True,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="hard",
        description="Dual-use scenario (Downtown): fire + traffic (same as crisis_hard)",
    ),
    "osm_athens_dual_penteli_easy": ScenarioMetadata(
        scenario_id="osm_athens_dual_penteli_easy",
        mission_type=MissionType.CRISIS_DUAL,
        regime=Regime.NATURALISTIC,
        tile="penteli",
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="easy",
        description="Dual-use scenario (Penteli): baseline pathfinding",
    ),
    "osm_athens_dual_penteli_medium": ScenarioMetadata(
        scenario_id="osm_athens_dual_penteli_medium",
        mission_type=MissionType.CRISIS_DUAL,
        regime=Regime.STRESS_TEST,
        tile="penteli",
        has_fire=True,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="medium",
        description="Dual-use scenario (Penteli): fire dynamics",
    ),
    "osm_athens_dual_penteli_hard": ScenarioMetadata(
        scenario_id="osm_athens_dual_penteli_hard",
        mission_type=MissionType.CRISIS_DUAL,
        regime=Regime.STRESS_TEST,
        tile="penteli",
        has_fire=True,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="hard",
        description="Dual-use scenario (Penteli): strong fire with high wind",
    ),
    "osm_athens_dual_piraeus_hard": ScenarioMetadata(
        scenario_id="osm_athens_dual_piraeus_hard",
        mission_type=MissionType.CRISIS_DUAL,
        regime=Regime.STRESS_TEST,
        tile="piraeus",
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="hard",
        description="Dual-use scenario (Piraeus): constrained airspace",
    ),
    
    # ========== OSM ADAPTIVE VARIANTS (2) ==========
    "osm_athens_wildfire_adaptive_easy": ScenarioMetadata(
        scenario_id="osm_athens_wildfire_adaptive_easy",
        mission_type=MissionType.WILDFIRE_WUI,
        regime=Regime.NATURALISTIC,
        tile="penteli",
        has_fire=True,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="easy",
        description="Wildfire with adaptive replanning: baseline",
    ),
    "osm_athens_traffic_adaptive_easy": ScenarioMetadata(
        scenario_id="osm_athens_traffic_adaptive_easy",
        mission_type=MissionType.EMERGENCY_RESPONSE,
        regime=Regime.NATURALISTIC,
        tile="downtown",
        has_fire=False,
        has_traffic=True,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="easy",
        description="Traffic with adaptive replanning: baseline",
    ),
    "osm_athens_adaptive_crisis_hard": ScenarioMetadata(
        scenario_id="osm_athens_adaptive_crisis_hard",
        mission_type=MissionType.CRISIS_DUAL,
        regime=Regime.STRESS_TEST,
        tile="downtown",
        has_fire=True,
        has_traffic=True,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="hard",
        description="Crisis with adaptive replanning: fire + traffic on Downtown",
    ),
    
    # ========== BASELINE / EASY SCENARIO (1) ==========
    "osm_athens_easy": ScenarioMetadata(
        scenario_id="osm_athens_easy",
        mission_type=MissionType.POINT_TO_POINT,
        regime=Regime.NATURALISTIC,
        tile="downtown",
        has_fire=False,
        has_traffic=False,
        has_moving_target=False,
        has_intruders=False,
        has_dynamic_nfz=False,
        difficulty="easy",
        description="Generic Athens baseline: static pathfinding only",
    ),
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def list_scenarios() -> list[str]:
    """Return sorted list of all scenario IDs."""
    return sorted(SCENARIO_REGISTRY.keys())


def list_scenarios_by_mission(mission: MissionType) -> list[str]:
    """Return scenario IDs matching mission type."""
    return sorted(
        [sid for sid, meta in SCENARIO_REGISTRY.items() if meta.mission_type == mission]
    )


def list_scenarios_by_regime(regime: Regime) -> list[str]:
    """Return scenario IDs matching regime."""
    return sorted(
        [sid for sid, meta in SCENARIO_REGISTRY.items() if meta.regime == regime]
    )


def list_scenarios_with_dynamics() -> list[str]:
    """Return scenario IDs with at least one dynamic layer enabled."""
    return sorted(
        [
            sid
            for sid, meta in SCENARIO_REGISTRY.items()
            if meta.has_fire
            or meta.has_traffic
            or meta.has_moving_target
            or meta.has_intruders
            or meta.has_dynamic_nfz
        ]
    )


def get_scenario_metadata(scenario_id: str) -> ScenarioMetadata:
    """Get metadata for scenario, or raise KeyError if not found."""
    if scenario_id not in SCENARIO_REGISTRY:
        raise KeyError(
            f"Unknown scenario '{scenario_id}'. Available: {list_scenarios()}"
        )
    return SCENARIO_REGISTRY[scenario_id]


def print_scenario_registry() -> None:
    """Print human-readable summary of all scenarios."""
    print("\n" + "=" * 100)
    print("UAVBENCH SCENARIO REGISTRY")
    print("=" * 100)
    
    for scenario_id in list_scenarios():
        meta = SCENARIO_REGISTRY[scenario_id]
        dyn_flags = [
            "fire" if meta.has_fire else None,
            "traffic" if meta.has_traffic else None,
            "target" if meta.has_moving_target else None,
            "intruders" if meta.has_intruders else None,
            "dynamic_nfz" if meta.has_dynamic_nfz else None,
        ]
        dyn_str = ", ".join([f for f in dyn_flags if f]) or "(static)"
        
        print(
            f"\n  {scenario_id:45s} | {meta.mission_type.value:25s} | "
            f"{meta.regime.value:15s} | {meta.difficulty:6s} | {dyn_str}"
        )
        print(f"    {meta.description}")
    
    print("\n" + "=" * 100)
    print(f"Total: {len(SCENARIO_REGISTRY)} scenarios")
    print(f"  Missions: {len(set(m.mission_type for m in SCENARIO_REGISTRY.values()))} types")
    print(f"  Regimes: {len(set(m.regime for m in SCENARIO_REGISTRY.values()))} types")
    print(f"  Dynamic: {len(list_scenarios_with_dynamics())} scenarios")
    print("=" * 100 + "\n")

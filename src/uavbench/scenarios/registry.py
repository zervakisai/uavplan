"""Scenario registry (SC-4).

Provides filter functions for the 9 government mission scenarios.
"""

from __future__ import annotations

from pathlib import Path

from uavbench.scenarios.loader import load_scenario
from uavbench.scenarios.schema import ScenarioConfig


_CONFIGS_DIR = Path(__file__).parent / "configs"

# All scenario IDs (derived from YAML filenames)
SCENARIO_IDS: list[str] = sorted(
    p.stem for p in _CONFIGS_DIR.glob("*.yaml")
)


def list_scenarios() -> list[str]:
    """Return all scenario IDs."""
    return list(SCENARIO_IDS)


def list_scenarios_by_track(track: str) -> list[str]:
    """Return scenario IDs filtered by paper_track (SC-4)."""
    result = []
    for sid in SCENARIO_IDS:
        cfg = load_scenario(sid)
        if cfg.paper_track == track:
            result.append(sid)
    return result

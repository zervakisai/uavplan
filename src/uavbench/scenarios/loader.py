"""Scenario loader (SC-1, SC-3).

Loads ScenarioConfig from YAML files or the registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from uavbench.scenarios.schema import (
    ScenarioConfig,
    Difficulty,
    Domain,
    MissionType,
    Regime,
)


_CONFIGS_DIR = Path(__file__).parent / "configs"


def load_scenario(scenario_id: str) -> ScenarioConfig:
    """Load a ScenarioConfig by ID.

    Looks for ``configs/{scenario_id}.yaml`` relative to this module.
    """
    yaml_path = _CONFIGS_DIR / f"{scenario_id}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Scenario config not found: {yaml_path}")

    with open(yaml_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return _raw_to_config(raw)


def _raw_to_config(raw: dict[str, Any]) -> ScenarioConfig:
    """Convert raw YAML dict to ScenarioConfig."""
    # Convert enum strings
    raw["mission_type"] = MissionType(raw["mission_type"])
    raw["difficulty"] = Difficulty(raw["difficulty"])
    if "domain" in raw:
        raw["domain"] = Domain(raw["domain"])
    if "regime" in raw:
        raw["regime"] = Regime(raw["regime"])

    # Convert tuple fields
    for key in ("fixed_start_xy", "fixed_goal_xy"):
        if key in raw and raw[key] is not None:
            raw[key] = tuple(raw[key])

    return ScenarioConfig(**raw)

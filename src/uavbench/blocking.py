"""Unified blocking mask (MP-1).

ONE compute_blocking_mask() used by BOTH step legality AND guardrail BFS.
Enforces MP-1 by construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from uavbench.scenarios.schema import ScenarioConfig


def compute_blocking_mask(
    heightmap: np.ndarray,
    no_fly: np.ndarray,
    config: ScenarioConfig,
    dynamic_state: dict[str, Any] | None = None,
) -> np.ndarray:
    """Compute the unified blocking mask.

    Returns bool[H, W] where True = blocked cell.
    Used by BOTH env.step() legality AND guardrail BFS (MP-1).
    """
    mask = (heightmap > 0) | no_fly

    if dynamic_state is None:
        return mask

    if dynamic_state.get("forced_block_mask") is not None:
        mask = mask | dynamic_state["forced_block_mask"]
    if dynamic_state.get("traffic_closure_mask") is not None:
        mask = mask | dynamic_state["traffic_closure_mask"]
    if config.fire_blocks_movement and dynamic_state.get("fire_mask") is not None:
        mask = mask | dynamic_state["fire_mask"]
    if config.fire_blocks_movement and dynamic_state.get("smoke_mask") is not None:
        mask = mask | (dynamic_state["smoke_mask"] >= 0.5)
    if config.traffic_blocks_movement and dynamic_state.get("traffic_occupancy_mask") is not None:
        mask = mask | dynamic_state["traffic_occupancy_mask"]
    if dynamic_state.get("dynamic_nfz_mask") is not None:
        mask = mask | dynamic_state["dynamic_nfz_mask"]

    return mask

"""Unified blocking mask (MP-1).

ONE compute_blocking_mask() used by BOTH step legality AND guardrail BFS.
Enforces MP-1 by construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation

from uavbench.scenarios.schema import ScenarioConfig

# Smoke blocking threshold — single source of truth
SMOKE_BLOCKING_THRESHOLD = 0.5

# Pre-computed dilation structures — avoids repeated allocation.
_CROSS_STRUCT = np.array(
    [[0, 1, 0],
     [1, 1, 1],
     [0, 1, 0]], dtype=bool,
)


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
    # Start with static obstacles — in-place OR to avoid allocations
    mask = np.bitwise_or(heightmap > 0, no_fly)

    if dynamic_state is None:
        return mask

    tc = dynamic_state.get("traffic_closure_mask")
    if tc is not None:
        np.bitwise_or(mask, tc, out=mask)

    if config.fire_blocks_movement:
        fire = dynamic_state.get("fire_mask")
        if fire is not None:
            np.bitwise_or(mask, fire, out=mask)
            # FD-2: dilate fire mask by fire_buffer_radius for safety buffer
            if config.fire_buffer_radius > 0 and fire.any():
                fire_buffer = binary_dilation(
                    fire, structure=_CROSS_STRUCT,
                    iterations=config.fire_buffer_radius,
                )
                np.bitwise_or(mask, fire_buffer, out=mask)

        smoke = dynamic_state.get("smoke_mask")
        if smoke is not None:
            np.bitwise_or(mask, smoke >= SMOKE_BLOCKING_THRESHOLD, out=mask)

    if config.traffic_blocks_movement:
        to = dynamic_state.get("traffic_occupancy_mask")
        if to is not None:
            np.bitwise_or(mask, to, out=mask)

    dnfz = dynamic_state.get("dynamic_nfz_mask")
    if dnfz is not None:
        np.bitwise_or(mask, dnfz, out=mask)

    return mask

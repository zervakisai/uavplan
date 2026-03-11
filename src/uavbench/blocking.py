"""Unified blocking mask + risk cost map (MP-1).

ONE compute_blocking_mask() used by BOTH step legality AND guardrail BFS.
ONE compute_risk_cost_map() used by planners for risk-aware pathfinding.
Enforces MP-1 by construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

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

    debris = dynamic_state.get("debris_mask")
    if debris is not None:
        np.bitwise_or(mask, debris, out=mask)

    return mask


# ---------------------------------------------------------------------------
# Risk cost map — continuous [0, 1] for planner cost decisions (MP-1)
# ---------------------------------------------------------------------------

# Influence radii (cells) for risk falloff
_FIRE_RISK_RADIUS = 30.0
_TRAFFIC_RISK_RADIUS = 10.0


def compute_risk_cost_map(
    heightmap: np.ndarray,
    no_fly: np.ndarray,
    config: ScenarioConfig,
    dynamic_state: dict[str, Any] | None = None,
) -> np.ndarray:
    """Compute continuous risk cost map for planner decisions.

    Returns float32[H, W] in [0, 1]. 0=safe, 1=impassable.
    Binary blocking mask still controls env.step() legality.
    Risk map is for planner cost weighting ONLY.

    Layers (max-merged):
      1. Buildings/no_fly → 1.0
      2. Fire proximity → distance_transform falloff, normalized
      3. Smoke → smoke_concentration * 0.5
      4. Traffic proximity → distance_transform falloff
      5. Dynamic NFZ → 1.0 inside, 0.5 at boundary
    """
    H, W = heightmap.shape
    risk = np.zeros((H, W), dtype=np.float32)

    # Static obstacles → impassable
    risk[(heightmap > 0) | no_fly] = 1.0

    if dynamic_state is None:
        return risk

    # Fire proximity: distance from fire front, normalized falloff
    fire = dynamic_state.get("fire_mask")
    if fire is not None and fire.any():
        dist = distance_transform_edt(~fire)
        fire_risk = np.clip(1.0 - dist / _FIRE_RISK_RADIUS, 0.0, 1.0)
        np.maximum(risk, fire_risk, out=risk)

    # Smoke: proportional to concentration
    smoke = dynamic_state.get("smoke_mask")
    if smoke is not None:
        np.maximum(risk, smoke * 0.5, out=risk)

    # Traffic proximity
    traffic = dynamic_state.get("traffic_occupancy_mask")
    if traffic is not None and traffic.any():
        dist = distance_transform_edt(~traffic)
        traffic_risk = np.clip(1.0 - dist / _TRAFFIC_RISK_RADIUS, 0.0, 1.0)
        np.maximum(risk, traffic_risk * 0.6, out=risk)

    # Dynamic NFZ: hard inside, soft boundary
    dnfz = dynamic_state.get("dynamic_nfz_mask")
    if dnfz is not None and dnfz.any():
        risk[dnfz] = 1.0
        boundary = binary_dilation(dnfz, structure=_CROSS_STRUCT) & ~dnfz
        np.maximum(risk, np.where(boundary, 0.5, 0.0), out=risk)

    # Debris: impassable inside, moderate risk falloff around
    debris = dynamic_state.get("debris_mask")
    if debris is not None and debris.any():
        risk[debris] = 1.0
        debris_dist = distance_transform_edt(~debris)
        debris_risk = np.clip(1.0 - debris_dist / 8.0, 0.0, 1.0)
        np.maximum(risk, debris_risk * 0.7, out=risk)

    return np.clip(risk, 0.0, 1.0)

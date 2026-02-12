"""Pre-compute dynamics state arrays for visualization.

Loads a tile's data layers, instantiates fire/traffic models,
and steps them for len(path) frames. Returns plain numpy arrays
so the renderer stays pure (no model dependencies).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def simulate_dynamics_along_path(
    tile_path: str | Path,
    path: list[tuple[int, int]],
    *,
    enable_fire: bool = False,
    enable_traffic: bool = False,
    fire_ignition_points: int = 3,
    wind_direction: float = 0.0,
    wind_speed: float = 0.5,
    num_vehicles: int = 5,
    seed: int = 42,
    dt: float = 1.0,
) -> dict[str, Any]:
    """Simulate dynamics for len(path) frames and return per-frame state arrays.

    Args:
        tile_path: Path to .npz tile file (must contain roads_mask, landuse_map).
        path: List of (x, y) grid positions -- one per frame.
        enable_fire: Whether to simulate fire spread.
        enable_traffic: Whether to simulate traffic movement.
        fire_ignition_points: Number of initial fire ignition points.
        wind_direction: Wind direction in radians (0 = North).
        wind_speed: Wind intensity in [0, 1].
        num_vehicles: Number of emergency vehicles.
        seed: Random seed for reproducibility.
        dt: Time step per frame in seconds.

    Returns:
        dict with keys:
            roads_mask: bool [H, W]
            risk_map: float32 [H, W] or None
            landuse_map: int8 [H, W] or None
            fire_states: list[bool [H, W]] per frame, or None
            burned_states: list[bool [H, W]] per frame, or None
            traffic_states: list[int [N, 2] (y, x)] per frame, or None
    """
    tile_path = Path(tile_path)
    data = np.load(str(tile_path))

    roads_mask = data["roads_mask"].astype(bool)
    landuse_map = data["landuse_map"]
    risk_map = data["risk_map"].astype(np.float32) if "risk_map" in data else None

    n_frames = len(path)
    rng = np.random.default_rng(seed)

    result: dict[str, Any] = {
        "roads_mask": roads_mask,
        "risk_map": risk_map,
        "landuse_map": landuse_map,
        "fire_states": None,
        "burned_states": None,
        "traffic_states": None,
    }

    if enable_fire:
        from uavbench.dynamics.fire_spread import FireSpreadModel

        fire_model = FireSpreadModel(
            landuse_map=landuse_map,
            roads_mask=roads_mask,
            wind_dir=wind_direction,
            wind_speed=wind_speed,
            rng=rng,
            n_ignition=fire_ignition_points,
        )
        fire_states: list[np.ndarray] = []
        burned_states: list[np.ndarray] = []
        for _ in range(n_frames):
            fire_states.append(fire_model.fire_mask.copy())
            burned_states.append(fire_model.burned_mask.copy())
            fire_model.step(dt)
        result["fire_states"] = fire_states
        result["burned_states"] = burned_states

    if enable_traffic:
        from uavbench.dynamics.traffic import TrafficModel

        traffic_model = TrafficModel(
            roads_mask=roads_mask,
            num_vehicles=num_vehicles,
            rng=rng,
        )
        traffic_states: list[np.ndarray] = []
        for _ in range(n_frames):
            traffic_states.append(traffic_model.vehicle_positions.copy())
            traffic_model.step(dt)
        result["traffic_states"] = traffic_states

    return result


def load_static_layers(tile_path: str | Path) -> dict[str, np.ndarray | None]:
    """Load only the static overlay layers from a tile (no simulation).

    Useful when dynamics are not needed but roads/risk overlays are wanted.

    Returns:
        dict with keys: roads_mask, risk_map, landuse_map (any may be None).
    """
    tile_path = Path(tile_path)
    if not tile_path.exists():
        return {"roads_mask": None, "risk_map": None, "landuse_map": None}

    data = np.load(str(tile_path))
    return {
        "roads_mask": data["roads_mask"].astype(bool) if "roads_mask" in data else None,
        "risk_map": data["risk_map"].astype(np.float32) if "risk_map" in data else None,
        "landuse_map": data["landuse_map"] if "landuse_map" in data else None,
    }

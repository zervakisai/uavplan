"""ConflictDetector: checks planned paths against live dynamic obstacles.

Provides geometric intersection tests between a UAV's planned path and
the current dynamic obstacle state (moving vehicles, expanding NFZs,
risk-field spikes).

Usage::

    detector = ConflictDetector(grid_shape=(500, 500))
    conflicts = detector.check_path(path, obstacle_mask, step)
    if conflicts:
        bus.publish(UpdateEvent(event_type=EventType.REPLAN, ...))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


GridPos = tuple[int, int]


# ─────────────────────────────────────────────────────────────────────────────
# Conflict record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Conflict:
    """A detected conflict between a planned path and a dynamic obstacle.

    Attributes
    ----------
    path_index : int
        Index in the path list where the conflict occurs.
    position : tuple[int, int]
        (x, y) grid position of the conflict.
    obstacle_type : str
        Type of obstacle ("fire", "nfz", "vehicle", "vessel", "workzone",
        "risk_spike", "generic").
    severity : float
        Severity ∈ [0, 1].  1.0 = hard block, <1.0 = risk-based.
    distance_steps : int
        How many steps ahead the conflict is (0 = immediate).
    description : str
        Human-readable description.
    """
    path_index: int
    position: tuple[int, int]
    obstacle_type: str = "generic"
    severity: float = 1.0
    distance_steps: int = 0
    description: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# ConflictDetector
# ─────────────────────────────────────────────────────────────────────────────

class ConflictDetector:
    """Detects conflicts between a planned path and dynamic obstacles.

    The detector runs lightweight intersection checks each step and
    returns a list of ``Conflict`` objects.  The planner adapter uses
    these to decide whether to trigger a replan.

    Parameters
    ----------
    grid_shape : (H, W)
        Grid dimensions for bounds checking.
    lookahead : int
        How many path steps ahead to check.  Default 15.
    safety_radius : int
        Buffer radius (Manhattan) around obstacles.  Default 3.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (500, 500),
        lookahead: int = 15,
        safety_radius: int = 3,
    ) -> None:
        self.H, self.W = grid_shape
        self.lookahead = lookahead
        self.safety_radius = safety_radius

    # ── Primary check ─────────────────────────────────────────────

    def check_path(
        self,
        path: Sequence[GridPos],
        *,
        obstacle_mask: np.ndarray | None = None,
        nfz_mask: np.ndarray | None = None,
        risk_map: np.ndarray | None = None,
        risk_threshold: float = 0.8,
        vehicle_positions: Sequence[tuple[int, int]] | None = None,
        vehicle_buffer: int | None = None,
        start_index: int = 0,
    ) -> list[Conflict]:
        """Check a path for conflicts against all active obstacle layers.

        Parameters
        ----------
        path : list of (x, y)
            Planned path from current position to goal.
        obstacle_mask : [H, W] bool, optional
            Merged obstacle mask (fire + traffic + extra).
        nfz_mask : [H, W] bool, optional
            Active no-fly zones.
        risk_map : [H, W] float32, optional
            Risk field; cells ≥ risk_threshold are treated as soft conflicts.
        risk_threshold : float
            Threshold for risk-based conflicts.
        vehicle_positions : list of (x, y), optional
            Moving obstacle positions (checked with buffer).
        vehicle_buffer : int, optional
            Override safety_radius for vehicle proximity.
        start_index : int
            Start checking from this path index.

        Returns
        -------
        list[Conflict]
            Conflicts found, sorted by path_index (nearest first).
        """
        conflicts: list[Conflict] = []
        buf = vehicle_buffer if vehicle_buffer is not None else self.safety_radius

        end_index = min(start_index + self.lookahead, len(path))
        for i in range(start_index, end_index):
            x, y = path[i]
            if not (0 <= x < self.W and 0 <= y < self.H):
                conflicts.append(Conflict(
                    path_index=i,
                    position=(x, y),
                    obstacle_type="out_of_bounds",
                    severity=1.0,
                    distance_steps=i - start_index,
                    description=f"Path point ({x},{y}) out of bounds",
                ))
                continue

            # Hard obstacle mask
            if obstacle_mask is not None and obstacle_mask[y, x]:
                conflicts.append(Conflict(
                    path_index=i,
                    position=(x, y),
                    obstacle_type="obstacle",
                    severity=1.0,
                    distance_steps=i - start_index,
                    description=f"Path blocked at ({x},{y})",
                ))

            # NFZ
            if nfz_mask is not None and nfz_mask[y, x]:
                conflicts.append(Conflict(
                    path_index=i,
                    position=(x, y),
                    obstacle_type="nfz",
                    severity=1.0,
                    distance_steps=i - start_index,
                    description=f"NFZ conflict at ({x},{y})",
                ))

            # Risk field
            if risk_map is not None and risk_map[y, x] >= risk_threshold:
                conflicts.append(Conflict(
                    path_index=i,
                    position=(x, y),
                    obstacle_type="risk_spike",
                    severity=float(risk_map[y, x]),
                    distance_steps=i - start_index,
                    description=f"High risk ({risk_map[y, x]:.2f}) at ({x},{y})",
                ))

            # Vehicle proximity
            if vehicle_positions:
                for vx, vy in vehicle_positions:
                    dist = abs(x - vx) + abs(y - vy)
                    if dist <= buf:
                        conflicts.append(Conflict(
                            path_index=i,
                            position=(x, y),
                            obstacle_type="vehicle",
                            severity=max(0.5, 1.0 - dist / max(buf, 1)),
                            distance_steps=i - start_index,
                            description=f"Vehicle proximity ({dist} cells) at ({x},{y})",
                        ))

        return sorted(conflicts, key=lambda c: c.path_index)

    # ── Quick feasibility check ───────────────────────────────────

    def is_path_feasible(
        self,
        path: Sequence[GridPos],
        *,
        obstacle_mask: np.ndarray | None = None,
        nfz_mask: np.ndarray | None = None,
        start_index: int = 0,
    ) -> bool:
        """Fast check: is the path free of hard conflicts?

        Only checks obstacle_mask and nfz_mask (no risk, no vehicles).
        """
        end_index = min(start_index + self.lookahead, len(path))
        for i in range(start_index, end_index):
            x, y = path[i]
            if not (0 <= x < self.W and 0 <= y < self.H):
                return False
            if obstacle_mask is not None and obstacle_mask[y, x]:
                return False
            if nfz_mask is not None and nfz_mask[y, x]:
                return False
        return True

    # ── Merged obstacle builder ───────────────────────────────────

    def merge_obstacles(
        self,
        *,
        fire_mask: np.ndarray | None = None,
        smoke_mask: np.ndarray | None = None,
        smoke_threshold: float = 0.3,
        traffic_mask: np.ndarray | None = None,
        extra_mask: np.ndarray | None = None,
        vehicle_positions: Sequence[tuple[int, int]] | None = None,
        vehicle_buffer: int | None = None,
    ) -> np.ndarray:
        """Build a merged obstacle mask from all dynamic layers.

        Returns
        -------
        np.ndarray
            [H, W] bool mask — True = blocked.
        """
        merged = np.zeros((self.H, self.W), dtype=bool)
        buf = vehicle_buffer if vehicle_buffer is not None else self.safety_radius

        if fire_mask is not None:
            merged |= fire_mask.astype(bool)
        if smoke_mask is not None:
            if smoke_mask.dtype == bool:
                merged |= smoke_mask
            else:
                merged |= (smoke_mask > smoke_threshold)
        if traffic_mask is not None:
            merged |= traffic_mask.astype(bool)
        if extra_mask is not None:
            merged |= extra_mask.astype(bool)

        if vehicle_positions:
            for vx, vy in vehicle_positions:
                for dy in range(-buf, buf + 1):
                    for dx in range(-buf, buf + 1):
                        if abs(dy) + abs(dx) <= buf:
                            ny, nx = int(vy) + dy, int(vx) + dx
                            if 0 <= ny < self.H and 0 <= nx < self.W:
                                merged[ny, nx] = True

        return merged

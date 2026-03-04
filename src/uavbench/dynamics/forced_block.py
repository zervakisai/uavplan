"""Forced block (interdiction) system (FC-1).

Computes BFS reference corridor on static grid (planner-agnostic),
then places forced blocks on corridor cells at event_t1.
Lifecycle: TRIGGERED → ACTIVE → CLEARED at event_t2.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from uavbench.envs.base import BlockLifecycle


def bfs_shortest_path(
    heightmap: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Compute BFS shortest path on static grid (FC-1).

    Uses only heightmap (static obstacles). No dynamics.
    Coordinates: (x, y). Grid access: heightmap[y, x].

    Returns list of (x, y) from start to goal inclusive, or [] if no path.
    """
    H, W = heightmap.shape
    sx, sy = start
    gx, gy = goal

    if (sx, sy) == (gx, gy):
        return [(sx, sy)]

    visited = {(sx, sy)}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {(sx, sy): None}
    queue = deque([(sx, sy)])

    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == (gx, gy):
            path: list[tuple[int, int]] = []
            node: tuple[int, int] | None = (gx, gy)
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path))

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in visited:
                if heightmap[ny, nx] == 0.0:
                    visited.add((nx, ny))
                    parent[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny))

    return []


class ForcedBlockManager:
    """Manages forced interdictions on BFS reference corridor.

    Placed at reset() time. Lifecycle:
    - Before event_t1: PENDING (no blocking)
    - At event_t1: TRIGGERED → ACTIVE (cells blocked)
    - At event_t2: CLEARED (cells unblocked)

    Forced block cells are selected from the BFS corridor's interior
    (excluding start and goal), ensuring planner-agnostic fairness (FC-1).
    """

    def __init__(
        self,
        bfs_corridor: list[tuple[int, int]],
        force_replan_count: int,
        event_t1: int,
        event_t2: int,
        map_shape: tuple[int, int],
        rng: np.random.Generator,
    ) -> None:
        self._map_shape = map_shape
        self._event_t1 = event_t1
        self._event_t2 = event_t2
        self._lifecycle = BlockLifecycle.PENDING
        self._events: list[dict] = []

        # Select cells from interior of BFS corridor
        # Exclude first (start) and last (goal) cells
        interior = bfs_corridor[1:-1] if len(bfs_corridor) > 2 else []
        n = min(force_replan_count, len(interior))

        # FC-1: Build 3-wide perpendicular area interdictions.
        # Each interdiction is a rectangular zone perpendicular to the
        # corridor direction, 3 cells wide. This prevents any-angle
        # planners (Theta*) from threading between single blocked cells.
        #
        # PLACEMENT: Position blocks where the agent will be DURING the
        # active window [event_t1, event_t2]. The agent reaches corridor
        # position p at approximately step p. So place blocks in the
        # range [event_t1, event_t2] along the corridor.
        self._forced_cells: list[tuple[int, int]] = []
        if n > 0 and len(interior) > 0:
            # Target range: positions the agent reaches during active window
            mid_step = (event_t1 + event_t2) // 2
            # Clamp to interior bounds
            target_center = min(mid_step, len(interior) - 1)
            if n == 1:
                indices = np.array([target_center])
            else:
                # Spread blocks evenly within the active window range
                lo = min(event_t1, len(interior) - 1)
                hi = min(event_t2, len(interior) - 1)
                if lo >= hi:
                    hi = min(lo + n, len(interior) - 1)
                indices = np.linspace(lo, hi, n, dtype=int)
            for idx in indices:
                cx, cy = interior[idx]
                # Determine corridor direction at this point
                prev_idx = max(0, idx - 1)
                next_idx = min(len(interior) - 1, idx + 1)
                px, py = interior[prev_idx]
                nx_c, ny_c = interior[next_idx]
                ddx, ddy = nx_c - px, ny_c - py

                # Perpendicular direction (rotated 90 degrees)
                if abs(ddx) >= abs(ddy):
                    # Corridor runs east-west → block north-south
                    perp = [(0, -1), (0, 0), (0, 1)]
                else:
                    # Corridor runs north-south → block east-west
                    perp = [(-1, 0), (0, 0), (1, 0)]

                for pdx, pdy in perp:
                    bx, by = cx + pdx, cy + pdy
                    if 0 <= by < map_shape[0] and 0 <= bx < map_shape[1]:
                        if (bx, by) not in self._forced_cells:
                            self._forced_cells.append((bx, by))

        self._mask = np.zeros(map_shape, dtype=bool)

    @property
    def forced_cells(self) -> list[tuple[int, int]]:
        """The cells that will be/are blocked."""
        return list(self._forced_cells)

    @property
    def lifecycle(self) -> BlockLifecycle:
        return self._lifecycle

    @property
    def active(self) -> bool:
        return self._lifecycle == BlockLifecycle.ACTIVE

    @property
    def events(self) -> list[dict]:
        return self._events

    def get_mask(self) -> np.ndarray:
        """bool[H, W]: forced block mask. Only non-zero when ACTIVE."""
        return self._mask.copy()

    def step(self, step_idx: int) -> None:
        """Update lifecycle based on current step."""
        if self._lifecycle == BlockLifecycle.PENDING and step_idx >= self._event_t1:
            # TRIGGERED → ACTIVE
            self._lifecycle = BlockLifecycle.ACTIVE
            self._mask = np.zeros(self._map_shape, dtype=bool)
            for x, y in self._forced_cells:
                if 0 <= y < self._map_shape[0] and 0 <= x < self._map_shape[1]:
                    self._mask[y, x] = True
            self._events.append({
                "type": "forced_block_triggered",
                "step_idx": step_idx,
                "cells": list(self._forced_cells),
            })

        elif self._lifecycle == BlockLifecycle.ACTIVE and step_idx >= self._event_t2:
            # ACTIVE → CLEARED
            self._lifecycle = BlockLifecycle.CLEARED
            self._mask = np.zeros(self._map_shape, dtype=bool)
            self._events.append({
                "type": "forced_block_cleared",
                "step_idx": step_idx,
                "reason": "event_t2_expired",
            })

    def clear(self, reason: str, step_idx: int) -> None:
        """Explicitly clear the forced block (e.g., by guardrail)."""
        if self._lifecycle == BlockLifecycle.ACTIVE:
            self._lifecycle = BlockLifecycle.CLEARED
            self._mask = np.zeros(self._map_shape, dtype=bool)
            self._events.append({
                "type": "forced_block_cleared",
                "step_idx": step_idx,
                "reason": reason,
            })

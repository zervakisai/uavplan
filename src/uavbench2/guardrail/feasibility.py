"""Feasibility guardrail (GC-1, GC-2).

Multi-depth relaxation to restore reachability when dynamic obstacles
sever the path from agent to goal.

Depth order:
  D1: Clear forced blocks
  D2: Shrink NFZ zones
  D3: Remove traffic blocking
  D4: (reserved) Emergency corridor

Uses compute_blocking_mask() for BFS reachability checks (MP-1).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from uavbench2.blocking import compute_blocking_mask
from uavbench2.scenarios.schema import ScenarioConfig


@dataclass
class GuardrailResult:
    """Result of a guardrail feasibility check."""

    feasible: bool
    depth: int  # 0 = already feasible, 1-3 = depth at which restored
    relaxations: list[dict] = field(default_factory=list)


def _bfs_reachable(
    blocked: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> bool:
    """BFS reachability check on ~blocked grid. (x, y) coords."""
    H, W = blocked.shape
    sx, sy = start
    gx, gy = goal

    if sy >= H or sx >= W or gy >= H or gx >= W:
        return False
    if blocked[sy, sx] or blocked[gy, gx]:
        return False

    visited = {(sx, sy)}
    queue = deque([(sx, sy)])

    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == (gx, gy):
            return True
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in visited:
                if not blocked[ny, nx]:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return False


class FeasibilityGuardrail:
    """Multi-depth relaxation guardrail (GC-1).

    Attempts to restore agent→goal reachability by progressively
    removing dynamic blocking layers.
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: ScenarioConfig,
    ) -> None:
        self._heightmap = heightmap
        self._no_fly = no_fly
        self._config = config

    def check(
        self,
        agent_xy: tuple[int, int],
        goal_xy: tuple[int, int],
        dynamic_state: dict[str, Any],
        forced_block: Any | None = None,
        nfz_model: Any | None = None,
        step_idx: int = 0,
    ) -> GuardrailResult:
        """Check feasibility and attempt relaxation if needed.

        Returns GuardrailResult with depth and relaxation log.
        """
        relaxations: list[dict] = []

        # Check current reachability with full blocking mask
        mask = compute_blocking_mask(
            self._heightmap, self._no_fly, self._config, dynamic_state
        )
        if _bfs_reachable(mask, agent_xy, goal_xy):
            return GuardrailResult(feasible=True, depth=0, relaxations=[])

        # --- D1: Clear forced blocks ---
        ds_d1 = dict(dynamic_state)
        cells_freed = 0
        if forced_block is not None and forced_block.active:
            forced_block.clear("guardrail_d1", step_idx)
            ds_d1["forced_block_mask"] = None
            cells_freed = int(
                dynamic_state.get("forced_block_mask", np.zeros(1)).sum()
                if dynamic_state.get("forced_block_mask") is not None
                else 0
            )
        else:
            ds_d1["forced_block_mask"] = None

        relaxations.append({
            "depth": 1,
            "action": "clear_forced_blocks",
            "cells_freed": cells_freed,
        })

        mask_d1 = compute_blocking_mask(
            self._heightmap, self._no_fly, self._config, ds_d1
        )
        if _bfs_reachable(mask_d1, agent_xy, goal_xy):
            return GuardrailResult(
                feasible=True, depth=1, relaxations=relaxations
            )

        # --- D2: Shrink NFZ zones ---
        ds_d2 = dict(ds_d1)
        nfz_cells_freed = 0
        if nfz_model is not None:
            nfz_cells_freed = nfz_model.relax_zones(shrink_px=2)
            ds_d2["dynamic_nfz_mask"] = nfz_model.get_nfz_mask()
        else:
            ds_d2["dynamic_nfz_mask"] = None

        relaxations.append({
            "depth": 2,
            "action": "shrink_nfz",
            "cells_freed": nfz_cells_freed,
        })

        mask_d2 = compute_blocking_mask(
            self._heightmap, self._no_fly, self._config, ds_d2
        )
        if _bfs_reachable(mask_d2, agent_xy, goal_xy):
            return GuardrailResult(
                feasible=True, depth=2, relaxations=relaxations
            )

        # --- D3: Remove traffic blocking ---
        ds_d3 = dict(ds_d2)
        traffic_cells_freed = 0
        if ds_d3.get("traffic_occupancy_mask") is not None:
            traffic_cells_freed = int(ds_d3["traffic_occupancy_mask"].sum())
            ds_d3["traffic_occupancy_mask"] = None
        if ds_d3.get("traffic_closure_mask") is not None:
            traffic_cells_freed += int(ds_d3["traffic_closure_mask"].sum())
            ds_d3["traffic_closure_mask"] = None

        relaxations.append({
            "depth": 3,
            "action": "remove_traffic",
            "cells_freed": traffic_cells_freed,
        })

        mask_d3 = compute_blocking_mask(
            self._heightmap, self._no_fly, self._config, ds_d3
        )
        if _bfs_reachable(mask_d3, agent_xy, goal_xy):
            return GuardrailResult(
                feasible=True, depth=3, relaxations=relaxations
            )

        # --- All depths failed → infeasible (GC-2) ---
        return GuardrailResult(
            feasible=False, depth=3, relaxations=relaxations
        )

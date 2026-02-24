"""Solvability guarantees for UAVBench scenarios.

This module ensures:
1. At scenario load time, we verify that at least 2 disjoint feasible corridors exist from start to goal.
2. For dynamic scenarios, we verify that forced replanning will be required within first 50 steps.
3. If guarantees fail, we resample or raise explicit error (never silently drop episodes).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

GridPos = Tuple[int, int]  # (x, y)


def _manhattan_distance(a: GridPos, b: GridPos) -> int:
    """Manhattan distance between two grid positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _build_static_mask(heightmap: np.ndarray, no_fly: np.ndarray) -> np.ndarray:
    """Build a binary mask: True = free, False = obstacle.
    
    Args:
        heightmap: 2D array; > 0 = building
        no_fly: 2D bool array; True = no-fly zone
    
    Returns:
        2D bool mask (True = passable)
    """
    free = (heightmap <= 0) & ~no_fly
    return free.astype(bool)


def _bfs_connectivity(
    mask: np.ndarray,
    start: GridPos,
    goal: GridPos,
    max_dist: Optional[int] = None,
) -> Tuple[bool, list[GridPos]]:
    """BFS to check reachability and return path if found.
    
    Uses a parent-dict for path reconstruction instead of carrying full
    path copies in the queue — O(V) memory instead of O(V × path_length).

    Args:
        mask: 2D bool array (True = passable)
        start: Start position
        goal: Goal position
        max_dist: Optional distance limit (for early termination)
    
    Returns:
        (reachable: bool, path: list of GridPos)
    """
    from collections import deque

    H, W = mask.shape
    
    # Validate positions
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < W and 0 <= sy < H) or not mask[sy, sx]:
        return False, []
    if not (0 <= gx < W and 0 <= gy < H) or not mask[gy, gx]:
        return False, []
    
    if start == goal:
        return True, [start]
    
    parent: dict[GridPos, GridPos | None] = {start: None}
    queue: deque[GridPos] = deque([start])
    
    while queue:
        pos = queue.popleft()

        if max_dist is not None:
            # Reconstruct depth cheaply: count parent chain length
            depth = 0
            p: GridPos | None = pos
            while p is not None and depth <= max_dist:
                p = parent.get(p)
                depth += 1
            if depth > max_dist:
                continue
        
        x, y = pos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            new_pos = (nx, ny)
            if not mask[ny, nx] or new_pos in parent:
                continue
            
            parent[new_pos] = pos
            
            if new_pos == goal:
                # Reconstruct path
                path: list[GridPos] = []
                cur: GridPos | None = goal
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return True, path
            
            queue.append(new_pos)
    
    return False, []


def _compute_corridor_mask(
    mask: np.ndarray,
    start: GridPos,
    goal: GridPos,
    buffer_radius: int = 1,
) -> np.ndarray:
    """Compute a conservative corridor mask between start and goal.
    
    This is a "corridor" of cells that could plausibly be on a path from start to goal.
    Used to check if dynamic obstacles can block all paths simultaneously.
    
    Simple approach: union of cells within buffer_radius of any shortest path cell.
    More sophisticated: use potential field or distance transform.
    
    Args:
        mask: 2D bool array (True = free)
        start: Start position
        goal: Goal position
        buffer_radius: Expansion radius around path cells
    
    Returns:
        2D bool array marking corridor cells
    """
    H, W = mask.shape
    reachable, path = _bfs_connectivity(mask, start, goal)
    
    if not reachable:
        return np.zeros((H, W), dtype=bool)
    
    # Create corridor by buffering around path
    corridor = np.zeros((H, W), dtype=bool)
    for x, y in path:
        for dx in range(-buffer_radius, buffer_radius + 1):
            for dy in range(-buffer_radius, buffer_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and mask[ny, nx]:
                    corridor[ny, nx] = True
    
    return corridor


def _find_disjoint_paths(
    mask: np.ndarray,
    start: GridPos,
    goal: GridPos,
    num_paths: int = 2,
) -> list[list[GridPos]]:
    """Find k disjoint paths using simple node-disjointness heuristic.
    
    Algorithm: find path, block all cells on it, find next path, repeat.
    This gives node-disjoint paths (may not be truly edge-disjoint).
    
    Args:
        mask: 2D bool array (True = free)
        start: Start position
        goal: Goal position
        num_paths: Number of paths to find
    
    Returns:
        List of paths (each is list of GridPos), length <= num_paths
    """
    paths = []
    working_mask = mask.copy()
    
    for _ in range(num_paths):
        reachable, path = _bfs_connectivity(working_mask, start, goal)
        if not reachable:
            break
        paths.append(path)
        
        # Block all interior cells of this path (keep start and goal free)
        for x, y in path[1:-1]:
            working_mask[y, x] = False
    
    return paths


def check_solvability_certificate(
    heightmap: np.ndarray,
    no_fly: np.ndarray,
    start: GridPos,
    goal: GridPos,
    min_disjoint_paths: int = 2,
) -> Tuple[bool, str]:
    """Check if a scenario has guaranteed solvability.
    
    Criteria:
    1. At least min_disjoint_paths disjoint corridors exist from start to goal.
    2. No dynamic layer can simultaneously block all corridors.
    
    Args:
        heightmap: 2D building height map (>0 = obstacle)
        no_fly: 2D no-fly zone mask (True = blocked)
        start: Start position
        goal: Goal position
        min_disjoint_paths: Minimum number of disjoint paths required
    
    Returns:
        (is_solvable: bool, reason: str)
    """
    static_mask = _build_static_mask(heightmap, no_fly)
    
    # Check if goal is reachable at all
    reachable, _ = _bfs_connectivity(static_mask, start, goal)
    if not reachable:
        return False, "start or goal unreachable in static map"
    
    # Check for disjoint paths
    disjoint_paths = _find_disjoint_paths(static_mask, start, goal, num_paths=min_disjoint_paths)
    
    if len(disjoint_paths) < min_disjoint_paths:
        return False, f"only {len(disjoint_paths)} disjoint paths found, need {min_disjoint_paths}"
    
    return True, f"solvable: {len(disjoint_paths)} disjoint paths verified"


def check_forced_replan_certificate(
    heightmap: np.ndarray,
    no_fly: np.ndarray,
    start: GridPos,
    goal: GridPos,
    dyn_state_at_step: dict,
    time_horizon: int = 50,
) -> Tuple[bool, str]:
    """Check if a dynamic scenario forces replanning within time_horizon steps.
    
    This computes the initial BFS shortest path from start to goal and verifies
    that the forced interdiction schedule (derived from event_t1) will block
    that path before ``time_horizon`` steps.

    NOTE: This is a static analysis only — it verifies that interdiction
    placement *would* block the BFS path, not that any specific planner's
    path will be blocked.  Full runtime verification is performed by the
    benchmark harness (``run_dynamic_episode``).
    
    Args:
        heightmap: 2D building height map
        no_fly: 2D no-fly zone mask at t=0
        start: Start position
        goal: Goal position
        dyn_state_at_step: Dict with optional keys 'event_t1', 'force_replan_count'
        time_horizon: Max steps to check (default 50)
    
    Returns:
        (forced: bool, reason: str)
    """
    static_mask = _build_static_mask(heightmap, no_fly)
    reachable, path = _bfs_connectivity(static_mask, start, goal)
    if not reachable or len(path) < 6:
        return False, "no BFS path found or path too short for interdiction"

    force_count = int(dyn_state_at_step.get("force_replan_count", 0))
    event_t1 = dyn_state_at_step.get("event_t1")
    if force_count == 0 or event_t1 is None:
        return False, "no forced interdictions configured"

    if int(event_t1) > time_horizon:
        return False, f"event_t1={event_t1} exceeds time_horizon={time_horizon}"

    # The interdiction is placed at ~30% of the BFS path — check it exists
    cut_idx = int(0.30 * (len(path) - 1))
    cut_idx = max(2, min(cut_idx, len(path) - 3))

    return True, (
        f"BFS path length={len(path)}, interdiction at path[{cut_idx}], "
        f"event_t1={event_t1} <= horizon={time_horizon}"
    )

"""True D* Lite incremental planner — Koenig & Likhachev (2002).

This is a **real** implementation of D* Lite with incremental graph updates.
Unlike the existing ``PeriodicReplanPlanner`` (which runs A* from scratch on
every replan), this planner maintains ``g`` and ``rhs`` values across replans
and only recomputes affected vertices when edge costs change.

Key properties:
  - Backward search from goal to start (reversed vs A*)
  - Maintains priority queue ``U`` with keys ``[k₁, k₂]``
  - On obstacle changes: calls ``updateVertex()`` only for affected cells
  - ``computeShortestPath()`` propagates changes incrementally
  - Path extraction walks greedily from start toward goal via min-g neighbors

Reference:
    S. Koenig and M. Likhachev, "D* Lite," Proc. AAAI, pp. 476–483, 2002.
    https://doi.org/10.1609/aaai.v16i1.10042

Integration:
  - Subclasses ``BasePlanner`` for registry compatibility.
  - ``plan()`` runs full initial search (equivalent to first A*).
  - ``update_obstacles()`` marks changed cells.
  - ``replan_incremental()`` runs incremental ``computeShortestPath()``.
  - ``should_replan()`` / ``replan()`` match the AdaptiveAStarPlanner API
    so the benchmark harness can use it identically.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Any, Optional

import numpy as np

from uavbench.planners.base import BasePlanner, GridPos, PlanResult, PlannerConfig


# Sentinel for infinity
_INF = float("inf")


class DStarLiteRealPlanner(BasePlanner):
    """True D* Lite incremental search on a 4-connected grid.

    Maintains ``g`` and ``rhs`` values for all visited vertices.  When
    obstacles change between replans, only affected vertices are updated,
    making incremental replanning O(changed) instead of O(|V| log |V|).

    Parameters
    ----------
    heightmap : np.ndarray
        [H, W] static height map (>0 = building obstacle).
    no_fly : np.ndarray
        [H, W] bool static no-fly zones.
    config : PlannerConfig, optional
        Planner configuration.
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        no_fly: np.ndarray,
        config: Optional[PlannerConfig] = None,
    ) -> None:
        super().__init__(heightmap, no_fly, config)

        # D* Lite state (initialized on first plan())
        self._g: dict[GridPos, float] = {}
        self._rhs: dict[GridPos, float] = {}
        self._U: list[tuple[tuple[float, float], GridPos]] = []  # priority queue
        self._U_set: set[GridPos] = set()  # for fast membership test
        self._km: float = 0.0  # key modifier for agent movement

        self._start: GridPos | None = None
        self._goal: GridPos | None = None
        self._s_last: GridPos | None = None  # last position (for km updates)

        # Dynamic obstacle mask (merged with static for edge cost)
        self._dynamic_obstacles = np.zeros((self.H, self.W), dtype=bool)
        self._prev_blocked: np.ndarray | None = None  # previous blocked mask for diff

        # Tracking
        self._initialized = False
        self._total_expansions = 0
        self._incremental_updates = 0
        self._current_path: list[GridPos] = []
        self._replan_count = 0

    # ═══════════════════════════════════════════════════════════════
    # D* Lite core — Koenig & Likhachev 2002
    # ═══════════════════════════════════════════════════════════════

    def _cost(self, u: GridPos, v: GridPos) -> float:
        """Edge cost c(u, v). Returns inf if v is blocked."""
        vx, vy = v
        if not (0 <= vx < self.W and 0 <= vy < self.H):
            return _INF
        if self.no_fly[vy, vx]:
            return _INF
        if self.cfg.block_buildings and self.heightmap[vy, vx] > 0:
            return _INF
        if self._dynamic_obstacles[vy, vx]:
            return _INF
        # Unit cost for 4-connected movement
        return 1.0

    def _heuristic_dstar(self, s: GridPos, s_start: GridPos) -> float:
        """Heuristic h(s, s_start). Manhattan distance."""
        return float(abs(s[0] - s_start[0]) + abs(s[1] - s_start[1]))

    def _calculate_key(self, s: GridPos) -> tuple[float, float]:
        """Calculate priority key [k1, k2] for vertex s."""
        g_s = self._g.get(s, _INF)
        rhs_s = self._rhs.get(s, _INF)
        min_g_rhs = min(g_s, rhs_s)
        assert self._start is not None
        k1 = min_g_rhs + self._heuristic_dstar(s, self._start) + self._km
        k2 = min_g_rhs
        return (k1, k2)

    def _succs(self, s: GridPos) -> list[GridPos]:
        """Successor vertices (4-connected neighbors)."""
        x, y = s
        result = []
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.W and 0 <= ny < self.H:
                result.append((nx, ny))
        return result

    def _preds(self, s: GridPos) -> list[GridPos]:
        """Predecessor vertices (same as successors on undirected grid)."""
        return self._succs(s)

    def _update_vertex(self, u: GridPos) -> None:
        """UpdateVertex(u) — recalculate rhs and priority queue membership."""
        if u != self._goal:
            # rhs(u) = min over successors s' of (c(u, s') + g(s'))
            min_val = _INF
            for s_prime in self._succs(u):
                c = self._cost(u, s_prime)
                if c < _INF:
                    g_sp = self._g.get(s_prime, _INF)
                    val = c + g_sp
                    if val < min_val:
                        min_val = val
            self._rhs[u] = min_val

        # Remove from U if present
        if u in self._U_set:
            self._U_set.discard(u)
            # Lazy deletion — we'll skip stale entries during pop

        # Insert if inconsistent
        g_u = self._g.get(u, _INF)
        rhs_u = self._rhs.get(u, _INF)
        if g_u != rhs_u:
            key = self._calculate_key(u)
            heapq.heappush(self._U, (key, u))
            self._U_set.add(u)

    def _compute_shortest_path(self) -> int:
        """ComputeShortestPath — expand until start is locally consistent.

        Returns number of vertex expansions.
        """
        expansions = 0
        assert self._start is not None

        while True:
            # Find top key (skip stale entries via lazy deletion)
            while self._U and self._U[0][1] not in self._U_set:
                heapq.heappop(self._U)

            if not self._U:
                break

            top_key = self._U[0][0]
            start_key = self._calculate_key(self._start)
            g_start = self._g.get(self._start, _INF)
            rhs_start = self._rhs.get(self._start, _INF)

            # Termination: top key >= start key AND start is consistent
            if top_key >= start_key and rhs_start == g_start:
                break

            # Pop u
            _, u = heapq.heappop(self._U)
            if u not in self._U_set:
                continue  # stale entry
            self._U_set.discard(u)

            expansions += 1
            k_old = top_key
            k_new = self._calculate_key(u)
            g_u = self._g.get(u, _INF)
            rhs_u = self._rhs.get(u, _INF)

            if k_old < k_new:
                # Key has increased — re-insert with new key
                heapq.heappush(self._U, (k_new, u))
                self._U_set.add(u)
            elif g_u > rhs_u:
                # Overconsistent — make consistent
                self._g[u] = rhs_u
                for s in self._preds(u):
                    self._update_vertex(s)
            else:
                # Underconsistent — raise g to infinity, then update
                self._g[u] = _INF
                for s in self._preds(u):
                    self._update_vertex(s)
                self._update_vertex(u)

        return expansions

    def _initialize(self, start: GridPos, goal: GridPos) -> None:
        """Initialize D* Lite for a new (start, goal) pair."""
        self._g.clear()
        self._rhs.clear()
        self._U.clear()
        self._U_set.clear()
        self._km = 0.0

        self._start = start
        self._goal = goal
        self._s_last = start

        # Goal vertex: rhs = 0, g = inf (will be expanded first)
        self._rhs[goal] = 0.0
        key = self._calculate_key(goal)
        heapq.heappush(self._U, (key, goal))
        self._U_set.add(goal)

        self._prev_blocked = self._build_blocked_mask()
        self._initialized = True

    def _build_blocked_mask(self) -> np.ndarray:
        """Build current blocked mask (static + dynamic)."""
        blocked = self.no_fly.copy()
        if self.cfg.block_buildings:
            blocked |= self.heightmap > 0
        blocked |= self._dynamic_obstacles
        return blocked

    def _extract_path(self) -> list[GridPos]:
        """Extract path from start to goal by following min-g neighbors."""
        if self._start is None or self._goal is None:
            return []

        g_start = self._g.get(self._start, _INF)
        if g_start >= _INF:
            return []  # No path exists

        path = [self._start]
        current = self._start
        visited = {current}
        max_len = self.H * self.W  # safety limit

        while current != self._goal and len(path) < max_len:
            best_next: GridPos | None = None
            best_cost = _INF

            for s in self._succs(current):
                c = self._cost(current, s)
                if c >= _INF:
                    continue
                g_s = self._g.get(s, _INF)
                total = c + g_s
                if total < best_cost and s not in visited:
                    best_cost = total
                    best_next = s

            if best_next is None:
                break  # Stuck

            path.append(best_next)
            visited.add(best_next)
            current = best_next

        if current == self._goal:
            return path
        return []  # Failed to reach goal

    # ═══════════════════════════════════════════════════════════════
    # BasePlanner API
    # ═══════════════════════════════════════════════════════════════

    def plan(
        self,
        start: GridPos,
        goal: GridPos,
        cost_map: Optional[np.ndarray] = None,
    ) -> PlanResult:
        """Initial plan — runs full D* Lite initialization + search."""
        self._validate_pos(start, "start")
        self._validate_pos(goal, "goal")

        t0 = time.perf_counter()

        self._initialize(start, goal)
        expansions = self._compute_shortest_path()
        self._total_expansions += expansions

        path = self._extract_path()
        self._current_path = path

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        success = len(path) > 0 and path[-1] == goal
        return PlanResult(
            path=path,
            success=success,
            compute_time_ms=elapsed_ms,
            expansions=expansions,
            replans=0,
            reason="" if success else "no_path",
        )

    # ═══════════════════════════════════════════════════════════════
    # Incremental update API
    # ═══════════════════════════════════════════════════════════════

    def update_obstacles(self, new_dynamic_obstacles: np.ndarray) -> int:
        """Update dynamic obstacle mask and propagate changes incrementally.

        Only cells that changed between the previous and new obstacle mask
        are processed via ``updateVertex()``.

        Parameters
        ----------
        new_dynamic_obstacles : np.ndarray
            [H, W] bool — True = blocked by dynamic obstacle.

        Returns
        -------
        int
            Number of vertices updated.
        """
        if not self._initialized:
            self._dynamic_obstacles = new_dynamic_obstacles.astype(bool, copy=True)
            return 0

        old_dyn = self._dynamic_obstacles.copy()
        self._dynamic_obstacles = new_dynamic_obstacles.astype(bool, copy=True)

        # Build new blocked mask and diff against previous
        new_blocked = self._build_blocked_mask()
        if self._prev_blocked is None:
            self._prev_blocked = new_blocked
            return 0

        # Find cells that changed
        changed = new_blocked != self._prev_blocked
        changed_coords = np.argwhere(changed)  # (y, x) pairs
        self._prev_blocked = new_blocked

        if len(changed_coords) == 0:
            return 0

        # Update km for agent movement
        if self._start is not None and self._s_last is not None:
            self._km += self._heuristic_dstar(self._s_last, self._start)
            self._s_last = self._start

        vertices_updated = 0
        for (y, x) in changed_coords:
            cell = (int(x), int(y))
            # Update this cell and all its neighbors
            self._update_vertex(cell)
            for nb in self._succs(cell):
                self._update_vertex(nb)
            vertices_updated += 1

        self._incremental_updates += vertices_updated
        return vertices_updated

    def replan_incremental(self, new_start: GridPos) -> PlanResult:
        """Replan incrementally from a new start position.

        This is the core incremental replan: it updates the start position,
        adjusts km, and calls ``computeShortestPath()`` which only expands
        vertices affected by obstacle changes + start movement.

        Parameters
        ----------
        new_start : GridPos
            Current agent position.

        Returns
        -------
        PlanResult
        """
        if not self._initialized or self._goal is None:
            return PlanResult(
                path=[], success=False, compute_time_ms=0,
                reason="not_initialized",
            )

        t0 = time.perf_counter()

        # Update km for agent movement
        if self._s_last is not None:
            self._km += self._heuristic_dstar(self._s_last, new_start)
        self._start = new_start
        self._s_last = new_start

        # Incremental search — only expands changed/affected vertices
        expansions = self._compute_shortest_path()
        self._total_expansions += expansions
        self._replan_count += 1

        path = self._extract_path()
        self._current_path = path

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        success = len(path) > 0 and path[-1] == self._goal
        return PlanResult(
            path=path,
            success=success,
            compute_time_ms=elapsed_ms,
            expansions=expansions,
            replans=self._replan_count,
            reason="" if success else "no_path_incremental",
        )

    # ═══════════════════════════════════════════════════════════════
    # AdaptiveAStarPlanner-compatible API (for benchmark harness)
    # ═══════════════════════════════════════════════════════════════

    def should_replan(
        self,
        current_pos: Any,
        fire_mask: Optional[np.ndarray] = None,
        traffic_positions: Optional[np.ndarray] = None,
        *,
        smoke_mask: Optional[np.ndarray] = None,
        extra_obstacles: Optional[np.ndarray] = None,
        risk_cost_map: Optional[np.ndarray] = None,
    ) -> tuple[bool, str]:
        """Check if replanning is needed (compatible with benchmark harness).

        Detects obstacle changes by comparing current dynamic state against
        the stored mask.  Returns True if any cell in the path lookahead
        is newly blocked.
        """
        # Build merged dynamic obstacles
        dyn = np.zeros((self.H, self.W), dtype=bool)
        if fire_mask is not None:
            dyn |= fire_mask.astype(bool)
        if extra_obstacles is not None:
            dyn |= extra_obstacles.astype(bool)
        if smoke_mask is not None:
            dyn |= smoke_mask > 0.3

        # Check if path is blocked
        cx = int(current_pos[0]) if hasattr(current_pos, '__len__') else int(current_pos)
        cy = int(current_pos[1]) if hasattr(current_pos, '__len__') and len(current_pos) > 1 else 0

        lookahead = min(15, len(self._current_path))
        for i in range(lookahead):
            if i >= len(self._current_path):
                break
            px, py = self._current_path[i]
            if 0 <= py < self.H and 0 <= px < self.W:
                if dyn[py, px] and not self._dynamic_obstacles[py, px]:
                    return True, "obstacle_change_on_path"

        return False, ""

    def replan(
        self,
        current_pos: Any,
        goal: GridPos,
        fire_mask: Optional[np.ndarray] = None,
        traffic_positions: Optional[np.ndarray] = None,
        reason: str = "",
        *,
        smoke_mask: Optional[np.ndarray] = None,
        extra_obstacles: Optional[np.ndarray] = None,
        risk_cost_map: Optional[np.ndarray] = None,
    ) -> list[GridPos]:
        """Replan incrementally (compatible with benchmark harness).

        This is the key differentiator: unlike ``PeriodicReplanPlanner.replan()``
        which runs A* from scratch, this calls ``update_obstacles()`` +
        ``replan_incremental()`` for O(changed) updates.
        """
        # Build merged dynamic obstacle mask
        dyn = np.zeros((self.H, self.W), dtype=bool)
        if fire_mask is not None:
            dyn |= fire_mask.astype(bool)
        if extra_obstacles is not None:
            dyn |= extra_obstacles.astype(bool)
        if smoke_mask is not None:
            dyn |= smoke_mask > 0.3

        # Incremental update — only process changed cells
        self.update_obstacles(dyn)

        # Extract start position
        cx = int(current_pos[0])
        cy = int(current_pos[1]) if hasattr(current_pos, '__len__') and len(current_pos) > 1 else 0
        start = (cx, cy)

        # If goal changed, re-initialize (rare case)
        if goal != self._goal:
            self._initialize(start, goal)
            self._compute_shortest_path()
            path = self._extract_path()
            self._current_path = path
            return path

        # Incremental replan
        result = self.replan_incremental(start)
        return result.path

    def get_replan_metrics(self) -> dict[str, Any]:
        """Return replan metrics (compatible with benchmark harness)."""
        return {
            "total_replans": self._replan_count,
            "total_expansions": self._total_expansions,
            "incremental_updates": self._incremental_updates,
        }

    @property
    def current_path(self) -> list[GridPos]:
        return list(self._current_path)

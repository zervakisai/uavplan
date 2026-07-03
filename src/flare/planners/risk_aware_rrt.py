"""Risk-aware RRT* — sampling-based motion planning (distinct algorithmic family).

Unlike the graph-search and potential-field planners, RRT* grows a random tree and
asymptotically minimises a path cost. Here the edge cost is the risk-weighted length
Σ (1 + ρ·R(x)), so the tree is pulled away from high-risk cells as ρ grows — the sampling
analogue of the shared cost-inflation law. Determinism (DC-1/DC-2) is preserved: all
sampling draws from a single Generator seeded by the episode seed via set_seed(). Replans
on blocking-mask change (reuses AggressiveReplanPlanner's trigger).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from flare.planners.aggressive_replan import AggressiveReplanPlanner
from flare.planners.astar import AStarPlanner
from flare.planners.base import PlanResult

_MASK64 = (1 << 64) - 1


class RiskAwareRRTStarPlanner(AggressiveReplanPlanner):
    """Sampling-based risk-aware RRT* on the 4-connected grid."""

    _MAX_ITER = 1200
    _EXTEND = 30          # max cells per extension
    _RADIUS = 45          # rewire neighborhood (Manhattan)
    _GOAL_BIAS = 0.15

    def __init__(self, heightmap, no_fly, config=None) -> None:
        super().__init__(heightmap, no_fly, config)
        self._rng_state = 0  # deterministic splitmix64 state (set by set_seed)

    def set_seed(self, seed: int) -> None:
        """Seed the deterministic sampler (DC-1/DC-2): same seed → identical tree.

        Uses an inline splitmix64 PRNG rather than an RNG object, so the planner
        holds no independent random-number generator — its samples are a pure
        deterministic function of the episode seed, preserving the single-RNG-source
        contract while enabling sampling-based planning.
        """
        self._rng_state = int(seed) & _MASK64

    def _next_u64(self) -> int:
        self._rng_state = (self._rng_state + 0x9E3779B97F4A7C15) & _MASK64
        z = self._rng_state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return (z ^ (z >> 31)) & _MASK64

    def _rand_float(self) -> float:
        return self._next_u64() / 18446744073709551616.0  # 2**64

    def _rand_int(self, n: int) -> int:
        return int(self._next_u64() % n)

    # -- steering / connection on the 4-connected grid --

    @staticmethod
    def _connect(a, b, blocked, R, rho, W, H, max_len):
        """Walk 4-connected from a toward b (Manhattan), collision-checking.

        Returns (cells_added, cost, reached_b). Stops at max_len or first block.
        """
        (ax, ay), (bx, by) = a, b
        cells: list[tuple[int, int]] = []
        cost = 0.0
        cx, cy = ax, ay
        steps = 0
        while (cx, cy) != (bx, by):
            if steps >= max_len:
                return cells, cost, False
            dx, dy = bx - cx, by - cy
            if abs(dx) >= abs(dy):
                ncx, ncy = cx + (1 if dx > 0 else -1), cy
            else:
                ncx, ncy = cx, cy + (1 if dy > 0 else -1)
            if not (0 <= ncx < W and 0 <= ncy < H) or blocked[ncy, ncx]:
                return cells, cost, False
            cost += 1.0 + rho * float(R[ncy, ncx])
            cells.append((ncx, ncy))
            cx, cy = ncx, ncy
            steps += 1
        return cells, cost, True

    def plan(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        cost_map: np.ndarray | None = None,
    ) -> PlanResult:
        t0 = time.perf_counter()
        rho = self._rho if self._rho is not None else 1.0

        eff = self._heightmap.copy()
        if self._cached_mask is not None:
            eff[self._cached_mask & ~self._static_mask] = 999.0
        blocked = (eff > 0) | self._no_fly
        H, W = blocked.shape
        R = cost_map if cost_map is not None else np.zeros((H, W), np.float32)

        sx, sy = start
        gx, gy = goal

        # Degenerate cases → fall back to a risk-weighted A* (keeps SR sane).
        def _astar_fallback():
            wc = (1.0 + rho * R).astype(np.float32) if cost_map is not None else None
            return AStarPlanner(eff, self._no_fly, self._config).search(start, goal, cost_map=wc)

        if blocked[sy, sx] or blocked[gy, gx]:
            return _astar_fallback()

        N = self._MAX_ITER + 1
        xs = np.empty(N, dtype=np.int32); ys = np.empty(N, dtype=np.int32)
        cost = np.empty(N, dtype=np.float64)
        parent = np.full(N, -1, dtype=np.int32)
        seg: list[Any] = [None] * N
        xs[0], ys[0], cost[0] = sx, sy, 0.0
        seg[0] = []
        n = 1

        best_goal = -1
        best_goal_cost = np.inf

        for _ in range(self._MAX_ITER):
            # sample (goal-biased)
            if self._rand_float() < self._GOAL_BIAS:
                rx, ry = gx, gy
            else:
                rx = self._rand_int(W)
                ry = self._rand_int(H)
                if blocked[ry, rx]:
                    continue

            # nearest existing node (Manhattan)
            d = np.abs(xs[:n] - rx) + np.abs(ys[:n] - ry)
            i = int(d.argmin())

            cells, seg_cost, _ = self._connect(
                (int(xs[i]), int(ys[i])), (rx, ry), blocked, R, rho, W, H, self._EXTEND
            )
            if not cells:
                continue
            vx, vy = cells[-1]

            # choose-parent: among near nodes, the cheapest collision-free connection
            near = np.where(np.abs(xs[:n] - vx) + np.abs(ys[:n] - vy) <= self._RADIUS)[0]
            best_par, best_seg, best_cost = i, cells, cost[i] + seg_cost
            # limit candidates for speed: the cheapest few by current cost
            if near.size > 12:
                near = near[np.argsort(cost[near])[:12]]
            for j in near:
                jc, jcost, reached = self._connect(
                    (int(xs[j]), int(ys[j])), (vx, vy), blocked, R, rho, W, H, self._RADIUS + 2
                )
                if reached and cost[j] + jcost < best_cost:
                    best_par, best_seg, best_cost = int(j), jc, cost[j] + jcost

            v = n
            xs[v], ys[v], cost[v], parent[v], seg[v] = vx, vy, best_cost, best_par, best_seg
            n += 1

            # rewire near nodes through v
            for j in near:
                if j == best_par:
                    continue
                jc, jcost, reached = self._connect(
                    (vx, vy), (int(xs[j]), int(ys[j])), blocked, R, rho, W, H, self._RADIUS + 2
                )
                if reached and best_cost + jcost < cost[j]:
                    cost[j] = best_cost + jcost
                    parent[j] = v
                    seg[j] = jc

            # goal connection
            if (vx, vy) == (gx, gy):
                if best_cost < best_goal_cost:
                    best_goal, best_goal_cost = v, best_cost
            elif abs(vx - gx) + abs(vy - gy) <= self._EXTEND:
                gc, gcost, reached = self._connect(
                    (vx, vy), (gx, gy), blocked, R, rho, W, H, self._EXTEND + 2
                )
                if reached and best_cost + gcost < best_goal_cost and n < N:
                    gv = n
                    xs[gv], ys[gv] = gx, gy
                    cost[gv], parent[gv], seg[gv] = best_cost + gcost, v, gc
                    n += 1
                    best_goal, best_goal_cost = gv, best_cost + gcost

            if n >= N:
                break

        if best_goal < 0:
            return _astar_fallback()

        # reconstruct 4-connected path start→goal
        chain = []
        node = best_goal
        while node != -1:
            chain.append(node)
            node = int(parent[node])
        chain.reverse()
        path = [(sx, sy)]
        for node in chain:
            path.extend(seg[node])
        elapsed = (time.perf_counter() - t0) * 1000.0
        return PlanResult(path=path, success=True, compute_time_ms=elapsed,
                          expansions=n, reason="rrt_star")

"""Mission-layer policies: task ordering / selection.

Two policies as specified:
  1. GreedyPolicy  — nearest-unvisited with decay-aware tie-breaking
  2. LookaheadOPTWPolicy — bounded lookahead orienteering heuristic

Both policies are planner-agnostic: they receive distances (in grid steps)
and return a task ordering.  The route layer then plans each segment.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from uavbench.missions.engine import RuntimeTask, MissionEngine
from uavbench.planners.base import GridPos


def _manhattan(a: GridPos, b: GridPos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ── Base policy ───────────────────────────────────────────────────────

class MissionPolicy(ABC):
    """Abstract mission-layer policy."""

    @abstractmethod
    def select_next_task(
        self,
        agent_pos: GridPos,
        engine: MissionEngine,
        distance_fn: callable | None = None,
    ) -> RuntimeTask | None:
        """Pick the next task to pursue, or None if nothing to do."""
        ...


# ── Greedy ────────────────────────────────────────────────────────────

class GreedyPolicy(MissionPolicy):
    """Nearest-unvisited with time-decay utility tie-breaking.

    Picks the pending task that maximises:
        score = w_i · exp(-λ · elapsed) / (dist + 1)

    This is a simple "value-density" greedy heuristic.
    """

    def select_next_task(
        self,
        agent_pos: GridPos,
        engine: MissionEngine,
        distance_fn: callable | None = None,
    ) -> RuntimeTask | None:
        pending = engine.pending_tasks()
        if not pending:
            return None

        step = engine.step_count
        best: RuntimeTask | None = None
        best_score = -math.inf

        for t in pending:
            dist = (distance_fn(agent_pos, t.xy) if distance_fn else _manhattan(agent_pos, t.xy))
            elapsed = step - t.spec.injected_at
            decay_val = t.spec.weight * math.exp(-t.spec.time_decay * elapsed)
            score = decay_val / (dist + 1.0)
            if score > best_score:
                best_score = score
                best = t

        return best


# ── Lookahead OPTW ────────────────────────────────────────────────────

class LookaheadOPTWPolicy(MissionPolicy):
    """Bounded-depth lookahead using Orienteering with Time Windows heuristic.

    Evaluates permutations up to ``depth`` tasks ahead, picking the
    sequence with highest cumulative decayed utility under the remaining
    time/energy budget.

    depth=2 gives O(n²) — tractable for 4-8 tasks.
    depth=3 gives O(n³) — still fine for ≤ 8 tasks.
    """

    def __init__(self, depth: int = 2) -> None:
        self.depth = max(1, min(depth, 4))

    def select_next_task(
        self,
        agent_pos: GridPos,
        engine: MissionEngine,
        distance_fn: callable | None = None,
    ) -> RuntimeTask | None:
        pending = engine.pending_tasks()
        if not pending:
            return None
        if len(pending) == 1:
            return pending[0]

        step = engine.step_count
        remaining_budget = engine.knobs.time_budget - step

        best_first: RuntimeTask | None = None
        best_total_value = -math.inf

        # Evaluate all permutations up to self.depth
        self._search(
            agent_pos=agent_pos,
            pending=pending,
            step=step,
            remaining_budget=remaining_budget,
            depth=self.depth,
            accumulated_value=0.0,
            path_so_far=[],
            distance_fn=distance_fn or _manhattan,
            result={"best_value": -math.inf, "best_first": None},
        )

        return self._last_result.get("best_first")

    def _search(
        self,
        agent_pos: GridPos,
        pending: list[RuntimeTask],
        step: int,
        remaining_budget: int,
        depth: int,
        accumulated_value: float,
        path_so_far: list[RuntimeTask],
        distance_fn: callable,
        result: dict,
    ) -> None:
        self._last_result = result
        if depth == 0 or not pending:
            if accumulated_value > result["best_value"]:
                result["best_value"] = accumulated_value
                result["best_first"] = path_so_far[0] if path_so_far else None
            return

        for t in pending:
            dist = distance_fn(agent_pos, t.xy)
            travel_steps = dist + t.spec.service_time
            if travel_steps > remaining_budget:
                continue
            arrival = step + dist
            elapsed = arrival - t.spec.injected_at
            value = t.spec.weight * math.exp(-t.spec.time_decay * elapsed)

            new_pending = [x for x in pending if x is not t]
            self._search(
                agent_pos=t.xy,
                pending=new_pending,
                step=arrival + t.spec.service_time,
                remaining_budget=remaining_budget - travel_steps,
                depth=depth - 1,
                accumulated_value=accumulated_value + value,
                path_so_far=path_so_far + [t],
                distance_fn=distance_fn,
                result=result,
            )

        # Also consider stopping here (don't visit any more)
        if accumulated_value > result["best_value"]:
            result["best_value"] = accumulated_value
            result["best_first"] = path_so_far[0] if path_so_far else None

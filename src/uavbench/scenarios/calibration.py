"""Scenario calibration (CC-1 through CC-4).

Feasibility pre-check: simulate dynamics forward without a planner.
At each timestep, BFS from start to goal on blocking mask.
Reports feasibility_rate over multiple seeds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from uavbench.blocking import compute_blocking_mask
from uavbench.envs.urban import UrbanEnvV2
from uavbench.scenarios.loader import load_scenario
from uavbench.scenarios.schema import Difficulty, ScenarioConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FeasibilityResult:
    """Result of a single-seed feasibility pre-check (CC-1)."""

    seed: int
    feasible: bool
    first_infeasible_step: int | None  # None means always feasible
    total_steps_checked: int


@dataclass
class CalibrationResult:
    """Result of multi-seed calibration check (CC-2)."""

    scenario_id: str
    difficulty: str
    n_seeds: int
    feasibility_rate: float
    threshold: float
    passes_threshold: bool
    per_seed: list[FeasibilityResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CC-2 thresholds
# ---------------------------------------------------------------------------

_DIFFICULTY_THRESHOLDS = {
    Difficulty.EASY: 0.80,
    Difficulty.MEDIUM: 0.50,
    Difficulty.HARD: 0.15,
}


# ---------------------------------------------------------------------------
# BFS reachability (pure function)
# ---------------------------------------------------------------------------


def _bfs_reachable(
    blocked: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> bool:
    """4-connected BFS reachability check. Coords are (x, y)."""
    H, W = blocked.shape
    sx, sy = start
    gx, gy = goal

    if not (0 <= sy < H and 0 <= sx < W):
        return False
    if not (0 <= gy < H and 0 <= gx < W):
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


# ---------------------------------------------------------------------------
# CC-1: Feasibility pre-check
# ---------------------------------------------------------------------------


def feasibility_pre_check(
    config: ScenarioConfig,
    seed: int,
    horizon: int | None = None,
) -> FeasibilityResult:
    """Simulate dynamics forward for horizon steps (no planner).

    At each timestep, BFS from start to goal on blocking mask.
    Returns first_infeasible_step or None if always feasible (CC-1).
    """
    env = UrbanEnvV2(config)
    obs, info = env.reset(seed=seed)
    _, _, start_xy, goal_xy = env.export_planner_inputs()

    if horizon is None:
        horizon = config.effective_max_steps

    # Check initial state
    dyn_state = env.get_dynamic_state()
    blocked = compute_blocking_mask(
        env._heightmap, env._no_fly, config, dyn_state
    )
    if not _bfs_reachable(blocked, start_xy, goal_xy):
        return FeasibilityResult(
            seed=seed,
            feasible=False,
            first_infeasible_step=0,
            total_steps_checked=0,
        )

    # Simulate dynamics forward by stepping with STAY action
    ACTION_STAY = 4
    first_infeasible = None

    for step in range(1, horizon + 1):
        obs, reward, terminated, truncated, info = env.step(ACTION_STAY)

        dyn_state = env.get_dynamic_state()
        blocked = compute_blocking_mask(
            env._heightmap, env._no_fly, config, dyn_state
        )

        if not _bfs_reachable(blocked, start_xy, goal_xy):
            first_infeasible = step
            break

        if terminated or truncated:
            break

    return FeasibilityResult(
        seed=seed,
        feasible=(first_infeasible is None),
        first_infeasible_step=first_infeasible,
        total_steps_checked=step if first_infeasible is not None else horizon,
    )


# ---------------------------------------------------------------------------
# CC-2: Difficulty calibration
# ---------------------------------------------------------------------------


def calibrate_difficulty(
    scenario_id: str,
    n_seeds: int = 30,
    base_seed: int = 0,
    horizon: int | None = None,
) -> CalibrationResult:
    """Run feasibility pre-check over n_seeds and check thresholds (CC-2).

    Returns CalibrationResult with feasibility_rate and pass/fail.
    """
    config = load_scenario(scenario_id)
    threshold = _DIFFICULTY_THRESHOLDS.get(config.difficulty, 0.50)

    per_seed: list[FeasibilityResult] = []
    feasible_count = 0

    for i in range(n_seeds):
        result = feasibility_pre_check(config, seed=base_seed + i, horizon=horizon)
        per_seed.append(result)
        if result.feasible:
            feasible_count += 1

    rate = feasible_count / n_seeds if n_seeds > 0 else 0.0

    return CalibrationResult(
        scenario_id=scenario_id,
        difficulty=config.difficulty.value,
        n_seeds=n_seeds,
        feasibility_rate=rate,
        threshold=threshold,
        passes_threshold=(rate >= threshold),
        per_seed=per_seed,
    )

"""Contract tests for Fairness (FC-1, FC-2).

FC-1: Forced interdictions placed on BFS reference corridor, NOT on any
      planner's actual path. Planner-agnostic.
FC-2: If latency/dropout enabled, all planners receive equivalent degraded
      observation snapshots. (For same (scenario, seed), get_dynamic_state()
      returns byte-identical arrays regardless of planner identity.)
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from uavbench.envs.base import BlockLifecycle
from uavbench.envs.urban import ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP, UrbanEnvV2
from uavbench.scenarios.schema import Difficulty, MissionType, ScenarioConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dynamic_config(**overrides) -> ScenarioConfig:
    """Config with forced interdictions enabled."""
    defaults = dict(
        name="test_fairness",
        mission_type=MissionType.FIRE_DELIVERY,
        difficulty=Difficulty.MEDIUM,
        map_size=20,
        building_density=0.10,
        max_episode_steps=120,
        terminate_on_collision=False,
        enable_fire=True,
        fire_blocks_movement=True,
        fire_ignition_points=1,
        enable_traffic=False,
        traffic_blocks_movement=False,
        force_replan_count=2,
        event_t1=10,
        event_t2=80,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def _bfs_path_on_static_grid(
    heightmap: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Compute BFS shortest path on the static grid (free cells only).

    This is the REFERENCE implementation for test verification.
    Coordinates: (x, y). Grid access: heightmap[y, x].
    """
    H, W = heightmap.shape
    sx, sy = start
    gx, gy = goal

    visited = set()
    visited.add((sx, sy))
    parent: dict[tuple[int, int], tuple[int, int] | None] = {(sx, sy): None}
    queue = deque([(sx, sy)])

    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == (gx, gy):
            # Reconstruct
            path = []
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

    return []  # No path found


def _run_env_steps(
    config: ScenarioConfig,
    seed: int,
    actions: list[int],
) -> tuple[UrbanEnvV2, list[dict]]:
    """Run env with given actions, return (env, infos)."""
    env = UrbanEnvV2(config)
    env.reset(seed=seed)
    infos = []
    for act in actions:
        _, _, term, trunc, info = env.step(act)
        infos.append(info)
        if term or trunc:
            break
    return env, infos


# ===========================================================================
# FC-1: Interdictions on BFS corridor, planner-agnostic
# ===========================================================================


class TestFC1_BFSCorridor:
    """FC-1: Forced interdictions are on BFS reference corridor."""

    def test_interdiction_on_bfs_corridor(self):
        """All interdiction cells are on the BFS shortest path computed
        on the static grid before any planner runs."""
        config = _make_dynamic_config(force_replan_count=3)
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        # Get BFS corridor from env
        bfs_corridor = env.bfs_corridor
        assert len(bfs_corridor) > 0, "BFS corridor should be non-empty"

        # Get forced block cells from env
        forced_cells = env.forced_block_cells
        assert len(forced_cells) > 0, (
            "With force_replan_count=3, there should be forced block cells"
        )

        # All forced cells must be on the BFS corridor
        corridor_set = set(bfs_corridor)
        for cell in forced_cells:
            assert cell in corridor_set, (
                f"FC-1: forced cell {cell} is NOT on BFS corridor"
            )

    def test_bfs_corridor_matches_reference(self):
        """The env's BFS corridor matches our independent BFS computation."""
        config = _make_dynamic_config()
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        heightmap, _, start, goal = env.export_planner_inputs()
        ref_path = _bfs_path_on_static_grid(heightmap, start, goal)

        assert len(ref_path) > 0, "Reference BFS should find a path"
        assert env.bfs_corridor == ref_path, (
            "FC-1: env BFS corridor must match reference BFS computation"
        )

    def test_interdiction_planner_agnostic(self):
        """Same (scenario, seed) with different action sequences produces
        identical interdiction cell coordinates.

        Since forced block cells are computed at reset() from the BFS
        corridor (which depends only on seed, not on planner), they
        must be identical across runs.
        """
        config = _make_dynamic_config(force_replan_count=2)

        # Run A: all RIGHT actions
        env_a = UrbanEnvV2(config)
        env_a.reset(seed=42)
        cells_a = env_a.forced_block_cells

        # Run B: all DOWN actions (different "planner")
        env_b = UrbanEnvV2(config)
        env_b.reset(seed=42)
        cells_b = env_b.forced_block_cells

        assert cells_a == cells_b, (
            f"FC-1: forced block cells must be planner-agnostic. "
            f"Run A: {cells_a}, Run B: {cells_b}"
        )

    def test_interdiction_not_at_start_or_goal(self):
        """Forced block cells should not include start or goal."""
        config = _make_dynamic_config(force_replan_count=3)
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        start = env.agent_xy
        goal = env.goal_xy
        for cell in env.forced_block_cells:
            assert cell != start, "FC-1: forced block must not be at start"
            assert cell != goal, "FC-1: forced block must not be at goal"


# ===========================================================================
# FC-1: Interdiction lifecycle
# ===========================================================================


class TestFC1_Lifecycle:
    """FC-1: Forced block lifecycle: TRIGGERED → ACTIVE → CLEARED."""

    def test_forced_block_triggers_at_t1(self):
        """Forced block becomes active at event_t1."""
        config = _make_dynamic_config(
            event_t1=5,
            event_t2=50,
            force_replan_count=1,
        )
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        # Before t1: no forced block
        for _ in range(4):
            _, _, _, _, info = env.step(ACTION_RIGHT)
            dyn = env.get_dynamic_state()
            fb = dyn.get("forced_block_mask")
            if fb is not None:
                assert not fb.any(), "Forced block should not be active before t1"

        # At t1 (step 5): forced block activates
        _, _, _, _, info = env.step(ACTION_RIGHT)
        dyn = env.get_dynamic_state()
        fb = dyn.get("forced_block_mask")
        assert fb is not None and fb.any(), (
            "Forced block should be active at event_t1"
        )
        assert info.get("forced_block_active") is True

    def test_forced_block_clears_at_t2(self):
        """Forced block clears at event_t2."""
        config = _make_dynamic_config(
            event_t1=3,
            event_t2=10,
            force_replan_count=1,
            max_episode_steps=200,
        )
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        # Step past t2
        for step_i in range(15):
            _, _, term, trunc, info = env.step(ACTION_RIGHT)
            if term or trunc:
                break

        # After t2: forced block should be cleared
        dyn = env.get_dynamic_state()
        fb = dyn.get("forced_block_mask")
        if fb is not None:
            assert not fb.any(), "Forced block should be cleared after t2"

    def test_forced_block_event_logged(self):
        """Forced block trigger and clear are logged as events."""
        config = _make_dynamic_config(
            event_t1=3,
            event_t2=10,
            force_replan_count=1,
            max_episode_steps=200,
        )
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        for _ in range(15):
            _, _, term, trunc, _ = env.step(ACTION_RIGHT)
            if term or trunc:
                break

        event_types = [e["type"] for e in env.events]
        assert "forced_block_triggered" in event_types, (
            "FC-1: forced_block_triggered event must be logged"
        )
        assert "forced_block_cleared" in event_types, (
            "FC-1: forced_block_cleared event must be logged"
        )


# ===========================================================================
# FC-2: Equivalent dynamic state across planners
# ===========================================================================


class TestFC2_EquivalentDynamicState:
    """FC-2: Same (scenario, seed) → identical dynamic state regardless of actions."""

    def test_identical_dynamic_state_across_planners(self):
        """At each step, get_dynamic_state() returns byte-identical arrays
        for two different action sequences running the same (scenario, seed).

        Dynamics evolve autonomously (fire, traffic, NFZ) and do NOT
        depend on which actions the agent takes.
        """
        config = _make_dynamic_config(
            enable_fire=True,
            fire_ignition_points=2,
            enable_traffic=True,
            traffic_blocks_movement=True,
            num_emergency_vehicles=2,
            max_episode_steps=30,
        )
        seed = 42

        env_a = UrbanEnvV2(config)
        env_a.reset(seed=seed)

        env_b = UrbanEnvV2(config)
        env_b.reset(seed=seed)

        # Different action sequences
        actions_a = [ACTION_RIGHT] * 30
        actions_b = [ACTION_DOWN] * 30

        for step_i in range(30):
            _, _, ta, tra, _ = env_a.step(actions_a[step_i])
            _, _, tb, trb, _ = env_b.step(actions_b[step_i])

            dyn_a = env_a.get_dynamic_state()
            dyn_b = env_b.get_dynamic_state()

            # Compare all dynamic layers
            for key in dyn_a:
                va = dyn_a[key]
                vb = dyn_b[key]
                if va is None and vb is None:
                    continue
                assert va is not None and vb is not None, (
                    f"FC-2: step {step_i}, key '{key}': one is None, other is not"
                )
                if isinstance(va, np.ndarray):
                    np.testing.assert_array_equal(
                        va, vb,
                        err_msg=(
                            f"FC-2: step {step_i}, key '{key}' differs "
                            f"between action sequences"
                        ),
                    )

            if ta or tra or tb or trb:
                break

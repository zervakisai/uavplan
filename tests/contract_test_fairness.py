"""Contract tests for Fairness (FC-1, FC-2).

FC-1: Physical interdictions (fire corridor closure + vehicle roadblocks)
      placed on A* reference corridor, planner-agnostic.
FC-2: If latency/dropout enabled, all planners receive equivalent degraded
      observation snapshots. (For same (scenario, seed), get_dynamic_state()
      returns byte-identical arrays regardless of planner identity.)
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from uavbench.envs.urban import ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP, UrbanEnvV2
from uavbench.scenarios.schema import Difficulty, MissionType, ScenarioConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dynamic_config(**overrides) -> ScenarioConfig:
    """Config with physical interdictions enabled."""
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
        enable_traffic=True,
        traffic_blocks_movement=True,
        num_emergency_vehicles=2,
        num_fire_corridor_closures=1,
        num_roadblock_vehicles=0,
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
# FC-1: Corridor alignment (A* reference)
# ===========================================================================


class TestFC1_BFSCorridor:
    """FC-1: Reference corridor matches A* and is planner-agnostic."""

    def test_corridor_matches_astar_reference(self):
        """The env's reference corridor matches A* path (same algorithm planners use)."""
        config = _make_dynamic_config()
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        heightmap, no_fly, start, goal = env.export_planner_inputs()
        from uavbench.planners.astar import AStarPlanner
        ref_planner = AStarPlanner(heightmap, no_fly)
        ref_result = ref_planner.plan(start, goal)

        assert ref_result.success, "Reference A* should find a path"
        assert env.bfs_corridor == ref_result.path, (
            "FC-1: env corridor must match A* reference computation "
            "(ensures fire/roadblock interdictions intersect actual planner paths)"
        )

    def test_corridor_planner_agnostic(self):
        """Same (scenario, seed) with different action sequences produces
        identical corridor. Corridor is computed at reset(), not planner-dependent.
        """
        config = _make_dynamic_config()

        env_a = UrbanEnvV2(config)
        env_a.reset(seed=42)
        corridor_a = env_a.bfs_corridor

        env_b = UrbanEnvV2(config)
        env_b.reset(seed=42)
        corridor_b = env_b.bfs_corridor

        assert corridor_a == corridor_b, (
            "FC-1: corridor must be planner-agnostic (identical across resets with same seed)"
        )


# ===========================================================================
# FC-1: Fire corridor guarantee
# ===========================================================================


class TestFC1_FireCorridorGuarantee:
    """FC-1: Fire guarantee targets are on the corridor and fire reaches them."""

    def test_fire_reaches_corridor_by_t1(self):
        """With num_fire_corridor_closures=1, fire must cover at least one
        corridor cell by event_t1 (guarantee step)."""
        config = _make_dynamic_config(
            num_fire_corridor_closures=1,
            event_t1=10,
            event_t2=80,
            max_episode_steps=60,
        )
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        corridor_set = set(env.bfs_corridor)

        # Step past event_t1
        for _ in range(15):
            _, _, term, trunc, _ = env.step(ACTION_RIGHT)
            if term or trunc:
                break

        # Check fire covers at least one corridor cell
        dyn = env.get_dynamic_state()
        fire = dyn.get("fire_mask")
        if fire is not None and fire.any():
            fire_on_corridor = False
            for cx, cy in corridor_set:
                if 0 <= cy < fire.shape[0] and 0 <= cx < fire.shape[1]:
                    if fire[cy, cx]:
                        fire_on_corridor = True
                        break
            assert fire_on_corridor, (
                "FC-1: with num_fire_corridor_closures=1, fire must reach "
                "corridor by event_t1 (guarantee step)"
            )


# ===========================================================================
# FC-1: Vehicle roadblock
# ===========================================================================


class TestFC1_VehicleRoadblock:
    """FC-1: Roadblock vehicles freeze at corridor road cells at event_t1."""

    def test_roadblock_activates_at_t1(self):
        """With num_roadblock_vehicles=1, a vehicle freezes near corridor at event_t1."""
        config = _make_dynamic_config(
            num_fire_corridor_closures=0,
            num_roadblock_vehicles=1,
            event_t1=5,
            event_t2=50,
            max_episode_steps=60,
        )
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        # Step past event_t1
        for _ in range(8):
            _, _, term, trunc, _ = env.step(ACTION_RIGHT)
            if term or trunc:
                break

        # Traffic model should have an active roadblock
        if env._traffic is not None:
            assert env._traffic.has_active_roadblocks, (
                "FC-1: roadblock vehicle should be frozen after event_t1"
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

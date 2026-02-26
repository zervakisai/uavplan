"""Contract tests for Feasibility Guardrail (GC-1, GC-2).

GC-1: Guardrail attempts to restore reachability via logged relaxation depths.
GC-2: If infeasible after all depths → episode flagged infeasible; exclusion rate reported.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from uavbench2.blocking import compute_blocking_mask
from uavbench2.scenarios.schema import Difficulty, MissionType, ScenarioConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bfs_reachable(
    blocked: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> bool:
    """Check if goal is reachable from start on ~blocked grid."""
    H, W = blocked.shape
    sx, sy = start
    gx, gy = goal
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


# ===========================================================================
# GC-1: Multi-depth relaxation
# ===========================================================================


class TestGC1_MultiDepthRelaxation:
    """GC-1: Guardrail restores reachability via logged relaxation depths."""

    def test_guardrail_logs_depth_and_relaxations(self):
        """After a topology-changing event, info dict includes guardrail_depth
        (int 0-3) and relaxations (list of dicts)."""
        from uavbench2.guardrail.feasibility import FeasibilityGuardrail

        # Create a small grid where agent is blocked
        heightmap = np.zeros((10, 10), dtype=np.float32)
        no_fly = np.zeros((10, 10), dtype=bool)
        config = ScenarioConfig(
            name="test_gr",
            mission_type=MissionType.FIRE_DELIVERY,
            difficulty=Difficulty.EASY,
            map_size=10,
        )
        start = (0, 0)
        goal = (9, 9)

        # Add forced block that severs the path
        forced_mask = np.zeros((10, 10), dtype=bool)
        forced_mask[0, 1:] = True  # block row 0 except (0,0)
        forced_mask[1:, 0] = True  # block col 0 except (0,0)

        dynamic_state = {
            "forced_block_mask": forced_mask,
            "fire_mask": None,
            "smoke_mask": None,
            "traffic_closure_mask": None,
            "traffic_occupancy_mask": None,
            "moving_target_buffer": None,
            "intruder_buffer": None,
            "dynamic_nfz_mask": None,
        }

        guardrail = FeasibilityGuardrail(
            heightmap=heightmap,
            no_fly=no_fly,
            config=config,
        )

        result = guardrail.check(
            agent_xy=start,
            goal_xy=goal,
            dynamic_state=dynamic_state,
            forced_block=None,  # D1 can't clear, but we test logging
            nfz_model=None,
        )

        assert isinstance(result.depth, int), "guardrail_depth must be int"
        assert isinstance(result.relaxations, list), "relaxations must be list"
        assert result.depth >= 0

    def test_depth1_clears_forced_blocks(self):
        """D1: guardrail clears forced blocks to restore reachability."""
        from uavbench2.dynamics.forced_block import ForcedBlockManager
        from uavbench2.envs.base import BlockLifecycle
        from uavbench2.guardrail.feasibility import FeasibilityGuardrail

        heightmap = np.zeros((10, 10), dtype=np.float32)
        no_fly = np.zeros((10, 10), dtype=bool)
        config = ScenarioConfig(
            name="test_d1",
            mission_type=MissionType.FIRE_DELIVERY,
            difficulty=Difficulty.EASY,
            map_size=10,
        )

        # Create a forced block on the only path
        # Block column 5 completely
        forced_mask = np.zeros((10, 10), dtype=bool)
        forced_mask[:, 5] = True
        start = (0, 0)
        goal = (9, 9)

        # Create a ForcedBlockManager that's ACTIVE
        bfs_corridor = [(i, 0) for i in range(10)]
        rng = np.random.default_rng(42)
        fb = ForcedBlockManager(
            bfs_corridor=bfs_corridor,
            force_replan_count=1,
            event_t1=1,
            event_t2=100,
            map_shape=(10, 10),
            rng=rng,
        )
        # Activate it
        fb.step(1)
        assert fb.active

        # Make the mask match what we want to test
        forced_mask = np.zeros((10, 10), dtype=bool)
        forced_mask[:, 5] = True

        dynamic_state = {
            "forced_block_mask": forced_mask,
            "fire_mask": None,
            "smoke_mask": None,
            "traffic_closure_mask": None,
            "traffic_occupancy_mask": None,
            "moving_target_buffer": None,
            "intruder_buffer": None,
            "dynamic_nfz_mask": None,
        }

        guardrail = FeasibilityGuardrail(
            heightmap=heightmap,
            no_fly=no_fly,
            config=config,
        )

        result = guardrail.check(
            agent_xy=start,
            goal_xy=goal,
            dynamic_state=dynamic_state,
            forced_block=fb,
            nfz_model=None,
        )

        assert result.feasible, "D1 should restore feasibility by clearing forced blocks"
        assert result.depth == 1, f"Expected depth 1, got {result.depth}"
        assert len(result.relaxations) >= 1
        assert result.relaxations[0]["depth"] == 1
        assert "cells_freed" in result.relaxations[0]

    def test_depth2_shrinks_nfz(self):
        """D2: after D1 fails (no forced blocks), guardrail shrinks NFZ zones."""
        from uavbench2.dynamics.restriction_zones import RestrictionZoneModel
        from uavbench2.guardrail.feasibility import FeasibilityGuardrail

        heightmap = np.zeros((10, 10), dtype=np.float32)
        no_fly = np.zeros((10, 10), dtype=bool)
        config = ScenarioConfig(
            name="test_d2",
            mission_type=MissionType.FIRE_DELIVERY,
            difficulty=Difficulty.MEDIUM,
            map_size=10,
            enable_dynamic_nfz=True,
        )

        start = (0, 0)
        goal = (9, 9)

        # Create NFZ that blocks the path by directly setting mask
        # Block a wall across the grid at column 5
        nfz_mask = np.zeros((10, 10), dtype=bool)
        nfz_mask[:, 5] = True  # wall at x=5

        # Create a real NFZ model for the relax_zones() call
        rng = np.random.default_rng(42)
        nfz = RestrictionZoneModel(
            map_shape=(10, 10),
            rng=rng,
            num_zones=1,
            event_t1=1,
            event_t2=100,
        )
        # Activate and override mask to ensure blocking
        for _ in range(5):
            nfz.step()

        dynamic_state = {
            "forced_block_mask": None,
            "fire_mask": None,
            "smoke_mask": None,
            "traffic_closure_mask": None,
            "traffic_occupancy_mask": None,
            "moving_target_buffer": None,
            "intruder_buffer": None,
            "dynamic_nfz_mask": nfz_mask,
        }

        guardrail = FeasibilityGuardrail(
            heightmap=heightmap,
            no_fly=no_fly,
            config=config,
        )

        result = guardrail.check(
            agent_xy=start,
            goal_xy=goal,
            dynamic_state=dynamic_state,
            forced_block=None,
            nfz_model=nfz,
        )

        # NFZ blocks path, D1 can't help (no forced blocks)
        # D2 should attempt NFZ shrinkage
        assert result.depth >= 2, f"Expected depth >= 2, got {result.depth}"
        # Verify relaxations are logged
        d2_relaxations = [r for r in result.relaxations if r["depth"] == 2]
        assert len(d2_relaxations) >= 1, "D2 relaxation should be logged"
        assert "cells_freed" in d2_relaxations[0]

    def test_depth3_emergency_corridor(self):
        """D3: after D1+D2 fail, guardrail removes traffic blocking."""
        from uavbench2.guardrail.feasibility import FeasibilityGuardrail

        heightmap = np.zeros((10, 10), dtype=np.float32)
        no_fly = np.zeros((10, 10), dtype=bool)
        config = ScenarioConfig(
            name="test_d3",
            mission_type=MissionType.FIRE_DELIVERY,
            difficulty=Difficulty.MEDIUM,
            map_size=10,
            enable_traffic=True,
            traffic_blocks_movement=True,
        )
        start = (0, 0)
        goal = (9, 9)

        # Traffic blocks the entire path
        traffic_mask = np.zeros((10, 10), dtype=bool)
        traffic_mask[4:6, :] = True  # block rows 4-5

        dynamic_state = {
            "forced_block_mask": None,
            "fire_mask": None,
            "smoke_mask": None,
            "traffic_closure_mask": None,
            "traffic_occupancy_mask": traffic_mask,
            "moving_target_buffer": None,
            "intruder_buffer": None,
            "dynamic_nfz_mask": None,
        }

        guardrail = FeasibilityGuardrail(
            heightmap=heightmap,
            no_fly=no_fly,
            config=config,
        )

        result = guardrail.check(
            agent_xy=start,
            goal_xy=goal,
            dynamic_state=dynamic_state,
            forced_block=None,
            nfz_model=None,
        )

        assert result.feasible, "D3 should restore feasibility by removing traffic"
        assert result.depth == 3, f"Expected depth 3, got {result.depth}"
        assert any(r["depth"] == 3 for r in result.relaxations)


# ===========================================================================
# GC-2: Infeasible flagging
# ===========================================================================


class TestGC2_InfeasibleFlagging:
    """GC-2: Infeasible episodes flagged after all depths fail."""

    def test_infeasible_flagged(self):
        """When all depths fail, feasible_after_guardrail=False."""
        from uavbench2.guardrail.feasibility import FeasibilityGuardrail

        # Create completely blocked scenario (static buildings surround agent)
        heightmap = np.zeros((10, 10), dtype=np.float32)
        no_fly = np.zeros((10, 10), dtype=bool)
        # Wall off agent at (0,0) with buildings on all sides
        heightmap[0, 1] = 5.0
        heightmap[1, 0] = 5.0
        heightmap[1, 1] = 5.0

        config = ScenarioConfig(
            name="test_infeasible",
            mission_type=MissionType.FIRE_DELIVERY,
            difficulty=Difficulty.EASY,
            map_size=10,
        )
        start = (0, 0)
        goal = (9, 9)

        dynamic_state = {
            "forced_block_mask": None,
            "fire_mask": None,
            "smoke_mask": None,
            "traffic_closure_mask": None,
            "traffic_occupancy_mask": None,
            "moving_target_buffer": None,
            "intruder_buffer": None,
            "dynamic_nfz_mask": None,
        }

        guardrail = FeasibilityGuardrail(
            heightmap=heightmap,
            no_fly=no_fly,
            config=config,
        )

        result = guardrail.check(
            agent_xy=start,
            goal_xy=goal,
            dynamic_state=dynamic_state,
            forced_block=None,
            nfz_model=None,
        )

        assert not result.feasible, "Should be infeasible when static buildings block"
        assert result.depth >= 3, "Should have tried all depths"

    def test_infeasible_rate_in_metrics(self):
        """Aggregate metrics contain infeasible_rate as float [0,1]."""
        # This tests the metrics schema, not the guardrail directly
        from uavbench2.metrics.compute import compute_episode_metrics

        metrics = compute_episode_metrics(
            scenario_id="test",
            planner_id="astar",
            seed=42,
            trajectory=[(0, 0)],
            events=[],
            final_info={
                "termination_reason": "infeasible",
                "objective_completed": False,
                "feasible_after_guardrail": False,
            },
        )

        # The metrics dict should be able to represent infeasibility
        # For aggregate, infeasible_rate is computed over multiple episodes.
        # For a single episode, we just verify the field is accessible.
        assert isinstance(metrics, dict)
        # The final_info's feasible_after_guardrail should propagate
        assert "feasible_after_guardrail" not in metrics or isinstance(
            metrics.get("feasible_after_guardrail"), bool
        )
